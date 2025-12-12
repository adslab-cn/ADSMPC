#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cassert>
#include <algorithm>
#include <random>
#include <iomanip> // 为了 std::fixed 和 std::setprecision

// 包含您的MPC框架的核心头文件
#include "../../nn/backend/FSS_extended.h"
#include "../../crypto/FSS/api/api.h"

// --- 辅助函数 (与您提供的文件相同) ---
std::vector<double> generate_random_double_array(size_t size, double min_val, double max_val) {
    std::vector<double> data;
    data.reserve(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(min_val, max_val);
    for (size_t i = 0; i < size; ++i) {
        data.push_back(distrib(gen));
    }
    return data;
}

void print_double_array(const std::string& title, const std::vector<double>& arr, int limit = 10) {
    std::cout << "\n--- " << title << " ---" << std::endl;
    std::cout << std::fixed << std::setprecision(8); // 设置输出精度
    for (size_t i = 0; i < arr.size() && i < limit; ++i) {
        std::cout << "  [" << i << "]: " << arr[i] << std::endl;
    }
    if (arr.size() > limit) {
        std::cout << "  ..." << std::endl;
    }
}

// =========================================================================
// == 明文参考实现: BUMBLEBEE SOFTMAX
// =========================================================================
void plaintext_softmax_bumblebee(
    const std::vector<double>& input, 
    std::vector<double>& output,
    int taylor_n)
{
    if (input.empty()) return;

    // 1. 找到最大值
    double max_val = *std::max_element(input.begin(), input.end());

    // 2. 中心化并计算指数
    std::vector<double> exp_values(input.size());
    double sum_exp = 0.0;
    
    for (size_t i = 0; i < input.size(); ++i) {
        double x_prime = input[i] - max_val;
        
        // 3. 使用 (1 + x'/2^n)^(2^n) 逼近 exp(x')
        // 注意：这里没有T_exp裁剪，与我们简化的MPC协议保持一致
        double term = x_prime / (1 << taylor_n);
        double base = 1.0 + term;
        // 重复平方n次
        double taylor_res = base;
        for(int j = 0; j < taylor_n; ++j) {
            taylor_res *= taylor_res;
        }
        
        exp_values[i] = taylor_res;
        sum_exp += exp_values[i];
    }
    
    // 4. 除以总和
    if (sum_exp == 0.0) sum_exp = 1e-9; // 避免除以零
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = exp_values[i] / sum_exp;
    }
}
// =========================================================================
// == 明文参考实现: 标准精确 SOFTMAX (Ground Truth)
// =========================================================================
void plaintext_softmax_precise(const std::vector<double>& input, std::vector<double>& output) {
    if (input.empty()) return;

    // 1. 找到最大值 (为了数值稳定性，防止 exp 溢出)
    double max_val = *std::max_element(input.begin(), input.end());

    // 2. 计算标准 exp
    std::vector<double> exp_values(input.size());
    double sum_exp = 0.0;
    
    for (size_t i = 0; i < input.size(); ++i) {
        // 使用标准库的 std::exp
        exp_values[i] = std::exp(input[i] - max_val);
        sum_exp += exp_values[i];
    }
    
    // 3. 归一化
    if (sum_exp == 0.0) sum_exp = 1e-9;
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = exp_values[i] / sum_exp;
    }
}
// --- 主测试函数 ---

void test_softmax_bumblebee(int party) {
    std::cout << "\n\n>> BumbleBee Softmax Protocol Test - Start" << std::endl;

    // --- 1. 初始化MPC环境 ---
    using FSSVersion = FSSExtended<u64>;
    FSSVersion *FSS = new FSSVersion();
    FSSConfig::bitlength = 64;
    FSSConfig::party = party;
    FSSConfig::num_threads = 4;
    std::string ip = "127.0.0.1";
    FSS->init(ip, true);

    // --- 2. 准备数据 ---
    const int size = 2048;       // 建议先用小一点的 size (如 128) 方便看日志，没问题了再开 2048
    const int scale = 16;       
    const int taylor_n = 6;     
    
    std::vector<double> plain_input_double;
    std::vector<GroupElement> plain_input_fixed(size);

    if (party == SERVER) {
        // 生成随机输入 (-8 到 8 是 BumbleBee 优化较好的区间，超过这个区间误差会增大)
        plain_input_double = generate_random_double_array(size, -8.0, 8.0);
        for (int i = 0; i < size; ++i) {
            plain_input_fixed[i] = double_to_fixed(plain_input_double[i], scale);
        }
    }

    // --- 3. 计算期望结果 (Server) ---
    std::vector<double> output_precise(size);
    std::vector<double> output_bumblebee_logic(size);

    if (party == SERVER) {
        // A. 计算标准精确值
        plaintext_softmax_precise(plain_input_double, output_precise);
        
        // B. 计算 BumbleBee 逻辑模拟值
        plaintext_softmax_bumblebee(plain_input_double, output_bumblebee_logic, taylor_n);

        // 打印输入
        print_double_array("Input Data", plain_input_double, 5);
    }

    // --- 4. 秘密分享 ---
    GroupElement* input_shares = new GroupElement[size]();
    if(party != DEALER) {
        SecretShare(size, plain_input_fixed.data(), input_shares, SERVER);
    }

    // --- 5. 执行MPC协议 ---
    GroupElement* output_shares = new GroupElement[size]();
    GroupElement* in_mask = new GroupElement[size]();
    GroupElement* out_mask = new GroupElement[size]();
    
    std::cout << "\n   Party " << party << ": Starting secure computation..." << std::endl;
    FSS::start();
    auto start_time = std::chrono::high_resolution_clock::now();

    // 你的核心调用
    SoftmaxBumbleBee(size, input_shares, in_mask, output_shares, out_mask, scale, taylor_n);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    FSS::end();
    
    // --- 6. 重构与验证 ---
    if(party != DEALER){
        std::cout << "\n   Party " << party << ": Reconstructing..." << std::endl;
        reconstruct(size, output_shares, FSSConfig::bitlength);
    }
    
    if (party == SERVER) {
        std::vector<double> output_mpc(size);
        for (int i = 0; i < size; ++i) {
            output_mpc[i] = fixed_to_double(output_shares[i], scale);
        }

        // --- 详细对比分析 ---
        std::cout << "\n================ COMPARISON REPORT ================" << std::endl;
        std::cout << std::setw(6) << "Idx" 
                  << std::setw(15) << "Precise" 
                  << std::setw(15) << "BB_Logic" 
                  << std::setw(15) << "MPC_Result" 
                  << std::setw(15) << "Alg_Err"   // |Precise - BB_Logic|
                  << std::setw(15) << "MPC_Err"   // |BB_Logic - MPC|
                  << std::endl;
        std::cout << "---------------------------------------------------------------------------------" << std::endl;

        double max_alg_error = 0.0;
        double max_mpc_error = 0.0;
        double max_total_error = 0.0;

        for (int i = 0; i < size; ++i) {
            double alg_err = std::abs(output_precise[i] - output_bumblebee_logic[i]);
            double mpc_err = std::abs(output_bumblebee_logic[i] - output_mpc[i]);
            double total_err = std::abs(output_precise[i] - output_mpc[i]);

            max_alg_error = std::max(max_alg_error, alg_err);
            max_mpc_error = std::max(max_mpc_error, mpc_err);
            max_total_error = std::max(max_total_error, total_err);

            if (i < 10) { // 只打印前10行详细数据
                std::cout << std::setw(6) << i 
                          << std::setw(15) << output_precise[i] 
                          << std::setw(15) << output_bumblebee_logic[i] 
                          << std::setw(15) << output_mpc[i] 
                          << std::setw(15) << alg_err 
                          << std::setw(15) << mpc_err 
                          << std::endl;
            }
        }
        std::cout << "..." << std::endl;
        std::cout << "---------------------------------------------------------------------------------" << std::endl;
        std::cout << "Max Algorithm Error (Logic vs Precise): " << std::fixed << std::setprecision(8) << max_alg_error << std::endl;
        std::cout << "Max MPC Impl Error  (MPC vs Logic)    : " << std::fixed << std::setprecision(8) << max_mpc_error << std::endl;
        std::cout << "Max Total Error     (MPC vs Precise)  : " << std::fixed << std::setprecision(8) << max_total_error << std::endl;
        std::cout << "===================================================" << std::endl;
    }

    // --- 7. 清理 ---
    delete[] input_shares;
    delete[] output_shares;
    delete[] in_mask;
    delete[] out_mask;
    FSS->finalize();
    delete FSS;

    std::cout << ">> BumbleBee Softmax Protocol Test - End\n" << std::endl;
}

// --- main函数 (与您提供的文件相同) ---
int main(int argc, char** argv) {
    int party = 0;
    if (argc > 1) {
        party = atoi(argv[1]);
        if (party < 1 || party > 3) {
            std::cerr << "Error: Party ID must be 1, 2, or 3." << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Usage: " << argv[0] << " <party_id> (1=Dealer, 2=Server, 3=Client)" << std::endl;
        return 1;
    }

    test_softmax_bumblebee(party);

    return 0;
}