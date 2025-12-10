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
    const int size = 128;       // 向量大小
    const int scale = 16;       // 定点数小数位数
    const int taylor_n = 6;     // Bumblebee论文中的参数 n=6
    
    std::vector<double> plain_input_double;
    std::vector<GroupElement> plain_input_fixed(size);

    if (party == SERVER) {
        // Server生成随机输入数据
        plain_input_double = generate_random_double_array(size, -8.0, 8.0);
        for (int i = 0; i < size; ++i) {
            plain_input_fixed[i] = double_to_fixed(plain_input_double[i], scale);
        }
    }

    // --- 3. 计算期望结果 ---
    std::vector<double> expected_output_double(size);
    if (party == SERVER) {
        plaintext_softmax_bumblebee(plain_input_double, expected_output_double, taylor_n);
        print_double_array("Plaintext Input (double)", plain_input_double);
        print_double_array("Expected Plaintext Output (double, after BumbleBee)", expected_output_double);
    }

    // --- 4. 秘密分享 ---
    GroupElement* input_shares = new GroupElement[size]();
    if(party != DEALER) {
        // Server持有明文，并负责分发
        SecretShare(size, plain_input_fixed.data(), input_shares, SERVER);
    }

    // --- 5. 执行MPC协议 ---
    GroupElement* output_shares = new GroupElement[size]();
    // 准备空的掩码数组给Dealer使用
    GroupElement* in_mask = new GroupElement[size]();
    GroupElement* out_mask = new GroupElement[size]();
    
    std::cout << "\n   Party " << party << ": Starting secure computation..." << std::endl;
    FSS::start();
    auto start_time = std::chrono::high_resolution_clock::now();

    SoftmaxBumbleBee(size, input_shares, in_mask, output_shares, out_mask, scale, taylor_n);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    FSS::end();
    std::cout << "   Party " << party << ": Secure computation finished." << std::endl;

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    if (party != DEALER) {
        std::cout << "================================================" << std::endl;
        std::cout << "Total MPC execution time: " << duration.count() << " milliseconds" << std::endl;
        std::cout << "================================================" << std::endl;
    }
    
    // --- 6. 重构与验证 ---
    if(party != DEALER){
        std::cout << "\n   Party " << party << ": Reconstructing final output..." << std::endl;
        reconstruct(size, output_shares, FSSConfig::bitlength);
    }
    
    if (party == SERVER) {
        // Server (Party 0) 拥有了重构后的明文结果，可以进行验证
        std::vector<double> mpc_output_double(size);
        for (int i = 0; i < size; ++i) {
            mpc_output_double[i] = fixed_to_double(output_shares[i], scale);
        }
        print_double_array("MPC Reconstructed Output (double)", mpc_output_double);

        // 验证
        bool all_ok = true;
        double max_error = 0.0;
        // 误差容限。逼近算法和定点数都会引入误差，所以容限需要大一些
        const double tolerance = 1e-3; // 0.001

        for (int i = 0; i < size; i++) {
            double error = std::abs(mpc_output_double[i] - expected_output_double[i]);
            if (error > max_error) {
                max_error = error;
            }
            if (error > tolerance) {
                all_ok = false;
                std::cerr << "  [FAILURE] at index " << i << ": Expected " << expected_output_double[i] 
                          << ", but MPC got " << mpc_output_double[i] << " (Error: " << error << ")" << std::endl;
            }
        }

        std::cout << "\n--- [Party " << party << "] Verification ---" << std::endl;
        std::cout << "  Max error found: " << max_error << std::endl;
        std::cout << "  Acceptable tolerance: " << tolerance << std::endl;
        if (all_ok) {
            std::cout << "  SUCCESS: All elements are within the acceptable error tolerance." << std::endl;
        } else {
            std::cout << "  FAILURE: One or more elements exceeded the error tolerance." << std::endl;
        }
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