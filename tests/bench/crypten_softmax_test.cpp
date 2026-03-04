#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cassert>
#include <algorithm>
#include <random>
#include <iomanip>

// 包含你的 MPC 框架核心头文件
#include "../../nn/backend/FSS_extended.h"
#include "../../crypto/FSS/api/api.h"

// --- 辅助函数 ---
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
    std::cout << std::fixed << std::setprecision(8); 
    for (size_t i = 0; i < arr.size() && i < limit; ++i) {
        std::cout << "  [" << i << "]: " << arr[i] << std::endl;
    }
    if (arr.size() > limit) {
        std::cout << "  ..." << std::endl;
    }
}

// =========================================================================
// == 明文参考实现: 标准精确 SOFTMAX (Ground Truth)
// =========================================================================
void plaintext_softmax_precise(const std::vector<double>& input, std::vector<double>& output) {
    if (input.empty()) return;
    output.resize(input.size());

    // 1. 找最大值 (数值稳定性)
    double max_val = input[0];
    for (double val : input) {
        if (val > max_val) max_val = val;
    }

    // 2. 计算 exp(x - max) 并求和
    double sum_exp = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum_exp += output[i];
    }

    // 3. 归一化 (除以和)
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] /= sum_exp;
    }
}

// --- 主测试函数 ---
void test_softmax_crypten(int party) {
    std::cout << "\n\n>> CrypTen-Style Softmax Protocol Test - Start" << std::endl;

    // --- 1. 初始化 MPC 环境 ---
    using FSSVersion = FSSExtended<u64>;
    FSSVersion *FSS = new FSSVersion();
    FSSConfig::bitlength = 64;
    FSSConfig::party = party;
    FSSConfig::num_threads = 4;
    std::string ip = "127.0.0.1";
    FSS->init(ip, true);

    // --- 2. 准备数据 ---
    // Softmax 注意事项：输入范围不宜过大，否则定点数 exp 容易溢出或精度丢失
    // 建议测试范围 [-3.0, 3.0] 或 [-5.0, 5.0]
    const int size = 8;  
    const int scale = 16;       
    
    std::vector<double> plain_input_double;
    std::vector<GroupElement> plain_input_fixed(size);

    if (party == SERVER) {
        plain_input_double = generate_random_double_array(size, -3.0, 3.0);
        for (int i = 0; i < size; ++i) {
            plain_input_fixed[i] = double_to_fixed(plain_input_double[i], scale);
        }
        print_double_array("Input Data (First 10)", plain_input_double, 10);
    }

    // --- 3. 计算期望结果 (Server) ---
    std::vector<double> output_precise(size);

    if (party == SERVER) {
        plaintext_softmax_precise(plain_input_double, output_precise);
        // print_double_array("Expected Softmax Output", output_precise, 5);
    }

    // --- 4. 秘密分享 ---
    GroupElement* input_shares = new GroupElement[size]();
    if(party != DEALER) {
        SecretShare(size, plain_input_fixed.data(), input_shares, SERVER);
    }

    // --- 5. 执行 MPC 协议 ---
    GroupElement* output_shares = new GroupElement[size]();
    GroupElement* in_mask = new GroupElement[size](); // 如果 api 需要 mask 数组
    GroupElement* out_mask = new GroupElement[size]();
    
    std::cout << "\n   Party " << party << ": Starting secure computation..." << std::endl;
    FSS::start();
    auto start_time = std::chrono::high_resolution_clock::now();

    // ==========================================================
    // 调用我们刚才实现的 CrypTen 风格 Softmax
    // 注意：Mask 参数根据你的 MASK_PAIR 宏定义传入
    // 这里的 input_shares 对应 MASK_PAIR 的第一项，in_mask 对应第二项
    // ==========================================================
    SoftmaxCrypTenStyle(size, input_shares, in_mask, output_shares, out_mask, scale);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    FSS::end();
    std::cout<<"   Party " << party << ": Secure computation finished in "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
        << " ms." << std::endl;

    // --- 6. 重构与验证 ---
    if(party != DEALER){
        std::cout << "\n   Party " << party << ": Reconstructing..." << std::endl;
        reconstruct(size, output_shares, FSSConfig::bitlength);
    }
    
    if (party == SERVER) {
        std::vector<double> output_mpc(size);
        double sum_check = 0.0;
        for (int i = 0; i < size; ++i) {
            output_mpc[i] = fixed_to_double(output_shares[i], scale);
            sum_check += output_mpc[i];
        }

        std::cout << "\n   [Check] Sum of MPC Softmax Output: " << sum_check << " (Should be close to 1.0)" << std::endl;

        // --- 详细对比分析 ---
        std::cout << "\n================ COMPARISON REPORT ================" << std::endl;
        std::cout << std::setw(6) << "Idx" 
                  << std::setw(15) << "Precise" 
                  << std::setw(15) << "MPC_Result" 
                  << std::setw(15) << "Error"   // |Precise - MPC|
                  << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;

        double max_error = 0.0;
        double total_error = 0.0;

        for (int i = 0; i < size; ++i) {
            double err = std::abs(output_precise[i] - output_mpc[i]);
            max_error = std::max(max_error, err);
            total_error += err;

            if (i < 15) { // 打印前 15 行
                std::cout << std::setw(6) << i 
                          << std::setw(15) << output_precise[i] 
                          << std::setw(15) << output_mpc[i] 
                          << std::setw(15) << err 
                          << std::endl;
            }
        }
        std::cout << "..." << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;
        std::cout << "Max Error: " << std::fixed << std::setprecision(8) << max_error << std::endl;
        std::cout << "Avg Error: " << std::fixed << std::setprecision(8) << (total_error / size) << std::endl;
        
        // 简单的 Pass/Fail 判断 (阈值取决于 scale 和迭代次数，scale=16 时 1e-3 到 1e-4 是合理的)
        if (max_error < 1e-2) { 
            std::cout << "\n[SUCCESS] Protocol implementation looks correct!" << std::endl;
        } else {
            std::cout << "\n[WARNING] Error might be too high. Check scale or iterations." << std::endl;
        }
        std::cout << "===================================================" << std::endl;
    }

    // --- 7. 清理 ---
    delete[] input_shares;
    delete[] output_shares;
    delete[] in_mask;
    delete[] out_mask;
    FSS->finalize();
    delete FSS;

    std::cout << ">> CrypTen-Style Softmax Test - End\n" << std::endl;
}

// --- main 函数 ---
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

    test_softmax_crypten(party);

    return 0;
}