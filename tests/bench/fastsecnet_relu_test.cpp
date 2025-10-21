#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>
#include "../../nn/backend/FSS_extended.h"
#include "../../crypto/FSS/api/api.h"

// 这是一个辅助函数，用于打印数组内容，方便调试
// 我会稍微修改它，让负数（以补码形式存在的大正数）能以更可读的方式打印出来
void print_array_signed(const std::string& title, int party, int size, const GroupElement* arr, int bitlength) {
    if (party == DEALER) return;

    std::cout << "\n--- [Party " << party << "] " << title << " (size=" << size << ") ---" << std::endl;
    for (int i = 0; i < size; ++i) {
        // 将 u64 转换为有符号数来打印
        int64_t signed_val = arr[i];
        // 如果最高位是1，说明是负数
        if (arr[i] & (1ULL << (bitlength - 1))) {
            // 计算其对应的负数值
            signed_val = arr[i] - (1ULL << bitlength);
        }
        std::cout << "  [" << i << "]: " << signed_val << " (raw: " << arr[i] << ")" << std::endl;
    }
}

void print_array_signed(const std::string& title, GroupElement ** mat, int row, int column) {
    if (FSSConfig::party == DEALER) return;
    row = 10;
    column = 10;
    std::cout << "\n--- [Party " << FSSConfig::party << "] " << title << " ---" << std::endl;
    for (int i = 0; i < row; ++i) {
        std::cout << "  Row " << std::setw(2) << i << ": [ ";
        for (int j = 0; j < column; ++j) {
            // 打印为有符号整数，更容易看懂份额
            std::cout << std::setw(5) << static_cast<int64_t>(mat[i][j]) << " ";
        }
        std::cout << "]" << std::endl;
    }
}


// 主测试函数
void test_fast_relu(int party) {
    std::cout << "\n\n>> FastSecNet ReLU Protocol Test - Start" << std::endl;

    using FSSVersion = FSSExtended<u64>;
    FSSVersion *FSS = new FSSVersion();
    FSSConfig::bitlength = 64;
    FSSConfig::party = party;
    FSSConfig::num_threads = 4;
    std::string ip = "127.0.0.1";
    FSS->init(ip, true);
    

    // --- 2. 准备测试数据 ---
    const int size = 10;
    
    // 原始明文输入 (包含正、负、零)
    std::vector<GroupElement> plain_input(size);
    // 期望的明文输出
    std::vector<GroupElement> expected_output(size);

    if (party != DEALER) {
        for (int i = 0; i < size; ++i) {
            int64_t val = i - 5; // 生成数据: -5, -4, ..., 0, 1, ..., 4
            plain_input[i] = (GroupElement)val;
            mod(plain_input[i], FSSConfig::bitlength); // 确保负数被正确转换为环上的元素

            // 计算期望结果
            expected_output[i] = (val > 0) ? (GroupElement)val : 0;
        }
    }

    // 用于存放秘密份额的数组
    GroupElement* input_shares = new GroupElement[size]();
    GroupElement* output_shares = new GroupElement[size]();

    // 打印原始数据用于验证
    if (party != DEALER) {
        print_array_signed("Original Plaintext Input", party, size, plain_input.data(), FSSConfig::bitlength);
        print_array_signed("Expected Plaintext Output", party, size, expected_output.data(), FSSConfig::bitlength);
    }
    
    // --- 3. 秘密分享输入 ---
    std::cout << "   Party " << party << ": Secret sharing inputs..." << std::endl;
    // 这里我们让 SERVER (party 2) 作为数据持有者
    SecretShare(size, plain_input.data(), input_shares, SERVER);
    std::cout << "   Party " << party << ": Secret sharing finished." << std::endl;

    // --- 4. 执行 FastRelu 协议 ---
    // Dealer会生成并发送密钥，Server/Client会接收密钥并执行计算
    FSS::start(); // 开始计时和通信统计
    
    FastRelu(size, input_shares, nullptr, output_shares, nullptr);

    FSS::end();   // 结束计时和通信统计

    // --- 5. 重构并验证结果 ---
    if (party != DEALER) {
        std::cout << "\n   Party " << party << ": Reconstructing final output..." << std::endl;
        reconstruct(size, output_shares, FSSConfig::bitlength);
        
        print_array_signed("Reconstructed Plaintext Output", party, size, output_shares, FSSConfig::bitlength);

        // 验证结果
        bool all_ok = true;
        for (int i = 0; i < size; i++) {
            if (output_shares[i] != expected_output[i]) {
                all_ok = false;
                std::cerr << "  [FAILURE] at index " << i << ": Expected " << expected_output[i] 
                          << ", but got " << output_shares[i] << std::endl;
            }
        }

        std::cout << "\n--- [Party " << party << "] Verification ---" << std::endl;
        if (all_ok) {
            std::cout << "  SUCCESS: All " << size << " elements match the expected ReLU output." << std::endl;
        } else {
            std::cout << "  FAILURE: Output does not match expected values." << std::endl;
        }
    }

    // --- 6. 清理 ---
    delete[] input_shares;
    delete[] output_shares;
    FSS->finalize();
    delete FSS;

    std::cout << ">> FastSecNet ReLU Protocol Test - End\n" << std::endl;
}


// main函数，用于启动不同角色的进程
int main(int argc, char** argv) {
    int party = 0;
    if (argc > 1) {
        party = atoi(argv[1]);
        if (party < 1 || party > 3) {
            std::cerr << "Error: Party ID must be 1 (Dealer), 2 (Server), or 3 (Client)." << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Usage: " << argv[0] << " <party_id> (1=Dealer, 2=Server, 3=Client)" << std::endl;
        return 1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    
    test_fast_relu(party);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (party != DEALER) {
        std::cout << "================================================" << std::endl;
        std::cout << "Total execution time for Party " << party << ": " << duration.count() << " ms" << std::endl;
        std::cout << "================================================" << std::endl;
    }

    return 0;
}