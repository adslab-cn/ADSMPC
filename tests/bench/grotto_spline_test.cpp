#include "../../crypto/FSS/api/api.h"
#include <cstdint> // 或者 <stdint.h>
#define USE_CLEARTEXT
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include "../../nn/backend/FSS_extended.h"
#include <random>
#include <algorithm>
#include <chrono>

// 辅助函数，如果您的代码库没有，需要添加
// Dealer 调用，将一个秘密值拆分为两方的份额
std::pair<GroupElement, GroupElement> splitShare(GroupElement secret, int bw) {
    GroupElement s0 = FSSConfig::prngs[0].get<GroupElement>(); // 假设 Dealer 的 PRNG 是 0
    mod(s0, bw);
    GroupElement s1 = secret - s0;
    mod(s1, bw);
    return {s0, s1};
}

void test_grotto_spline(int party) {
    std::cerr << "\n\n>> Grotto Spline/Interval Test - Start (Party " << party << ")" << std::endl;
    using FSSVersion = FSSExtended<u64>;
    FSSVersion *FSS = new FSSVersion();
    FSSConfig::bitlength = 64;
    FSSConfig::party = party;
    FSSConfig::num_threads = 4;
    FSS->init("127.0.0.1", true); // true 表示使用内存IO
    auto bin = 64;

    // --- 1. 准备数据 ---
    GroupElement x_plain = 50;
    GroupElement a = 30, b = 70;
    GroupElement x_share = 0; // 初始化

    if (party == CLIENT) {
        // 只有 CLIENT 持有明文 x
        SecretShare(1, &x_plain, &x_share, CLIENT);
    } else {
        // 其他方参与 secret sharing 协议，但没有明文输入
        SecretShare(1, nullptr, &x_share, CLIENT);
    }
    


    // --- 4. 执行协议 ---
    OneHotShares result_shares;
    FSS::start();
    if (party != DEALER) {
        three_interval_check(party, x_share, a, b, bin, result_shares);
    }
    FSS::end();

    // --- 5. 验证结果 ---
    if (party != DEALER) {
        GroupElement s_shares[3] = {result_shares.s0, result_shares.s1, result_shares.s2};
        reconstruct(3, s_shares, 1); // 结果是 0/1，位宽为 1

        if (party == SERVER) {
            std::cout << "\n--- Verification ---" << std::endl;
            std::cout << "Plaintext x = " << x_plain << ", a = " << a << ", b = " << b << std::endl;
            std::cout << "Reconstructed one-hot vector: [" 
                      << (int)s_shares[0] << ", " << (int)s_shares[1] << ", " << (int)s_shares[2] << "]\n";
            
            // 验证逻辑
            bool success = false;
            if (x_plain < a && s_shares[0] == 1 && s_shares[1] == 0 && s_shares[2] == 0) success = true;
            if (x_plain >= a && x_plain < b && s_shares[0] == 0 && s_shares[1] == 1 && s_shares[2] == 0) success = true;
            if (x_plain >= b && s_shares[0] == 0 && s_shares[1] == 0 && s_shares[2] == 1) success = true;
            
            std::cout << "Verification result: " << (success ? "SUCCESS!" : "FAILURE!") << std::endl;
        }
    }
    
    FSS->finalize();
}


int main(int argc, char** argv) {
    // 1. 解析命令行参数以确定 party ID
    //    默认是 Dealer (party=1)
    int party = 1; 
    if (argc > 1) {
        party = atoi(argv[1]);
    }
    
    // 检查 party ID 是否有效
    if (party != 1 && party != 2 && party != 3) {
        std::cerr << "Error: Invalid party ID. Must be 1 (Dealer), 2 (Server), or 3 (Client)." << std::endl;
        return 1; // 返回错误码
    }

    // 2. 调用我们的主测试函数
    try {
        test_grotto_spline(party);
    } catch (const std::exception& e) {
        // 捕获并打印任何可能发生的异常
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }

    // 3. 正常退出
    return 0;
}