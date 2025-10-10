// 在 tests/CNN.cpp 文件顶部，其他函数之前
#define USE_CLEARTEXT
/*
CNN训练
    fptraining_init()：float point 初始化
    microbenchmark_conv(int party)：conv层测试
    cifar10_fill_images(Tensor4D<T>& trainImages, Tensor<u64> &trainLabels, int datasetOffset = 0)：cifar10数据集处理
    FSS_test_3layer(int party)：三层的CNN网络的端到端训练
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include "../../nn/backend/FSS_extended.h"
#include <random>
#include <algorithm>
#include "../../crypto/FSS/api/api.h"
#include <chrono> // 1. 包含 chrono 头文件

// =================================================================
//                 DPF SORTING PROTOCOL TEST
// =================================================================


void test_dpf_sort(int party) {
    std::cerr << "\n\n>> DPF Route Sort Test - Start" << std::endl;

    // --- 1. 初始化 FSS 后端 ---
    using FSSVersion = FSSExtended<u64>;
    FSSVersion *FSS = new FSSVersion();
    FSSConfig::bitlength = 64;
    FSSConfig::party = party;
    FSSConfig::num_threads = 4;
    std::string ip = "127.0.0.1";
    FSS->init(ip, true);

    const int size = 1;
    const int rank_bw = 14; 
    const int data_bw = 32;

    // --- 3. 准备明文和份额数组 ---
    GroupElement* y_in_plain = new GroupElement[size]();
    GroupElement* y_in_shares = new GroupElement[size]();
    GroupElement* z_in_plain = new GroupElement[size]();
    GroupElement* z_in_shares = new GroupElement[size]();

    if (party != DEALER) {
        std::vector<int> p(size);
        for(int i=0; i<size; ++i) p[i] = i;
        std::mt19937 g(1337); 
        std::shuffle(p.begin(), p.end(), g);
        for(int i=0; i<size; ++i) {
            y_in_plain[i] = p[i];
            z_in_plain[i] = 10000 + p[i];
        }
    }
    print_array("Original Plaintext 'y_in'", party, size, y_in_plain);
    
    std::cerr << "   Party " << party << ": Secret sharing inputs..." << std::endl;
    SecretShare(size, y_in_plain, y_in_shares, SERVER);
    SecretShare(size, z_in_plain, z_in_shares, SERVER);
    std::cerr << "   Party " << party << ": Secret sharing finished." << std::endl;
    // print_array("Secret Shares 'y_in_shares'", party, size, y_in_shares);

    // // --- 5. 重构并验证 ---
    // if (party != DEALER) {
    //     GroupElement* y_in_reconstructed = new GroupElement[size]();
    //     memcpy(y_in_reconstructed, y_in_shares, size * sizeof(GroupElement));
    //     reconstruct(size, y_in_reconstructed, FSSConfig::bitlength);
    //     print_array("Reconstructed Plaintext 'y_in'", party, size, y_in_reconstructed);
    //     bool ok = true;
    //     for (int i = 0; i < size; ++i) {
    //         if (y_in_plain[i] != y_in_reconstructed[i]) {
    //             ok = false;
    //             break;
    //         }
    //     }
    //     std::cout << "\n--- [Party " << party << "] Verification of Secret Sharing ---" << std::endl;
    //     if (ok) std::cout << "  SUCCESS: Reconstructed plaintext matches the original." << std::endl;
    //     else std::cout << "  FAILURE: Reconstructed plaintext does NOT match." << std::endl;
    // }

    
    GroupElement* z_out = new GroupElement[size]();
    GroupElement* y_in_mask = new GroupElement[size]();
    GroupElement* z_in_mask = new GroupElement[size]();
    GroupElement* z_out_mask = new GroupElement[size]();
    FSS::start();
    DpfRoute(
        size,
        y_in_shares, y_in_mask, rank_bw,
        z_in_shares, z_in_mask, data_bw,
        z_out, z_out_mask
    );
    FSS::end();

    if (party != DEALER) {
        reconstruct(size, z_out, data_bw); 
        //reconstruct(size, z_out, bitlength);
        mod_array(z_out,size,data_bw);
        print_array("Original Plaintext 'z'", party, size, z_out,50);
        std::cerr << "\n11\n" << std::endl;
    }
    FSS->finalize();
    //...
}

void fptraining_init() {
    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));
}

int main(int argc, char** argv) {
    // 2. 记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();

    fptraining_init();
    int party = 0;
    if (argc > 1) {
        party = atoi(argv[1]);
    }
    test_dpf_sort(party);

    // 3. 记录结束时间
    auto end_time = std::chrono::high_resolution_clock::now();

    // 4. 计算时间差并打印
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // 为了防止多个参与方都打印时间，可以只让一个 party (例如 party 0) 打印
    if (party == 2) {
        std::cout << "================================================" << std::endl;
        std::cout << "Total execution time: " << duration.count() << " milliseconds" << std::endl;
        std::cout << "Total execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
        std::cout << "================================================" << std::endl;
    }

    return 0;
}