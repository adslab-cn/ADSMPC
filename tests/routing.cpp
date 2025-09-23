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
#include <backend/FSS_extended.h>
#include <backend/FSS_improved.h>
#include <sequential.h>
#include <random>
#include <algorithm>
// =================================================================
//                 DPF SORTING PROTOCOL TEST
// =================================================================

void test_dpf_sort(int party) {
    std::cerr << "\n\n>> DPF Route Sort Test - Start" << std::endl;

    // --- 1. 初始化 FSS 后端 ---
    using FSSVersion = FSSExtended<u64>;
    FSSVersion *FSS = new FSSVersion();
    const u64 scale = 24; // 虽然用不到，但为了与其他代码一致
    FSSConfig::bitlength = 64; // 全局位宽
    FSSConfig::party = party;
    FSSConfig::num_threads = 4;
    std::string ip = "127.0.0.1";
    FSS->init(ip, true);

    // --- 2. 定义测试参数 ---
    const int size = 100; // 待排序的元素数量
    
    // 排名位宽必须能容纳 size-1
    // log2(100) 约等于 6.64，所以需要 7 位
    const int rank_bw = 7; 
    
    // 数据位宽
    const int data_bw = 32;

    // --- 3. 准备输入数据 (只有计算方需要) ---
    // Dealer 不需要知道真实数据
    GroupElement* y_in = new GroupElement[size]; // 排名向量
    GroupElement* z_in = new GroupElement[size]; // 载荷向量
    GroupElement* z_out = new GroupElement[size]; // 输出向量

    if (party != DEALER) {
        std::cerr << "   Party " << party << ": Preparing input data..." << std::endl;
        // 创建一个乱序的排名向量 [0, 1, ..., size-1] 和对应的载荷
        std::vector<int> p(size);
        for(int i=0; i<size; ++i) p[i] = i;
        
        // 使用固定的随机种子确保 Server 和 Client 生成相同的乱序
        std::mt19937 g(1337); 
        std::shuffle(p.begin(), p.end(), g);

        std::cout << "   Original (Rank, Payload) pairs:" << std::endl;
        for(int i=0; i<size; ++i) {
            y_in[i] = p[i]; // 乱序的排名
            z_in[i] = 1000 + p[i]; // 载荷 = 1000 + 排名，方便验证
            if (i < 10) { // 只打印前10个
                 std::cout << "      (" << y_in[i] << ", " << z_in[i] << ")" << std::endl;
            }
        }
        std::cout << "      ..." << std::endl;

        // 对输入数据进行秘密分享
        FSS->inputA(size, y_in); // 默认由 Server (party=2) 输入
        FSS->inputA(size, z_in); // 默认由 Server (party=2) 输入
    }

    // --- 4. 执行协议 ---
    FSS::start();
    
    DpfRoute(
        size,
        y_in, nullptr, rank_bw, // Dealer 端 y_in_mask 是 nullptr
        z_in, nullptr, data_bw, // Dealer 端 z_in_mask 是 nullptr
        z_out, nullptr          // Dealer 端 z_out_mask 是 nullptr
    );

    FSS::end();

    // --- 5. 验证结果 (只有计算方需要) ---
    if (party != DEALER) {
        std::cerr << "   Party " << party << ": Verifying output..." << std::endl;
        bool success = true;
        std::cout << "   Sorted Payload (first 10 elements):" << std::endl;
        for (int i = 0; i < size; ++i) {
            // 验证排好序的载荷 z_out[i] 是否等于 1000 + i
            if (z_out[i] != (GroupElement)(1000 + i)) {
                std::cerr << "   ERROR at index " << i << ": Expected " << (1000 + i) 
                          << ", but got " << z_out[i] << std::endl;
                success = false;
            }
            if (i < 10) {
                std::cout << "      z_out[" << i << "] = " << z_out[i] << std::endl;
            }
        }

        if (success) {
            std::cerr << "   SUCCESS: Sorting was correct!" << std::endl;
        } else {
            std::cerr << "   FAILURE: Sorting failed!" << std::endl;
        }
    }

    // --- 6. 清理 ---
    FSS->finalize();
    delete[] y_in;
    delete[] z_in;
    delete[] z_out;
    delete FSS;

    std::cerr << ">> DPF Route Sort Test - End\n\n" << std::endl;
}

void fptraining_init() {
    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));
}

int main(int argc, char** argv) {
    fptraining_init();
    int party = 0;
    if (argc > 1) {
        party = atoi(argv[1]);
    }
    test_dpf_sort(party);
    return 0;
}