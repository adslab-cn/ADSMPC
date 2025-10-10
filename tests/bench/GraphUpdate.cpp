// 在 tests/CNN.cpp 文件顶部，其他函数之前
#define USE_CLEARTEXT
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include "../../nn/backend/FSS_extended.h"
#include <random>
#include <algorithm>
#include "../../crypto/FSS/api/api.h"
#include <chrono>
using Matrix = std::vector<std::vector<GroupElement>>; 

void test_graph_update(int party) {
    std::cerr << "\n\n>> Graph Update Protocol Test - Start" << std::endl;

    // --- 1. 初始化 FSS 后端 (与你的 test_dpf_sort 相同) ---
    using FSSVersion = FSSExtended<u64>;
    FSSVersion *FSS = new FSSVersion();
    FSSConfig::bitlength = 64;
    FSSConfig::party = party;
    // ... 其他初始化 ...
    FSS->init("127.0.0.1", true);

    // --- 2. 定义图的尺寸 ---
    const int N = 128; // 节点数
    const int C = 64;  // 特征维度

    // --- 3. 准备数据 ---
    Matrix A(N, std::vector<GroupElement>(N, 0));
    Matrix F(N, std::vector<GroupElement>(C, 0));
    
    Matrix A_share(N, std::vector<GroupElement>(N, 0));
    Matrix F_share(N, std::vector<GroupElement>(C, 0));

    // DELEAR (或客户端) 初始化明文图
    if (party == DEALER) {
        // 简单初始化，比如创建一个环形图
        for(int i = 0; i < N; ++i) {
            A[i][(i + 1) % N] = 1;
            for(int j=0; j<C; ++j) F[i][j] = i*100 + j;
        }
    }
    
    // --- 4. 秘密共享初始图 ---
    // 你需要一个辅助函数来对二维矩阵进行秘密共享
    // 这里简化处理
    if (party == DEALER) {
        // DEALER 生成份额并发给 SERVER 和 CLIENT
    } else {
        // SERVER 和 CLIENT 接收份额
    }

    // --- 5. 客户端定义更新目标 ---
    int target_node = 5;
    Matrix A_new = A;
    Matrix F_new = F;
    if (party == DEALER) {
        // 修改连接：删除 (5,6) 的边，增加 (5,10) 的边
        A_new[5][6] = 0;
        A_new[5][10] = 1;
        // 修改特征
        F_new[5][0] = 999;
    }

    // --- 6. 客户端生成并分发密钥 ---
    if (party == DEALER) {
        auto key_pairs_A = ...; // 调用 keyGenForUpdate
        auto key_pairs_F = ...;
        // send keys to SERVER and CLIENT
    } else {
        // SERVER/CLIENT recv keys
        std::vector<DPFKeyPack> keys_A = ...;
        std::vector<DPFKeyPack> keys_F = ...;

        // --- 7. 服务器执行不经意更新 ---
        FSS->start();
        obliviousUpdate(party, A_share, F_share, keys_A, keys_F);
        FSS->end();
        
        // --- 8. 验证结果 ---
        // 重构 A_share 和 F_share 来检查更新是否正确
        // ... reconstruct logic for Matrix ...
        
        // 打印 A_new[5][6], A_new[5][10], F_new[5][0] 等关键位置的值进行验证
    }

    FSS->finalize();
    std::cerr << ">> Graph Update Protocol Test - End" << std::endl;
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
    test_graph_update(party);

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