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

// =================================================================
//                 辅助函数 (Helper Functions)
// =================================================================
void printMatrix(const std::string& title, const Matrix& mat) {
    if (FSSConfig::party == DEALER) return;
    
    std::cout << "\n--- [Party " << FSSConfig::party << "] " << title << " ---" << std::endl;
    if (mat.empty()) {
        std::cout << "  (Matrix is empty)" << std::endl;
        return;
    }
    for (int i = 0; i < mat.size(); ++i) {
        std::cout << "  Row " << std::setw(2) << i << ": [ ";
        for (int j = 0; j < mat[0].size(); ++j) {
            // 打印为有符号整数，更容易看懂份额
            std::cout << std::setw(5) << static_cast<int64_t>(mat[i][j]) << " ";
        }
        std::cout << "]" << std::endl;
    }
}

// 辅助函数：对二维矩阵进行秘密共享 (基于你的 SecretShare)
void secretShareMatrix(const Matrix& plain, Matrix& share, int owner) {
    if (FSSConfig::party == DEALER) {
        // Dealer 不持有明文，也不需要份额，但需要参与 PRNG 的同步
        int rows = plain.size();
        int cols = plain[0].size();
        SecretShare(rows * cols, nullptr, nullptr, owner);
        return;
    }

    if (plain.empty()) return;
    int rows = plain.size();
    int cols = plain[0].size();
    int size = rows * cols;

    std::vector<GroupElement> plain_flat(size);
    std::vector<GroupElement> share_flat(size);

    if (FSSConfig::party == owner) {
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                plain_flat[i * cols + j] = plain[i][j];
            }
        }
    }
    
    // 调用你已有的、经过测试的 SecretShare 函数
    SecretShare(size, plain_flat.data(), share_flat.data(), owner);

    // 将一维份额转换回二维
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            share[i][j] = share_flat[i * cols + j];
        }
    }
}

// 辅助函数：重构二维矩阵 (基于你的 reconstruct)
void reconstructMatrix(Matrix& share) {
    if (FSSConfig::party == DEALER) return;
    if (share.empty()) return;
    int rows = share.size();
    int cols = share[0].size();
    int size = rows * cols;
    
    std::vector<GroupElement> share_flat(size);
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            share_flat[i * cols + j] = share[i][j];
        }
    }
    
    // 调用全局的 reconstruct 函数
    reconstruct(size, share_flat.data(), FSSConfig::bitlength);
    
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            share[i][j] = share_flat[i * cols + j];
        }
    }
}



// =================================================================
//                 主测试函数 (Main Test Function)
// =================================================================

void test_graph_update(int party) {
    std::cerr << "\n\n>> Graph Update Protocol Test - Start (Party " << party << ")" << std::endl;

    // --- 1. 初始化 FSS 后端 ---
    // 完全复用你的 test_dpf_sort 中的初始化逻辑
    using FSSVersion = FSSExtended<u64>;
    FSSVersion *FSS = new FSSVersion();
    FSSConfig::bitlength = 64;
    FSSConfig::party = party;
    FSSConfig::num_threads = 4;
    FSS->init("127.0.0.1", true); // true 表示使用内存IO

    // --- 2. 定义图的尺寸和参数 ---
    const int N = 128;
    const int C = 64;
    const int A_bw = static_cast<int>(ceil(log2(N)));
    const int F_bw = A_bw;
    const int A_data_bw = 64;
    const int F_data_bw = 64;
    int target_node_to_update = 5;

    // --- 3. 准备数据 ---
    Matrix A_old(N, std::vector<GroupElement>(N, 0));
    Matrix F_old(N, std::vector<GroupElement>(C, 0));
    Matrix A_new(N, std::vector<GroupElement>(N, 0));
    Matrix F_new(N, std::vector<GroupElement>(C, 0));
    Matrix A_share(N, std::vector<GroupElement>(N, 0));
    Matrix F_share(N, std::vector<GroupElement>(C, 0));

    Matrix A_temp(N, std::vector<GroupElement>(N, 0));
    Matrix F_temp(N, std::vector<GroupElement>(C, 0));

    // DEALER 和 SERVER (作为 owner) 都需要明文
    if (party == DEALER || party == SERVER) {
        std::cout << "[Party " << party << "] Initializing plaintext graphs..." << std::endl;
        for(int i = 0; i < N; ++i) {
            A_old[i][(i + 1) % N] = 1;
            for(int j = 0; j < C; ++j) F_old[i][j] = i * 100 + j;
        }
        A_new = A_old; F_new = F_old;
        A_new[target_node_to_update][(target_node_to_update + 1) % N] = 0;
        A_new[target_node_to_update][10] = 1;
        F_new[target_node_to_update][0] = 999;
    }

    // --- 4. 秘密共享初始图 ---
    // 假设 SERVER 是旧图的持有者 (owner)
    std::cout << "[Party " << party << "] Secret sharing initial graph..." << std::endl;
    secretShareMatrix(A_old, A_share, SERVER);
    secretShareMatrix(F_old, F_share, SERVER);

    A_temp = A_old;
    F_temp = F_old;
    reconstructMatrix(A_temp);
    reconstructMatrix(F_temp);
    if (party == SERVER) {
        printMatrix("A_old (Plaintext)", A_old);
        printMatrix("F_old (Plaintext)", F_old);
        printMatrix("A_new (Target)", A_new);
        printMatrix("F_new (Target)", F_new);
        printMatrix("A_old (Reconstructed)", A_temp);
        printMatrix("F_old (Reconstructed)", F_temp);
    }
    
    // --- 5. 执行协议并计时 ---
    FSS::start(); // 开始计时和通信统计

    obliviousGraphUpdate(
        party, target_node_to_update,
        A_old, A_new, A_bw, A_data_bw,
        F_old, F_new, F_bw, F_data_bw,
        A_share, F_share
    );

    FSS::end(); // 结束计时和通信统计

    // --- 6. 验证结果 ---
    if (party != DEALER) {
        std::cout << "[Party " << party << "] Reconstructing results for verification..." << std::endl;
        
        reconstructMatrix(A_share);
        reconstructMatrix(F_share);

        if (party == SERVER) { // 只有一方打印验证结果
            std::cout << "\n\n--- Verification of Graph Update ---" << std::endl;
            
            bool success = true;
            // 检查被修改的点
            if (A_share[target_node_to_update][(target_node_to_update + 1) % N] != 0) success = false;
            if (A_share[target_node_to_update][10] != 1) success = false;
            if (F_share[target_node_to_update][0] != 999) success = false;
            
            // 检查一个未被修改的点，确保它保持原样
            int other_node = target_node_to_update + 2;
            if (A_share[other_node][(other_node + 1) % N] != 1) success = false;
            if (F_share[other_node][0] != other_node * 100) success = false;
            
            std::cout << "  Verification result: " << (success ? "SUCCESS!" : "FAILURE!") << std::endl;
        }
        if (party == SERVER) {
            printMatrix("A_new (Result)", A_share);
            printMatrix("F_new (Result)", F_share);
        }
    }
    
    FSS->finalize();
    std::cerr << ">> Graph Update Protocol Test - End (Party " << party << ")" << std::endl;
}
void fptraining_init() {
    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));
}
// 在你的 main 函数中
int main(int argc, char** argv) {
    fptraining_init();
    int party = 0;
    if (argc > 1) {
        party = atoi(argv[1]);
    }

    // 将 main 函数中的调用从 test_dpf_sort 改为 test_graph_update
    test_graph_update(party);

    return 0;
}