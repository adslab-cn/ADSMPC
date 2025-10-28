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

// =================================================================
//                 辅助函数 (Helper Functions)
// =================================================================
void printMatrix(const std::string& title, GroupElement ** mat, int row, int column) {
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

// 辅助函数：对二维矩阵进行秘密共享 (基于你的 SecretShare)
void secretShareMatrix( int rows, int cols, GroupElement ** plain, GroupElement ** share, int owner) {
    if (FSSConfig::party == DEALER) {
        SecretShare(rows * cols, nullptr, nullptr, owner);
        return;
    }

    int size = rows * cols;

    GroupElement *plain_flat = new GroupElement[size];
    GroupElement* share_flat = new GroupElement[size];

    if (FSSConfig::party == owner) {
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                plain_flat[i * cols + j] = plain[i][j];
            }
        }
    }
    
    // 调用你已有的、经过测试的 SecretShare 函数
    SecretShare(size, plain_flat, share_flat, owner);

    // 将一维份额转换回二维
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            share[i][j] = share_flat[i * cols + j];
        }
    }
}

// 辅助函数：重构二维矩阵 (基于你的 reconstruct)
void reconstructMatrix(int rows, int cols, GroupElement ** share) {
    if (FSSConfig::party == DEALER) return;
    int size = rows * cols;
    
    GroupElement* share_flat = new GroupElement[size];
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            share_flat[i * cols + j] = share[i][j];
        }
    }
    
    // 调用全局的 reconstruct 函数
    reconstruct(size, share_flat, FSSConfig::bitlength);
    
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            share[i][j] = share_flat[i * cols + j];
        }
    }
}


GroupElement** allocateMatrix(int rows, int cols) {
    if (rows == 0 || cols == 0) return nullptr;
    GroupElement** mat = new GroupElement*[rows];
    for (int i = 0; i < rows; ++i) {
        mat[i] = new GroupElement[cols](); // () for zero-initialization
    }
    return mat;
}
 

/**
 * @brief 对一个 GroupElement** C-style 二维数组进行深拷贝。
 * 
 * @param src 要拷贝的源矩阵。
 * @param rows 源矩阵的行数。
 * @param cols 源矩阵的列数。
 * @return GroupElement** 指向新创建的、完全独立的矩阵副本的指针。
 *         如果源指针为 null 或维度无效，则返回 nullptr。
 */
GroupElement** deepCopyMatrix(GroupElement** src, int rows, int cols) {
    // --- 安全检查 ---
    // 如果源指针为空或维度无效，则无法进行拷贝。
    if (!src || rows <= 0 || cols <= 0) {
        return nullptr;
    }

    // --- 步骤 1: 分配外层数组 (指针数组) ---
    // 这个数组将持有指向每一行的指针。
    GroupElement** dest = new GroupElement*[rows];

    // --- 步骤 2: 循环分配每一行并复制数据 ---
    for (int i = 0; i < rows; ++i) {
        // 为目标矩阵的第 i 行分配内存。
        dest[i] = new GroupElement[cols];

        // 检查源矩阵的当前行是否为空指针，增加健壮性。
        if (!src[i]) {
            std::cerr << "Error: Source matrix has a null row at index " << i << std::endl;
            // 清理已分配的内存以避免泄漏
            for (int k = 0; k < i; ++k) {
                delete[] dest[k];
            }
            delete[] dest;
            return nullptr;
        }

        // 使用 memcpy 高效地将整行数据从源复制到目标。
        // 这通常比逐个元素复制的 for 循环更快。
        memcpy(dest[i], src[i], cols * sizeof(GroupElement));
    }

    return dest;
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
    const int N = 3000;
    const int C = 1000;
    const int A_bw = static_cast<int>(ceil(log2(N)));
    const int F_bw = A_bw;
    const int A_data_bw = 64;
    const int F_data_bw = 64;
    int target_node_to_update = 1;

    // --- 3. 准备数据 ---
    GroupElement ** A_old = allocateMatrix(N,N);
    GroupElement ** F_old = allocateMatrix(N,C);
    GroupElement ** A_new = allocateMatrix(N,N);
    GroupElement ** F_new = allocateMatrix(N,C);
    GroupElement ** A_share = allocateMatrix(N,N);
    GroupElement ** F_share = allocateMatrix(N,C);

    GroupElement ** A_temp = allocateMatrix(N,N);
    GroupElement ** F_temp = allocateMatrix(N,C);

    // DEALER 和 SERVER (作为 owner) 都需要明文
    if (party == DEALER || party == SERVER) {
        std::cout << "[Party " << party << "] Initializing plaintext graphs..." << std::endl;
        for(int i = 0; i < N; ++i) {
            A_old[i][(i + 1) % N] = 1;
            for(int j = 0; j < C; ++j) F_old[i][j] = i * 100 + j;
        }
        A_new = deepCopyMatrix(A_old,N,N); F_new = deepCopyMatrix(F_old,N,C);
        A_new[target_node_to_update][(target_node_to_update + 1) % N] = 0;
        A_new[target_node_to_update][0] = 1;
        F_new[target_node_to_update][0] = 999;
    }

    // --- 4. 秘密共享初始图 ---
    // 假设 SERVER 是旧图的持有者 (owner)
    std::cout << "[Party " << party << "] Secret sharing initial graph..." << std::endl;
    secretShareMatrix(N,N,A_old, A_share, SERVER);
    secretShareMatrix(N,C,F_old, F_share, SERVER);

    A_temp = deepCopyMatrix(A_old,N,N);
    F_temp = deepCopyMatrix(F_old,N,C);
    reconstructMatrix(N,N,A_temp);
    reconstructMatrix(N,C,F_temp);
    if (party == SERVER) {
        printMatrix("A_old (Plaintext)", A_old,N,N);
        printMatrix("F_old (Plaintext)", F_old,N,C);
        printMatrix("A_new (Target)", A_new,N,N);
        printMatrix("F_new (Target)", F_new,N,C);
        printMatrix("A_old (Reconstructed)", A_temp,N,N);
        printMatrix("F_old (Reconstructed)", F_temp,N,C);
    }
    
    // --- 5. 执行协议并计时 ---
    FSS::start(); // 开始计时和通信统计
    auto start_time = std::chrono::high_resolution_clock::now();
    obliviousGraphUpdate(
        party, target_node_to_update,
        N, C,
        A_old, A_new, A_bw, A_data_bw,
        F_old, F_new, F_bw, F_data_bw,
        A_share, F_share
    );



    // --- 6. 验证结果 ---
    if (party != DEALER) {
        std::cout << "[Party " << party << "] Reconstructing results for verification..." << std::endl;
        
        reconstructMatrix(N,N,A_share);
        reconstructMatrix(N,C,F_share);

        if (party == SERVER) { // 只有一方打印验证结果
            std::cout << "\n\n--- Verification of Graph Update ---" << std::endl;
            
            bool success = true;
            // 检查被修改的点
            if (A_share[target_node_to_update][(target_node_to_update + 1) % N] != 0) success = false;
            if (A_share[target_node_to_update][0] != 1) success = false;
            if (F_share[target_node_to_update][0] != 999) success = false;
            
            // 检查一个未被修改的点，确保它保持原样
            int other_node = target_node_to_update + 2;
            if (A_share[other_node][(other_node + 1) % N] != 1) success = false;
            if (F_share[other_node][0] != other_node * 100) success = false;
            
            std::cout << "  Verification result: " << (success ? "SUCCESS!" : "FAILURE!") << std::endl;
        }
        if (party == SERVER) {
            printMatrix("A_new (Result)", A_share,N,N);
            printMatrix("F_new (Result)", F_share,N,C);
        }
    }
    //FSS::end(); // 结束计时和通信统计
    if (party == DEALER) {
        FSS->finalize();
    }
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
    std::cerr << ">> Graph Update Protocol Test - End (Party " << party << ")" << std::endl;
}
void fptraining_init() {
    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));
}
// 在你的 main 函数中
int main(int argc, char** argv) {
    //fptraining_init();
    int party = 0;
    if (argc > 1) {
        party = atoi(argv[1]);
    }

    // 将 main 函数中的调用从 test_dpf_sort 改为 test_graph_update
    test_graph_update(party);

    return 0;
}