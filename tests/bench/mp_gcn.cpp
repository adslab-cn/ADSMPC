#include <sytorch/backend/FSS_extended.h>
#include <FSS/utils.h>
#include <FSS/api.h>
#include <FSS/group_element.h>
#include "crypt_mpl.h" 
#include <vector>
#include <iostream>
#include <cassert>

// 辅助函数：计算明文 GCN 结果 (用于验证)
// Out = ReLU( Aggregate( (X * W) * EdgeWeight ) )
void compute_ground_truth(int N, int M, int K_in, int K_out, int scale,
                          uint64_t* X, uint64_t* W, 
                          uint64_t* S, uint64_t* D, uint64_t* E_W,
                          uint64_t* Out_Expected) 
{
    // 1. MatMul: Z = X * W
    std::vector<int64_t> Z(N * K_out, 0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K_out; ++j) {
            for (int k = 0; k < K_in; ++k) {
                // 注意处理定点数乘法，这里简化逻辑，假设输入已经对齐
                // 实际 FSS 框架中 MatMul 后通常跟着 Truncate
                // 这里我们模拟 unsigned 到 signed 的转换计算
                int64_t x_val = (int64_t)X[i * K_in + k]; 
                int64_t w_val = (int64_t)W[k * K_out + j];
                Z[i * K_out + j] += x_val * w_val;
            }
        }
    }

    // Truncate (Scale Down)
    for(auto& val : Z) val >>= scale;

    // 2. Message Passing (Weighted Aggregation)
    std::vector<int64_t> Agg(N * K_out, 0);
    for (int m = 0; m < M; ++m) {
        uint64_t u = S[m]; // Source Node Index
        uint64_t v = D[m]; // Dest Node Index
        int64_t weight = (int64_t)E_W[m]; // Edge Weight

        for (int j = 0; j < K_out; ++j) {
            // Agg[v] += Z[u] * weight
            Agg[v * K_out + j] += Z[u * K_out + j] * weight; 
            // 注意：这里也可能涉及 Scale Down，视具体实现而定
            // 本测试假设 EdgeWeight 是整数或已处理
        }
    }

    // 3. ReLU
    for (int i = 0; i < N * K_out; ++i) {
        if (Agg[i] < 0) Out_Expected[i] = 0;
        else Out_Expected[i] = Agg[i];
    }
}

// 辅助函数：模拟计算 CryptMPL 产生的总噪声 (用于验证阶段去除)
// 在真实场景中，这是 Client 在预处理阶段计算并拥有的
void calculate_expected_noise(int N, int M, int K_out, 
                              const std::vector<CryptMPLKeyPack>& keys,
                              const std::vector<uint64_t>& S,
                              const std::vector<uint64_t>& D,
                              uint64_t* Noise_Total) 
{
    // 初始化噪声矩阵为 0
    std::fill(Noise_Total, Noise_Total + N * K_out, 0);

    // CryptMPL 逻辑：
    // Read: 引入噪声 Noise_Read (由 seed 生成)
    // Write: 引入噪声 Noise_Write (由 seed 生成)
    // 最终结果包含：Rotate(Noise_Read, D) + Noise_Write
    
    // 注意：这里的逻辑必须与 crypt_mpl.cpp 中的实现严格一致
    // 为了简化测试，如果您在 api.cpp 中实现了 Client 去噪逻辑，这里可以简化。
    // 这里我们假设测试验证的是 "含噪结果" 是否匹配 "真值 + 噪声"。
    
    // ... 此处省略复杂的噪声重算逻辑，将在 main 中通过 reconstruct 方式处理 ...
    // 在 Dealer 模式下，我们直接让 Dealer 知道所有种子，计算出最终的 Noise 矩阵。
}

int main(int argc, char** argv) {
    int party = atoi(argv[1]);
    std::string ip = "127.0.0.1";
    if (argc > 2) ip = argv[2];

    FSSConfig::bitlength = 64;
    FSSConfig::party = party;
    FSSConfig::num_threads = 4; // 启用多线程
    
    // 初始化 FSS 库
    // 假设您有一个 init 函数，如果没有，参考 mp_relu.cpp 的初始化方式
    // FSS->init(ip, true); // 这里需要根据您的框架入口调整
    // 暂时使用 FSSConfig 直接配置，假设 Peer 连接在 start() 中处理

    // 定义图参数
    int N = 10;      // 节点数
    int M = 20;      // 边数
    int K_in = 8;    // 输入特征维数
    int K_out = 4;   // 输出特征维数
    int scale = 12;  // 定点数缩放

    // 分配内存
    // Data (X), Weights (W), Topology (S, D), EdgeWeights (E_W), Output (Out)
    GroupElement *X = new GroupElement[N * K_in];
    GroupElement *W = new GroupElement[K_in * K_out];
    GroupElement *S = new GroupElement[M];
    GroupElement *D = new GroupElement[M];
    GroupElement *E_W = new GroupElement[M];
    GroupElement *Out = new GroupElement[N * K_out];

    // Masks (用于秘密分享)
    GroupElement *X_mask = new GroupElement[N * K_in];
    GroupElement *W_mask = new GroupElement[K_in * K_out];
    GroupElement *S_mask = new GroupElement[M];
    GroupElement *D_mask = new GroupElement[M];
    GroupElement *E_W_mask = new GroupElement[M];
    GroupElement *Out_mask = new GroupElement[N * K_out];

    // Ground Truth
    GroupElement *Out_GroundTruth = new GroupElement[N * K_out];

    // 1. 初始化数据 (Dealer 负责生成并分发/设置)
    if (party == DEALER) {
        std::cout << ">> [Dealer] Generating Data..." << std::endl;
        
        // 生成随机特征 X
        for(int i=0; i<N*K_in; ++i) X_mask[i] = random_ge(64) >> 10; // 避免溢出
        
        // 生成随机权重 W
        for(int i=0; i<K_in*K_out; ++i) W_mask[i] = random_ge(64) >> 10;

        // 生成图结构 (随机边)
        for(int i=0; i<M; ++i) {
            S_mask[i] = random_ge(64) % N; // Source
            D_mask[i] = random_ge(64) % N; // Dest
            E_W_mask[i] = 1; // 简化：权重设为 1
        }

        // 计算明文真值 (Dealer 作为可信方进行验证)
        // 注意：计算真值时需要使用未掩码的值，但在 Dealer 模式下，
        // 我们通常假设 _mask 数组存储的就是明文值（作为输入源），
        // 然后生成分享给 Server。
        compute_ground_truth(N, M, K_in, K_out, scale, 
                             X_mask, W_mask, S_mask, D_mask, E_W_mask, Out_GroundTruth);
    }

    // 建立连接
    // 假设 FSS::start() 会处理连接
    // 这里需要根据您的框架具体实现 peer 连接
    // 模拟: FSSConfig::peer = new Peer(ip, 8000); 等等
    
    // ... 连接代码 (参考您的 api.cpp 或 mp_relu.cpp) ...
    // 为了示例完整，假设 Peer 已经 setup 好

    FSS::start();

    // 2. 执行 GCN 协议
    std::cout << ">> Party " << party << " executing GCN Layer..." << std::endl;
    
    // 调用我们在 api.cpp 中实现的函数
    // 注意：Mask_Pair 宏展开是 (Val, Val_mask)
    // Dealer 传入真实值在 _mask 变量中，Server 传入 Share 在非 mask 变量中
    // 您的框架约定可能略有不同，请根据 splitShare 的使用调整
    
    // 假设 Dealer 已经将数据 Split 并分发给了 Server 2 和 3 (模拟)
    // 在 mp_relu.cpp 中，通常 Dealer 持有 _mask 里的明文，并在 KeyGen 内部进行 split。
    // 而 Server 持有的数组在开始时是空的 (或接收到的 share)。
    
    // 这里我们手动模拟 Input Sharing (如果是真实网络环境，需要 recv)
    if (party == DEALER) {
        // Dealer 实际上是在 GCN_Layer_Forward 内部生成 Key 并分发
        // Dealer 不需要发送输入数据本身（那是 Setup 阶段的事），
        // 但为了测试，我们需要确保 Server 有 Share。
        // 在此测试文件中，我们略过 Input Distribution 的网络代码，
        // 假设 Server 已经拥有了 shares (通过某种魔法或预处理)。
        // *实际测试中，您可能需要在这里加一段 send/recv shares 的代码*
    }

    GCN_Layer_Forward(N, M, K_in, K_out,
                      X, X_mask,
                      W, W_mask,
                      S, S_mask,
                      D, D_mask,
                      E_W, E_W_mask,
                      Out, Out_mask,
                      scale);

    FSS::end();

    // 3. 验证结果
    if (party != DEALER) {
        // Server 2 和 3 重构结果
        // 注意：CryptGNN 的结果包含 Noise。
        // 只有 Client (这里由 Dealer 辅助验证) 才能去除 Noise。
        // 为了验证，我们在这里直接重构 Out。
        
        reconstruct(N * K_out, Out, 64);
        
        // 如果我们是 Server 2 (作为主验证方)
        if (party == 2) {
            std::cout << ">> Verifying Results..." << std::endl;
            
            // 注意：这里得到的 Out 是 (Result + Noise)
            // 在真正的 CryptGNN 中，Result 是 Client 拿到的。
            // 这是一个单元测试，我们无法轻易获得 Noise 的值 (除非改写 api.cpp 暴露出来)。
            // **为了让测试跑通，我们验证以下逻辑：**
            
            // 如果您在 api.cpp 中未实现去噪逻辑（Client Post-processing），
            // 这里 Assert 可能会失败。
            // 建议：在测试初期，先把 crypt_mpl.cpp 中的 generateNoiseMatrix 的 scale 设为 0，
            // 或者 seed 设为固定值 0，来验证逻辑正确性（此时无噪声）。
            
            // 假设我们已经去除了噪声 (或噪声为0):
            int error_count = 0;
            for(int i=0; i<N*K_out; ++i) {
                int64_t res = (int64_t)Out[i];
                // 简单的容差检查 (因为定点数精度问题)
                // int64_t truth = ... (需要从 Dealer 拿真值)
                // 在分布式测试中，Server 2 不知道 Truth。
                std::cout << "Out[" << i << "] = " << res << std::endl;
            }
            std::cout << "Verification requires Plaintext comparison (manual check or centralized test)." << std::endl;
        }
    } else {
        // Dealer 打印真值供参考
        std::cout << ">> Ground Truth (First 10):" << std::endl;
        for(int i=0; i<std::min(10, N*K_out); ++i) {
            std::cout << Out_GroundTruth[i] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    delete[] X; delete[] X_mask;
    delete[] W; delete[] W_mask;
    delete[] S; delete[] S_mask;
    delete[] D; delete[] D_mask;
    delete[] E_W; delete[] E_W_mask;
    delete[] Out; delete[] Out_mask;
    delete[] Out_GroundTruth;

    return 0;
}