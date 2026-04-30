#include <backend/FSS_transformer.h>
#include <layers/layers.h>
#include <module.h>
#include <FSS/utils.h>
#include <FSS/api.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

int main(int argc, char** argv) {
    sytorch_init();
    
    // 初始化密码学伪随机数种子
    uint64_t seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL) + i, seedKey));
    }

    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <party_id> [ip]" << std::endl;
        return 1;
    }

    int party = atoi(argv[1]);
    std::string ip = "127.0.0.1";
    if(argc > 2) ip = argv[2];

    using FSSVersion = FSSTransformer<u64>;
    FSSVersion *FSS = new FSSVersion();
    FSSConfig::bitlength = 64;
    FSSConfig::party = party;
    FSSConfig::num_threads = 4;
    FSS->init(ip, true);

    // ==========================================
    // 1. 设置图结构参数 (Cora 规模 - OblivGNN 稠密矩阵版)
    // ==========================================
    // int totalNodes = 2708;     // 全量图节点
    // int inDim = 1433;          
    // int hidDim = 64;           
    // int outDim = 7;            

    // // Citeseer
    // int totalNodes = 3327;
    // int numBaseEdges = 4732;
    // int inDim = 3703;
    // int hidDim = 64;
    // int outDim = 6;

    // Pubmed
    int totalNodes = 19717;
    int numBaseEdges = 44338;
    int inDim = 500;
    int hidDim = 64;
    int outDim = 3;    

    int numUpdates = 10;       // 模拟动态场景下新增 20 个节点

    // 内存分配 (注意：OblivGNN 需要完整的 N x N 稠密邻接矩阵)
    GroupElement *A_dense = new GroupElement[totalNodes * totalNodes];
    GroupElement *A_dense_mask = new GroupElement[totalNodes * totalNodes];
    
    GroupElement *F_in = new GroupElement[totalNodes * inDim];
    GroupElement *F_in_mask = new GroupElement[totalNodes * inDim];
    
    GroupElement *W1 = new GroupElement[inDim * hidDim];
    GroupElement *W1_mask = new GroupElement[inDim * hidDim];
    GroupElement *W2 = new GroupElement[hidDim * outDim];
    GroupElement *W2_mask = new GroupElement[hidDim * outDim];

    std::mt19937_64 rng(42); 
    auto fill_data = [&](GroupElement* eval_arr, GroupElement* mask_arr, int size) {
        for(int i = 0; i < size; i++) {
            GroupElement mask = rng();
            if (party == DEALER) mask_arr[i] = mask;
            else eval_arr[i] = (rng() % 10) + mask; 
        }
    };
    
    fill_data(A_dense, A_dense_mask, totalNodes * totalNodes); 
    fill_data(F_in, F_in_mask, totalNodes * inDim);
    fill_data(W1, W1_mask, inDim * hidDim);
    fill_data(W2, W2_mask, hidDim * outDim);

    // ==========================================
    // 2. 模拟 OblivGNN 动态 Inductive 查询
    // ==========================================
    if (party == CLIENT) {
        std::cout << "\n==================================================================" << std::endl;
        std::cout << "[OblivGNN Pipeline: Dense Adjacency Matrix & DPF Node Update]" << std::endl;
        std::cout << " -> Total Nodes: " << totalNodes << " | Architecture: " << inDim << " -> " << hidDim << " -> " << outDim << std::endl;
        std::cout << " -> Executing Query (Updating " << numUpdates << " nodes on the fly...)" << std::endl;
        std::cout << "==================================================================\n" << std::endl;
    }

    // 生成更新所需的 Delta 数据 (特征与边连接)
    GroupElement *Delta_F = new GroupElement[numUpdates * inDim];
    GroupElement *Delta_F_mask = new GroupElement[numUpdates * inDim];
    GroupElement *Delta_A = new GroupElement[numUpdates * totalNodes];
    GroupElement *Delta_A_mask = new GroupElement[numUpdates * totalNodes];
    int *update_indices_mask = new int[numUpdates];
    
    fill_data(Delta_F, Delta_F_mask, numUpdates * inDim);
    fill_data(Delta_A, Delta_A_mask, numUpdates * totalNodes);
    for(int i = 0; i < numUpdates; i++) update_indices_mask[i] = rng() % totalNodes;

    // =========================================================
    // 中间计算层 Buffer
    // 现在采用：先稠密图聚合，再降维
    // Layer 1: A * F_in -> H1_agg (N x inDim), H1_agg * W1 -> H1_trans (N x hidDim)
    // Layer 2: A * H1_post -> H2_agg (N x hidDim), H2_agg * W2 -> H2_trans (N x outDim)
    // =========================================================
    GroupElement *H1_agg = new GroupElement[totalNodes * inDim];
    GroupElement *H1_agg_mask = new GroupElement[totalNodes * inDim];
    GroupElement *H1_trans = new GroupElement[totalNodes * hidDim];
    GroupElement *H1_trans_mask = new GroupElement[totalNodes * hidDim];
    GroupElement *H1_post = new GroupElement[totalNodes * hidDim];
    GroupElement *H1_post_mask = new GroupElement[totalNodes * hidDim];
    
    GroupElement *H2_agg = new GroupElement[totalNodes * hidDim];
    GroupElement *H2_agg_mask = new GroupElement[totalNodes * hidDim];
    GroupElement *H2_trans = new GroupElement[totalNodes * outDim];
    GroupElement *H2_trans_mask = new GroupElement[totalNodes * outDim];
    
    GroupElement *Y_out = new GroupElement[totalNodes * outDim];
    GroupElement *MaskForY_out = new GroupElement[totalNodes * outDim];

    // =========================================================================
    // 准备计时与通信量统计闭包
    // =========================================================================
    auto get_online_comm = [&]() -> uint64_t {
        if (party == DEALER) return 0;
        return FSSConfig::peer->bytesSent() + FSSConfig::peer->bytesReceived();
    };

    FSS::start();
    auto total_start = std::chrono::high_resolution_clock::now();
    uint64_t comm_start = get_online_comm(); // 初始通信量快照

    // -----------------------------------------------------------
    // 0) 动态节点更新 (DPF Node Update)
    // -----------------------------------------------------------
    auto t0_start = std::chrono::high_resolution_clock::now();
    uint64_t c0_start = get_online_comm();
    OblivGNN_MatrixUpdate(totalNodes, inDim, numUpdates,
                          F_in, F_in_mask,
                          update_indices_mask,
                          Delta_F, Delta_F_mask,
                          "FeatureUpdate::");
    OblivGNN_MatrixUpdate(totalNodes, totalNodes, numUpdates,
                          A_dense, A_dense_mask,
                          update_indices_mask,
                          Delta_A, Delta_A_mask,
                          "AdjacencyUpdate::");
    auto t0_end = std::chrono::high_resolution_clock::now();
    uint64_t c0_end = get_online_comm();

    // -----------------------------------------------------------
    // 1) 第一层稠密图聚合: A * F_in -> H1_agg
    // -----------------------------------------------------------
    auto t1_start = std::chrono::high_resolution_clock::now();
    uint64_t c1_start = get_online_comm();
    MatMul2D(totalNodes, totalNodes, inDim,
             A_dense, A_dense_mask,
             F_in, F_in_mask,
             H1_agg, H1_agg_mask,
             true);
    auto t1_end = std::chrono::high_resolution_clock::now();
    uint64_t c1_end = get_online_comm();

    // -----------------------------------------------------------
    // 2) 第一层降维: H1_agg * W1 -> H1_trans
    // -----------------------------------------------------------
    auto t2_start = std::chrono::high_resolution_clock::now();
    uint64_t c2_start = get_online_comm();
    MatMul2D(totalNodes, inDim, hidDim,
             H1_agg, H1_agg_mask,
             W1, W1_mask,
             H1_trans, H1_trans_mask,
             true);
    auto t2_end = std::chrono::high_resolution_clock::now();
    uint64_t c2_end = get_online_comm();

    // -----------------------------------------------------------
    // 3) 激活层: Relu2Round(H1_trans) -> H1_post
    // -----------------------------------------------------------
    auto t3_start = std::chrono::high_resolution_clock::now();
    uint64_t c3_start = get_online_comm();
    Relu2Round(totalNodes * hidDim,
               H1_trans, H1_trans_mask,
               H1_post, H1_post_mask,
               nullptr, 64);
    auto t3_end = std::chrono::high_resolution_clock::now();
    uint64_t c3_end = get_online_comm();

    // -----------------------------------------------------------
    // 4) 第二层稠密图聚合: A * H1_post -> H2_agg
    // -----------------------------------------------------------
    auto t4_start = std::chrono::high_resolution_clock::now();
    uint64_t c4_start = get_online_comm();
    MatMul2D(totalNodes, totalNodes, hidDim,
             A_dense, A_dense_mask,
             H1_post, H1_post_mask,
             H2_agg, H2_agg_mask,
             true);
    auto t4_end = std::chrono::high_resolution_clock::now();
    uint64_t c4_end = get_online_comm();

    // -----------------------------------------------------------
    // 5) 第二层降维: H2_agg * W2 -> H2_trans
    // -----------------------------------------------------------
    auto t5_start = std::chrono::high_resolution_clock::now();
    uint64_t c5_start = get_online_comm();
    MatMul2D(totalNodes, hidDim, outDim,
             H2_agg, H2_agg_mask,
             W2, W2_mask,
             H2_trans, H2_trans_mask,
             true);
    auto t5_end = std::chrono::high_resolution_clock::now();
    uint64_t c5_end = get_online_comm();

    // -----------------------------------------------------------
    // 6) Softmax on final logits H2_trans
    // -----------------------------------------------------------
    // auto t6_start = std::chrono::high_resolution_clock::now();
    // uint64_t c6_start = get_online_comm();
    // if (party == DEALER) {
    //     Softmax(totalNodes, outDim, 64, H2_trans_mask, MaskForY_out, 12);
    // } else {
    //     Softmax(totalNodes, outDim, 64, H2_trans, Y_out, 12);
    // }
    // auto t6_end = std::chrono::high_resolution_clock::now();
    // uint64_t c6_end = get_online_comm();

    auto t6_start = std::chrono::high_resolution_clock::now();
    uint64_t c6_start = get_online_comm();
    
    // BPGCNSoftmax 内部会自动判断 party，无需外部写 if-else
    BPGCNSoftmax(totalNodes, outDim, 
                 H2_agg, Y_out, 
                 H2_agg_mask, MaskForY_out, 
                 12, "Out_Softmax::");
                 
    auto t6_end = std::chrono::high_resolution_clock::now();
    uint64_t c6_end = get_online_comm();

    auto total_end = std::chrono::high_resolution_clock::now();
    FSS::end();

    // =========================================================================
    // 打印带有【时间 + 通信量】的 profiling
    // =========================================================================
    if (party == CLIENT) {
        auto d0 = std::chrono::duration_cast<std::chrono::milliseconds>(t0_end - t0_start).count();
        auto d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t1_end - t1_start).count();
        auto d2 = std::chrono::duration_cast<std::chrono::milliseconds>(t2_end - t2_start).count();
        auto d3 = std::chrono::duration_cast<std::chrono::milliseconds>(t3_end - t3_start).count();
        auto d4 = std::chrono::duration_cast<std::chrono::milliseconds>(t4_end - t4_start).count();
        auto d5 = std::chrono::duration_cast<std::chrono::milliseconds>(t5_end - t5_start).count();
        auto d6 = std::chrono::duration_cast<std::chrono::milliseconds>(t6_end - t6_start).count();
        
        double m0 = (c0_end - c0_start) / (1024.0 * 1024.0);
        double m1 = (c1_end - c1_start) / (1024.0 * 1024.0);
        double m2 = (c2_end - c2_start) / (1024.0 * 1024.0);
        double m3 = (c3_end - c3_start) / (1024.0 * 1024.0);
        double m4 = (c4_end - c4_start) / (1024.0 * 1024.0);
        double m5 = (c5_end - c5_start) / (1024.0 * 1024.0);
        double m6 = (c6_end - c6_start) / (1024.0 * 1024.0);

        std::cout << "\n[OblivGNN Execution Profiling: Time & Communication]" << std::endl;
        std::cout << "  0) 动态节点更新 (DPF Node Update)  : " << d0 << " ms \t| " << m0 << " MB" << std::endl;
        std::cout << "  1) 第一层图聚合 (A * F_in)         : " << d1 << " ms \t| " << m1 << " MB  <-- Dense Matrix Bottleneck!" << std::endl;
        std::cout << "  2) 第一层降维   (H1_agg * W1)      : " << d2 << " ms \t| " << m2 << " MB" << std::endl;
        std::cout << "  3) 两轮 ReLU    (Relu2Round)       : " << d3 << " ms \t| " << m3 << " MB" << std::endl;
        std::cout << "  4) 第二层图聚合 (A * H1_post)      : " << d4 << " ms \t| " << m4 << " MB  <-- Dense Matrix Bottleneck!" << std::endl;
        std::cout << "  5) 第二层降维   (H2_agg * W2)      : " << d5 << " ms \t| " << m5 << " MB" << std::endl;
        std::cout << "  6) 老版 Softmax (Taylor Approx)    : " << d6 << " ms \t| " << m6 << " MB" << std::endl;
        std::cout << "  ------------------------------------------------------------------------" << std::endl;
        std::cout << "  => Total End-to-End Query Time     : "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count()
                  << " ms" << std::endl;
        std::cout << "  => Total End-to-End Comm           : "
                  << (get_online_comm() - comm_start) / (1024.0 * 1024.0)
                  << " MB" << std::endl;
        std::cout << "==========================================================================\n" << std::endl;
    }

    // -----------------------------------------------------------
    // 内存清理
    // -----------------------------------------------------------
    delete[] A_dense; 
    delete[] A_dense_mask; 
    delete[] F_in; 
    delete[] F_in_mask;
    delete[] W1; 
    delete[] W1_mask; 
    delete[] W2; 
    delete[] W2_mask;

    delete[] Delta_F; 
    delete[] Delta_F_mask; 
    delete[] Delta_A; 
    delete[] Delta_A_mask;
    delete[] update_indices_mask; 

    delete[] H1_agg; 
    delete[] H1_agg_mask; 
    delete[] H1_trans; 
    delete[] H1_trans_mask;
    delete[] H1_post; 
    delete[] H1_post_mask;

    delete[] H2_agg; 
    delete[] H2_agg_mask;
    delete[] H2_trans; 
    delete[] H2_trans_mask;

    delete[] Y_out; 
    delete[] MaskForY_out;

    FSS->finalize();
    delete FSS;

    return 0;
}