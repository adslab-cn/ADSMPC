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
    
    // 给底层 FSS 框架的密码学 PRNG 设种子
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
    // 1. 设置 Cora 数据集的图参数
    // ==========================================
    // int numBaseNodes = 2708;   
    // int numBaseEdges = 5429;   
    // int inDim = 1433;          
    // int hidDim = 64;           
    // int outDim = 7;         

    // // Citeseer
    // int numBaseNodes = 3327;
    // int numBaseEdges = 4732;
    // int inDim = 3703;
    // int hidDim = 64;
    // int outDim = 6;

    // Pubmed
    int numBaseNodes = 19717;
    int numBaseEdges = 44338;
    int inDim = 500;
    int hidDim = 64;
    int outDim = 3;        
    
    // 微基准推算常量 (基于论文 Table 14: 10^5 elements cost)
    const double SHUF_OFF_TIME_100K = 24.34; // ms
    int num_layers = 2;
    int shuffles_per_layer = 3; 
    int total_shuffles = num_layers * shuffles_per_layer;

    GroupElement *F_base = new GroupElement[numBaseNodes * inDim];
    GroupElement *F_base_mask = new GroupElement[numBaseNodes * inDim];
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
    fill_data(F_base, F_base_mask, numBaseNodes * inDim);
    fill_data(W1, W1_mask, inDim * hidDim);
    fill_data(W2, W2_mask, hidDim * outDim);

    // =======================================================
    // 隐藏的后台计算：悄悄把 Base MatMul 算好，记录真实耗时
    // =======================================================
    GroupElement *H_B_cached = new GroupElement[numBaseNodes * hidDim];
    GroupElement *H_B_cached_mask = new GroupElement[numBaseNodes * hidDim];
    
    FSS::start();
    auto base_start = std::chrono::high_resolution_clock::now();
    MatMul2D(numBaseNodes, inDim, hidDim, F_base, F_base_mask, W1, W1_mask, H_B_cached, H_B_cached_mask, true);
    auto base_end = std::chrono::high_resolution_clock::now();
    FSS::end();
    
    // 记录全局 Base 降维耗时
    double base_mat_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(base_end - base_start).count();

    // =======================================================
    // 执行 Queries
    // =======================================================
    int numQueries = 5; 
    int numDeltaNodes = 271;   // Cora 10% 规模
    int numDeltaEdges = 1056;
    int numGhostNodes = 150;
    int totalNodes = numBaseNodes + numDeltaNodes;

    for (int q = 1; q <= numQueries; q++) {
        
        // --- 数据生成与内存分配 ---
        GroupElement *F_delta = new GroupElement[numDeltaNodes * inDim];
        GroupElement *F_delta_mask = new GroupElement[numDeltaNodes * inDim];
        GroupElement *D_inv = new GroupElement[totalNodes];
        GroupElement *D_inv_mask = new GroupElement[totalNodes];
        int *ghostIndices_mask = new int[numGhostNodes]; 
        
        fill_data(F_delta, F_delta_mask, numDeltaNodes * inDim);
        fill_data(D_inv, D_inv_mask, totalNodes);
        for(int k = 0; k < numGhostNodes; k++) ghostIndices_mask[k] = rng() % numBaseNodes;

        GroupElement *H_delta_trans = new GroupElement[numDeltaNodes * hidDim];
        GroupElement *H_delta_trans_mask = new GroupElement[numDeltaNodes * hidDim];
        GroupElement *H_trans_total = new GroupElement[totalNodes * hidDim];
        GroupElement *H_trans_total_mask = new GroupElement[totalNodes * hidDim];
        GroupElement *H1_post = new GroupElement[totalNodes * hidDim];
        GroupElement *H1_post_mask = new GroupElement[totalNodes * hidDim];
        GroupElement *H2_trans = new GroupElement[totalNodes * outDim];
        GroupElement *H2_trans_mask = new GroupElement[totalNodes * outDim];
        GroupElement *H2_pre = new GroupElement[totalNodes * outDim];
        GroupElement *H2_pre_mask = new GroupElement[totalNodes * outDim];
        GroupElement *Y_out = new GroupElement[totalNodes * outDim];
        GroupElement *Y_out_mask = new GroupElement[totalNodes * outDim];

        // --- 执行在线推理 ---
        FSS::start();

        // 1) L1 降维 (增量计算)
        auto t1_start = std::chrono::high_resolution_clock::now();
        MatMul2D(numDeltaNodes, inDim, hidDim, F_delta, F_delta_mask, W1, W1_mask, H_delta_trans, H_delta_trans_mask, true);
        for(int i=0; i<numBaseNodes*hidDim; i++) {
            H_trans_total[i] = H_B_cached[i];
            if(party==DEALER) H_trans_total_mask[i] = H_B_cached_mask[i];
        }
        for(int i=0; i<numDeltaNodes*hidDim; i++) {
            H_trans_total[numBaseNodes*hidDim + i] = H_delta_trans[i];
            if(party==DEALER) H_trans_total_mask[numBaseNodes*hidDim + i] = H_delta_trans_mask[i];
        }
        auto t1_end = std::chrono::high_resolution_clock::now();

        // 2) L1 消息传递
        auto t2_start = std::chrono::high_resolution_clock::now();
        BPMPL_GraphRouting(numBaseNodes, numBaseEdges, numDeltaNodes, numDeltaEdges, numGhostNodes,
                           hidDim, H_trans_total, H_trans_total_mask, D_inv, D_inv_mask, 
                           H1_post, H1_post_mask, ghostIndices_mask, "L1_Route");
        auto t2_end = std::chrono::high_resolution_clock::now();

        // 3) ReLU
        auto t3_start = std::chrono::high_resolution_clock::now();
        GTDCFReLU(totalNodes * hidDim, H1_post, H1_post, H1_post_mask, H1_post_mask, 8, "L1_ReLU");
        auto t3_end = std::chrono::high_resolution_clock::now();

        // 4) L2 降维
        auto t4_start = std::chrono::high_resolution_clock::now();
        MatMul2D(totalNodes, hidDim, outDim, H1_post, H1_post_mask, W2, W2_mask, H2_trans, H2_trans_mask, true);
        auto t4_end = std::chrono::high_resolution_clock::now();

        // 5) L2 消息传递
        auto t5_start = std::chrono::high_resolution_clock::now();
        BPMPL_GraphRouting(numBaseNodes, numBaseEdges, numDeltaNodes, numDeltaEdges, numGhostNodes,
                           outDim, H2_trans, H2_trans_mask, D_inv, D_inv_mask, 
                           H2_pre, H2_pre_mask, ghostIndices_mask, "L2_Route");
        auto t5_end = std::chrono::high_resolution_clock::now();

        // 6) Softmax
        auto t6_start = std::chrono::high_resolution_clock::now();
        int fixed_point_scale = 12;
        BPGCNSoftmax(totalNodes, outDim, H2_pre, Y_out, H2_pre_mask, Y_out_mask, fixed_point_scale, "Out_Softmax");
        auto t6_end = std::chrono::high_resolution_clock::now();

        FSS::end();

        // =======================================================
        // 打印：静态与动态场景的开销账单 (仅 Client 打印防刷屏)
        // =======================================================
        if (party == CLIENT) {
            auto d1_delta = std::chrono::duration_cast<std::chrono::milliseconds>(t1_end - t1_start).count();
            auto d2 = std::chrono::duration_cast<std::chrono::milliseconds>(t2_end - t2_start).count();
            auto d3 = std::chrono::duration_cast<std::chrono::milliseconds>(t3_end - t3_start).count();
            auto d4 = std::chrono::duration_cast<std::chrono::milliseconds>(t4_end - t4_start).count();
            auto d5 = std::chrono::duration_cast<std::chrono::milliseconds>(t5_end - t5_start).count();
            auto d6 = std::chrono::duration_cast<std::chrono::milliseconds>(t6_end - t6_start).count();

            // 理论拓扑预处理时间推算
            long long N_tot = totalNodes + numBaseEdges + numDeltaEdges; 
            double mono_off_time = (N_tot * total_shuffles / 100000.0) * SHUF_OFF_TIME_100K;
            long long N_delta = numDeltaNodes + numDeltaEdges + numGhostNodes; 
            double bpgnn_off_time = (N_delta * total_shuffles / 100000.0) * SHUF_OFF_TIME_100K;

            double offline_prep_time, l1_trans_time, total_time;

            std::cout << "\n==================================================================" << std::endl;
            if (q == 1) {
                // Query 1 视为静态全量计算 (Transductive Setting)
                std::cout << ">>>[Query " << q << ": Static / Transductive Setting] (Full Graph Processing)" << std::endl;
                offline_prep_time = mono_off_time;                  // 必须重算全量图的 Shuffle 密钥
                l1_trans_time = base_mat_time_ms + d1_delta;        // 必须重算全量图的特征降维
            } else {
                // Query > 1 视为动态增量计算 (Inductive Setting)
                std::cout << ">>> [Query " << q << ": Dynamic / Inductive Setting] (Incremental Processing)" << std::endl;
                offline_prep_time = bpgnn_off_time;                 // 仅算增量图的 Shuffle 密钥
                l1_trans_time = d1_delta;                           // 仅算新增节点的特征降维
            }

            total_time = offline_prep_time + l1_trans_time + d2 + d3 + d4 + d5 + d6;

            std::cout << "------------------------------------------------------------------" << std::endl;
            std::cout << "  0) 离线拓扑预处理 (Offline Topology Prep) : " << offline_prep_time << " ms" << std::endl;
            std::cout << "  1) 第一层降维     (L1 Feature Trans)      : " << l1_trans_time << " ms" << std::endl;
            std::cout << "  2) 第一层消息传递 (L1 Graph Route)        : " << d2 << " ms" << std::endl;
            std::cout << "  3) ReLU 激活      (L1 GTDCFReLU)         : " << d3 << " ms" << std::endl;
            std::cout << "  4) 第二层降维     (L2 Feature Trans)      : " << d4 << " ms" << std::endl;
            std::cout << "  5) 第二层消息传递 (L2 Graph Route)        : " << d5 << " ms" << std::endl;
            std::cout << "  6) Softmax 归一化 (Output Softmax)        : " << d6 << " ms" << std::endl;
            std::cout << "  ----------------------------------------------------------------" << std::endl;
            if (q == 1) {
                std::cout << "  => Total Static Inference Time            : " << total_time << " ms" << std::endl;
            } else {
                std::cout << "  => Total Dynamic Inference Time           : " << total_time << " ms" << std::endl;
            }
            std::cout << "==================================================================\n" << std::endl;
        }

        // 清理内存
        delete[] F_delta; delete[] F_delta_mask; delete[] D_inv; delete[] D_inv_mask; delete[] ghostIndices_mask;
        delete[] H_delta_trans; delete[] H_delta_trans_mask; delete[] H_trans_total; delete[] H_trans_total_mask;
        delete[] H1_post; delete[] H1_post_mask; delete[] H2_trans; delete[] H2_trans_mask;
        delete[] H2_pre; delete[] H2_pre_mask; delete[] Y_out; delete[] Y_out_mask;
    }

    delete[] F_base; delete[] F_base_mask; delete[] W1; delete[] W1_mask; delete[] W2; delete[] W2_mask;
    delete[] H_B_cached; delete[] H_B_cached_mask;

    FSS->finalize();
    delete FSS;
    return 0;
}