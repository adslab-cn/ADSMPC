#include "dpfsort.h"
#include "../primitives/dpf.h"
#include <assert.h>
#include "../aux_parameter/utils.h" // 需要 modularInverse
#include <omp.h> // 为了并行化


std::pair<DpfRouteKeyPack, DpfRouteKeyPack> keyGenDpfRoute(
    int size, int data_bin, int rank_bin
) {
    // 1. 使用构造函数创建和分配内存
    DpfRouteKeyPack k0(size, data_bin, rank_bin);
    DpfRouteKeyPack k1(size, data_bin, rank_bin);

    // 2. 准备临时变量
    GroupElement* r_masks = new GroupElement[size];
    GroupElement* s_masks = new GroupElement[size];
    GroupElement* s_inv_masks = new GroupElement[size];
    GroupElement data_mod = 1ULL << data_bin;

    // 3. 生成随机掩码 (并行化)
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        r_masks[i] = random_ge(rank_bin);
        
        s_masks[i] = random_ge(data_bin);
        if (s_masks[i] == 0) s_masks[i] = 1; // 确保非零
        
        // 修复：调用 modularInverse
        //s_inv_masks[i] = modularInverse(s_masks[i], data_mod);
    }

    // 4. 生成 DPF 密钥 (并行化)
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        GroupElement alpha = r_masks[i];
        GroupElement beta = s_inv_masks[i];

        auto dpf_keys = keyGenDPF(rank_bin, data_bin, alpha, beta);
        
        k0.routing_keys[i] = dpf_keys.first;
        k1.routing_keys[i] = dpf_keys.second;
    }

    // 5. 对掩码进行秘密分享 (并行化)
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        auto r_split = splitShare(r_masks[i], rank_bin);
        k0.r_shares[i] = r_split.first;
        k1.r_shares[i] = r_split.second;

        auto s_split = splitShare(s_masks[i], data_bin);
        k0.s_shares[i] = s_split.first;
        k1.s_shares[i] = s_split.second;
    }
    
    // 6. 清理临时内存
    delete[] r_masks;
    delete[] s_masks;
    delete[] s_inv_masks;

    return std::make_pair(k0, k1);
}


// --- 在线协议 - Round 1 (本地准备) ---
// 注意：安全乘法部分被移到了 ElemWiseMul API 中，这里只是纯本地计算
void online_round1_prepare(
    int size,
    const DpfRouteKeyPack &key,
    const GroupElement* y_in_shares,
    GroupElement* y_plus_r_shares) 
{
    // 这个函数现在只负责计算 y+r 的份额，这是一个纯本地操作
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        y_plus_r_shares[i] = y_in_shares[i] + key.r_shares[i];
    }
}


// --- 在线协议 - Round 2 (本地计算) ---
void online_round2_compute(
    int party,
    const DpfRouteKeyPack &key,
    const GroupElement* y_hat_public,
    const GroupElement* z_tilde_public,
    GroupElement* z_out_shares) 
{
    int size = key.size;
    int rank_bin = key.rank_bin;
    int data_bin = key.data_bin;
    
    // 对每一个目标排名 k 进行计算
    #pragma omp parallel for
    for (int k = 0; k < size; ++k) {
        GroupElement target_rank_k = k;
        GroupElement result_share_k = 0;

        for (int i = 0; i < size; ++i) {
            GroupElement dpf_input = y_hat_public[i] - target_rank_k;
            mod(dpf_input, rank_bin);
            
            GroupElement v_share_i = evalDPF_with_payload(party, key.routing_keys[i], dpf_input);
            mod(v_share_i, data_bin);

            GroupElement term = z_tilde_public[i] * v_share_i;
            mod(term, data_bin);
            
            result_share_k += term;
        }
        mod(result_share_k, data_bin);
        z_out_shares[k] = result_share_k;
    }
}
