#include <aux_parameter/keypack.h>
// Dealer 端的密钥生成函数
std::pair<DpfRouteKeyPack, DpfRouteKeyPack> keyGenDpfRoute(
    int size, int data_bin, int rank_bin
);

// // 在线协议 - Round 1 准备阶段 (本地计算)
// void online_round1_prepare(
//     int size,
//     const DpfRouteKeyPack &key,
//     const GroupElement* y_in_shares,
//     const GroupElement* z_in_shares,
//     GroupElement* y_plus_r_shares, // 输出: y+r 的份额
//     GroupElement* z_mul_s_shares   // 输出: z*s 的份额 (Beaver Triple 中间值)
// );


// 在线协议 - Round 2 计算阶段 (本地计算)
void online_round2_compute(
    int party,
    const DpfRouteKeyPack &key,
    const GroupElement* y_hat_public, // 输入: 重构后的 y_hat
    const GroupElement* z_tilde_public, // 输入: 重构后的 z_tilde
    GroupElement* z_out_shares      // 输出: 最终排序结果的份额
);
