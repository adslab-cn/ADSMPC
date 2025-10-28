#include "../aux_parameter/comms.h"
#include "../aux_parameter/array.h"
#include "../aux_parameter/config.h"
#include "ars.h"

inline GroupElement count_local_wrap(GroupElement a, GroupElement b) {
    // 将无符号的 GroupElement 转换为有符号的 int64_t 来进行判断
    int64_t signed_a = static_cast<int64_t>(a);
    int64_t signed_b = static_cast<int64_t>(b);
    
    // 加法仍然在 uint64_t 上进行，以模拟环的行为
    GroupElement next_unsigned = a + b;
    int64_t next_signed = static_cast<int64_t>(next_unsigned);

    // 检查上溢: 两个正数相加，结果为负数
    if (signed_a > 0 && signed_b > 0 && next_signed < 0) {
        return 1; // 上溢
    }
    
    // 检查下溢: 两个负数相加，结果为正数
    if (signed_a < 0 && signed_b < 0 && next_signed > 0) {
        return -1; // 下溢，返回 -1 (在环上是一个大正数)
    }

    return 0; // 没有溢出
}

// ===================================================================
//              密钥生成 (Dealer离线执行)
// ===================================================================
std::pair<ARS_CrypTen_Style_KeyPack, ARS_CrypTen_Style_KeyPack> 
keyGenARS_CrypTen_Style(int Bin) {
    ARS_CrypTen_Style_KeyPack k0, k1;
    GroupElement r = random_ge(Bin);

    auto r_split = splitShare(r, Bin);
    k0.r_share = r_split.first;
    k1.r_share = r_split.second;
    GroupElement theta_r = count_local_wrap( k0.r_share,k1.r_share); 

    auto theta_r_split = splitShare(theta_r, Bin); // 环绕次数本身只需要1位或几位，但为了简单，可以用GroupElement分享
    k0.theta_r_share = theta_r_split.first;
    k1.theta_r_share = theta_r_split.second;
    
    return std::make_pair(k0, k1);
}