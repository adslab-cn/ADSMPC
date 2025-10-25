#include "../aux_parameter/comms.h"
#include "../aux_parameter/array.h"
#include "../aux_parameter/config.h"
#include "ars.h"



// ===================================================================
//              密钥生成 (Dealer离线执行)
// ===================================================================
std::pair<ARS_CrypTen_Style_KeyPack, ARS_CrypTen_Style_KeyPack> 
keyGenARS_CrypTen_Style(int Bin) {
    ARS_CrypTen_Style_KeyPack k0, k1;
    GroupElement r = random_ge(Bin);
    GroupElement theta_r = 0; // r < RING_MAX, so its wrap count is 0

    auto r_split = splitShare(r, Bin);
    k0.r_share = r_split.first;
    k1.r_share = r_split.second;

    auto theta_r_split = splitShare(theta_r, Bin); // 环绕次数本身只需要1位或几位，但为了简单，可以用GroupElement分享
    k0.theta_r_share = theta_r_split.first;
    k1.theta_r_share = theta_r_split.second;

    k0.theta_r_share = 1;
    k1.theta_r_share = -1;
    
    return std::make_pair(k0, k1);
}