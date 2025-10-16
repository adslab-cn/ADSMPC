#include "../aux_parameter/keypack.h"


std::pair<GraphUpdateKeyPack, GraphUpdateKeyPack> keyGenForGraphUpdate(
    int target_node_v_star,int n, int c,
    GroupElement ** A_old, GroupElement ** A_new, int A_bw, int A_data_bw,
    GroupElement ** F_old, GroupElement ** F_new, int F_bw, int F_data_bw
);
void obliviousUpdate(
    int party,
    int n,
    int c,
    GroupElement ** A_share,
    GroupElement ** F_share,
    DPFKeyPack* keys_A,
    DPFKeyPack* keys_F
);