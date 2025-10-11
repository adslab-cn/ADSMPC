#include "../aux_parameter/keypack.h"
using Matrix = std::vector<std::vector<GroupElement>>; 


std::pair<GraphUpdateKeyPack, GraphUpdateKeyPack> keyGenForGraphUpdate(
    int target_node_v_star,
    const Matrix& A_old, const Matrix& A_new, int A_bw, int A_data_bw,
    const Matrix& F_old, const Matrix& F_new, int F_bw, int F_data_bw
);
void obliviousUpdate(
    int party,
    Matrix& A_share,
    Matrix& F_share,
    std::vector<DPFKeyPack>& keys_A,
    std::vector<DPFKeyPack>& keys_F
);