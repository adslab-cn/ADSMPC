#include "graphupdate.h"
#include "../primitives/dpf.h"
#include <assert.h>
#include <omp.h> 
using Matrix = std::vector<std::vector<GroupElement>>; 


std::pair<GraphUpdateKeyPack, GraphUpdateKeyPack> keyGenForGraphUpdate(
    int target_node_v_star,
    const Matrix& A_old, const Matrix& A_new, int A_bw, int A_data_bw,
    const Matrix& F_old, const Matrix& F_new, int F_bw, int F_data_bw
) {
    int n = A_old.size();
    int c = F_old[0].size();
    
    GraphUpdateKeyPack k0(n, c, A_bw, F_bw, A_data_bw, F_data_bw);
    GraphUpdateKeyPack k1(n, c, A_bw, F_bw, A_data_bw, F_data_bw);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        GroupElement delta = A_new[target_node_v_star][i] - A_old[target_node_v_star][i];
        
        auto key_pair = keyGenDPF(A_bw, A_data_bw, target_node_v_star, delta);
        
        k0.keys_A[i] = key_pair.first;
        k1.keys_A[i] = key_pair.second;
    }
    #pragma omp parallel for
    for (int i = 0; i < c; ++i) {
        GroupElement delta = F_new[target_node_v_star][i] - F_old[target_node_v_star][i];
        
        auto key_pair = keyGenDPF(F_bw, F_data_bw, target_node_v_star, delta);
        
        k0.keys_F[i] = key_pair.first;
        k1.keys_F[i] = key_pair.second;
    }

    return std::make_pair(k0, k1);
}

void obliviousUpdate(
    int party,
    Matrix& A_share,
    Matrix& F_share,
    std::vector<DPFKeyPack>& keys_A,
    std::vector<DPFKeyPack>& keys_F
) {
    int n = A_share.size();
    int c = F_share[0].size();
    int dpf_party = party - 2;

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        std::vector<GroupElement> delta_column_share(n, 0);
        evalAll(dpf_party, keys_A[i], 0, delta_column_share.data());
        for (int j = 0; j < n; ++j) {
            A_share[j][i] += delta_column_share[j];
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < c; ++i) {
        std::vector<GroupElement> delta_column_share(n, 0);
        evalAll(dpf_party, keys_F[i], 0, delta_column_share.data());
        for (int j = 0; j < n; ++j) {
            F_share[j][i] += delta_column_share[j];
        }
    }
}