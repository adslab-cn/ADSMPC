#include "softmax.h"
#include <FSS/dcf.h>
#include "mult.h"
#include "taylor.h"
#include "select.h"

// --- OblivSoftmax Implementation ---
// Logic: y = ReLU(x), S = sum(y), invS = 1/S (if S>0 else 1/L), out = y * invS
std::pair<OblivSoftmaxKeyPack, OblivSoftmaxKeyPack> keyGenOblivSoftmax(
    int s1, int s2, int Bin, int Bout, 
    GroupElement* rin, GroupElement* rout, 
    int sf, int logk)
{
    OblivSoftmaxKeyPack k0, k1;
    k0.s1 = s1; k0.s2 = s2; k0.Bin = Bin; k0.Bout = Bout;
    k1.s1 = s1; k1.s2 = s2; k1.Bin = Bin; k1.Bout = Bout;
    
    int size_total = s1 * s2;
    int size_batch = s1;
    int m = logk + 1; // Taylor expansion param

    k0.reluKeys = new OblivReLUKeyPack[size_total];
    k1.reluKeys = new OblivReLUKeyPack[size_total];
    k0.finalMultKeys = new MultKey[size_total];
    k1.finalMultKeys = new MultKey[size_total];
    
    k0.inverseKeys = new TaylorKeyPack[size_batch];
    k1.inverseKeys = new TaylorKeyPack[size_batch];
    k0.sumCheckKeys = new DCFKeyPack[size_batch];
    k1.sumCheckKeys = new DCFKeyPack[size_batch];
    k0.r_sumCheck = new GroupElement[size_batch];
    k1.r_sumCheck = new GroupElement[size_batch];
    k0.selectKeys = new SelectKeyPack[size_batch];
    k1.selectKeys = new SelectKeyPack[size_batch];

    // Intermediate masks
    GroupElement* r_relu_out = new GroupElement[size_total]; // y
    GroupElement* r_sum = new GroupElement[size_batch];      // S
    GroupElement* r_inv = new GroupElement[size_batch];      // Inv
    GroupElement* r_sel = new GroupElement[size_batch];      // Selected Inv

    for(int i=0; i<size_total; ++i) r_relu_out[i] = random_ge(Bout);
    
    // 1. KeyGen for ReLU
    for(int i=0; i<size_total; ++i) {
        auto keys = keyGenOblivReLU(Bin, Bout, rin[i], r_relu_out[i]);
        k0.reluKeys[i] = keys.first;
        k1.reluKeys[i] = keys.second;
    }

    // Calculate mask for Sum (r_sum = sum(r_relu_out))
    for(int i=0; i<s1; ++i) {
        r_sum[i] = 0;
        for(int j=0; j<s2; ++j) {
            r_sum[i] += r_relu_out[i*s2 + j];
        }
    }

    // 2. KeyGen for Sum Check (S > 0) & Inverse & Select
    for(int i=0; i<s1; ++i) {
        // A. Sum Check (S > 0)
        GroupElement r_bit = random_ge(1);
        auto r_bit_split = splitShare(r_bit, 1);
        k0.r_sumCheck[i] = r_bit_split.first;
        k1.r_sumCheck[i] = r_bit_split.second;
        
        auto dcfK = keyGenDCF(Bin, 1, r_sum[i], 1);
        k0.sumCheckKeys[i] = dcfK.first;
        k1.sumCheckKeys[i] = dcfK.second;

        // B. Inverse (1/S) via Taylor
        r_inv[i] = random_ge(Bout);
        // Note: TaylorKeyGen params: a=2.63, b=-5.8, c=4.2 are fixed approx params
        // Adjust params as per paper if needed. Using standard framework ones here.
        auto tayK = keyGenTaylor(Bin, Bout, 2.630, -5.857, 4.245, r_sum[i], r_inv[i], sf, logk);
        k0.inverseKeys[i] = tayK.first;
        k1.inverseKeys[i] = tayK.second;

        // C. Select (bit ? inv : 1/L)
        // 1/L in fixed point
        double one_over_L = 1.0 / (double)s2;
        uint64_t default_val = (uint64_t)(one_over_L * (1ULL << sf));
        
        // KeyGenSelect expects inputs: selector mask, input mask, output mask.
        // But here one input is PUBLIC CONSTANT (1/L). 
        // This requires a Select variant or standard Select where one input mask is 0.
        // Standard Select: out = s * x + (1-s) * y
        // Here x = inv (mask r_inv), y = default_val (mask 0).
        // Selector s (mask r_bit).
        r_sel[i] = random_ge(Bout);
        // Warning: keyGenSelect usually for s * x. We might need logic adaptation in API.
        // Assuming we select between Secret(Inv) and Public(1/L).
        // Let's rely on standard Select logic: res = Select(bit, Inv)
        // Then manually add (1-bit) * 1/L in Eval.
        // So we KeyGen for selecting 'Inv' based on 'bit'.
        auto selK = keyGenSelect(Bin, r_bit, r_inv[i], r_sel[i]); 
        k0.selectKeys[i] = selK.first;
        k1.selectKeys[i] = selK.second;
    }

    // 3. KeyGen for Final Multiplication (y * selected_inv)
    for(int i=0; i<s1; ++i) {
        for(int j=0; j<s2; ++j) {
            int idx = i*s2 + j;
            auto mulK = MultGen(r_relu_out[idx], r_sel[i], rout[idx]);
            k0.finalMultKeys[idx] = mulK.first;
            k1.finalMultKeys[idx] = mulK.second;
        }
    }

    delete[] r_relu_out;
    delete[] r_sum;
    delete[] r_inv;
    delete[] r_sel;

    return std::make_pair(k0, k1);
}