// tests/mp_oblivgnn.cpp
#include <FSS/api.h>
#include <FSS/utils.h>
#include <iostream>
#include <vector>

void test_obliv_relu(int party) {
    int size = 10;
    GroupElement *in = new GroupElement[size];
    GroupElement *in_mask = new GroupElement[size];
    GroupElement *out = new GroupElement[size];
    GroupElement *out_mask = new GroupElement[size];

    if (party == DEALER) {
        for(int i=0; i<size; ++i) in_mask[i] = random_ge(64);
    } else {
        for(int i=0; i<size; ++i) {
            int64_t val = (i % 2 == 0) ? (100 * (i+1)) : (-100 * (i+1)); // alternating signs
            in[i] = val; 
        }
    }

    OblivReLUWrapper(size, in, in_mask, out, out_mask);

    if (party == CLIENT) {
        std::cout << "--- OblivReLU Results ---" << std::endl;
        for(int i=0; i<size; ++i) {
            int64_t val = (i % 2 == 0) ? (100 * (i+1)) : (-100 * (i+1));
            int64_t expected = (val > 0) ? val : 0;
            std::cout << "In: " << val << " Out: " << (int64_t)out[i] << " Exp: " << expected << std::endl;
        }
    }
    
    delete[] in; delete[] in_mask; delete[] out; delete[] out_mask;
}

void test_obliv_softmax(int party) {
    int s1 = 2; // batch
    int s2 = 5; // classes
    int size = s1 * s2;
    int sf = 12; // scale factor
    
    GroupElement *in = new GroupElement[size];
    GroupElement *in_mask = new GroupElement[size];
    GroupElement *out = new GroupElement[size];
    GroupElement *out_mask = new GroupElement[size];

    if (party == DEALER) {
        for(int i=0; i<size; ++i) in_mask[i] = random_ge(64);
    } else {
        // Inputs are fixed point
        for(int i=0; i<size; ++i) in[i] = (i+1) * (1ULL << sf); 
    }

    OblivSoftmaxWrapper(s1, s2, in, in_mask, out, out_mask, sf);

    if (party == CLIENT) {
        std::cout << "--- OblivSoftmax Results (Scaled) ---" << std::endl;
        for(int i=0; i<s1; ++i) {
            std::cout << "Batch " << i << ": ";
            for(int j=0; j<s2; ++j) {
                std::cout << (int64_t)out[i*s2+j] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    delete[] in; delete[] in_mask; delete[] out; delete[] out_mask;
}

int main(int argc, char** argv) {
    int party = atoi(argv[1]);
    FSSConfig::party = party;
    FSSConfig::bitlength = 64;
    FSSConfig::num_threads = 1;

    // Setup connection logic here (similar to existing tests)
    // ... init network ...
    
    // FSS::start();
    test_obliv_relu(party);
    test_obliv_softmax(party);
    // FSS::end();
    
    return 0;
}