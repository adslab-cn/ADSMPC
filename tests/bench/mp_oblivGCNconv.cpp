#include <sytorch/backend/FSS_extended.h>
#include <sytorch/backend/FSS_transformer.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <FSS/utils.h>
#include <FSS/api.h>

// Helper for plaintext matrix multiplication
void plain_matmul(int s1, int s2, int s3, u64* A, u64* B, u64* C) {
    for (int i = 0; i < s1; ++i) {
        for (int j = 0; j < s3; ++j) {
            C[i * s3 + j] = 0;
            for (int k = 0; k < s2; ++k) {
                C[i * s3 + j] += A[i * s2 + k] * B[k * s3 + j];
            }
        }
    }
}


int main(int __argc, char**__argv){

    sytorch_init();

    int party = atoi(__argv[1]);
    std::string ip = "127.0.0.1";

    using FSSVersion = FSSTransformer<u64>;
    FSSVersion *FSS = new FSSVersion();
    srand(time(NULL));

    FSSConfig::bitlength = 64;
    FSSConfig::party = party;
    FSSConfig::num_threads = 4;
    
    if(__argc > 2){
        ip = __argv[2];
    }
    FSS->init(ip, true);

    // Define dimensions for GCN layer
    // For simplicity, let's use small dimensions for testing
    const int N = 19717;      // Number of nodes 2708  3327  19717
    const int C_in = 500;   // Input features 1433  3703  500
    const int C_out = 256;  // Output features 256

    // Input Tensors
    Tensor<u64> A_hat({N, N});
    Tensor<u64> F_in({N, C_in});
    Tensor<u64> W({C_in, C_out});
    
    // Plaintext tensors for verification
    Tensor<u64> A_hat_pt({N, N});
    Tensor<u64> F_in_pt({N, C_in});
    Tensor<u64> W_pt({C_in, C_out});

    // Output Tensor
    Tensor<u64> F_out({N, C_out});

    if(party == CLIENT)
    {
        // Initialize with random data on the client side
        for (int i = 0; i < N * N; ++i) {
            A_hat.data[i] = random_ge(10); // small values to avoid overflow
            A_hat_pt.data[i] = A_hat.data[i];
        }
        for (int i = 0; i < N * C_in; ++i) {
            F_in.data[i] = random_ge(10);
            F_in_pt.data[i] = F_in.data[i];
        }
        for (int i = 0; i < C_in * C_out; ++i) {
            W.data[i] = random_ge(10);
            W_pt.data[i] = W.data[i];
        }
    }

    // Share the inputs
    FSS->initializeInferencePartyB(A_hat);
    FSS->initializeInferencePartyB(F_in);
    FSS->initializeInferencePartyB(W);

    FSS::start();
    
    // Execute the secure GCNConv layer
    GCNConv(N, C_in, C_out, 
                 A_hat.data, A_hat.data,
                 F_in.data, F_in.data,
                 W.data, W.data,
                 F_out.data, F_out.data);
    
    FSS::end();

    // Reconstruct the output
    FSS->outputA(F_out);
    
    // Verification on the client side
    if (party == CLIENT) {
        std::cout << "Verification Phase" << std::endl;
        
        // Plaintext computation
        Tensor<u64> Temp_pt({N, C_in});
        Tensor<u64> F_out_pt({N, C_out});

        plain_matmul(N, N, C_in, A_hat_pt.data, F_in_pt.data, Temp_pt.data);
        plain_matmul(N, C_in, C_out, Temp_pt.data, W_pt.data, F_out_pt.data);

        // Compare results
        bool pass = true;
        for (int i = 0; i < N * C_out; ++i) {
            mod(F_out.data[i], FSSConfig::bitlength);
            mod(F_out_pt.data[i], FSSConfig::bitlength);
            if (F_out.data[i] != F_out_pt.data[i]) {
                pass = false;
                std::cout << "Mismatch at index " << i << ": ";
                std::cout << "Secure=" << F_out.data[i] << ", Plaintext=" << F_out_pt.data[i] << std::endl;
            }
        }

        if (pass) {
            std::cout << "GCNConv Test Passed!" << std::endl;
        } else {
            std::cout << "GCNConv Test Failed!" << std::endl;
        }
    }

    FSS->finalize();

    return 0;
}