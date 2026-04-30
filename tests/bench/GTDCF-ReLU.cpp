// Table 5: The communication cost and runtime of different secure ReLU protocols under various input dimensions
#include <backend/FSS_transformer.h>
#include <layers/layers.h>
#include <module.h>
#include <FSS/utils.h>
#include <FSS/api.h>
#include <iostream>
#include <vector>
#include <cstdlib>

uint64_t plain_relu(uint64_t x, int bitlength) {
    if ((x >> (bitlength - 1)) & 1) return 0;
    return x;
}

uint64_t my_rand() {
    uint64_t r1 = (uint64_t)rand();
    uint64_t r2 = (uint64_t)rand();
    uint64_t r3 = (uint64_t)rand();
    uint64_t r4 = (uint64_t)rand();
    return (r1 << 48) ^ (r2 << 32) ^ (r3 << 16) ^ r4;
}

int main(int __argc, char**__argv){
    sytorch_init();

    if(__argc < 2) {
        std::cerr << "Usage: " << __argv[0] << " <party_id> [ip]" << std::endl;
        return 1;
    }

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

    u64 num_samples = 10000;
    
    GroupElement *input = new GroupElement[num_samples];       
    GroupElement *output = new GroupElement[num_samples];      
    GroupElement *input_mask = new GroupElement[num_samples];  
    GroupElement *output_mask = new GroupElement[num_samples]; 
    GroupElement *real_x = new GroupElement[num_samples];      

    srand(0); 

    for(int i=0; i<num_samples; ++i) {
        GroupElement x = my_rand() % (1ULL << 40);
        if (rand() % 2 != 0) x = -x; 
        real_x[i] = x;
        
        GroupElement rin = my_rand(); 

        if (party == DEALER) {
            input_mask[i] = rin; 
        } else {
            input[i] = x + rin; 
        }
        output[i] = 0; output_mask[i] = 0;
    }

    FSS::start();
    auto start = std::chrono::high_resolution_clock::now();
    
    GTDCFReLU(num_samples, input, output, input_mask, output_mask, 8, "GTDCF-ReLU::");

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Protocol Time=" << duration << " ms" << std::endl;
    FSS::end();
    
    if (party == DEALER) {
        FSSConfig::client->send_batched_input(output_mask, num_samples, 64);
    } 
    else if (party == CLIENT) {
        GroupElement *rout_recv = new GroupElement[num_samples];
        FSSConfig::dealer->recv_ge_array(rout_recv, num_samples);

        int correct_cnt = 0;
        for(int i=0; i<num_samples; ++i) {
            GroupElement y_masked = output[i];
            GroupElement y = y_masked - rout_recv[i];
            GroupElement expected = plain_relu(real_x[i], 64);
            
            if (y != expected) {
                if (num_samples - correct_cnt <= 10) {
                    std::cout << "[Error] Idx=" << i 
                              << " Expected=" << (int64_t)expected 
                              << " Got=" << (int64_t)y << std::endl;
                }
            } else {
                correct_cnt++;
            }
        }
        delete[] rout_recv;

        std::cout << "Accuracy: " << correct_cnt << "/" << num_samples << std::endl;
        if (correct_cnt == num_samples) std::cout << "Result: SUCCESS" << std::endl;
        else std::cout << "Result: FAILED" << std::endl;
    }

    FSS->finalize();
    
    delete[] input; delete[] output;
    delete[] input_mask; delete[] output_mask;
    delete[] real_x; delete FSS;

    return 0;
}