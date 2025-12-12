#include "../../nn/backend/FSS_extended.h"
// #include <backend/FSS_transformer.h>
#include "../../nn/layers/layers.h"
#include "../../nn/module.h"
#include "../../crypto/FSS/aux_parameter/utils.h"
#include "../../crypto/FSS/api/api.h"



int main(int __argc, char**__argv){

    //sytorch_init();

    int party = atoi(__argv[1]);
    std::string ip = "127.0.0.1";

    using FSSVersion = FSSExtended<u64>;
    FSSVersion *FSS = new FSSVersion();
    srand(time(NULL));

    FSSConfig::bitlength = 64;
    FSSConfig::party = party;
    FSSConfig::num_threads = 4;
    
    if(__argc > 2){
        ip = __argv[2];
    }
    FSS->init(ip, true);

    u64 n_seq = 2048;

    Tensor4D<u64> input(1, 1, 1, n_seq); 
    Tensor4D<i64> input_ct(1, 1, 1, n_seq);

    u64 scale = 12;

    if(party == CLIENT)
    {
        for (int i = 0; i < input.size(); ++i) {
            // input.data[i] = i * (1LL << scale);
            input.data[i] = rand();
            if ((rand() % 2) == 0)
                input.data[i] = -input.data[i];
            input_ct.data[i] = input.data[i];
        }

    }
Tensor4D<u64> output(input.d1, input.d2, input.d3, input.d4);
Tensor4D<i64> output_ct(input.d1, input.d2, input.d3, input.d4);
    FSS->initializeInferencePartyB(input);

    
    auto start_time = std::chrono::high_resolution_clock::now();
    FSS::start();
    //for (int i = 0; i < 144; ++i)
    FSS->softmax(input, output, scale, 0);
    FSS::end();


    auto end_time = std::chrono::high_resolution_clock::now();

    // 4. 计算时间差并打印
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    

    std::cout << "================================================" << std::endl;
    std::cout << "Total execution time: " << duration.count() << " milliseconds" << std::endl;
    std::cout << "Total execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
    std::cout << "================================================" << std::endl;

    FSS->finalize();

    return 0;
}