// SIGMA-CPU-应该是（

#include <backend/FSS_extended.h>
// #include <backend/FSS_transformer.h>
// #include <layers/layers.h>
#include <module.h>
#include <aux_parameter/utils.h>
#include <api/api.h>

int main(int __argc, char**__argv){

    sytorch_init();

    int party = atoi(__argv[1]);
    std::string ip = "127.0.0.1";

    // using FSSVersion = FSSTransformer<u64>;
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

    u64 num_samples = 10000;

    Tensor<u64> input({num_samples});
    Tensor<u64> input_ct({num_samples});

    if(party == CLIENT)
    {
        for (int i = 0; i < num_samples; ++i) {
            input.data[i] = random_ge(FSSConfig::bitlength);
            input_ct.data[i] = input.data[i];
        }
    }
    Tensor<u64> output({num_samples});
    Tensor<u64> drelu({num_samples});
    FSS->initializeInferencePartyB(input);

    FSS::start();
    // Relu(num_samples, 64, input.data, output.data);
    Relu(10000, input.data, input.data, output.data, output.data, drelu.data);
    // Relu(64, input.data, output.data, drelu.data);
    FSS::end();

    FSS->outputA(output);
    
    if (party == CLIENT) {
        for (int i = 0; i < num_samples; ++i) {
            mod(output.data[i], FSSConfig::bitlength);
            if (input_ct.data[i] < (1ULL << (FSSConfig::bitlength - 1)))
            {
                always_assert(output.data[i] == input_ct.data[i]);
            }
            else
            {
                always_assert(output.data[i] == 0);
            }
        }
    }
    FSS->finalize();

    return 0;
}