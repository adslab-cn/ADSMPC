#include <backend/FSS_extended.h>
// #include <backend/FSS_transformer.h>
#include <layers/layers.h>
#include <module.h>
#include <aux_parameter/utils.h>
#include <api/api.h>



int main(int __argc, char**__argv){

    sytorch_init();

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

    u64 n_seq = 10;

    Tensor<u64> input({n_seq, n_seq});
    Tensor<i64> input_ct(input.shape);

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
    Tensor<u64> output(input.shape);
    Tensor<i64> output_ct(input.shape);
    FSS->initializeInferencePartyB(input);

    FSS::start();
    for (int i = 0; i < 144; ++i)
        FSS->softmax(input, output, scale, 0);
    FSS::end();

    ClearText<i64> *ct = new ClearText<i64>();
    ct->softmax(input_ct, output_ct, scale, 1);

    FSS->outputA(output);
    if (party == CLIENT) {
        for (int i = 0; i < input.size(); ++i) {
            i64 diff = std::abs((i64)output.data[i] - output_ct.data[i]);
            always_assert(diff == 0);
        }
    }
    FSS->finalize();

    return 0;
}