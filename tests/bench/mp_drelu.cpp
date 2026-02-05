// Authors: Kanav Gupta, Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <sytorch/backend/FSS_extended.h>
#include <sytorch/backend/FSS_transformer.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <FSS/utils.h>
#include <FSS/api.h>

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

    u64 num_samples = 100;

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
    FSS->initializeInferencePartyB(input);

    FSS::start();
    SlothDrelu(num_samples, 64, input.data, output.data, "sloth_drelu_");
    // NewDrelu(num_samples, 64, input.data, output.data, "new_drelu_");
    FSS::end();

    FSS->outputA(output);
    
    if (party == CLIENT) {
        for (int i = 0; i < num_samples; ++i) {
            mod(output.data[i], 1);
            std::cout << "input=" << i << ", r=" << input_ct.data[i] << ", res=" << output.data[i] << std::endl;
            if (input_ct.data[i] < (1ULL << u64(FSSConfig::bitlength - 1)))
            {
                always_assert(output.data[i] == 1);
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