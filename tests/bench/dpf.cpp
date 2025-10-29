/* DPF评估 */
#include "../../crypto/FSS/primitives/dpf.h"
#include <iostream>
#include "../../nn/backend/FSS_base.h"

void DPF_TEST()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int samples = 3000;
    int bin = 64;
    int bout = 64;

    std::pair<DPFKeyPack, DPFKeyPack> *keys = new std::pair<DPFKeyPack, DPFKeyPack>[samples];

    GroupElement *r = new GroupElement[samples];
    GroupElement *op = new GroupElement[samples];

    

    for (int idx = 0; idx < samples; ++idx)
    {
        r[idx] = rand();
        keys[idx] = keyGenDPF(bin, bout, r[idx], 1);
        // key0[idx] = keys[idx].first;
        // key1 = keys.second;
    }

    auto startEQ = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; ++i)
    {
        // 分别计算两方的评估结果
             
        // GroupElement res1 = evalDPF_EQ(1, keys[i].second, i);
        // GroupElement xor_result = res0 ^ res1;  // 异或结果应为 0 或 1
        auto y = (evalDPF_EQ(0, keys[i].first, i) ^ evalDPF_EQ(1, keys[i].second, i));
        if (i == r[i])
        {
            always_assert(y == 1);
        }
        else
        {
            always_assert(y == 0);
        }
                
    }
    // 计时结束
    auto endEQ = std::chrono::high_resolution_clock::now();
    auto durationEQ = std::chrono::duration_cast<std::chrono::nanoseconds>(endEQ - startEQ).count();
    std::cout << "EQTime=" << durationEQ << " ns" << std::endl;
    

    // auto startGT = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < samples; ++i)
    // {
    //     auto y = (evalDPF_GT(0, keys[i].first, i) ^ evalDPF_GT(1, keys[i].second, i));
    //     if (i > r[i])
    //     {
    //         always_assert(y == 1);
    //     }
    //     else
    //     {
    //         always_assert(y == 0);
    //     }
    // }
    // // 计时结束
    // auto endGT = std::chrono::high_resolution_clock::now();
    // auto durationGT = std::chrono::duration_cast<std::chrono::nanoseconds>(endGT - startGT).count();
    // std::cout << "GTTime=" << durationGT << " ns" << std::endl;

        // for (int i = 0; i < 16; ++i)
        // {
        //     auto y = (evalDPF_LT(0, key0, i) ^ evalDPF_LT(1, key1, i));
        //     if (i < idx)
        //     {
        //         always_assert(y == 1);
        //     }
        //     else
        //     {
        //         always_assert(y == 0);
        //     }
        // }

        // GroupElement out0[16];
        // GroupElement out1[16];
        // evalAll(0, key0, 0, out0);
        // evalAll(1, key1, 0, out1);
        // for (int i = 0; i < 16; ++i)
        // {
        //     auto y = (out0[i] + out1[i]);
        //     if (i == idx)
        //     {
        //         always_assert(y == 1);
        //     }
        //     else
        //     {
        //         always_assert(y == 0);
        //     }
        // }

        // evalAll(0, key0, 7, out0);
        // evalAll(1, key1, 7, out1);
        // for (int i = 0; i < 16; ++i)
        // {
        //     auto y = (out0[i] + out1[i]);
        //     if (i == ((idx+7)%16))
        //     {
        //         always_assert(y == 1);
        //     }
        //     else
        //     {
        //         always_assert(y == 0);
        //     }
        // }

        // GroupElement res0, res1;
        // std::vector<GroupElement> tab(16);
        // for (int i = 0; i < 16; ++i)
        // {
        //     tab[i] = rand();
        // }
        // res0 = evalAll_reduce(0, key0, 0, tab);
        // res1 = evalAll_reduce(1, key1, 0, tab);
        // always_assert(res0 + res1 == tab[idx]);
        // GroupElement final_result = res0 + res1; // 计算最终结果
        
        // // 添加输出语句
        // std::cout << "Result for idx=" << idx << ": " << final_result << std::endl;

    
    // 输出详细信息
            // std::cout << "idx=" << idx << ", i=" << i 
            //         << ": res0=" << res0 << ", res1=" << res1 
            //         << ", XOR=" << xor_result << std::endl;


    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    // std::cout << ", Time=" << duration << " ns" << std::endl;
}


void DPFET_TEST()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int samples = 10;
    int bin = 64;
    int bout = 64;

    std::pair<DPFETKeyPack, DPFETKeyPack> *keys = new std::pair<DPFETKeyPack, DPFETKeyPack>[samples];

    GroupElement *alpha = new GroupElement[samples];
    GroupElement *r = new GroupElement[samples];
    GroupElement *op = new GroupElement[samples];

    

    for (int idx = 0; idx < samples; ++idx)
    {
        // r[idx] = rand();
        alpha[idx] = 5;
        // keys[idx] = keyGenDPF(bin, bout, r[idx], 1);
        keys[idx] = keyGenDPFET(bin, alpha[idx]);
        // key0[idx] = keys[idx].first;
        // key1 = keys.second;
    }

    auto startGT = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; i++)
    {   
        auto y = (evalDPFET_LT(0, keys[i].first, i) ^ evalDPFET_LT(1, keys[i].second, i));
        std::cout << "x=" << i << ", alpha=" << alpha[i] << ", res=" << y << std::endl;
        if (i < alpha[i])
        {
            always_assert(y == 1);
        }
        else
        {
            always_assert(y == 0);
        }
    }
    // 计时结束
    auto endGT = std::chrono::high_resolution_clock::now();
    auto durationGT = std::chrono::duration_cast<std::chrono::nanoseconds>(endGT - startGT).count();
    std::cout << "GTTime=" << durationGT << " ns" << std::endl;

}



void DPFIC_TEST()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int samples = 10;
    int bin = 64;
    int bout = 64;

    std::pair<DPFETKeyPack, DPFETKeyPack> *keys = new std::pair<DPFETKeyPack, DPFETKeyPack>[samples];

    GroupElement *alpha = new GroupElement[samples];
    GroupElement *r = new GroupElement[samples];
    GroupElement *op = new GroupElement[samples];
    GroupElement *j = new GroupElement[samples];

    

    for (int idx = 0; idx < samples; ++idx)
    {
        // r[idx] = rand();
        alpha[idx] = 5000;
        // keys[idx] = keyGenDPF(bin, bout, r[idx], 1);
        keys[idx] = keyGenDPFET(bin, alpha[idx]);
        // key0[idx] = keys[idx].first;
        // key1 = keys.second;
    }

    GroupElement a = 2;
    GroupElement b = 6;
    auto startGT = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; i++)
    {   j[i] = i - alpha[i];
        auto res1 = (evalDPFET_LT(0, keys[i].first, a-j[i]) ^ evalDPFET_LT(1, keys[i].second, a-j[i]));
        auto res2 = (evalDPFET_LT(0, keys[i].first, b-j[i]) ^ evalDPFET_LT(1, keys[i].second, b-j[i]));
        // std::cout << "a-j[i]=" << a-j[i] << ", b-j[i]=" << b-j[i] << std::endl;
        // std::cout << "res1=" << res1 << ", res2=" << res2 << std::endl;
        
        auto res = res1 ^ res2;
        std::cout << "x=" << i << ", a=" << a << ", b=" << b << ", res=" << res << std::endl;
        std::cout << "==========" << std::endl;
        // if (i >= alpha[i])
        // {
        //     always_assert(y == 1);
        // }
        // else
        // {
        //     always_assert(y == 0);
        // }
    }
    // 计时结束
    auto endGT = std::chrono::high_resolution_clock::now();
    auto durationGT = std::chrono::duration_cast<std::chrono::nanoseconds>(endGT - startGT).count();
    std::cout << "GTTime=" << durationGT << " ns" << std::endl;

}

int main(){
    DPF_TEST();
    // DPFET_TEST();
    // DPFIC_TEST();
}
