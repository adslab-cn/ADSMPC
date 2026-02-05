// DPF正确性测试

#include <FSS/dpf.h>
#include <iostream>
#include <sytorch/backend/FSS_base.h>

void DPF_TEST()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int bin = 4;
    int bout = 64;

    for (int idx = 0; idx < 16; ++idx)
    {
        auto keys = keyGenDPF(bin, bout, idx, 1);
        auto& key0 = keys.first;
        auto& key1 = keys.second;

        for (int i = 0; i < 16; ++i)
        {
            auto y = (evalDPF_EQ(0, key0, i) ^ evalDPF_EQ(1, key1, i));
            if (i == idx)
            {
                always_assert(y == 1);
            }
            else
            {
                always_assert(y == 0);
            }
        }

        for (int i = 0; i < 16; ++i)
        {
            auto y = (evalDPF_GT(0, key0, i) ^ evalDPF_GT(1, key1, i));
            if (i > idx)
            {
                always_assert(y == 1);
            }
            else
            {
                always_assert(y == 0);
            }
        }

        for (int i = 0; i < 16; ++i)
        {
            auto y = (evalDPF_LT(0, key0, i) ^ evalDPF_LT(1, key1, i));
            if (i < idx)
            {
                always_assert(y == 1);
            }
            else
            {
                always_assert(y == 0);
            }
        }

        GroupElement out0[16];
        GroupElement out1[16];
        evalAll(0, key0, 0, out0);
        evalAll(1, key1, 0, out1);
        for (int i = 0; i < 16; ++i)
        {
            auto y = (out0[i] + out1[i]);
            if (i == idx)
            {
                always_assert(y == 1);
            }
            else
            {
                always_assert(y == 0);
            }
        }

        evalAll(0, key0, 7, out0);
        evalAll(1, key1, 7, out1);
        for (int i = 0; i < 16; ++i)
        {
            auto y = (out0[i] + out1[i]);
            if (i == ((idx+7)%16))
            {
                always_assert(y == 1);
            }
            else
            {
                always_assert(y == 0);
            }
        }

        GroupElement res0, res1;
        std::vector<GroupElement> tab(16);
        for (int i = 0; i < 16; ++i)
        {
            tab[i] = rand();
        }
        res0 = evalAll_reduce(0, key0, 0, tab);
        res1 = evalAll_reduce(1, key1, 0, tab);
        always_assert(res0 + res1 == tab[idx]);

    }
}


void DPFET_TEST()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int samples = 1000;
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
        // std::cout << "x=" << i << ", alpha=" << alpha[i] << ", res=" << y << std::endl;
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

void DPFGT_TEST()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int samples = 10000;
    int bin = 64;
    int bout = 64;

    std::pair<DPFETKeyPack, DPFETKeyPack> *keys = new std::pair<DPFETKeyPack, DPFETKeyPack>[samples];

    GroupElement *alpha = new GroupElement[samples];
    GroupElement *r = new GroupElement[samples];
    GroupElement *op = new GroupElement[samples];

    

    for (int idx = 0; idx < samples; ++idx)
    {
        r[idx] = rand();
        // alpha[idx] = 5;
        // keys[idx] = keyGenDPF(bin, bout, r[idx], 1);
        // keys[idx] = keyGenGTDPF(bin, alpha[idx]);
        keys[idx] = keyGenGTDPF(bin, r[idx]);
        // key0[idx] = keys[idx].first;
        // key1 = keys.second;
    }

    auto startGT = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; i++)
    {   
        auto y = (evalGTDPF(0, keys[i].first, i) ^ evalGTDPF(1, keys[i].second, i));
        std::cout << "x=" << i << ", r=" << r[i] << ", res=" << y << std::endl;
        if (i >= r[i])
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

int main(){
    // DPF_TEST();
    // DPFET_TEST();
    DPFGT_TEST();
}