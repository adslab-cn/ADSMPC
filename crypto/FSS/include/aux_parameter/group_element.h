#pragma once
#include <vector>
#include <cstdint>
#include <iostream>
#include <cryptoTools/Common/Defines.h>
#include <cryptoTools/Crypto/PRNG.h>
#include <aux_parameter/config.h>
#include <aux_parameter/prng.h>
#include <omp.h>

using GroupElement = uint64_t;

inline void mod(GroupElement &a, int bw)
{
    if (bw != 64)
        a = a & ((uint64_t(1) << bw) - 1); 
}

inline GroupElement random_ge(int bw)
{
    GroupElement a;
    int tid = omp_get_thread_num();
    a = FSSConfig::prngs[tid].get<uint64_t>();
    mod(a, bw);
    return a;
}

inline std::pair<GroupElement, GroupElement> splitShare(const GroupElement& a, int bw)
{
    GroupElement a1, a2;
    a1 = random_ge(bw);
    // a1 = 0;
    mod(a1, bw);
    a2 = a - a1;
    mod(a2, bw);
    return std::make_pair(a1, a2);
}

inline std::pair<GroupElement, GroupElement> splitShareXor(const GroupElement& a, int bw)
{
    GroupElement a1, a2;
    a1 = random_ge(bw);
    a2 = a ^ a1;
    return std::make_pair(a1, a2);
}

inline std::pair<GroupElement, GroupElement> splitShareCommonPRNG(const GroupElement& a, int bw)
{
    GroupElement a1, a2;
    a1 = prngShared.get<uint64_t>();
    // a1 = 0;
    mod(a1, bw);
    a2 = a - a1;
    mod(a2, bw);
    return std::make_pair(a1, a2);
}

inline GroupElement pow(GroupElement x, uint64_t e)
{
    if (e == 0)
    {
        return 1;
    }
    GroupElement res = pow(x, e / 2);
    if (e % 2 == 0)
    {
        return res * res;
    }
    else
    {
        return res * res * x;
    }
}

inline GroupElement msb(GroupElement a, int bw)
{
    return (a >> (bw - 1)) & 1;
}