/*
    MPC框架中的伪随机数生成器（PRNG）配置
*/

#pragma once
#include <cryptoTools/Crypto/PRNG.h>

namespace FSSConfig {
    extern osuCrypto::PRNG prngs[256];
}

extern osuCrypto::PRNG prngShared;
