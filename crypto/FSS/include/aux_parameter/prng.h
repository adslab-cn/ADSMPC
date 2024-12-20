#pragma once
#include <cryptoTools/Crypto/PRNG.h>

namespace FSSConfig {
    extern osuCrypto::PRNG prngs[256];
}

extern osuCrypto::PRNG prngShared;
