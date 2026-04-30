#include <FSS/dpf.h>
#include <FSS/assert.h>
#include <cassert>

using namespace osuCrypto;

inline u8 lsb(const block &b)
{
    return _mm_cvtsi128_si64x(b) & 1;
}

/*
 * lambda = 127
 */
std::pair<DPFKeyPack, DPFKeyPack> keyGenDPF(int bin, int bout, GroupElement idx, GroupElement payload)
{
    always_assert(bin <= 64);
    always_assert(bout <= 64);
    static const block notOneBlock = toBlock(~0, ~1);
    const static block pt[2] = {ZeroBlock, OneBlock};

    DPFKeyPack key0(bin, bout);
    DPFKeyPack key1(bin, bout);

    int tid = omp_get_thread_num();
    auto s = FSSConfig::prngs[tid].get<std::array<block, 2>>();
    auto s0 = s[0];
    auto s1 = s[1];

    s0 = s0 & notOneBlock;
    s1 = s1 & notOneBlock;
    key0.s[0] = s0;
    key1.s[0] = s1;
    
    u8 t0 = 0;
    u8 t1 = 1;

    block ct0[2];
    block ct1[2];

    for (int i = 0; i < bin; ++i)
    {
        const u8 keep = static_cast<uint8_t>(idx >> (bin - 1 - i)) & 1;
        const u8 loose = keep ^ 1;

        AES ak0(s0);
        AES ak1(s1);

        ak0.ecbEncTwoBlocks(pt, ct0);
        ak1.ecbEncTwoBlocks(pt, ct1);

        auto scw = (ct0[loose] ^ ct1[loose]) & notOneBlock;
        u64 tcw[2];
        u64 &tLcw = tcw[0];
        u64 &tRcw = tcw[1];
        tLcw = lsb(ct0[0]) ^ lsb(ct1[0]) ^ keep ^ 1;
        tRcw = lsb(ct0[1]) ^ lsb(ct1[1]) ^ keep;

        key0.s[i+1] = scw;
        key1.s[i+1] = scw;
        key0.tLcw |= (tLcw << (bin - 1 - i));
        key0.tRcw |= (tRcw << (bin - 1 - i));

        if (t0 == 0)
        {
            s0 = ct0[keep] & notOneBlock;
            t0 = lsb(ct0[keep]);
        }
        else
        {
            s0 = (ct0[keep] & notOneBlock) ^ scw;
            t0 = lsb(ct0[keep]) ^ tcw[keep];
        }

        if (t1 == 0)
        {
            s1 = ct1[keep] & notOneBlock;
            t1 = lsb(ct1[keep]);
        }
        else
        {
            s1 = (ct1[keep] & notOneBlock) ^ scw;
            t1 = lsb(ct1[keep]) ^ tcw[keep];
        }
    }

    key1.tLcw = key0.tLcw;
    key1.tRcw = key0.tRcw;

    key0.payload = payload - _mm_extract_epi64(s0, 0) + _mm_extract_epi64(s1, 0);
    if (t1 == 1) key0.payload = -key0.payload;
    key1.payload = key0.payload;

    return std::make_pair(key0, key1);
}

GroupElement evalDPF_EQ(int party, DPFKeyPack &key, GroupElement x)
{
    static const block notOneBlock = toBlock(~0, ~1);
    int bin = key.bin;
    int bout = key.bout;

    auto s = _mm_loadu_si128(key.s);
    u8 t = party;
    for (int i = 0; i < bin; ++i)
    {
        assert(lsb(s) == 0);
        const u8 x_i = static_cast<uint8_t>(x >> (bin - 1 - i)) & 1;
        
        AES ak(s);
        block ct = ak.ecbEncBlock(toBlock(0, x_i));
        s = ct & notOneBlock;
        u8 t_old = t;
        t = lsb(ct);

        if (t_old) {
            s = s ^ _mm_loadu_si128(key.s + i + 1);
            t = t ^ ((key.tcw[x_i] >> (bin - 1 - i)) & 1);
        }
    }

    return t;
}

GroupElement evalDPF_GT(int party, DPFKeyPack &key, GroupElement x)
{
    static const block notOneBlock = toBlock(~0, ~1);
    int bin = key.bin;
    int bout = key.bout;

    auto s = _mm_loadu_si128(key.s);
    u8 t = party;
    
    u8 x_prev = 0;
    u8 t_dcf = 0;

    for (int i = 0; i < bin; ++i)
    {
        assert(lsb(s) == 0);
        const u8 x_i = static_cast<uint8_t>(x >> (bin - 1 - i)) & 1;

        if (x_prev != x_i)
        {
            t_dcf = t_dcf ^ t;
        }
        x_prev = x_i;

        AES ak(s);
        block ct = ak.ecbEncBlock(toBlock(0, x_i));
        s = ct & notOneBlock;
        u8 t_old = t;
        t = lsb(ct);
        

        if (t_old) {
            s = s ^ _mm_loadu_si128(key.s + i + 1);
            t = t ^ ((key.tcw[x_i] >> (bin - 1 - i)) & 1);
        }
    }

    if (x_prev == 1)
    {
        t_dcf = t_dcf ^ t;
    }
    return t_dcf;
}

GroupElement evalDPF_LT(int party, DPFKeyPack &key, GroupElement x)
{
    static const block notOneBlock = toBlock(~0, ~1);
    int bin = key.bin;
    int bout = key.bout;

    auto s = _mm_loadu_si128(key.s);
    u8 t = party;
    
    u8 x_prev = 1;
    u8 t_dcf = 0;

    for (int i = 0; i < bin; ++i)
    {
        assert(lsb(s) == 0);
        const u8 x_i = static_cast<uint8_t>(x >> (bin - 1 - i)) & 1;

        if (x_prev != x_i)
        {
            t_dcf = t_dcf ^ t;
        }
        x_prev = x_i;

        AES ak(s);
        block ct = ak.ecbEncBlock(toBlock(0, x_i));
        s = ct & notOneBlock;
        u8 t_old = t;
        t = lsb(ct);
        

        if (t_old) {
            s = s ^ _mm_loadu_si128(key.s + i + 1);
            t = t ^ ((key.tcw[x_i] >> (bin - 1 - i)) & 1);
        }
    }

    if (x_prev == 0)
    {
        t_dcf = t_dcf ^ t;
    }
    return t_dcf;
}

void evalAll_helper(int party, DPFKeyPack &key, GroupElement rightShift, GroupElement *out, block s_prev, u8 t_prev, int i, GroupElement acc)
{
    if (i == key.bin)
    {
        GroupElement idx = acc + rightShift;
        mod(idx, key.bin);
        out[idx] = (1 - 2 * party) * (_mm_extract_epi64(s_prev, 0) + key.payload * t_prev);
        return;
    }

    const static block pt[2] = {ZeroBlock, OneBlock};
    static const block notOneBlock = toBlock(~0, ~1);

    AES ak(s_prev);
    block ct[2];
    ak.ecbEncTwoBlocks(pt, ct);

    for (int x_i = 0; x_i < 2; ++x_i)
    {
        block s = ct[x_i] & notOneBlock;
        u8 t = lsb(ct[x_i]);

        if (t_prev) {
            s = s ^ _mm_loadu_si128(key.s + i + 1);
            t = t ^ ((key.tcw[x_i] >> (key.bin - 1 - i)) & 1);
        }

        evalAll_helper(party, key, rightShift, out, s, t, i+1, 2 * acc + x_i);
    }
}

void evalAll(int party, DPFKeyPack &key, GroupElement rightShift, GroupElement *out)
{
    auto s = _mm_loadu_si128(key.s);
    u8 t = party;
    
    evalAll_helper(party, key, rightShift, out, s, t, 0, 0);
}

void evalAll_reduce_helper(int party, DPFKeyPack &key, GroupElement rightShift, const std::vector<GroupElement> &tab, GroupElement &out, block &s_prev, u8 t_prev, int i, GroupElement acc)
{
    if (i == key.bin)
    {
        GroupElement idx = acc + rightShift;
        mod(idx, key.bin);
        out = out + tab[idx] * ((1 - 2 * party) * (_mm_extract_epi64(s_prev, 0) + key.payload * t_prev));
        return;
    }

    const static block pt[2] = {ZeroBlock, OneBlock};
    static const block notOneBlock = toBlock(~0, ~1);

    AES ak(s_prev);
    block ct[2];
    ak.ecbEncTwoBlocks(pt, ct);

    for (int x_i = 0; x_i < 2; ++x_i)
    {
        block s = ct[x_i] & notOneBlock;
        u8 t = lsb(ct[x_i]);

        if (t_prev) {
            s = s ^ _mm_loadu_si128(key.s + i + 1);
            t = t ^ ((key.tcw[x_i] >> (key.bin - 1 - i)) & 1);
        }

        evalAll_reduce_helper(party, key, rightShift, tab, out, s, t, i+1, 2 * acc + x_i);
    }
}

GroupElement evalAll_reduce(int party, DPFKeyPack &key, GroupElement rightShift, const std::vector<GroupElement> &tab)
{
    auto s = _mm_loadu_si128(key.s);
    u8 t = party;
    GroupElement out = 0;
    
    evalAll_reduce_helper(party, key, rightShift, tab, out, s, t, 0, 0);
    return out;
}

std::pair<DPFETKeyPack, DPFETKeyPack> keyGenDPFET(int bin, GroupElement idx)
{
    always_assert(bin <= 64);
    always_assert(bin >= 8);
    static const block notOneBlock = toBlock(~0, ~1);
    const static block pt[2] = {ZeroBlock, OneBlock};

    DPFETKeyPack key0(bin);
    DPFETKeyPack key1(bin);

    int tid = omp_get_thread_num();
    auto s = FSSConfig::prngs[tid].get<std::array<block, 2>>();
    auto s0 = s[0];
    auto s1 = s[1];

    s0 = s0 & notOneBlock;
    s1 = s1 & notOneBlock;
    key0.s[0] = s0;
    key1.s[0] = s1;
    
    u8 t0 = 0;
    u8 t1 = 1;

    block ct0[2];
    block ct1[2];

    for (int i = 0; i < bin - 7; ++i)
    {
        const u8 keep = static_cast<uint8_t>(idx >> (bin - 1 - i)) & 1;
        const u8 loose = keep ^ 1;

        AES ak0(s0);
        AES ak1(s1);

        ak0.ecbEncTwoBlocks(pt, ct0);
        ak1.ecbEncTwoBlocks(pt, ct1);

        auto scw = (ct0[loose] ^ ct1[loose]) & notOneBlock;
        u64 tcw[2];
        u64 &tLcw = tcw[0];
        u64 &tRcw = tcw[1];
        tLcw = lsb(ct0[0]) ^ lsb(ct1[0]) ^ keep ^ 1;
        tRcw = lsb(ct0[1]) ^ lsb(ct1[1]) ^ keep;

        key0.s[i+1] = scw;
        key1.s[i+1] = scw;
        key0.tLcw |= (tLcw << (bin - 1 - i));
        key0.tRcw |= (tRcw << (bin - 1 - i));

        if (t0 == 0)
        {
            s0 = ct0[keep] & notOneBlock;
            t0 = lsb(ct0[keep]);
        }
        else
        {
            s0 = (ct0[keep] & notOneBlock) ^ scw;
            t0 = lsb(ct0[keep]) ^ tcw[keep];
        }

        if (t1 == 0)
        {
            s1 = ct1[keep] & notOneBlock;
            t1 = lsb(ct1[keep]);
        }
        else
        {
            s1 = (ct1[keep] & notOneBlock) ^ scw;
            t1 = lsb(ct1[keep]) ^ tcw[keep];
        }
    }

    key1.tLcw = key0.tLcw;
    key1.tRcw = key0.tRcw;

    if (t0 == 1) s0 = s0 ^ OneBlock;
    if (t1 == 1) s1 = s1 ^ OneBlock;
    uint64_t e0, e1;
    GroupElement ip = idx % 128;
    if (ip >= 64) {
        e0 = 0;
        e1 = 1ULL << (127 - ip);
    }
    else {
        e0 = 1ULL << (63 - ip);
        e1 = 0;
    }
    key0.leaf = s0 ^ s1 ^ osuCrypto::toBlock(e0, e1);
    key1.leaf = key0.leaf;

    return std::make_pair(key0, key1);
}

GroupElement evalDPFET_LT(int party, const DPFETKeyPack &key, GroupElement x)
{
    static const block notOneBlock = toBlock(~0, ~1);
    int bin = key.bin;

    auto s = _mm_loadu_si128(key.s);
    u8 t = party;
    
    u8 x_prev = 1;
    u8 t_dcf = 0;

    for (int i = 0; i < bin - 7; ++i)
    {
        assert(lsb(s) == 0);
        const u8 x_i = static_cast<uint8_t>(x >> (bin - 1 - i)) & 1;

        if (x_prev != x_i)
        {
            t_dcf = t_dcf ^ t;
        }
        x_prev = x_i;

        AES ak(s);
        block ct = ak.ecbEncBlock(toBlock(0, x_i));
        s = ct & notOneBlock;
        u8 t_old = t;
        t = lsb(ct);

        if (t_old) {
            s = s ^ _mm_loadu_si128(key.s + i + 1);
            t = t ^ ((key.tcw[x_i] >> (bin - 1 - i)) & 1);
        }
    }

    osuCrypto::block leaf = s;
    if (t) leaf = leaf ^ OneBlock ^ key.leaf;
    uint64_t b;

    {
        const u8 x_i = static_cast<uint8_t>(x >> 6) & 1;
        if (x_prev != x_i)
        {
            t_dcf = t_dcf ^ t;
        }
        x_prev = x_i;
        if (x_i) 
        {
            b = _mm_extract_epi64(leaf, 0);
        }
        else 
        {
            b = _mm_extract_epi64(leaf, 1);
        }
        t = __builtin_parityll(b);
    }

    GroupElement xp = x % 64;

    if (x_prev == 0)
    {
        for (int i = 0; i <= xp; ++i)
        {
            t_dcf = t_dcf ^ ((b >> (63 - i)) & 1);
        }
    }
    else
    {
        for (int i = xp + 1; i < 64; ++i)
        {
            t_dcf = t_dcf ^ ((b >> (63 - i)) & 1);
        }
    }

    return t_dcf;
}

void evalAll_reduce_helper_et(int party, DPFETKeyPack &key, GroupElement rightShift, const std::vector<GroupElement> &tab, GroupElement &out, GroupElement &corr, block &s_prev, u8 t_prev, int i, GroupElement acc)
{
    if (i == key.bin - 7)
    {
        osuCrypto::block leaf = s_prev;
        if (t_prev) leaf = leaf ^ OneBlock ^ key.leaf;
        
        uint64_t b = _mm_extract_epi64(leaf, 1);
        for (int j = 0; j < 64; ++j) {
            GroupElement idx = 128 * acc + j + rightShift;
            mod(idx, key.bin);
            GroupElement e = ((1 - 2 * party) * ((b >> (63 - j)) & 1));
            out = out + tab[idx] * e;
            corr = corr + e;
        }

        b = _mm_extract_epi64(leaf, 0);
        for (int j = 64; j < 128; ++j) {
            GroupElement idx = 128 * acc + j + rightShift;
            mod(idx, key.bin);
            GroupElement e = ((1 - 2 * party) * ((b >> (127 - j)) & 1));
            out = out + tab[idx] * e;
            corr = corr + e;
        }
        return;
    }

    const static block pt[2] = {ZeroBlock, OneBlock};
    static const block notOneBlock = toBlock(~0, ~1);

    AES ak(s_prev);
    block ct[2];
    ak.ecbEncTwoBlocks(pt, ct);

    for (int x_i = 0; x_i < 2; ++x_i)
    {
        block s = ct[x_i] & notOneBlock;
        u8 t = lsb(ct[x_i]);

        if (t_prev) {
            s = s ^ _mm_loadu_si128(key.s + i + 1);
            t = t ^ ((key.tcw[x_i] >> (key.bin - 1 - i)) & 1);
        }

        evalAll_reduce_helper_et(party, key, rightShift, tab, out, corr, s, t, i+1, 2 * acc + x_i);
    }
}

std::pair<GroupElement, GroupElement> evalAll_reduce_et(int party, DPFETKeyPack &key, GroupElement rightShift, const std::vector<GroupElement> &tab)
{
    auto s = _mm_loadu_si128(key.s);
    u8 t = party;
    GroupElement out = 0;
    GroupElement corr = 0;
    
    evalAll_reduce_helper_et(party, key, rightShift, tab, out, corr, s, t, 0, 0);
    return std::make_pair(out, corr);
}

std::pair<DPFETKeyPack, DPFETKeyPack> keyGenGTDPF(int bin, GroupElement idx)
{
    always_assert(bin <= 64);
    always_assert(bin >= 8);
    static const block notOneBlock = toBlock(~0, ~1);
    const static block pt[2] = {ZeroBlock, OneBlock};

    DPFETKeyPack key0(bin);
    DPFETKeyPack key1(bin);

    int tid = omp_get_thread_num();
    auto s = FSSConfig::prngs[tid].get<std::array<block, 2>>();
    auto s0 = s[0];
    auto s1 = s[1];

    s0 = s0 & notOneBlock;
    s1 = s1 & notOneBlock;
    key0.s[0] = s0;
    key1.s[0] = s1;
    
    u8 t0 = 0;
    u8 t1 = 1;

    block ct0[2];
    block ct1[2];

    for (int i = 0; i < bin - 7; ++i)
    {
        const u8 keep = static_cast<uint8_t>(idx >> (bin - 1 - i)) & 1;
        const u8 loose = keep ^ 1;

        AES ak0(s0);
        AES ak1(s1);

        ak0.ecbEncTwoBlocks(pt, ct0);
        ak1.ecbEncTwoBlocks(pt, ct1);

        auto scw = (ct0[loose] ^ ct1[loose]) & notOneBlock;
        u64 tcw[2];
        u64 &tLcw = tcw[0];
        u64 &tRcw = tcw[1];
        tLcw = lsb(ct0[0]) ^ lsb(ct1[0]) ^ keep ^ 1;
        tRcw = lsb(ct0[1]) ^ lsb(ct1[1]) ^ keep;

        key0.s[i+1] = scw;
        key1.s[i+1] = scw;
        key0.tLcw |= (tLcw << (bin - 1 - i));
        key0.tRcw |= (tRcw << (bin - 1 - i));

        if (t0 == 0)
        {
            s0 = ct0[keep] & notOneBlock;
            t0 = lsb(ct0[keep]);
        }
        else
        {
            s0 = (ct0[keep] & notOneBlock) ^ scw;
            t0 = lsb(ct0[keep]) ^ tcw[keep];
        }

        if (t1 == 0)
        {
            s1 = ct1[keep] & notOneBlock;
            t1 = lsb(ct1[keep]);
        }
        else
        {
            s1 = (ct1[keep] & notOneBlock) ^ scw;
            t1 = lsb(ct1[keep]) ^ tcw[keep];
        }
    }

    key1.tLcw = key0.tLcw;
    key1.tRcw = key0.tRcw;

    if (t0 == 1) s0 = s0 ^ OneBlock;
    if (t1 == 1) s1 = s1 ^ OneBlock;
    uint64_t e0, e1;
    GroupElement ip = idx % 128;
    if (ip >= 64) {
        e0 = 0;
        e1 = 1ULL << (127 - ip);
    }
    else {
        e0 = 1ULL << (63 - ip);
        e1 = 0;
    }
    key0.leaf = s0 ^ s1 ^ osuCrypto::toBlock(e0, e1);
    key1.leaf = key0.leaf;

    return std::make_pair(key0, key1);
}


GroupElement evalGTDPF(int party, const DPFETKeyPack &key, GroupElement x)
{
    static const block notOneBlock = toBlock(~0, ~1);
    int bin = key.bin;

    auto s = _mm_loadu_si128(key.s);
    u8 t = party;
    
    u8 x_prev = 1;
    u8 t_dcf = 0;

    for (int i = 0; i < bin - 7; ++i)
    {
        assert(lsb(s) == 0);
        const u8 x_i = static_cast<uint8_t>(x >> (bin - 1 - i)) & 1;

        if (x_prev != x_i)
        {
            t_dcf = t_dcf ^ t;
        }
        x_prev = x_i;

        AES ak(s);
        block ct = ak.ecbEncBlock(toBlock(0, x_i));
        s = ct & notOneBlock;
        u8 t_old = t;
        t = lsb(ct);

        if (t_old) {
            s = s ^ _mm_loadu_si128(key.s + i + 1);
            t = t ^ ((key.tcw[x_i] >> (bin - 1 - i)) & 1);
        }
    }

    osuCrypto::block leaf = s;
    if (t) leaf = leaf ^ OneBlock ^ key.leaf;
    uint64_t b;

    {
        const u8 x_i = static_cast<uint8_t>(x >> 6) & 1;
        if (x_prev != x_i)
        {
            t_dcf = t_dcf ^ t;
        }
        x_prev = x_i;
        if (x_i) 
        {
            b = _mm_extract_epi64(leaf, 0);
        }
        else 
        {
            b = _mm_extract_epi64(leaf, 1);
        }
        t = __builtin_parityll(b);
    }

    GroupElement xp = x % 64;

    if (x_prev == 0)
    {
        for (int i = 0; i <= xp; ++i)
        {
            t_dcf = t_dcf ^ ((b >> (63 - i)) & 1);
        }
    }
    else
    {
        for (int i = xp + 1; i < 64; ++i)
        {
            t_dcf = t_dcf ^ ((b >> (63 - i)) & 1);
        }
    }
    if (party==1)
    {
       t_dcf = t_dcf ^ 1;
    }
    else{
        t_dcf = t_dcf ^ 0;
    }
    

    return t_dcf;
}

// =========================================================================
// GTDCF
// =========================================================================

std::pair<GTDCFKeyPack, GTDCFKeyPack> keyGenGTDCF(
    int bin, int w, int groupSize, GroupElement idx, const GroupElement* beta)
{
    always_assert(bin <= 64);
    always_assert(groupSize == 2); // 强制为2以启用极限优化
    static const block notOneBlock = toBlock(~0, ~1);
    
    // 0:左分支, 1:右分支, 2:负载掩码
    const static block pt[3] = {ZeroBlock, OneBlock, toBlock(0, 2)};

    GTDCFKeyPack key0(bin, w, 2);
    GTDCFKeyPack key1(bin, w, 2);

    int tid = omp_get_thread_num();
    auto s = FSSConfig::prngs[tid].get<std::array<block, 2>>();
    
    key0.seed = s[0] & notOneBlock;
    key1.seed = s[1] & notOneBlock;
    block s0 = key0.seed, s1 = key1.seed;
    u8 t0 = 0, t1 = 1;
    int d = bin - w;

    for (int i = 0; i < d; ++i)
    {
        const u8 keep_dir = static_cast<uint8_t>(idx >> (bin - 1 - i)) & 1;
        const u8 loose_dir = keep_dir ^ 1;

        AES ak0(s0); AES ak1(s1);
        block ct0[3], ct1[3];
        ak0.ecbEncBlocks(pt, 3, ct0);
        ak1.ecbEncBlocks(pt, 3, ct1);

        block scw = (ct0[loose_dir] ^ ct1[loose_dir]) & notOneBlock;
        uint8_t tcwL = lsb(ct0[0]) ^ lsb(ct1[0]) ^ keep_dir ^ 1;
        uint8_t tcwR = lsb(ct0[1]) ^ lsb(ct1[1]) ^ keep_dir;

        key0.scw[i] = key1.scw[i] = scw;
        key0.tcw[2 * i] = key1.tcw[2 * i] = tcwL;
        key0.tcw[2 * i + 1] = key1.tcw[2 * i + 1] = tcwR;

        // 极限优化：直接用 SSE 指令提取 64 位整数，告别 memcpy
        GroupElement v0_0 = _mm_extract_epi64(ct0[2], 0);
        GroupElement v0_1 = _mm_extract_epi64(ct0[2], 1);
        GroupElement v1_0 = _mm_extract_epi64(ct1[2], 0);
        GroupElement v1_1 = _mm_extract_epi64(ct1[2], 1);

        int offset = i * 2;
        GroupElement target0 = (keep_dir == 0) ? beta[0] : 0;
        GroupElement target1 = (keep_dir == 0) ? beta[1] : 0;

        if (t0 == 1) {
            key0.vcw[offset]     = key1.vcw[offset]     = target0 - v0_0 + v1_0;
            key0.vcw[offset + 1] = key1.vcw[offset + 1] = target1 - v0_1 + v1_1;
        } else {
            key0.vcw[offset]     = key1.vcw[offset]     = -(target0 - v0_0 + v1_0);
            key0.vcw[offset + 1] = key1.vcw[offset + 1] = -(target1 - v0_1 + v1_1);
        }

        s0 = (ct0[keep_dir] & notOneBlock) ^ (t0 ? scw : ZeroBlock);
        t0 = lsb(ct0[keep_dir]) ^ (t0 ? (keep_dir == 0 ? tcwL : tcwR) : 0);
        
        s1 = (ct1[keep_dir] & notOneBlock) ^ (t1 ? scw : ZeroBlock);
        t1 = lsb(ct1[keep_dir]) ^ (t1 ? (keep_dir == 0 ? tcwL : tcwR) : 0);
    }

    AES ak0_leaf(s0); AES ak1_leaf(s1);
    int B = 1 << w;
    GroupElement alpha_lo = idx & (B - 1); 

    for (int j = 0; j < B; ++j) {
        block pt_leaf = toBlock(0, j);
        block ct0_leaf = ak0_leaf.ecbEncBlock(pt_leaf);
        block ct1_leaf = ak1_leaf.ecbEncBlock(pt_leaf);
        
        GroupElement v0_0 = _mm_extract_epi64(ct0_leaf, 0);
        GroupElement v0_1 = _mm_extract_epi64(ct0_leaf, 1);
        GroupElement v1_0 = _mm_extract_epi64(ct1_leaf, 0);
        GroupElement v1_1 = _mm_extract_epi64(ct1_leaf, 1);

        GroupElement target0 = (j >= alpha_lo) ? beta[0] : 0;
        GroupElement target1 = (j >= alpha_lo) ? beta[1] : 0;

        int offset = j * 2;
        if (t0 == 1) {
            key0.leaf_vcw[offset]     = key1.leaf_vcw[offset]     = target0 - v0_0 + v1_0;
            key0.leaf_vcw[offset + 1] = key1.leaf_vcw[offset + 1] = target1 - v0_1 + v1_1;
        } else {
            key0.leaf_vcw[offset]     = key1.leaf_vcw[offset]     = -(target0 - v0_0 + v1_0);
            key0.leaf_vcw[offset + 1] = key1.leaf_vcw[offset + 1] = -(target1 - v0_1 + v1_1);
        }
    }

    return std::make_pair(key0, key1);
}

void evalGTDCF(int party, const GTDCFKeyPack &key, GroupElement x, GroupElement* res)
{
    static const block notOneBlock = toBlock(~0, ~1);
    int bin = key.bin, d = key.d;
    block s = key.seed;
    u8 t = party; 
    int64_t sign = (party == 0) ? 1 : -1;
    
    // 使用局部寄存器变量，避免内存读写
    GroupElement res0 = 0, res1 = 0;

    for (int i = 0; i < d; ++i) {
        const u8 x_i = static_cast<uint8_t>(x >> (bin - 1 - i)) & 1;
        AES ak(s);
        block s_next_raw;
        
        if (x_i == 1) {
            // 核心优化：只需要右分支(OneBlock)和掩码块(toBlock(0,2))
            block pt[2] = {OneBlock, toBlock(0, 2)};
            block ct[2];
            ak.ecbEncTwoBlocks(pt, ct); // 硬件加速，同时加密两块
            
            s_next_raw = ct[0]; 
            
            GroupElement v0 = _mm_extract_epi64(ct[1], 0);
            GroupElement v1 = _mm_extract_epi64(ct[1], 1);
            
            int offset = i * 2;
            if (t) {
                v0 += key.vcw[offset];
                v1 += key.vcw[offset + 1];
            }
            res0 += sign * v0;
            res1 += sign * v1;
        } else {
            // x_i == 0 极速模式：只需要左分支
            s_next_raw = ak.ecbEncBlock(ZeroBlock);
        }

        s = (s_next_raw & notOneBlock) ^ (t ? key.scw[i] : ZeroBlock);
        t = lsb(s_next_raw) ^ (t ? key.tcw[2 * i + x_i] : 0);
    }

    GroupElement x_lo = x & ((1 << key.w) - 1); 
    AES ak_leaf(s);
    block ct_leaf = ak_leaf.ecbEncBlock(toBlock(0, x_lo));
    
    GroupElement v_leaf0 = _mm_extract_epi64(ct_leaf, 0);
    GroupElement v_leaf1 = _mm_extract_epi64(ct_leaf, 1);

    int offset_leaf = x_lo * 2;
    if (t) {
        v_leaf0 += key.leaf_vcw[offset_leaf];
        v_leaf1 += key.leaf_vcw[offset_leaf + 1];
    }
    
    res[0] = res0 + sign * v_leaf0;
    res[1] = res1 + sign * v_leaf1;
}