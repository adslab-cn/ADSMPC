// 多叉树FSS的理论对比，这里没有用AES-IN的工程优化，每次AES加密都是独立实现的

#include <FSS/dcf.h>
#include <FSS/dpf.h>
#include <FSS/keypack.h>
#include <FSS/freekey.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <backend/FSS_base.h>

// 提取最低位的辅助函数
inline uint8_t get_lsb(const osuCrypto::block &b) {
    return _mm_cvtsi128_si64x(b) & 1;
}

// =========================================================================
// 1. 纯学术版 Binary DPF
// =========================================================================
std::pair<DPFKeyPack, DPFKeyPack> keyGenBinary_Academic(int bin, int bout, GroupElement idx, GroupElement payload) {
    static const osuCrypto::block notOneBlock = osuCrypto::toBlock(~0, ~1);
    const static osuCrypto::block pt[2] = {osuCrypto::toBlock(0, 0), osuCrypto::toBlock(0, 1)};

    DPFKeyPack key0(bin, bout), key1(bin, bout);
    int tid = omp_get_thread_num();
    auto s = FSSConfig::prngs[tid].get<std::array<osuCrypto::block, 2>>();
    osuCrypto::block s0 = s[0] & notOneBlock, s1 = s[1] & notOneBlock;
    key0.s[0] = s0; key1.s[0] = s1;
    uint8_t t0 = 0, t1 = 1;
    osuCrypto::block ct0[2], ct1[2];

    for (int i = 0; i < bin; ++i) {
        uint8_t keep = static_cast<uint8_t>(idx >> (bin - 1 - i)) & 1;
        uint8_t loose = keep ^ 1;

        // 【学术强制】：循环 2 次，强制实例化 2 个独立 AES 对象
        for (int c = 0; c < 2; ++c) {
            osuCrypto::AES ak0(s0), ak1(s1);
            ct0[c] = ak0.ecbEncBlock(pt[c]);
            ct1[c] = ak1.ecbEncBlock(pt[c]);
        }

        auto scw = (ct0[loose] ^ ct1[loose]) & notOneBlock;
        uint64_t tLcw = get_lsb(ct0[0]) ^ get_lsb(ct1[0]) ^ keep ^ 1;
        uint64_t tRcw = get_lsb(ct0[1]) ^ get_lsb(ct1[1]) ^ keep;

        key0.s[i+1] = scw; key1.s[i+1] = scw;
        key0.tcw[0] |= (tLcw << (bin - 1 - i));
        key0.tcw[1] |= (tRcw << (bin - 1 - i));

        s0 = (ct0[keep] & notOneBlock) ^ (t0 ? scw : osuCrypto::ZeroBlock);
        t0 = get_lsb(ct0[keep]) ^ (t0 ? (keep == 0 ? tLcw : tRcw) : 0);
        s1 = (ct1[keep] & notOneBlock) ^ (t1 ? scw : osuCrypto::ZeroBlock);
        t1 = get_lsb(ct1[keep]) ^ (t1 ? (keep == 0 ? tLcw : tRcw) : 0);
    }
    key1.tcw[0] = key0.tcw[0]; key1.tcw[1] = key0.tcw[1];
    key0.payload = payload - _mm_extract_epi64(s0, 0) + _mm_extract_epi64(s1, 0);
    if (t1 == 1) key0.payload = -key0.payload;
    key1.payload = key0.payload;

    return std::make_pair(key0, key1);
}

GroupElement evalBinary_Academic(int party, const DPFKeyPack &key, GroupElement x) {
    static const osuCrypto::block notOneBlock = osuCrypto::toBlock(~0, ~1);
    const static osuCrypto::block pt[2] = {osuCrypto::toBlock(0, 0), osuCrypto::toBlock(0, 1)};
    auto s = _mm_loadu_si128(key.s);
    uint8_t t = party;

    for (int i = 0; i < key.bin; ++i) {
        uint8_t x_i = static_cast<uint8_t>(x >> (key.bin - 1 - i)) & 1;
        // 【学术强制】：实例化 1 个 AES 对象
        osuCrypto::AES ak(s);
        osuCrypto::block ct = ak.ecbEncBlock(pt[x_i]);
        s = (ct & notOneBlock) ^ (t ? _mm_loadu_si128(key.s + i + 1) : osuCrypto::ZeroBlock);
        t = get_lsb(ct) ^ (t ? ((key.tcw[x_i] >> (key.bin - 1 - i)) & 1) : 0);
    }
    return t;
}

// =========================================================================
// 2. 纯学术版 Quad-Tree DPF
// =========================================================================
std::pair<QuadDPFKeyPack, QuadDPFKeyPack> keyGenQuad_Academic(int bin, int bout, GroupElement idx, GroupElement payload) {
    static const osuCrypto::block notOneBlock = osuCrypto::toBlock(~0, ~1);
    const static osuCrypto::block pt[4] = {osuCrypto::toBlock(0, 0), osuCrypto::toBlock(0, 1), osuCrypto::toBlock(0, 2), osuCrypto::toBlock(0, 3)};
    QuadDPFKeyPack key0(bin, bout), key1(bin, bout);
    int tid = omp_get_thread_num();
    auto s = FSSConfig::prngs[tid].get<std::array<osuCrypto::block, 2>>();
    osuCrypto::block s0 = s[0] & notOneBlock, s1 = s[1] & notOneBlock;
    key0.s0_initial = s0; key1.s0_initial = s1;
    uint8_t t0 = 0, t1 = 1;

    for (int i = 0; i < key0.depth; ++i) {
        int shift = bin - 2 - 2 * i;
        uint8_t keep = (shift >= 0) ? ((idx >> shift) & 3) : ((idx << 1) & 3);
        osuCrypto::block ct0[4], ct1[4];
        
        // 【学术强制】：循环 4 次，强制实例化 4 个独立 AES 对象
        for (int c = 0; c < 4; ++c) {
            osuCrypto::AES ak0(s0), ak1(s1);
            ct0[c] = ak0.ecbEncBlock(pt[c]);
            ct1[c] = ak1.ecbEncBlock(pt[c]);
        }

        uint8_t t_cw_bits = 0;
        uint8_t r_idx = (keep + 1) % 4; 
        for (int c = 0; c < 4; ++c) {
            osuCrypto::block scw;
            uint8_t tc0 = get_lsb(ct0[c]), tc1 = get_lsb(ct1[c]), tcw;
            if (c != keep) {
                scw = (ct0[c] ^ ct1[c]) & notOneBlock;
                tcw = tc0 ^ tc1;
            } else {
                scw = (ct0[r_idx] ^ ct1[r_idx]) & notOneBlock; 
                tcw = tc0 ^ tc1 ^ 1;
            }
            key0.scw[i * 4 + c] = scw; key1.scw[i * 4 + c] = scw;
            t_cw_bits |= (tcw << c);
        }
        key0.tcw[i] = t_cw_bits; key1.tcw[i] = t_cw_bits;
        osuCrypto::block scw_keep = key0.scw[i * 4 + keep];
        uint8_t tcw_keep = (t_cw_bits >> keep) & 1;

        s0 = (ct0[keep] & notOneBlock) ^ (t0 ? scw_keep : osuCrypto::ZeroBlock);
        t0 = get_lsb(ct0[keep]) ^ (t0 ? tcw_keep : 0);
        s1 = (ct1[keep] & notOneBlock) ^ (t1 ? scw_keep : osuCrypto::ZeroBlock);
        t1 = get_lsb(ct1[keep]) ^ (t1 ? tcw_keep : 0);
    }
    GroupElement cw_payload = payload - _mm_extract_epi64(s0, 0) + _mm_extract_epi64(s1, 0);
    if (t1 == 1) cw_payload = -cw_payload;
    key0.payload = cw_payload; key1.payload = cw_payload;
    return std::make_pair(key0, key1);
}

GroupElement evalQuad_Academic(int party, const QuadDPFKeyPack &key, GroupElement x) {
    static const osuCrypto::block notOneBlock = osuCrypto::toBlock(~0, ~1);
    osuCrypto::block s = key.s0_initial;
    uint8_t t = party;
    for (int i = 0; i < key.depth; ++i) {
        int shift = key.bin - 2 - 2 * i;
        uint8_t c = (shift >= 0) ? ((x >> shift) & 3) : ((x << 1) & 3);
        
        // 【学术强制】：实例化 1 个 AES 对象
        osuCrypto::AES ak(s);
        osuCrypto::block ct = ak.ecbEncBlock(osuCrypto::toBlock(0, c));
        s = (ct & notOneBlock) ^ (t ? key.scw[i * 4 + c] : osuCrypto::ZeroBlock);
        t = get_lsb(ct) ^ (t ? ((key.tcw[i] >> c) & 1) : 0);
    }
    GroupElement res = _mm_extract_epi64(s, 0) + t * key.payload;
    if (party == 1) res = -res;
    mod(res, key.bout);
    return res;
}

// =========================================================================
// 3. 纯学术版 Oct-Tree DPF
// =========================================================================
std::pair<OctDPFKeyPack, OctDPFKeyPack> keyGenOct_Academic(int bin, int bout, GroupElement idx, GroupElement payload) {
    static const osuCrypto::block notOneBlock = osuCrypto::toBlock(~0, ~1);
    const static osuCrypto::block pt[8] = {
        osuCrypto::toBlock(0, 0), osuCrypto::toBlock(0, 1), osuCrypto::toBlock(0, 2), osuCrypto::toBlock(0, 3),
        osuCrypto::toBlock(0, 4), osuCrypto::toBlock(0, 5), osuCrypto::toBlock(0, 6), osuCrypto::toBlock(0, 7)
    };
    OctDPFKeyPack key0(bin, bout), key1(bin, bout);
    int tid = omp_get_thread_num();
    auto s = FSSConfig::prngs[tid].get<std::array<osuCrypto::block, 2>>();
    osuCrypto::block s0 = s[0] & notOneBlock, s1 = s[1] & notOneBlock;
    key0.s0_initial = s0; key1.s0_initial = s1;
    uint8_t t0 = 0, t1 = 1;

    for (int i = 0; i < key0.depth; ++i) {
        int shift = bin - 3 - 3 * i;
        uint8_t keep = 0;
        if (shift >= 0) keep = (idx >> shift) & 7;
        else if (shift == -1) keep = ((idx << 1) & 7);
        else if (shift == -2) keep = ((idx << 2) & 7);

        osuCrypto::block ct0[8], ct1[8];
        
        // 【学术强制】：循环 8 次，强制实例化 8 个独立 AES 对象
        for (int c = 0; c < 8; ++c) {
            osuCrypto::AES ak0(s0), ak1(s1);
            ct0[c] = ak0.ecbEncBlock(pt[c]);
            ct1[c] = ak1.ecbEncBlock(pt[c]);
        }

        uint8_t t_cw_bits = 0;
        uint8_t r_idx = (keep + 1) % 8;
        for (int c = 0; c < 8; ++c) {
            osuCrypto::block scw;
            uint8_t tc0 = get_lsb(ct0[c]), tc1 = get_lsb(ct1[c]), tcw;
            if (c != keep) {
                scw = (ct0[c] ^ ct1[c]) & notOneBlock;
                tcw = tc0 ^ tc1;
            } else {
                scw = (ct0[r_idx] ^ ct1[r_idx]) & notOneBlock;
                tcw = tc0 ^ tc1 ^ 1;
            }
            key0.scw[i * 8 + c] = scw; key1.scw[i * 8 + c] = scw;
            t_cw_bits |= (tcw << c);
        }
        key0.tcw[i] = t_cw_bits; key1.tcw[i] = t_cw_bits;
        osuCrypto::block scw_keep = key0.scw[i * 8 + keep];
        uint8_t tcw_keep = (t_cw_bits >> keep) & 1;

        s0 = (ct0[keep] & notOneBlock) ^ (t0 ? scw_keep : osuCrypto::ZeroBlock);
        t0 = get_lsb(ct0[keep]) ^ (t0 ? tcw_keep : 0);
        s1 = (ct1[keep] & notOneBlock) ^ (t1 ? scw_keep : osuCrypto::ZeroBlock);
        t1 = get_lsb(ct1[keep]) ^ (t1 ? tcw_keep : 0);
    }
    GroupElement cw_payload = payload - _mm_extract_epi64(s0, 0) + _mm_extract_epi64(s1, 0);
    if (t1 == 1) cw_payload = -cw_payload;
    key0.payload = cw_payload; key1.payload = cw_payload;
    return std::make_pair(key0, key1);
}

GroupElement evalOct_Academic(int party, const OctDPFKeyPack &key, GroupElement x) {
    static const osuCrypto::block notOneBlock = osuCrypto::toBlock(~0, ~1);
    osuCrypto::block s = key.s0_initial;
    uint8_t t = party;
    for (int i = 0; i < key.depth; ++i) {
        int shift = key.bin - 3 - 3 * i;
        uint8_t c = 0;
        if (shift >= 0) c = (x >> shift) & 7;
        else if (shift == -1) c = ((x << 1) & 7);
        else if (shift == -2) c = ((x << 2) & 7);
        
        // 【学术强制】：实例化 1 个 AES 对象
        osuCrypto::AES ak(s);
        osuCrypto::block ct = ak.ecbEncBlock(osuCrypto::toBlock(0, c));
        s = (ct & notOneBlock) ^ (t ? key.scw[i * 8 + c] : osuCrypto::ZeroBlock);
        t = get_lsb(ct) ^ (t ? ((key.tcw[i] >> c) & 1) : 0);
    }
    GroupElement res = _mm_extract_epi64(s, 0) + t * key.payload;
    if (party == 1) res = -res;
    mod(res, key.bout);
    return res;
}

// =========================================================================
// Benchmark 测试主函数
// =========================================================================
void Academic_Benchmark_TEST() {
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "Starting [PURE THEORETICAL] DPF Benchmarks (No Hardware Pipelining)..." << std::endl;

    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));

    int samples = 50000;
    int bin = 64, bout = 64;  

    auto *std_keys = new std::pair<DPFKeyPack, DPFKeyPack>[samples];
    auto *quad_keys = new std::pair<QuadDPFKeyPack, QuadDPFKeyPack>[samples];
    auto *oct_keys = new std::pair<OctDPFKeyPack, OctDPFKeyPack>[samples];
    auto *alpha = new GroupElement[samples], *payload = new GroupElement[samples], *query_x = new GroupElement[samples];

    for (int i = 0; i < samples; ++i) {
        alpha[i] = rand(); payload[i] = rand();
        query_x[i] = (rand() % 2 == 0) ? alpha[i] : rand();
    }

    // --- KeyGen 测试 ---
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; ++i) std_keys[i] = keyGenBinary_Academic(bin, bout, alpha[i], payload[i]);
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; ++i) quad_keys[i] = keyGenQuad_Academic(bin, bout, alpha[i], payload[i]);
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; ++i) oct_keys[i] = keyGenOct_Academic(bin, bout, alpha[i], payload[i]);
    auto t4 = std::chrono::high_resolution_clock::now();

    double durStdGen  = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    double durQuadGen = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    double durOctGen  = std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();

    // --- Eval 测试 ---
    auto e1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; ++i) {
        GroupElement t0 = evalBinary_Academic(0, std_keys[i].first, query_x[i]);
        GroupElement t1 = evalBinary_Academic(1, std_keys[i].second, query_x[i]);
    }
    auto e2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; ++i) {
        GroupElement res = evalQuad_Academic(0, quad_keys[i].first, query_x[i]) + evalQuad_Academic(1, quad_keys[i].second, query_x[i]);
    }
    auto e3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; ++i) {
        GroupElement res = evalOct_Academic(0, oct_keys[i].first, query_x[i]) + evalOct_Academic(1, oct_keys[i].second, query_x[i]);
    }
    auto e4 = std::chrono::high_resolution_clock::now();

    double durStdEval  = std::chrono::duration_cast<std::chrono::nanoseconds>(e2 - e1).count();
    double durQuadEval = std::chrono::duration_cast<std::chrono::nanoseconds>(e3 - e2).count();
    double durOctEval  = std::chrono::duration_cast<std::chrono::nanoseconds>(e4 - e3).count();

    // ================= 输出结果 =================
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n[Pure Theoretical KeyGen Time (Offline)]" << std::endl;
    std::cout << "  Binary (128 PRG calls/key) : " << durStdGen / 1e6 << " ms" << std::endl;
    std::cout << "  Quad   (128 PRG calls/key) : " << durQuadGen / 1e6 << " ms" << std::endl;
    std::cout << "  Oct    (176 PRG calls/key) : " << durOctGen / 1e6 << " ms" << std::endl;

    std::cout << "\n[Pure Theoretical Evaluation Time (Online)]" << std::endl;
    std::cout << "  Binary (64 PRG calls/eval) : " << durStdEval / 1e6 << " ms" << std::endl;
    std::cout << "  Quad   (32 PRG calls/eval) : " << durQuadEval / 1e6 << " ms (Speedup: " << durStdEval / durQuadEval << "x)" << std::endl;
    std::cout << "  Oct    (22 PRG calls/eval) : " << durOctEval / 1e6 << " ms (Speedup: " << durStdEval / durOctEval << "x)" << std::endl;

    std::cout << "=======================================================\n" << std::endl;
}

int main(){
    Academic_Benchmark_TEST();
    return 0;
}