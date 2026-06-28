#include <FSS/dpf.h>
// #include <FSS/verdpf.h>
#include <FSS/freekey.h>
#include <FSS/group_element.h>
#include <FSS/comms.h>
#include <backend/FSS_base.h>

#include <cryptoTools/Crypto/AES.h>
#include <cryptoTools/Common/Defines.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstdlib>

using namespace osuCrypto;

static inline uint8_t test_lsb(const block &b)
{
    return _mm_cvtsi128_si64x(b) & 1;
}

static GroupElement rand_ge_local(std::mt19937_64 &rng, int bw)
{
    GroupElement x = static_cast<GroupElement>(rng());

    if (bw < 64) {
        x &= ((GroupElement(1) << bw) - 1);
    }

    return x;
}

static void make_distinct_points_for_alpha(std::vector<GroupElement> &xs,
                                           std::mt19937_64 &rng,
                                           int bin,
                                           int npoints,
                                           GroupElement alpha)
{
    xs.clear();
    xs.reserve(npoints);

    std::unordered_set<GroupElement> seen;
    seen.insert(alpha);
    xs.push_back(alpha);

    while (static_cast<int>(xs.size()) < npoints) {
        GroupElement x = rand_ge_local(rng, bin);
        if (seen.insert(x).second) {
            xs.push_back(x);
        }
    }

    std::shuffle(xs.begin(), xs.end(), rng);
}

static size_t actual_send_bytes_dpf(const DPFKeyPack &kp)
{
    const size_t BUF_SIZE = 1 << 20;
    char *buf = new char[BUF_SIZE];
    char *ptr = buf;

    Peer p(&ptr);
    p.send_dpf_keypack(kp);

    size_t bytes = p.bytesSent();

    delete p.keyBuf;
    delete[] buf;

    return bytes;
}

static size_t actual_send_bytes_verdpf(const VerDPFKeyPack &kp)
{
    const size_t BUF_SIZE = 1 << 20;
    char *buf = new char[BUF_SIZE];
    char *ptr = buf;

    Peer p(&ptr);
    p.send_verdpf_keypack(kp);

    size_t bytes = p.bytesSent();

    delete p.keyBuf;
    delete[] buf;

    return bytes;
}

static GroupElement evalDPF_PF(int party,
                               const DPFKeyPack &key,
                               GroupElement x)
{
    static const block notOneBlock = toBlock(~0, ~1);
    const static block pt[2] = {ZeroBlock, OneBlock};

    block s = key.s[0];
    uint8_t t = static_cast<uint8_t>(party);

    for (int i = 0; i < key.bin; ++i) {
        uint8_t x_i = static_cast<uint8_t>(x >> (key.bin - 1 - i)) & 1;

        AES ak(s);
        block ct = ak.ecbEncBlock(pt[x_i]);

        s = (ct & notOneBlock) ^ (t ? key.s[i + 1] : ZeroBlock);
        t = test_lsb(ct) ^ (t ? ((key.tcw[x_i] >> (key.bin - 1 - i)) & 1) : 0);
    }

    GroupElement y = static_cast<GroupElement>(_mm_extract_epi64(s, 0));
    if (t) {
        y += key.payload;
    }

    if (party == 1) {
        y = -y;
    }

    mod(y, key.bout);
    return y;
}

static void evalDPF_PF_Batch(int party,
                             const DPFKeyPack &key,
                             const std::vector<GroupElement> &xs,
                             std::vector<GroupElement> &ys)
{
    ys.resize(xs.size());

    for (size_t i = 0; i < xs.size(); ++i) {
        ys[i] = evalDPF_PF(party, key, xs[i]);
    }
}

static bool check_outputs(const std::vector<GroupElement> &xs,
                          const std::vector<GroupElement> &y0,
                          const std::vector<GroupElement> &y1,
                          GroupElement alpha,
                          GroupElement payload,
                          int bout)
{
    for (size_t i = 0; i < xs.size(); ++i) {
        GroupElement y = y0[i] + y1[i];
        mod(y, bout);

        GroupElement expected = (xs[i] == alpha) ? payload : 0;
        mod(expected, bout);

        if (y != expected) {
            std::cerr << "Mismatch at i=" << i
                      << ", x=" << xs[i]
                      << ", y=" << y
                      << ", expected=" << expected
                      << std::endl;
            return false;
        }
    }

    return true;
}

void DPF_vs_VerDPF_Final_Benchmark()
{
    const int trials = 50000;
    const int eval_points = 50000;
    const int bin = 64;
    const int bout = 64;

    u64 seedKey = 0xdeadbeefbadc0ffe;
    for (int i = 0; i < 256; ++i) {
        FSSConfig::prngs[i].SetSeed(toBlock(seedKey, seedKey ^ static_cast<u64>(i)));
    }

    std::mt19937_64 rng(0x123456789abcdefULL);

    volatile GroupElement sink = 0;

    // ============================================================
    // 1. ActualSend/Party(B)
    // ============================================================

    GroupElement size_alpha = rand_ge_local(rng, bin);
    GroupElement size_payload = rand_ge_local(rng, bout);

    auto dpf_size_keys = keyGenDPF(bin, bout, size_alpha, size_payload);
    auto verdpf_size_keys = keyGenVerDPF(bin, bout, size_alpha, size_payload);

    size_t dpf_send0 = actual_send_bytes_dpf(dpf_size_keys.first);
    size_t dpf_send1 = actual_send_bytes_dpf(dpf_size_keys.second);
    double dpf_actual_send_party =
        (static_cast<double>(dpf_send0) + static_cast<double>(dpf_send1)) / 2.0;

    size_t verdpf_send0 = actual_send_bytes_verdpf(verdpf_size_keys.first);
    size_t verdpf_send1 = actual_send_bytes_verdpf(verdpf_size_keys.second);
    double verdpf_actual_send_party =
        (static_cast<double>(verdpf_send0) + static_cast<double>(verdpf_send1)) / 2.0;

    sink ^= dpf_size_keys.first.payload;
    sink ^= dpf_size_keys.second.payload;
    sink ^= verdpf_size_keys.first.ocw;
    sink ^= verdpf_size_keys.second.ocw;

    freeDPFKeyPackPair(dpf_size_keys);
    freeVerDPFKeyPackPair(verdpf_size_keys);

    // ============================================================
    // 2. KeyGen(ms): total time for 50000 key generations
    // ============================================================

    auto dpf_kg_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < trials; ++i) {
        GroupElement alpha = rand_ge_local(rng, bin);
        GroupElement payload = rand_ge_local(rng, bout);

        auto keys = keyGenDPF(bin, bout, alpha, payload);

        sink ^= keys.first.payload;
        sink ^= keys.second.payload;

        freeDPFKeyPackPair(keys);
    }

    auto dpf_kg_end = std::chrono::high_resolution_clock::now();

    auto verdpf_kg_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < trials; ++i) {
        GroupElement alpha = rand_ge_local(rng, bin);
        GroupElement payload = rand_ge_local(rng, bout);

        auto keys = keyGenVerDPF(bin, bout, alpha, payload);

        sink ^= keys.first.ocw;
        sink ^= keys.second.ocw;

        freeVerDPFKeyPackPair(keys);
    }

    auto verdpf_kg_end = std::chrono::high_resolution_clock::now();

    double dpf_keygen_ms =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            dpf_kg_end - dpf_kg_start
        ).count() / 1e6;

    double verdpf_keygen_ms =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            verdpf_kg_end - verdpf_kg_start
        ).count() / 1e6;

    // ============================================================
    // 3. EvalBatch(ms): total time for 50000 evaluation points
    // ============================================================

    GroupElement eval_alpha = rand_ge_local(rng, bin);
    GroupElement eval_payload = rand_ge_local(rng, bout);

    std::vector<GroupElement> xs;
    make_distinct_points_for_alpha(xs, rng, bin, eval_points, eval_alpha);

    auto dpf_eval_keys = keyGenDPF(bin, bout, eval_alpha, eval_payload);
    auto verdpf_eval_keys = keyGenVerDPF(bin, bout, eval_alpha, eval_payload);

    std::vector<GroupElement> dpf_y0, dpf_y1;
    std::vector<GroupElement> verdpf_y0, verdpf_y1;

    block pi0[2], pi1[2];

    auto dpf_eval_start = std::chrono::high_resolution_clock::now();

    evalDPF_PF_Batch(0, dpf_eval_keys.first, xs, dpf_y0);
    evalDPF_PF_Batch(1, dpf_eval_keys.second, xs, dpf_y1);

    auto dpf_eval_end = std::chrono::high_resolution_clock::now();

    auto verdpf_eval_start = std::chrono::high_resolution_clock::now();

    evalVerDPF_Batch(0, verdpf_eval_keys.first, xs, verdpf_y0, pi0);
    evalVerDPF_Batch(1, verdpf_eval_keys.second, xs, verdpf_y1, pi1);

    auto verdpf_eval_end = std::chrono::high_resolution_clock::now();

    bool dpf_correct = check_outputs(xs, dpf_y0, dpf_y1,
                                     eval_alpha, eval_payload, bout);

    bool verdpf_correct = check_outputs(xs, verdpf_y0, verdpf_y1,
                                        eval_alpha, eval_payload, bout);

    bool verdpf_accept = verifyVerDPF(pi0, pi1);

    if (!dpf_correct || !verdpf_correct || !verdpf_accept) {
        std::cerr << "Correctness or verification failed." << std::endl;
        std::exit(1);
    }

    double dpf_eval_ms =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            dpf_eval_end - dpf_eval_start
        ).count() / 1e6;

    double verdpf_eval_ms =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            verdpf_eval_end - verdpf_eval_start
        ).count() / 1e6;

    sink ^= dpf_y0[0];
    sink ^= dpf_y1[0];
    sink ^= verdpf_y0[0];
    sink ^= verdpf_y1[0];
    sink ^= static_cast<GroupElement>(verdpf_accept ? 1 : 0);

    freeDPFKeyPackPair(dpf_eval_keys);
    freeVerDPFKeyPackPair(verdpf_eval_keys);

    // ============================================================
    // Final output: only requested columns
    // ============================================================

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "\n=======================================================\n";
    std::cout << "DPF vs VerDPF Benchmark\n";
    std::cout << "Parameters: bin=" << bin
              << ", bout=" << bout
              << ", trials=" << trials
              << ", eval_points=" << eval_points << "\n\n";

    std::cout << "Scheme | ActualSend/Party(B) | KeyGen(ms) | EvalBatch(ms)\n";

    std::cout << "DPF    | "
              << dpf_actual_send_party << " | "
              << dpf_keygen_ms << " | "
              << dpf_eval_ms << "\n";

    std::cout << "VerDPF | "
              << verdpf_actual_send_party << " | "
              << verdpf_keygen_ms << " | "
              << verdpf_eval_ms << "\n";

    std::cout << "Ratio  | "
              << (verdpf_actual_send_party / dpf_actual_send_party) << "x | "
              << (verdpf_keygen_ms / dpf_keygen_ms) << "x | "
              << (verdpf_eval_ms / dpf_eval_ms) << "x\n";

    std::cout << "=======================================================\n";

    // 防止优化；不想显示可删除这一行，但建议保留。
    // std::cerr << "sink = " << sink << std::endl;
}

int main()
{
    DPF_vs_VerDPF_Final_Benchmark();
    return 0;
}