#include <FSS/dpf.h>
// #include <FSS/verdpf.h>
#include <FSS/freekey.h>
#include <FSS/group_element.h>
#include <backend/FSS_base.h>

#include <iostream>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <algorithm>

static size_t ceil_div_8(size_t bits)
{
    return (bits + 7) / 8;
}

// 按发送/序列化逻辑估算的 key size。
// block 固定 16 bytes；tcw 按 bin bits；ocw 按 bout bits。
static size_t verdpf_key_size_serialized_bytes(int bin, int bout)
{
    size_t bytes = 0;

    // s[0...bin]
    bytes += static_cast<size_t>(bin + 1) * sizeof(osuCrypto::block);

    // tLcw, tRcw, each bin bits
    bytes += ceil_div_8(static_cast<size_t>(bin));
    bytes += ceil_div_8(static_cast<size_t>(bin));

    // cs[4], 4λ-bit correction seed
    bytes += 4 * sizeof(osuCrypto::block);

    // ocw, bout bits
    bytes += ceil_div_8(static_cast<size_t>(bout));

    return bytes;
}

// 按当前 C++ KeyPack 实际字段大小估算。
// 不包含 allocator overhead，只包含 key material。
static size_t verdpf_key_size_in_memory_bytes(int bin)
{
    size_t bytes = 0;

    // s[0...bin]
    bytes += static_cast<size_t>(bin + 1) * sizeof(osuCrypto::block);

    // tcw[2] stored as uint64_t[2]
    bytes += 2 * sizeof(uint64_t);

    // cs[4]
    bytes += 4 * sizeof(osuCrypto::block);

    // ocw stored as GroupElement
    bytes += sizeof(GroupElement);

    return bytes;
}

static GroupElement rand_ge_local(int bw)
{
    GroupElement x =
        (static_cast<GroupElement>(static_cast<uint32_t>(rand())) << 32) ^
        static_cast<GroupElement>(static_cast<uint32_t>(rand()));

    if (bw < 64) {
        x &= ((GroupElement(1) << bw) - 1);
    }

    return x;
}

static void make_distinct_points_for_alpha(std::vector<GroupElement> &xs,
                                           int bin,
                                           int batch_size,
                                           GroupElement alpha)
{
    xs.clear();
    xs.reserve(batch_size);

    std::unordered_set<GroupElement> seen;
    seen.insert(alpha);
    xs.push_back(alpha);

    while (static_cast<int>(xs.size()) < batch_size) {
        GroupElement x = rand_ge_local(bin);
        if (seen.insert(x).second) {
            xs.push_back(x);
        }
    }

    // Shuffle, so alpha is not always at position 0.
    for (int i = 0; i < batch_size; ++i) {
        int j = rand() % batch_size;
        std::swap(xs[i], xs[j]);
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
                      << " x=" << xs[i]
                      << " y=" << y
                      << " expected=" << expected
                      << std::endl;
            return false;
        }
    }

    return true;
}

void VerDPF_50000_Benchmark()
{
    std::cout << "\n=======================================================\n";
    std::cout << "Starting VerDPF 50000-Key Benchmark...\n";

    // =========================
    // Parameters
    // =========================
    const int trials = 50000;
    const int bin = 64;
    const int bout = 64;

    // 每组 key 在多少个 distinct evaluation points 上做一次 BVEval。
    // 如果只想测单点 Eval，可改为 1；如果更贴近 batch verification，可用 16/64。
    const int batch_size = 16;

    // =========================
    // Key size
    // =========================
    const size_t key_size_serialized = verdpf_key_size_serialized_bytes(bin, bout);
    const size_t keypair_size_serialized = 2 * key_size_serialized;

    const size_t key_size_memory = verdpf_key_size_in_memory_bytes(bin);
    const size_t keypair_size_memory = 2 * key_size_memory;

    // =========================
    // PRNG initialization
    // =========================
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for (int i = 0; i < 256; ++i) {
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey ^ i));
    }

    srand(static_cast<unsigned>(time(NULL)));

    uint64_t total_keygen_ns = 0;
    uint64_t total_eval_ns = 0;
    uint64_t total_verify_ns = 0;

    int correctness_ok = 0;
    int verify_ok = 0;

    // 防止编译器把 eval/verify 优化掉
    volatile GroupElement sink = 0;

    std::vector<GroupElement> xs;
    std::vector<GroupElement> y0;
    std::vector<GroupElement> y1;
    osuCrypto::block pi0[2], pi1[2];

    // =========================
    // Benchmark loop
    // =========================
    for (int rep = 0; rep < trials; ++rep)
    {
        GroupElement alpha = rand_ge_local(bin);
        GroupElement payload = rand_ge_local(bout);

        make_distinct_points_for_alpha(xs, bin, batch_size, alpha);

        // -------------------------
        // 1. KeyGen
        // -------------------------
        auto kg_start = std::chrono::high_resolution_clock::now();

        auto keys = keyGenVerDPF(bin, bout, alpha, payload);

        auto kg_end = std::chrono::high_resolution_clock::now();

        total_keygen_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            kg_end - kg_start
        ).count();

        // -------------------------
        // 2. Batch Eval, both parties
        // -------------------------
        auto ev_start = std::chrono::high_resolution_clock::now();

        evalVerDPF_Batch(0, keys.first, xs, y0, pi0);
        evalVerDPF_Batch(1, keys.second, xs, y1, pi1);

        auto ev_end = std::chrono::high_resolution_clock::now();

        total_eval_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            ev_end - ev_start
        ).count();

        // -------------------------
        // 3. Verify
        // -------------------------
        auto vf_start = std::chrono::high_resolution_clock::now();

        bool accept = verifyVerDPF(pi0, pi1);

        auto vf_end = std::chrono::high_resolution_clock::now();

        total_verify_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            vf_end - vf_start
        ).count();

        // -------------------------
        // 4. Correctness check
        // -------------------------
        bool correct = check_outputs(xs, y0, y1, alpha, payload, bout);

        if (correct) {
            correctness_ok++;
        }

        if (accept) {
            verify_ok++;
        }

        // 使用部分结果，防止优化
        sink ^= y0[0];
        sink ^= y1[0];
        sink ^= static_cast<GroupElement>(accept ? 1 : 0);

        freeVerDPFKeyPackPair(keys);

        if ((rep + 1) % 5000 == 0) {
            std::cout << "  Finished " << (rep + 1)
                      << " / " << trials << "\n";
        }
    }

    // =========================
    // Statistics
    // =========================
    double avg_keygen_ns = static_cast<double>(total_keygen_ns) / trials;
    double avg_batcheval_ns = static_cast<double>(total_eval_ns) / trials;
    double avg_verify_ns = static_cast<double>(total_verify_ns) / trials;

    // 两方各评估 batch_size 个点，因此每轮实际执行 2 * batch_size 次单点路径评估。
    double avg_single_eval_ns =
        static_cast<double>(total_eval_ns) /
        (static_cast<double>(trials) * 2.0 * static_cast<double>(batch_size));

    double total_serialized_key_mb =
        (static_cast<double>(keypair_size_serialized) * static_cast<double>(trials)) /
        (1024.0 * 1024.0);

    double total_memory_key_mb =
        (static_cast<double>(keypair_size_memory) * static_cast<double>(trials)) /
        (1024.0 * 1024.0);

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "\n[Correctness & Verification]\n";
    std::cout << "  Correctness OK : "
              << correctness_ok << " / " << trials << "\n";
    std::cout << "  Verify Accept  : "
              << verify_ok << " / " << trials << "\n";
    std::cout << "  sink           : "
              << sink << "\n";

    std::cout << "\n[Key Size]\n";
    std::cout << "  Serialized key size per party      : "
              << key_size_serialized << " Bytes\n";
    std::cout << "  Serialized key size per key-pair   : "
              << keypair_size_serialized << " Bytes\n";
    std::cout << "  In-memory key material per party   : "
              << key_size_memory << " Bytes\n";
    std::cout << "  In-memory key material per key-pair: "
              << keypair_size_memory << " Bytes\n";
    std::cout << "  Total serialized key material      : "
              << total_serialized_key_mb << " MB for " << trials << " key-pairs\n";
    std::cout << "  Total in-memory key material       : "
              << total_memory_key_mb << " MB for " << trials << " key-pairs\n";

    std::cout << "\n[Average Time]\n";
    std::cout << "  Trials                              : "
              << trials << "\n";
    std::cout << "  Batch size per key                  : "
              << batch_size << "\n";
    std::cout << "  Avg KeyGen time per key-pair        : "
              << avg_keygen_ns / 1e6 << " ms\n";
    std::cout << "  Avg BatchEval time per key-pair     : "
              << avg_batcheval_ns / 1e6 << " ms\n";
    std::cout << "  Avg Verify time per key-pair        : "
              << avg_verify_ns << " ns\n";
    std::cout << "  Avg Eval time per party per point   : "
              << avg_single_eval_ns << " ns/op\n";

    std::cout << "\n[Total Time]\n";
    std::cout << "  Total KeyGen  : "
              << static_cast<double>(total_keygen_ns) / 1e9 << " s\n";
    std::cout << "  Total Eval    : "
              << static_cast<double>(total_eval_ns) / 1e9 << " s\n";
    std::cout << "  Total Verify  : "
              << static_cast<double>(total_verify_ns) / 1e9 << " s\n";

    std::cout << "=======================================================\n";
}

int main()
{
    VerDPF_50000_Benchmark();
    return 0;
}