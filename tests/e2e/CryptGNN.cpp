#include <backend/FSS_transformer.h>
#include <FSS/api.h>
#include <FSS/comms.h>
#include <FSS/config.h>
#include <FSS/cryptgnn.h>
#include <FSS/prng.h>
#include <FSS/stats.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

struct Dataset {
    const char *name;
    int nodes;
    int edges;
    int input_dim;
    int classes;
};

static uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static void fill_masked(std::vector<GroupElement> &value,
                        std::vector<GroupElement> &mask,
                        uint64_t salt) {
    for (size_t i = 0; i < value.size(); ++i) {
        const GroupElement r = mix64(i + salt);
        const GroupElement x = GroupElement((i * 13 + salt) % 17);
        if (FSSConfig::party == DEALER) mask[i] = r;
        else value[i] = x + r;
    }
}

static void make_graph(const Dataset &ds, std::vector<int> &src,
                       std::vector<int> &dst) {
    src.resize(ds.edges);
    dst.resize(ds.edges);
    // Include one self-loop per node when possible, followed by deterministic
    // sparse edges. The benchmark retains the published dataset dimensions.
    for (int e = 0; e < ds.edges; ++e) {
        src[e] = e % ds.nodes;
        dst[e] = (e < ds.nodes) ? src[e]
                                 : (src[e] * 17 + e / ds.nodes + 1) % ds.nodes;
    }
}

static uint64_t keyread_us() {
    uint64_t total = 0;
    for (const auto &entry : FSS::stats) total += entry.second.keyread_time;
    return total;
}

static uint64_t online_comm_bytes() {
    return FSSConfig::party == DEALER ? 0 :
        FSSConfig::peer->bytesSent() + FSSConfig::peer->bytesReceived();
}

static void run_dataset(const Dataset &ds) {
    constexpr int hidden = 64;
    constexpr int scale = 12;
    constexpr int cryptmpl_batches = 20;

    std::vector<int> src, dst;
    make_graph(ds, src, dst);
    // Graph/index sharing is preprocessing and is deliberately outside the
    // requested online benchmark interval. The same plan is reused by both
    // GCN layers.
    CryptMPLPlan plan = CryptMPLPrepare2P1(ds.nodes, src, dst, cryptmpl_batches);

    std::vector<GroupElement> x(static_cast<size_t>(ds.nodes) * ds.input_dim);
    std::vector<GroupElement> xm(x.size());
    std::vector<GroupElement> w1(static_cast<size_t>(ds.input_dim) * hidden);
    std::vector<GroupElement> w1m(w1.size());
    std::vector<GroupElement> w2(static_cast<size_t>(hidden) * ds.classes);
    std::vector<GroupElement> w2m(w2.size());
    fill_masked(x, xm, 0x1000 + ds.nodes);
    fill_masked(w1, w1m, 0x2000 + ds.nodes);
    fill_masked(w2, w2m, 0x3000 + ds.nodes);

    std::vector<GroupElement> h1(static_cast<size_t>(ds.nodes) * hidden), h1m(h1.size());
    std::vector<GroupElement> a1(h1.size()), a1m(h1.size());
    std::vector<GroupElement> r1(h1.size()), r1m(h1.size());
    std::vector<GroupElement> h2(static_cast<size_t>(ds.nodes) * ds.classes), h2m(h2.size());
    std::vector<GroupElement> a2(h2.size()), a2m(h2.size());
    std::vector<GroupElement> prob(h2.size()), probm(h2.size());

    FSS::start();
    const uint64_t comm0 = FSSConfig::party == DEALER ? 0 :
        FSSConfig::peer->bytesSent() + FSSConfig::peer->bytesReceived();
    const uint64_t key0 = keyread_us();
    const auto begin = Clock::now();

    uint64_t first_mpl_elapsed_us = 0;
    uint64_t first_mpl_key_us = 0;
    uint64_t first_mpl_comm = 0;

    // First MPL accounting boundary:
    // layer-1 MatMul2D + layer-1 CryptMPL2P1; the following ReLU is excluded.
    const uint64_t mpl_comm0 = online_comm_bytes();
    const uint64_t mpl_key0 = keyread_us();
    const auto mpl_begin = Clock::now();

    MatMul2D(ds.nodes, ds.input_dim, hidden,
             x.data(), xm.data(), w1.data(), w1m.data(),
             h1.data(), h1m.data(), true);

    CryptMPL2P1(plan, hidden, h1.data(), h1m.data(),
                a1.data(), a1m.data(), "CryptGNN::L1::");
    first_mpl_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now() - mpl_begin).count();
    first_mpl_key_us = keyread_us() - mpl_key0;
    first_mpl_comm = online_comm_bytes() - mpl_comm0;

    CrypTenReLU(ds.nodes * hidden, a1.data(), r1.data(),
                a1m.data(), r1m.data(), 64, "CryptGNN::L1::");

    // GCN layer 2 and CrypTen-style output normalization.
    MatMul2D(ds.nodes, hidden, ds.classes,
             r1.data(), r1m.data(), w2.data(), w2m.data(),
             h2.data(), h2m.data(), true);
    CryptMPL2P1(plan, ds.classes, h2.data(), h2m.data(),
                a2.data(), a2m.data(), "CryptGNN::L2::");
    CrypTenSoftmax(ds.nodes, ds.classes, a2.data(), a2m.data(),
                   prob.data(), probm.data(), scale, "CryptGNN::Output::");

    const uint64_t elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now() - begin).count();
    const uint64_t key_us = keyread_us() - key0;
    const uint64_t comm = FSSConfig::party == DEALER ? 0 :
        FSSConfig::peer->bytesSent() + FSSConfig::peer->bytesReceived() - comm0;
    FSS::end();

    if (FSSConfig::party == CLIENT) {
        const double online_ms = (elapsed_us >= key_us ? elapsed_us - key_us : 0) / 1000.0;
        const double first_mpl_online_ms =
            (first_mpl_elapsed_us >= first_mpl_key_us
                 ? first_mpl_elapsed_us - first_mpl_key_us
                 : 0) / 1000.0;
        std::cout << "\n================ CryptGNN " << ds.name
                  << " ONLINE RESULT ================\n";
        std::cout << std::left << std::setw(18) << "Scope"
                  << std::setw(12) << "Dataset"
                  << std::right << std::setw(18) << "Online(ms)"
                  << std::setw(18) << "Online(s)"
                  << std::setw(20) << "OnlineComm(MB)" << '\n';
        std::cout << std::string(86, '-') << '\n';
        std::cout << std::left << std::setw(18) << "End-to-end"
                  << std::setw(12) << ds.name << std::right
                  << std::fixed << std::setprecision(3)
                  << std::setw(18) << online_ms
                  << std::setw(18) << online_ms / 1000.0
                  << std::setprecision(6)
                  << std::setw(20) << comm / 1048576.0 << '\n';
        std::cout << std::left << std::setw(18) << "First MPL"
                  << std::setw(12) << ds.name << std::right
                  << std::fixed << std::setprecision(3)
                  << std::setw(18) << first_mpl_online_ms
                  << std::setw(18) << first_mpl_online_ms / 1000.0
                  << std::setprecision(6)
                  << std::setw(20) << first_mpl_comm / 1048576.0 << '\n';
        std::cout << std::string(86, '=') << '\n';
    }
}

int main(int argc, char **argv) {
    sytorch_init();
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <party:1|2|3> [peer-ip]\n";
        return 1;
    }
    FSSConfig::party = std::atoi(argv[1]);
    FSSConfig::bitlength = 64;
    FSSConfig::num_threads = 4;
    for (int i = 0; i < 256; ++i)
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(0x4352595054474e4eULL + i,
                                                       0x123456789abcdef0ULL));
    FSSTransformer<u64> backend;
    backend.init(argc > 2 ? argv[2] : "127.0.0.1", false);

#if defined(CRYPTGNN_CORA_ONLY)
    run_dataset({"Cora", 2708, 5429, 1433, 7});
#elif defined(CRYPTGNN_CITESEER_ONLY)
    run_dataset({"Citeseer", 3327, 4732, 3703, 6});
#elif defined(CRYPTGNN_PUBMED_ONLY)
    run_dataset({"Pubmed", 19717, 44338, 500, 3});
#else
    run_dataset({"Cora", 2708, 5429, 1433, 7});
    run_dataset({"Citeseer", 3327, 4732, 3703, 6});
    run_dataset({"Pubmed", 19717, 44338, 500, 3});
#endif
    backend.finalize();
    return 0;
}
