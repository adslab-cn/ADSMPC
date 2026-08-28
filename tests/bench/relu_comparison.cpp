#include <algorithm>
#include <backend/FSS_transformer.h>
#include <FSS/api.h>
#include <FSS/comms.h>
#include <FSS/config.h>
#include <FSS/prng.h>
#include <FSS/stats.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
struct Shape { const char *name; int rows, cols; };
struct Result {
    std::string name;
    double total_ms, keyread_ms, online_ms, online_mb, offline_mb;
    uint64_t errors;
};

static uint64_t keyread_us() {
    uint64_t total = 0;
    for (const auto &entry : FSS::stats) total += entry.second.keyread_time;
    return total;
}

static uint64_t online_comm() {
    return FSSConfig::party == DEALER ? 0 :
        FSSConfig::peer->bytesSent() + FSSConfig::peer->bytesReceived();
}
static uint64_t offline_comm() {
    return FSSConfig::party == DEALER ?
        FSSConfig::server->bytesSent() + FSSConfig::client->bytesSent() :
        FSSConfig::dealer->bytesReceived();
}
static uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL; return x ^ (x >> 31);
}

template<class Protocol>
static Result run_protocol(const std::string &name, int size,
                           const std::vector<GroupElement> &plain,
                           const std::vector<GroupElement> &masked_input,
                           const std::vector<GroupElement> &input_mask,
                           Protocol protocol) {
    std::vector<GroupElement> in = masked_input, in_mask = input_mask;
    std::vector<GroupElement> out(size), out_mask(size), received_mask(size);
    FSS::start();
    const uint64_t on0 = online_comm(), off0 = offline_comm();
    const uint64_t key0 = keyread_us();
    const auto begin = Clock::now();
    protocol(in.data(), in_mask.data(), out.data(), out_mask.data());
    const auto end = Clock::now();
    const uint64_t key_us = keyread_us() - key0;
    const uint64_t on = online_comm() - on0, off = offline_comm() - off0;
    FSS::end();

    // Accuracy-only mask transfer; excluded from the protocol measurements.
    if (FSSConfig::party == DEALER) FSSConfig::client->send_ge_array(out_mask.data(), size);
    else if (FSSConfig::party == CLIENT) FSSConfig::dealer->recv_ge_array(received_mask.data(), size);

    const uint64_t total_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    Result result{name, total_us / 1000.0, key_us / 1000.0,
        (total_us >= key_us ? total_us - key_us : 0) / 1000.0,
        on / 1048576.0, off / 1048576.0, 0};
    if (FSSConfig::party == CLIENT) {
        for (int i = 0; i < size; ++i) {
            const GroupElement actual = out[i] - received_mask[i];
            const GroupElement expected = (int64_t(plain[i]) > 0) ? plain[i] : GroupElement(0);
            if (actual != expected) ++result.errors;
        }
    }
    return result;
}

int main(int argc, char **argv) {
    sytorch_init();
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <party:1|2|3> [ip] [cora|citeseer|pubmed]\n";
        return 1;
    }
    FSSConfig::party = std::atoi(argv[1]);
    FSSConfig::bitlength = 64; FSSConfig::num_threads = 4;
    for (int i = 0; i < 256; ++i)
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(0x31415926ULL + i, 0x27182818ULL));
    FSSTransformer<u64> backend;
    backend.init(argc > 2 ? argv[2] : "127.0.0.1", false);

    const std::string selected = argc > 3 ? argv[3] : "cora";
    const Shape shape = selected == "citeseer" ? Shape{"Citeseer", 3327, 64} :
                        selected == "pubmed" ? Shape{"Pubmed", 19717, 64} :
                        Shape{"Cora", 2708, 64};
    const int size = shape.rows * shape.cols;
    constexpr int scale = 12;
    std::vector<GroupElement> plain(size), masked(size), masks(size);
    for (int i = 0; i < size; ++i) {
        // Signed fixed-point activations in [-8,8], with exact zero cases.
        int64_t milli = int64_t(mix(i + 31) % 16001) - 8000;
        if (i % 257 == 0) milli = 0;
        const int64_t fixed = (milli * (int64_t(1) << scale)) / 1000;
        plain[i] = GroupElement(fixed);
        const GroupElement r = mix(i + 0x6a09e667f3bcc909ULL);
        if (FSSConfig::party == DEALER) masks[i] = r;
        else masked[i] = plain[i] + r;
    }

    std::vector<Result> results;
    results.push_back(run_protocol("OblivGNN-style", size, plain, masked, masks,
        [&](auto*x, auto*xm, auto*y, auto*ym) {
            OblivGNNReLU(size, x, y, xm, ym, 64, "Bench::OblivGNN::");
        }));
    results.push_back(run_protocol("CrypTen schedule adapter", size, plain, masked, masks,
        [&](auto*x, auto*xm, auto*y, auto*ym) {
            CrypTenReLU(size, x, y, xm, ym, 64, "Bench::CrypTen::");
        }));
    results.push_back(run_protocol("GROTTO", size, plain, masked, masks,
        [&](auto*x, auto*xm, auto*y, auto*ym) {
            GrottoReLU(size, x, y, xm, ym, "Bench::GROTTO::");
        }));
    results.push_back(run_protocol("SIGMA/SlothRelu", size, plain, masked, masks,
        [&](auto*x, auto*xm, auto*y, auto*ym) {
            SIGMAReLU(size, x, y, xm, ym, 64, "Bench::SIGMA::");
        }));
    results.push_back(run_protocol("BPGNN-GTDCF", size, plain, masked, masks,
        [&](auto*x, auto*xm, auto*y, auto*ym) {
            GTDCFReLU(size, x, y, xm, ym, 8, "Bench::GTDCF::");
        }));

    if (FSSConfig::party == CLIENT) {
        std::cout << "\nReLU comparison: " << shape.name << " (" << shape.rows << "x"
                  << shape.cols << ", scale=" << scale << ", ring=64)\n";
        std::cout << std::left << std::setw(22) << "Protocol" << std::right
                  << std::setw(14) << "Total(ms)" << std::setw(14) << "KeyRead(ms)"
                  << std::setw(14) << "Online(ms)" << std::setw(16) << "Online(MB)"
                  << std::setw(17) << "Offline(MB)" << std::setw(12) << "Errors\n";
        std::cout << std::string(109, '-') << '\n';
        for (const auto &r : results)
            std::cout << std::left << std::setw(22) << r.name << std::right << std::fixed
                      << std::setprecision(3) << std::setw(14) << r.total_ms
                      << std::setw(14) << r.keyread_ms << std::setw(14) << r.online_ms
                      << std::setprecision(6) << std::setw(16) << r.online_mb
                      << std::setw(17) << r.offline_mb << std::setw(12) << r.errors << '\n';
        std::cout << "Note: GROTTO is the exact two-piece degree-one spline: rotated\n"
                     "DPF prefix parity followed by ternary masked multiplication.\n"
                     "SIGMA is SlothDrelu+Select. CrypTen is a nine-round schedule\n"
                     "adapter, not a source-level port of the Python CrypTen runtime.\n";
    }
    backend.finalize();
    return 0;
}
