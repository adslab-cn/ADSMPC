#include <algorithm>
#include <backend/FSS_transformer.h>
#include <FSS/api.h>
#include <FSS/comms.h>
#include <FSS/config.h>
#include <FSS/prng.h>
#include <FSS/stats.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
struct Shape { const char *name; int rows, cols; };
struct Result {
    std::string name;
    double total_ms, keyread_ms, online_ms;
    double online_mb, offline_mb, mae, max_error, row_sum_error;
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
static Result run_protocol(const std::string &name, const Shape &shape, int scale,
                           const std::vector<double> &plain,
                           const std::vector<GroupElement> &masked_input,
                           const std::vector<GroupElement> &input_mask,
                           Protocol protocol) {
    const int size = shape.rows * shape.cols;
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

    // Correctness-only transfer; intentionally excluded from protocol metrics.
    if (FSSConfig::party == DEALER) FSSConfig::client->send_ge_array(out_mask.data(), size);
    else if (FSSConfig::party == CLIENT) FSSConfig::dealer->recv_ge_array(received_mask.data(), size);

    const uint64_t total_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    Result result{name, total_us / 1000.0, key_us / 1000.0,
        (total_us >= key_us ? total_us - key_us : 0) / 1000.0,
        on/1048576.0, off/1048576.0, 0, 0, 0};
    if (FSSConfig::party == CLIENT) {
        const double unit = double(uint64_t(1) << scale);
        double abs_sum = 0, max_err = 0, row_err = 0;
        for (int i = 0; i < shape.rows; ++i) {
            double maxv = plain[i * shape.cols];
            for (int j = 1; j < shape.cols; ++j) maxv = std::max(maxv, plain[i * shape.cols + j]);
            double denom = 0;
            for (int j = 0; j < shape.cols; ++j) denom += std::exp(plain[i * shape.cols + j] - maxv);
            double row_sum = 0;
            for (int j = 0; j < shape.cols; ++j) {
                const int idx = i * shape.cols + j;
                GroupElement clear = out[idx] - received_mask[idx];
                const double actual = double(clear) / unit;
                const double expected = std::exp(plain[idx] - maxv) / denom;
                const double err = std::abs(actual - expected);
                abs_sum += err; max_err = std::max(max_err, err); row_sum += actual;
            }
            row_err += std::abs(row_sum - 1.0);
        }
        result.mae = abs_sum / size; result.max_error = max_err;
        result.row_sum_error = row_err / shape.rows;
    }
    return result;
}

int main(int argc, char **argv) {
    sytorch_init();
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <party:1|2|3> [ip] [cora|citeseer|pubmed]\n";
        return 1;
    }
    FSSConfig::party = std::atoi(argv[1]); FSSConfig::bitlength = 64; FSSConfig::num_threads = 4;
    for (int i=0;i<256;++i) FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(0x12345678ULL+i,0xabcdefULL));
    FSSTransformer<u64> backend; backend.init(argc>2?argv[2]:"127.0.0.1", false);
    std::string selected = argc>3?argv[3]:"cora";
    Shape shape = selected=="citeseer"?Shape{"Citeseer",3327,6}:
                  selected=="pubmed"?Shape{"Pubmed",19717,3}:Shape{"Cora",2708,7};
    constexpr int scale = 12; const int size = shape.rows * shape.cols;
    std::vector<double> plain(size);
    std::vector<GroupElement> masked(size), masks(size);
    for (int i=0;i<size;++i) {
        // Stable logits in [-4,4], including negative values and nontrivial rows.
        plain[i] = (double(int(mix(i+17)%8001)-4000))/1000.0;
        GroupElement r = mix(i+0x5a5a5a5aULL);
        GroupElement fixed = GroupElement(int64_t(std::llround(plain[i]*(uint64_t(1)<<scale))));
        if (FSSConfig::party==DEALER) masks[i]=r; else masked[i]=fixed+r;
    }

    std::vector<Result> results;
    results.push_back(run_protocol("BPGNN",shape,scale,plain,masked,masks,
        [&](auto*x,auto*xm,auto*y,auto*ym){BPGCNSoftmax(shape.rows,shape.cols,x,y,xm,ym,scale,"Bench::BPGNN::");}));
    results.push_back(run_protocol("SIGMA",shape,scale,plain,masked,masks,
        [&](auto*x,auto*xm,auto*y,auto*ym){SIGMASoftmax(shape.rows,shape.cols,x,xm,y,ym,scale,"Bench::SIGMA::");}));
    results.push_back(run_protocol("BumbleBee",shape,scale,plain,masked,masks,
        [&](auto*x,auto*xm,auto*y,auto*ym){BumbleBeeSoftmax(shape.rows,shape.cols,x,xm,y,ym,scale,"Bench::BumbleBee::");}));
    results.push_back(run_protocol("CryptGNN/CrypTen",shape,scale,plain,masked,masks,
        [&](auto*x,auto*xm,auto*y,auto*ym){CrypTenSoftmax(shape.rows,shape.cols,x,xm,y,ym,scale,"Bench::CrypTen::");}));

    if (FSSConfig::party==CLIENT) {
        std::cout << "\nSoftmax comparison: " << shape.name << " (" << shape.rows << "x" << shape.cols << ", scale=" << scale << ")\n";
        std::cout << std::left << std::setw(20) << "Protocol" << std::right
                  << std::setw(13) << "Total(ms)" << std::setw(14) << "KeyRead(ms)"
                  << std::setw(14) << "Online(ms)"
                  << std::setw(15) << "Online(MB)" << std::setw(16) << "Offline(MB)"
                  << std::setw(14) << "MAE" << std::setw(14) << "MaxError" << std::setw(15) << "RowSumError\n";
        std::cout << std::string(135,'-') << '\n';
        for (const auto&r:results) std::cout << std::left << std::setw(20) << r.name << std::right << std::fixed
            << std::setprecision(3) << std::setw(13) << r.total_ms
            << std::setw(14) << r.keyread_ms << std::setw(14) << r.online_ms
            << std::setprecision(6)
            << std::setw(15) << r.online_mb << std::setw(16) << r.offline_mb
            << std::setw(14) << r.mae << std::setw(14) << r.max_error << std::setw(15) << r.row_sum_error << '\n';
    }
    backend.finalize(); return 0;
}
