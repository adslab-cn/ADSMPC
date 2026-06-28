#include <FSS/dpf.h>
#include <FSS/dcf.h>
#include <FSS/freekey.h>
#include <FSS/group_element.h>
#include <FSS/assert.h>
#include <backend/FSS_transformer.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <random>

struct IFSSBenchStats
{
    uint64_t dpf_correct = 0;
    uint64_t dcf_two_correct = 0;
    uint64_t dcf_vec_correct = 0;

    uint64_t dpf_mac_ok = 0;
    uint64_t dcf_two_mac_ok = 0;
    uint64_t dcf_vec_mac_ok = 0;

    uint64_t dpf_tamper_detected = 0;
    uint64_t dcf_two_tamper_detected = 0;
    uint64_t dcf_vec_tamper_detected = 0;

    uint64_t dpf_keygen_ns = 0;
    uint64_t dcf_two_keygen_ns = 0;
    uint64_t dcf_vec_keygen_ns = 0;

    uint64_t dpf_eval_ns = 0;
    uint64_t dcf_two_eval_ns = 0;
    uint64_t dcf_vec_eval_ns = 0;

    uint64_t mac_check_ns = 0;
    uint64_t tamper_check_ns = 0;
};

static GroupElement rand_ge_local(std::mt19937_64 &rng, int bw)
{
    GroupElement x = static_cast<GroupElement>(rng());
    mod(x, bw);
    return x;
}

static bool check_value(IFSSAuthShare s0,
                        IFSSAuthShare s1,
                        GroupElement expected,
                        int bout)
{
    GroupElement y = ifss_reconstruct_value(s0, s1, bout);
    mod(expected, bout);
    return y == expected;
}

static bool tamper_detected(IFSSAuthShare s0,
                            IFSSAuthShare s1,
                            GroupElement deltaA0,
                            GroupElement deltaA1,
                            int bout,
                            uint64_t seed)
{
    std::vector<IFSSAuthShare> batch0;
    std::vector<IFSSAuthShare> batch1;

    batch0.push_back(s0);
    batch1.push_back(s1);

    // Tamper P1's value share but keep its tag unchanged.
    batch1[0].value += 5;
    mod(batch1[0].value, bout);

    bool pass = ifss_batch_check_arithmetic(batch0,
                                            batch1,
                                            deltaA0,
                                            deltaA1,
                                            bout,
                                            seed);

    return !pass;
}

void IFSS_FSS_50000_TEST()
{
    std::cout << "\n=======================================================\n";
    std::cout << "Starting IFSS_DPF / IFSS_DCF 50000-Test...\n";

    const int trials = 50000;
    const int bin = 64;
    const int bout = 64;

    // Initialize framework PRNGs.
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for (int i = 0; i < 256; ++i)
    {
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey ^ i));
    }

    std::mt19937_64 rng(0x123456789abcdefULL);

    IFSSGlobalMACKey mac = ifss_setup_arithmetic_mac(bout);

    IFSSBenchStats stat;

    volatile GroupElement sink = 0;

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int rep = 0; rep < trials; ++rep)
    {
        // ------------------------------------------------------------
        // Random parameters
        // ------------------------------------------------------------

        GroupElement dpf_alpha = rand_ge_local(rng, bin);
        GroupElement dpf_beta = rand_ge_local(rng, bout);

        // Alternate hit and miss.
        GroupElement dpf_x;
        GroupElement dpf_expected;

        if (rep & 1)
        {
            dpf_x = dpf_alpha;
            dpf_expected = dpf_beta;
        }
        else
        {
            dpf_x = dpf_alpha + 1;
            mod(dpf_x, bin);
            dpf_expected = 0;
        }

        // Keep DCF alpha away from boundary for stable lt/ge cases.
        GroupElement dcf_alpha =
            (static_cast<GroupElement>(rng()) & ((GroupElement(1) << 48) - 1)) + 2;

        GroupElement dcf_beta = rand_ge_local(rng, bout);

        GroupElement dcf_x;
        GroupElement dcf_expected;

        if (rep & 1)
        {
            dcf_x = dcf_alpha - 1; // x < alpha
            dcf_expected = dcf_beta;
        }
        else
        {
            dcf_x = dcf_alpha;     // x >= alpha
            dcf_expected = 0;
        }

        mod(dcf_x, bin);
        mod(dcf_expected, bout);

        // ============================================================
        // 1. IFSS_DPF
        // ============================================================

        auto kg_start = std::chrono::high_resolution_clock::now();

        auto dpfKeys = keyGenIFSS_DPF(bin,
                                      bout,
                                      dpf_alpha,
                                      dpf_beta,
                                      mac.deltaA);

        auto kg_end = std::chrono::high_resolution_clock::now();

        stat.dpf_keygen_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            kg_end - kg_start
        ).count();

        auto eval_start = std::chrono::high_resolution_clock::now();

        IFSSAuthShare dpf0 = evalIFSS_DPF(0, dpfKeys.first, dpf_x);
        IFSSAuthShare dpf1 = evalIFSS_DPF(1, dpfKeys.second, dpf_x);

        auto eval_end = std::chrono::high_resolution_clock::now();

        stat.dpf_eval_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            eval_end - eval_start
        ).count();

        bool dpf_correct =
            check_value(dpf0, dpf1, dpf_expected, bout);

        bool dpf_mac =
            ifss_check_mac_single(dpf0,
                                  dpf1,
                                  mac.deltaA0,
                                  mac.deltaA1,
                                  bout);

        if (dpf_correct) stat.dpf_correct++;
        if (dpf_mac) stat.dpf_mac_ok++;

        auto dpf_tamper_start = std::chrono::high_resolution_clock::now();

        bool dpf_tamper =
            tamper_detected(dpf0,
                            dpf1,
                            mac.deltaA0,
                            mac.deltaA1,
                            bout,
                            0x11111111ULL ^ static_cast<uint64_t>(rep));

        auto dpf_tamper_end = std::chrono::high_resolution_clock::now();

        stat.tamper_check_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            dpf_tamper_end - dpf_tamper_start
        ).count();

        if (dpf_tamper) stat.dpf_tamper_detected++;

        sink ^= dpf0.value;
        sink ^= dpf1.value;
        sink ^= dpf0.tag;
        sink ^= dpf1.tag;

        freeIFSS_DPFKeyPackPair(dpfKeys);

        // ============================================================
        // 2. IFSS_DCF_TwoDCF
        // ============================================================

        kg_start = std::chrono::high_resolution_clock::now();

        auto dcfTwoKeys = keyGenIFSS_DCF_TwoDCF(bin,
                                                bout,
                                                dcf_alpha,
                                                dcf_beta,
                                                mac.deltaA);

        kg_end = std::chrono::high_resolution_clock::now();

        stat.dcf_two_keygen_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            kg_end - kg_start
        ).count();

        eval_start = std::chrono::high_resolution_clock::now();

        IFSSAuthShare dcfTwo0 = evalIFSS_DCF_TwoDCF(0, dcfTwoKeys.first, dcf_x);
        IFSSAuthShare dcfTwo1 = evalIFSS_DCF_TwoDCF(1, dcfTwoKeys.second, dcf_x);

        eval_end = std::chrono::high_resolution_clock::now();

        stat.dcf_two_eval_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            eval_end - eval_start
        ).count();

        bool dcf_two_correct =
            check_value(dcfTwo0, dcfTwo1, dcf_expected, bout);

        bool dcf_two_mac =
            ifss_check_mac_single(dcfTwo0,
                                  dcfTwo1,
                                  mac.deltaA0,
                                  mac.deltaA1,
                                  bout);

        if (dcf_two_correct) stat.dcf_two_correct++;
        if (dcf_two_mac) stat.dcf_two_mac_ok++;

        auto dcf_two_tamper_start = std::chrono::high_resolution_clock::now();

        bool dcf_two_tamper =
            tamper_detected(dcfTwo0,
                            dcfTwo1,
                            mac.deltaA0,
                            mac.deltaA1,
                            bout,
                            0x22222222ULL ^ static_cast<uint64_t>(rep));

        auto dcf_two_tamper_end = std::chrono::high_resolution_clock::now();

        stat.tamper_check_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            dcf_two_tamper_end - dcf_two_tamper_start
        ).count();

        if (dcf_two_tamper) stat.dcf_two_tamper_detected++;

        sink ^= dcfTwo0.value;
        sink ^= dcfTwo1.value;
        sink ^= dcfTwo0.tag;
        sink ^= dcfTwo1.tag;

        freeIFSS_DCF_TwoDCFKeyPackPair(dcfTwoKeys);

        // ============================================================
        // 3. IFSS_DCF vector-payload
        // ============================================================

        kg_start = std::chrono::high_resolution_clock::now();

        auto dcfVecKeys = keyGenIFSS_DCF(bin,
                                         bout,
                                         dcf_alpha,
                                         dcf_beta,
                                         mac.deltaA);

        kg_end = std::chrono::high_resolution_clock::now();

        stat.dcf_vec_keygen_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            kg_end - kg_start
        ).count();

        eval_start = std::chrono::high_resolution_clock::now();

        IFSSAuthShare dcfVec0 = evalIFSS_DCF(0, dcfVecKeys.first, dcf_x);
        IFSSAuthShare dcfVec1 = evalIFSS_DCF(1, dcfVecKeys.second, dcf_x);

        eval_end = std::chrono::high_resolution_clock::now();

        stat.dcf_vec_eval_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            eval_end - eval_start
        ).count();

        bool dcf_vec_correct =
            check_value(dcfVec0, dcfVec1, dcf_expected, bout);

        bool dcf_vec_mac =
            ifss_check_mac_single(dcfVec0,
                                  dcfVec1,
                                  mac.deltaA0,
                                  mac.deltaA1,
                                  bout);

        if (dcf_vec_correct) stat.dcf_vec_correct++;
        if (dcf_vec_mac) stat.dcf_vec_mac_ok++;

        auto dcf_vec_tamper_start = std::chrono::high_resolution_clock::now();

        bool dcf_vec_tamper =
            tamper_detected(dcfVec0,
                            dcfVec1,
                            mac.deltaA0,
                            mac.deltaA1,
                            bout,
                            0x33333333ULL ^ static_cast<uint64_t>(rep));

        auto dcf_vec_tamper_end = std::chrono::high_resolution_clock::now();

        stat.tamper_check_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            dcf_vec_tamper_end - dcf_vec_tamper_start
        ).count();

        if (dcf_vec_tamper) stat.dcf_vec_tamper_detected++;

        sink ^= dcfVec0.value;
        sink ^= dcfVec1.value;
        sink ^= dcfVec0.tag;
        sink ^= dcfVec1.tag;

        freeIFSS_DCFKeyPackPair(dcfVecKeys);

        if ((rep + 1) % 5000 == 0)
        {
            std::cout << "  Finished " << (rep + 1)
                      << " / " << trials << "\n";
        }

        always_assert(dpf_correct);
        always_assert(dpf_mac);
        always_assert(dpf_tamper);

        always_assert(dcf_two_correct);
        always_assert(dcf_two_mac);
        always_assert(dcf_two_tamper);

        always_assert(dcf_vec_correct);
        always_assert(dcf_vec_mac);
        always_assert(dcf_vec_tamper);
    }

    auto total_end = std::chrono::high_resolution_clock::now();

    uint64_t total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        total_end - total_start
    ).count();

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "\n=======================================================\n";
    std::cout << "IFSS_DPF / IFSS_DCF 50000-Test Results\n";
    std::cout << "=======================================================\n";

    std::cout << "\n[Parameters]\n";
    std::cout << "  trials      : " << trials << "\n";
    std::cout << "  bin         : " << bin << "\n";
    std::cout << "  bout        : " << bout << "\n";
    std::cout << "  deltaA      : " << mac.deltaA << "\n";

    std::cout << "\n[Correctness]\n";
    std::cout << "  IFSS_DPF correct             : "
              << stat.dpf_correct << " / " << trials << "\n";
    std::cout << "  IFSS_DCF_TwoDCF correct      : "
              << stat.dcf_two_correct << " / " << trials << "\n";
    std::cout << "  IFSS_DCF vector correct      : "
              << stat.dcf_vec_correct << " / " << trials << "\n";

    std::cout << "\n[MAC Check]\n";
    std::cout << "  IFSS_DPF MAC passed          : "
              << stat.dpf_mac_ok << " / " << trials << "\n";
    std::cout << "  IFSS_DCF_TwoDCF MAC passed   : "
              << stat.dcf_two_mac_ok << " / " << trials << "\n";
    std::cout << "  IFSS_DCF vector MAC passed   : "
              << stat.dcf_vec_mac_ok << " / " << trials << "\n";

    std::cout << "\n[Tamper Detection]\n";
    std::cout << "  IFSS_DPF tamper detected        : "
              << stat.dpf_tamper_detected << " / " << trials << "\n";
    std::cout << "  IFSS_DCF_TwoDCF tamper detected : "
              << stat.dcf_two_tamper_detected << " / " << trials << "\n";
    std::cout << "  IFSS_DCF vector tamper detected : "
              << stat.dcf_vec_tamper_detected << " / " << trials << "\n";

    std::cout << "\n[Total Time]\n";
    std::cout << "  Total wall-clock time        : "
              << static_cast<double>(total_ns) / 1e6 << " ms\n";

    std::cout << "\n  IFSS_DPF keygen time         : "
              << static_cast<double>(stat.dpf_keygen_ns) / 1e6 << " ms\n";
    std::cout << "  IFSS_DCF_TwoDCF keygen time  : "
              << static_cast<double>(stat.dcf_two_keygen_ns) / 1e6 << " ms\n";
    std::cout << "  IFSS_DCF vector keygen time  : "
              << static_cast<double>(stat.dcf_vec_keygen_ns) / 1e6 << " ms\n";

    std::cout << "\n  IFSS_DPF eval time           : "
              << static_cast<double>(stat.dpf_eval_ns) / 1e6 << " ms\n";
    std::cout << "  IFSS_DCF_TwoDCF eval time    : "
              << static_cast<double>(stat.dcf_two_eval_ns) / 1e6 << " ms\n";
    std::cout << "  IFSS_DCF vector eval time    : "
              << static_cast<double>(stat.dcf_vec_eval_ns) / 1e6 << " ms\n";

    std::cout << "\n[Average Time]\n";
    std::cout << "  Avg IFSS_DPF keygen          : "
              << (static_cast<double>(stat.dpf_keygen_ns) / trials) / 1e6
              << " ms\n";
    std::cout << "  Avg IFSS_DCF_TwoDCF keygen   : "
              << (static_cast<double>(stat.dcf_two_keygen_ns) / trials) / 1e6
              << " ms\n";
    std::cout << "  Avg IFSS_DCF vector keygen   : "
              << (static_cast<double>(stat.dcf_vec_keygen_ns) / trials) / 1e6
              << " ms\n";

    std::cout << "\n  Avg IFSS_DPF eval            : "
              << (static_cast<double>(stat.dpf_eval_ns) / trials) / 1e6
              << " ms\n";
    std::cout << "  Avg IFSS_DCF_TwoDCF eval     : "
              << (static_cast<double>(stat.dcf_two_eval_ns) / trials) / 1e6
              << " ms\n";
    std::cout << "  Avg IFSS_DCF vector eval     : "
              << (static_cast<double>(stat.dcf_vec_eval_ns) / trials) / 1e6
              << " ms\n";

    std::cout << "\n[Note]\n";
    std::cout << "  IFSS_DPF uses two ordinary DPF keys because current DPF has no vector payload.\n";
    std::cout << "  IFSS_DCF_TwoDCF uses two ordinary DCF keys.\n";
    std::cout << "  IFSS_DCF vector uses one DCF key with groupSize = 2.\n";
    std::cout << "  Each eval measures both parties evaluating one common public input.\n";
    std::cout << "  sink = " << sink << "\n";

    std::cout << "=======================================================\n";
}

int main()
{
    srand(time(NULL));
    IFSS_FSS_50000_TEST();
    return 0;
}