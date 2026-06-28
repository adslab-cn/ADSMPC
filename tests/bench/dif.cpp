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

struct DIFBenchStats
{
    uint64_t correct = 0;
    uint64_t keygen_ns = 0;
    uint64_t eval_ns = 0;

    uint64_t full_interval_cases = 0;
    uint64_t prefix_cases = 0;
    uint64_t suffix_cases = 0;
    uint64_t singleton_cases = 0;
    uint64_t general_cases = 0;

    uint64_t hit_cases = 0;
    uint64_t miss_cases = 0;
};

static inline GroupElement test_domain_max(int bin)
{
    always_assert(bin > 0);
    always_assert(bin <= 64);

    if (bin == 64)
    {
        return ~GroupElement(0);
    }

    return (GroupElement(1) << bin) - 1;
}

static GroupElement rand_ge_local(std::mt19937_64 &rng, int bw)
{
    GroupElement x = static_cast<GroupElement>(rng());
    mod(x, bw);
    return x;
}

static bool plain_interval_contains(GroupElement a,
                                    GroupElement b,
                                    GroupElement x)
{
    return (a <= x) && (x <= b);
}

static GroupElement plain_dif(GroupElement a,
                              GroupElement b,
                              GroupElement beta,
                              GroupElement x,
                              int bout)
{
    GroupElement expected = plain_interval_contains(a, b, x) ? beta : 0;
    mod(expected, bout);
    return expected;
}

static GroupElement reconstruct_dif(GroupElement y0,
                                    GroupElement y1,
                                    int bout)
{
    GroupElement y = y0 + y1;
    mod(y, bout);
    return y;
}

static void generate_interval_case(std::mt19937_64 &rng,
                                   int rep,
                                   int bin,
                                   GroupElement &a,
                                   GroupElement &b,
                                   GroupElement &x,
                                   bool &isHit,
                                   DIFBenchStats &stat)
{
    const GroupElement maxVal = test_domain_max(bin);

    // To avoid accidental overflow in test generation, most random
    // general intervals are sampled from a 48-bit subdomain.
    const GroupElement mask48 = (GroupElement(1) << 48) - 1;

    int type = rep % 10;

    if (type == 0)
    {
        // Full interval [0, max]
        a = 0;
        b = maxVal;
        x = rand_ge_local(rng, bin);
        isHit = true;
        stat.full_interval_cases++;
        stat.hit_cases++;
        return;
    }

    if (type == 1)
    {
        // Prefix interval [0, b]
        a = 0;
        b = static_cast<GroupElement>(rng()) & mask48;

        if (rep & 1)
        {
            x = b;
            isHit = true;
            stat.hit_cases++;
        }
        else
        {
            x = b + 1;
            mod(x, bin);
            isHit = false;
            stat.miss_cases++;
        }

        stat.prefix_cases++;
        return;
    }

    if (type == 2)
    {
        // Suffix interval [a, max]
        a = (static_cast<GroupElement>(rng()) & mask48) + 1;
        b = maxVal;

        if (rep & 1)
        {
            x = a;
            isHit = true;
            stat.hit_cases++;
        }
        else
        {
            x = a - 1;
            isHit = false;
            stat.miss_cases++;
        }

        stat.suffix_cases++;
        return;
    }

    if (type == 3)
    {
        // Singleton interval [a, a]
        a = static_cast<GroupElement>(rng()) & mask48;
        b = a;

        if (rep & 1)
        {
            x = a;
            isHit = true;
            stat.hit_cases++;
        }
        else
        {
            x = a + 1;
            mod(x, bin);
            isHit = false;
            stat.miss_cases++;
        }

        stat.singleton_cases++;
        return;
    }

    // General interval [a, b], a < b
    {
        GroupElement base = static_cast<GroupElement>(rng()) & mask48;
        GroupElement len = (static_cast<GroupElement>(rng()) % 100000) + 1;

        a = base;
        b = base + len;
        mod(a, bin);
        mod(b, bin);

        // Since base and len are small, a <= b holds.
        always_assert(a <= b);

        if (rep & 1)
        {
            GroupElement offset = static_cast<GroupElement>(rng()) % (len + 1);
            x = a + offset;
            mod(x, bin);
            isHit = true;
            stat.hit_cases++;
        }
        else
        {
            if (a > 0)
            {
                x = a - 1;
            }
            else
            {
                x = b + 1;
                mod(x, bin);
            }

            isHit = false;
            stat.miss_cases++;
        }

        stat.general_cases++;
        return;
    }
}

void DIF_50000_TEST()
{
    std::cout << "\n=======================================================\n";
    std::cout << "Starting DIF 50000-Test...\n";

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

    DIFBenchStats stat;

    volatile GroupElement sink = 0;

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int rep = 0; rep < trials; ++rep)
    {
        GroupElement a = 0;
        GroupElement b = 0;
        GroupElement x = 0;
        bool isHit = false;

        generate_interval_case(rng, rep, bin, a, b, x, isHit, stat);

        GroupElement beta = rand_ge_local(rng, bout);

        // ------------------------------------------------------------
        // KeyGen
        // ------------------------------------------------------------

        auto kg_start = std::chrono::high_resolution_clock::now();

        auto difKeys = keyGenDIF(bin, bout, a, b, beta);

        auto kg_end = std::chrono::high_resolution_clock::now();

        stat.keygen_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            kg_end - kg_start
        ).count();

        // ------------------------------------------------------------
        // Eval
        // ------------------------------------------------------------

        auto eval_start = std::chrono::high_resolution_clock::now();

        GroupElement y0 = evalDIF(0, difKeys.first, x);
        GroupElement y1 = evalDIF(1, difKeys.second, x);

        auto eval_end = std::chrono::high_resolution_clock::now();

        stat.eval_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            eval_end - eval_start
        ).count();

        // ------------------------------------------------------------
        // Correctness
        // ------------------------------------------------------------

        GroupElement got = reconstruct_dif(y0, y1, bout);
        GroupElement expected = plain_dif(a, b, beta, x, bout);

        if (got == expected)
        {
            stat.correct++;
        }
        else
        {
            std::cerr << "DIF mismatch at rep = " << rep << "\n";
            std::cerr << "  a        = " << a << "\n";
            std::cerr << "  b        = " << b << "\n";
            std::cerr << "  x        = " << x << "\n";
            std::cerr << "  beta     = " << beta << "\n";
            std::cerr << "  got      = " << got << "\n";
            std::cerr << "  expected = " << expected << "\n";
            std::cerr << "  isHit    = " << isHit << "\n";
            always_assert(false);
        }

        sink ^= y0;
        sink ^= y1;
        sink ^= got;
        sink ^= expected;

        freeDIFKeyPackPair(difKeys);

        if ((rep + 1) % 5000 == 0)
        {
            std::cout << "  Finished " << (rep + 1)
                      << " / " << trials << "\n";
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();

    uint64_t total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        total_end - total_start
    ).count();

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "\n=======================================================\n";
    std::cout << "DIF 50000-Test Results\n";
    std::cout << "=======================================================\n";

    std::cout << "\n[Parameters]\n";
    std::cout << "  trials              : " << trials << "\n";
    std::cout << "  bin                 : " << bin << "\n";
    std::cout << "  bout                : " << bout << "\n";

    std::cout << "\n[Case Distribution]\n";
    std::cout << "  full interval cases : " << stat.full_interval_cases << "\n";
    std::cout << "  prefix cases        : " << stat.prefix_cases << "\n";
    std::cout << "  suffix cases        : " << stat.suffix_cases << "\n";
    std::cout << "  singleton cases     : " << stat.singleton_cases << "\n";
    std::cout << "  general cases       : " << stat.general_cases << "\n";
    std::cout << "  hit cases           : " << stat.hit_cases << "\n";
    std::cout << "  miss cases          : " << stat.miss_cases << "\n";

    std::cout << "\n[Correctness]\n";
    std::cout << "  DIF correct         : "
              << stat.correct << " / " << trials << "\n";

    std::cout << "\n[Total Time]\n";
    std::cout << "  Total wall-clock    : "
              << static_cast<double>(total_ns) / 1e6 << " ms\n";
    std::cout << "  DIF keygen time     : "
              << static_cast<double>(stat.keygen_ns) / 1e6 << " ms\n";
    std::cout << "  DIF eval time       : "
              << static_cast<double>(stat.eval_ns) / 1e6 << " ms\n";

    std::cout << "\n[Average Time]\n";
    std::cout << "  Avg DIF keygen      : "
              << (static_cast<double>(stat.keygen_ns) / trials) / 1e6
              << " ms\n";
    std::cout << "  Avg DIF eval        : "
              << (static_cast<double>(stat.eval_ns) / trials) / 1e6
              << " ms\n";

    std::cout << "\n[Note]\n";
    std::cout << "  This DIF implementation realizes beta * {a <= x <= b}.\n";
    std::cout << "  Internally it uses beta*{x < b+1} - beta*{x < a}.\n";
    std::cout << "  For b = 2^bin-1, the upper term is represented as a constant share.\n";
    std::cout << "  sink = " << sink << "\n";

    std::cout << "=======================================================\n";
}

int main()
{
    srand(time(NULL));
    DIF_50000_TEST();
    return 0;
}