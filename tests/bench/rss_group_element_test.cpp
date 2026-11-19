#include <crypto/primitives/RSS/group_element.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

using Wide = unsigned __int128;
using ArithmeticShares = std::array<RssShare, 3>;
using BooleanShares = std::array<RssXorShare, 3>;

static_assert(!std::is_convertible<RssShare, RssXorShare>::value,
              "Arithmetic shares must not convert implicitly to boolean shares");
static_assert(!std::is_convertible<RssXorShare, RssShare>::value,
              "Boolean shares must not convert implicitly to arithmetic shares");

std::size_t checks = 0;
std::size_t inputCases = 0;
std::string context;

void require(bool condition, const char* message)
{
    ++checks;
    if (!condition) {
        throw std::runtime_error(context + ": " + message);
    }
}

// Independent reference arithmetic: no calls to the production mod() helper.
GroupElement maskFor(int bw)
{
    return std::numeric_limits<GroupElement>::max() >> (64 - bw);
}

GroupElement residue(Wide value, int bw)
{
    return static_cast<GroupElement>(value & maskFor(bw));
}

template<class Share>
void verifyLayout(const std::array<Share, 3>& shares, int bw)
{
    const auto highBits = ~maskFor(bw);
    for (std::size_t p = 0; p < 3; ++p) {
        require((shares[p].first & highBits) == 0, "first component is not normalized");
        require((shares[p].second & highBits) == 0, "second component is not normalized");
        require(shares[p].second == shares[(p + 1) % 3].first,
                "adjacent parties disagree on their replicated component");
    }
}

void verifyArithmetic(const ArithmeticShares& shares, GroupElement expected, int bw)
{
    verifyLayout(shares, bw);
    expected &= maskFor(bw);
    require(residue(Wide{shares[0].first} + shares[1].first + shares[2].first, bw)
                == expected, "arithmetic components encode the wrong plaintext");
    for (std::size_t p = 0; p < 3; ++p) {
        const auto missing = shares[(p + 2) % 3].first;
        require(rssCombine(shares[p], missing, bw) == expected,
                "arithmetic reconstruction failed");
        const auto highBits = ~maskFor(bw);
        const RssShare dirty{shares[p].first | highBits, shares[p].second | highBits};
        require(rssCombine(dirty, missing | highBits, bw) == expected,
                "arithmetic reconstruction did not normalize its inputs");
    }
}

void verifyBoolean(const BooleanShares& shares, GroupElement expected, int bw)
{
    verifyLayout(shares, bw);
    expected &= maskFor(bw);
    require((shares[0].first ^ shares[1].first ^ shares[2].first) == expected,
            "boolean components encode the wrong plaintext");
    for (std::size_t p = 0; p < 3; ++p) {
        const auto missing = shares[(p + 2) % 3].first;
        require(rssCombineXor(shares[p], missing, bw) == expected,
                "boolean reconstruction failed");
        const auto highBits = ~maskFor(bw);
        const RssXorShare dirty{shares[p].first | highBits, shares[p].second | highBits};
        require(rssCombineXor(dirty, missing | highBits, bw) == expected,
                "boolean reconstruction did not normalize its inputs");
    }
}

template<class Share>
void verifyPublicLayout(const std::array<Share, 3>& shares, GroupElement value, int bw)
{
    value &= maskFor(bw);
    require(shares[0].first == value && shares[0].second == 0,
            "wrong public value encoding for P0");
    require(shares[1].first == 0 && shares[1].second == 0,
            "wrong public value encoding for P1");
    require(shares[2].first == 0 && shares[2].second == value,
            "wrong public value encoding for P2");
}

template<class Share, class Operation>
std::array<Share, 3> forParties(Operation operation)
{
    std::array<Share, 3> result{};
    for (std::size_t p = 0; p < 3; ++p) {
        result[p] = operation(p, static_cast<RssParty>(p));
    }
    return result;
}

template<class Share>
void addUnusedHighBits(std::array<Share, 3>& shares, int bw)
{
    for (auto& share : shares) {
        share.first |= ~maskFor(bw);
        share.second |= ~maskFor(bw);
    }
}

void checkCase(GroupElement x, GroupElement y, GroupElement constant, int bw,
               osuCrypto::PRNG& arithmeticOwner, osuCrypto::PRNG& booleanOwner)
{
    ++inputCases;
    const std::string prefix = "bw=" + std::to_string(bw) + " case="
        + std::to_string(inputCases) + " ";
    context = prefix + "input sharing";
    auto ax = splitShareRss(x, bw, arithmeticOwner);
    auto ay = splitShareRss(y, bw, arithmeticOwner);
    auto bx = splitShareRssXor(x, bw, booleanOwner);
    auto by = splitShareRssXor(y, bw, booleanOwner);
    verifyArithmetic(ax, x, bw);
    verifyArithmetic(ay, y, bw);
    verifyBoolean(bx, x, bw);
    verifyBoolean(by, y, bw);

    // Exercise both canonical and non-canonical representations.
    if (inputCases % 2 == 0) {
        addUnusedHighBits(ax, bw);
        addUnusedHighBits(ay, bw);
        addUnusedHighBits(bx, bw);
        addUnusedHighBits(by, bw);
    }

    context = prefix + "public encoding";
    const auto pa = forParties<RssShare>([&](std::size_t, RssParty party) {
        return rssFromPublic(constant, party, bw);
    });
    const auto pb = forParties<RssXorShare>([&](std::size_t, RssParty party) {
        return rssFromPublicXor(constant, party, bw);
    });
    verifyPublicLayout(pa, constant, bw);
    verifyPublicLayout(pb, constant, bw);
    verifyArithmetic(pa, constant, bw);
    verifyBoolean(pb, constant, bw);

    context = prefix + "rssAdd";
    verifyArithmetic(forParties<RssShare>([&](std::size_t p, RssParty) {
        return rssAdd(ax[p], ay[p], bw);
    }), residue(Wide{x} + y, bw), bw);
    context = prefix + "rssSub";
    verifyArithmetic(forParties<RssShare>([&](std::size_t p, RssParty) {
        return rssSub(ax[p], ay[p], bw);
    }), residue(Wide{x} - y, bw), bw);
    context = prefix + "rssMulPublic";
    verifyArithmetic(forParties<RssShare>([&](std::size_t p, RssParty) {
        return rssMulPublic(ax[p], constant, bw);
    }), residue(Wide{x} * constant, bw), bw);
    context = prefix + "rssNeg";
    verifyArithmetic(forParties<RssShare>([&](std::size_t p, RssParty) {
        return rssNeg(ax[p], bw);
    }), residue(Wide{0} - x, bw), bw);
    context = prefix + "rssAddPublic";
    verifyArithmetic(forParties<RssShare>([&](std::size_t p, RssParty party) {
        return rssAddPublic(ax[p], constant, party, bw);
    }), residue(Wide{x} + constant, bw), bw);
    context = prefix + "rssSubPublic";
    verifyArithmetic(forParties<RssShare>([&](std::size_t p, RssParty party) {
        return rssSubPublic(ax[p], constant, party, bw);
    }), residue(Wide{x} - constant, bw), bw);

    context = prefix + "rssXor";
    verifyBoolean(forParties<RssXorShare>([&](std::size_t p, RssParty) {
        return rssXor(bx[p], by[p], bw);
    }), x ^ y, bw);
    context = prefix + "rssXorPublic";
    verifyBoolean(forParties<RssXorShare>([&](std::size_t p, RssParty party) {
        return rssXorPublic(bx[p], constant, party, bw);
    }), x ^ constant, bw);
    context = prefix + "rssAndPublic";
    verifyBoolean(forParties<RssXorShare>([&](std::size_t p, RssParty) {
        return rssAndPublic(bx[p], constant, bw);
    }), x & constant, bw);
    context = prefix + "rssNot";
    verifyBoolean(forParties<RssXorShare>([&](std::size_t p, RssParty party) {
        return rssNot(bx[p], party, bw);
    }), ~x, bw);
}

void testOperations()
{
    osuCrypto::PRNG arithmeticOwner(osuCrypto::toBlock(0x12345678, 0x9abcdef0));
    osuCrypto::PRNG booleanOwner(osuCrypto::toBlock(0x23456789, 0xabcdef01));
    std::mt19937_64 samples(0x5eed1234);

    context = "default initialization";
    const RssShare arithmeticZero{};
    const RssXorShare booleanZero{};
    require(arithmeticZero.first == 0 && arithmeticZero.second == 0,
            "arithmetic share is not initialized to zero");
    require(booleanZero.first == 0 && booleanZero.second == 0,
            "boolean share is not initialized to zero");

    // Known components also exercise reconstruction independently of the split helpers.
    context = "known arithmetic components";
    verifyArithmetic(ArithmeticShares{{{5, 7}, {7, 11}, {11, 5}}}, 23, 8);
    context = "known boolean components";
    verifyBoolean(BooleanShares{{{5, 7}, {7, 11}, {11, 5}}}, 9, 8);

    for (int bw = 1; bw <= 64; ++bw) {
        const auto mask = maskFor(bw);
        const std::array<GroupElement, 7> boundary{
            0, 1, mask, mask - 1, GroupElement{1} << (bw - 1),
            std::numeric_limits<GroupElement>::max(),
            bw == 64 ? mask - 1 : GroupElement{1} << bw
        };
        for (std::size_t i = 0; i < boundary.size(); ++i) {
            for (std::size_t j = 0; j < boundary.size(); ++j) {
                checkCase(boundary[i], boundary[j], boundary[(i + 2 * j) % boundary.size()],
                          bw, arithmeticOwner, booleanOwner);
            }
        }
        for (int i = 0; i < 32; ++i) {
            const auto x = samples();
            const auto y = samples();
            const auto constant = samples();
            checkCase(x, y, constant, bw, arithmeticOwner, booleanOwner);
        }
    }
}

void testPairwiseRandomness(bool boolean)
{
    // Six endpoint objects: prev[i] and next[(i+2)%3] share edge seed i.
    std::array<osuCrypto::PRNG, 3> prev;
    std::array<osuCrypto::PRNG, 3> next;
    std::array<osuCrypto::PRNG, 3> reference;
    const std::uint64_t domain = boolean ? 0x424f4f4c : 0x41524954;
    for (std::size_t p = 0; p < 3; ++p) {
        const auto seed = osuCrypto::toBlock(domain, p + 123);
        prev[p].SetSeed(seed);
        next[(p + 2) % 3].SetSeed(seed);
        reference[p].SetSeed(seed);
    }

    // 1280 draws per endpoint also cross the PRNG's default buffer boundary.
    for (int bw = 1; bw <= 64; ++bw) {
        for (int sample = 0; sample < 20; ++sample) {
            context = std::string(boolean ? "boolean" : "arithmetic")
                + " pairwise randomness bw=" + std::to_string(bw)
                + " sample=" + std::to_string(sample);
            std::array<GroupElement, 3> components{};
            for (std::size_t p = 0; p < 3; ++p) {
                components[p] = reference[p].get<GroupElement>() & maskFor(bw);
            }
            if (boolean) {
                const auto shares = forParties<RssXorShare>([&](std::size_t p, RssParty) {
                    return randomShareRssXor(bw, prev[p], next[p]);
                });
                verifyBoolean(shares, components[0] ^ components[1] ^ components[2], bw);
                for (std::size_t p = 0; p < 3; ++p) {
                    require(shares[p].first == components[p], "wrong previous-edge stream");
                    require(shares[p].second == components[(p + 1) % 3], "wrong next-edge stream");
                }
            } else {
                const auto shares = forParties<RssShare>([&](std::size_t p, RssParty) {
                    return randomShareRss(bw, prev[p], next[p]);
                });
                verifyArithmetic(shares, residue(Wide{components[0]} + components[1]
                    + components[2], bw), bw);
                for (std::size_t p = 0; p < 3; ++p) {
                    require(shares[p].first == components[p], "wrong previous-edge stream");
                    require(shares[p].second == components[(p + 1) % 3], "wrong next-edge stream");
                }
            }
        }
    }
}

template<class Operation>
void expectInvalid(Operation operation, const char* name)
{
    bool rejected = false;
    try {
        operation();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, name);
}

void verifyUnconsumed(osuCrypto::PRNG& actual, osuCrypto::PRNG& expected)
{
    for (int i = 0; i < 4; ++i) {
        const auto observed = actual.get<GroupElement>();
        const auto reference = expected.get<GroupElement>();
        require(observed == reference, "invalid argument consumed PRNG output");
    }
}

void testInvalidArguments()
{
    const auto seed = osuCrypto::toBlock(0x1111, 0x2222);
    const auto secondSeed = osuCrypto::toBlock(0x3333, 0x4444);
    osuCrypto::PRNG owner(seed), expectedOwner(seed);
    osuCrypto::PRNG prev(seed), expectedPrev(seed);
    osuCrypto::PRNG next(secondSeed), expectedNext(secondSeed);
    const RssShare a{7, 11};
    const RssXorShare b{7, 11};
    const auto p = RssParty::P0;

    for (int bw : {std::numeric_limits<int>::min(), -1, 0, 65, 128,
                   std::numeric_limits<int>::max()}) {
        context = "invalid bit width " + std::to_string(bw);
        expectInvalid([&] { splitShareRss(1, bw, owner); }, "splitShareRss accepted invalid bw");
        verifyUnconsumed(owner, expectedOwner);
        expectInvalid([&] { splitShareRssXor(1, bw, owner); }, "splitShareRssXor accepted invalid bw");
        verifyUnconsumed(owner, expectedOwner);
        expectInvalid([&] { randomShareRss(bw, prev, next); }, "randomShareRss accepted invalid bw");
        verifyUnconsumed(prev, expectedPrev);
        verifyUnconsumed(next, expectedNext);
        expectInvalid([&] { randomShareRssXor(bw, prev, next); }, "randomShareRssXor accepted invalid bw");
        verifyUnconsumed(prev, expectedPrev);
        verifyUnconsumed(next, expectedNext);
        expectInvalid([&] { rssFromPublic(1, p, bw); }, "rssFromPublic accepted invalid bw");
        expectInvalid([&] { rssAdd(a, a, bw); }, "rssAdd accepted invalid bw");
        expectInvalid([&] { rssSub(a, a, bw); }, "rssSub accepted invalid bw");
        expectInvalid([&] { rssMulPublic(a, 3, bw); }, "rssMulPublic accepted invalid bw");
        expectInvalid([&] { rssCombine(a, 5, bw); }, "rssCombine accepted invalid bw");
        expectInvalid([&] { rssNeg(a, bw); }, "rssNeg accepted invalid bw");
        expectInvalid([&] { rssAddPublic(a, 3, p, bw); }, "rssAddPublic accepted invalid bw");
        expectInvalid([&] { rssSubPublic(a, 3, p, bw); }, "rssSubPublic accepted invalid bw");
        expectInvalid([&] { rssFromPublicXor(1, p, bw); }, "rssFromPublicXor accepted invalid bw");
        expectInvalid([&] { rssXor(b, b, bw); }, "rssXor accepted invalid bw");
        expectInvalid([&] { rssXorPublic(b, 3, p, bw); }, "rssXorPublic accepted invalid bw");
        expectInvalid([&] { rssAndPublic(b, 3, bw); }, "rssAndPublic accepted invalid bw");
        expectInvalid([&] { rssNot(b, p, bw); }, "rssNot accepted invalid bw");
        expectInvalid([&] { rssCombineXor(b, 5, bw); }, "rssCombineXor accepted invalid bw");
    }
    for (int partyValue : {-1, 3, 1000}) {
        const auto party = static_cast<RssParty>(partyValue);
        for (int bw : {1, 32, 64}) {
            context = "invalid party " + std::to_string(partyValue) + " bw=" + std::to_string(bw);
            expectInvalid([&] { rssFromPublic(1, party, bw); }, "rssFromPublic accepted invalid party");
            expectInvalid([&] { rssAddPublic(a, 1, party, bw); }, "rssAddPublic accepted invalid party");
            expectInvalid([&] { rssSubPublic(a, 1, party, bw); }, "rssSubPublic accepted invalid party");
            expectInvalid([&] { rssFromPublicXor(1, party, bw); }, "rssFromPublicXor accepted invalid party");
            expectInvalid([&] { rssXorPublic(b, 1, party, bw); }, "rssXorPublic accepted invalid party");
            expectInvalid([&] { rssNot(b, party, bw); }, "rssNot accepted invalid party");
        }
    }
}

} // namespace

int main()
{
    try {
        testOperations();
        testPairwiseRandomness(false);
        testPairwiseRandomness(true);
        testInvalidArguments();
        std::cout << "RSS group element checks passed: " << inputCases
                  << " input cases across widths 1..64, 2560 pairwise samples, "
                  << checks << " explicit checks";
#ifdef NDEBUG
        std::cout << " (NDEBUG enabled)";
#endif
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RSS group element test failed: " << error.what() << '\n';
        return 1;
    }
}
