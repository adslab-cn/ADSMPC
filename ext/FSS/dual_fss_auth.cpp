#include <FSS/dual_fss_auth.h>
#include <FSS/freekey.h>
#include <chrono>
#include <random>
#include <stdexcept>

namespace {
GroupElement maskFor(int bits) {
    if (bits <= 0 || bits > 64) throw std::invalid_argument("bad ring width");
    return bits == 64 ? ~GroupElement(0) : ((GroupElement(1) << bits) - 1);
}
GroupElement red(GroupElement x, int bits) { return x & maskFor(bits); }

NativeB2AKey makeB2A(int bits, GroupElement multiplier, std::mt19937_64 &rng) {
    const auto mask = maskFor(bits); const std::uint8_t r = rng() & 1;
    NativeB2AKey k; k.multiplier = multiplier & mask;
    k.xor_mask_share[0] = rng() & 1;
    k.xor_mask_share[1] = k.xor_mask_share[0] ^ r;
    k.arithmetic_mask_share[0] = rng() & mask;
    k.arithmetic_mask_share[1] =
        (GroupElement(r) * k.multiplier - k.arithmetic_mask_share[0]) & mask;
    return k;
}
void b2aBoth(std::uint8_t b0, std::uint8_t b1, const NativeB2AKey &k,
             int bits, GroupElement &o0, GroupElement &o1) {
    const auto mask = maskFor(bits);
    const std::uint8_t c = (b0 ^ k.xor_mask_share[0]) ^
                           (b1 ^ k.xor_mask_share[1]);
    o0 = k.arithmetic_mask_share[0]; o1 = k.arithmetic_mask_share[1];
    if (c) { o0 = (-o0) & mask; o1 = (k.multiplier - o1) & mask; }
}
} // namespace

DualFSSDCFDealerKeys keyGenDualFSSDCF(int n, int vb, int, GroupElement a,
                                      GroupElement delta) {
    DualFSSDCFDealerKeys k; k.ring_bits = vb;
    k.value_fss = keyGenDCF(n, vb, a, 1);
    k.mac_fss = keyGenDCF(n, vb, a, red(delta, vb));
    return k;
}
DualFSSAuthShare evalDualFSSDCF(int p, const DualFSSDCFDealerKeys &k,
                                GroupElement x) {
    GroupElement v = 0, m = 0;
    evalDCF(p, &v, x, p ? k.value_fss.second : k.value_fss.first);
    evalDCF(p, &m, x, p ? k.mac_fss.second : k.mac_fss.first);
    return {red(v, k.ring_bits), red(m, k.ring_bits)};
}
void freeDualFSSDCF(DualFSSDCFDealerKeys &k) {
    freeDCFKeyPackPair(k.value_fss); freeDCFKeyPackPair(k.mac_fss);
}

DualFSSGrottoDealerKeys keyGenDualFSSGrotto(int n, int vb, int,
    GroupElement a, GroupElement delta, std::uint64_t randomness) {
    DualFSSGrottoDealerKeys k; k.ring_bits = vb;
    k.value_fss = keyGenDPFET(n, a);
    k.mac_fss = keyGenDPFET(n, a);
    std::mt19937_64 rng(randomness);
    k.value_b2a = makeB2A(vb, 1, rng);
    k.mac_b2a = makeB2A(vb, red(delta, vb), rng);
    return k;
}
void evalDualFSSGrotto(const DualFSSGrottoDealerKeys &k, GroupElement x,
    int vb, int, DualFSSAuthShare &a, DualFSSAuthShare &b,
    std::size_t *bytes, std::uint64_t *b2aNs) {
    const auto v0 = std::uint8_t(evalDPFET_LT(0, k.value_fss.first, x) & 1);
    const auto v1 = std::uint8_t(evalDPFET_LT(1, k.value_fss.second, x) & 1);
    const auto m0 = std::uint8_t(evalDPFET_LT(0, k.mac_fss.first, x) & 1);
    const auto m1 = std::uint8_t(evalDPFET_LT(1, k.mac_fss.second, x) & 1);
    const auto t0 = std::chrono::steady_clock::now();
    b2aBoth(v0, v1, k.value_b2a, vb, a.value, b.value);
    b2aBoth(m0, m1, k.mac_b2a, vb, a.mac, b.mac);
    const auto t1 = std::chrono::steady_clock::now();
    if (b2aNs) *b2aNs += std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
    if (bytes) *bytes += 1; // four online bits, rounded up to one byte
}
void freeDualFSSGrotto(DualFSSGrottoDealerKeys &k) {
    freeDPFKeyPackPair(k.value_fss); freeDPFKeyPackPair(k.mac_fss);
}

DualFSSGTDCFDealerKeys keyGenDualFSSGTDCF(int n, int w, int vb, int,
    GroupElement a, GroupElement delta) {
    DualFSSGTDCFDealerKeys k; k.ring_bits = vb;
    const GroupElement value_payload[2] = {1, 0};
    const GroupElement mac_payload[2] = {red(delta, vb), 0};
    k.value_fss = keyGenGTDCF(n, w, 2, a, value_payload);
    k.mac_fss = keyGenGTDCF(n, w, 2, a, mac_payload);
    return k;
}
DualFSSAuthShare evalDualFSSGTDCF(int p, const DualFSSGTDCFDealerKeys &k,
                                  GroupElement x) {
    GroupElement value[2]{0,0}, mac[2]{0,0};
    evalGTDCF(p, p ? k.value_fss.second : k.value_fss.first, x, value);
    evalGTDCF(p, p ? k.mac_fss.second : k.mac_fss.first, x, mac);
    return {red(value[0], k.ring_bits), red(mac[0], k.ring_bits)};
}
void freeDualFSSGTDCF(DualFSSGTDCFDealerKeys &k) {
    freeGTDCFKeyPackPair(k.value_fss);
    freeGTDCFKeyPackPair(k.mac_fss);
}

DualFSSBatchCheckResult batchCheckDualFSS(
    const std::vector<DualFSSAuthShare> &a,
    const std::vector<DualFSSAuthShare> &b,
    GroupElement d0, GroupElement d1, int vb, int, std::uint64_t seed) {
    if (a.size() != b.size()) throw std::invalid_argument("batch shape");
    const auto mask = maskFor(vb), delta = (d0 + d1) & mask;
    std::mt19937_64 rng(seed ? seed : std::random_device{}());
    GroupElement z = 0; DualFSSBatchCheckResult r; r.opened.reserve(a.size());
    for (std::size_t i=0; i<a.size(); ++i) {
        const auto x = (a[i].value + b[i].value) & mask;
        const auto tag = (a[i].mac + b[i].mac) & mask;
        z = (z + (rng() & mask) * (tag - delta*x)) & mask;
        r.opened.push_back(x);
    }
    const std::size_t bytes = (vb+7)/8;
    r.communication_bytes = 2*bytes*(a.size()+1);
    r.accepted = (z == 0); if (!r.accepted) r.opened.clear(); return r;
}
