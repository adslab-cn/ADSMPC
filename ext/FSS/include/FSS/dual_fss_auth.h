#pragma once

#include <FSS/dcf.h>
#include <FSS/dpf.h>
#include <FSS/group_element.h>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// Simplified dual-FSS authentication. Both value and MAC paths wrap the
// framework's native primitives; no replacement FSS implementation is used.
struct DualFSSAuthShare { GroupElement value = 0, mac = 0; };
struct DualFSSBatchCheckResult {
    bool accepted = false;
    std::vector<GroupElement> opened;
    std::size_t communication_bytes = 0;
};

struct DualFSSDCFDealerKeys {
    std::pair<DCFKeyPack, DCFKeyPack> value_fss, mac_fss;
    int ring_bits = 64;
};
DualFSSDCFDealerKeys keyGenDualFSSDCF(int input_bits, int value_bits,
    int stat_bits, GroupElement threshold, GroupElement delta);
DualFSSAuthShare evalDualFSSDCF(int party, const DualFSSDCFDealerKeys&, GroupElement x);
void freeDualFSSDCF(DualFSSDCFDealerKeys&);

struct NativeB2AKey {
    std::uint8_t xor_mask_share[2]{0, 0};
    GroupElement arithmetic_mask_share[2]{0, 0};
    GroupElement multiplier = 1;
};
struct DualFSSGrottoDealerKeys {
    std::pair<DPFETKeyPack, DPFETKeyPack> value_fss, mac_fss;
    NativeB2AKey value_b2a, mac_b2a;
    int ring_bits = 64;
};
DualFSSGrottoDealerKeys keyGenDualFSSGrotto(int input_bits, int value_bits,
    int stat_bits, GroupElement threshold, GroupElement delta, std::uint64_t randomness);
void evalDualFSSGrotto(const DualFSSGrottoDealerKeys&, GroupElement x,
    int value_bits, int stat_bits, DualFSSAuthShare &party0,
    DualFSSAuthShare &party1, std::size_t *b2a_online_bytes = nullptr,
    std::uint64_t *b2a_time_ns = nullptr);
void freeDualFSSGrotto(DualFSSGrottoDealerKeys&);

struct DualFSSGTDCFDealerKeys {
    // Two independent native GTDCF instances, matching the dual-FSS treatment
    // used by DCF and GROTTO. Only lane 0 of each native key is consumed.
    std::pair<GTDCFKeyPack, GTDCFKeyPack> value_fss, mac_fss;
    int ring_bits = 64;
};
DualFSSGTDCFDealerKeys keyGenDualFSSGTDCF(int input_bits, int suffix_bits,
    int value_bits, int stat_bits, GroupElement threshold, GroupElement delta);
DualFSSAuthShare evalDualFSSGTDCF(int party, const DualFSSGTDCFDealerKeys&, GroupElement x);
void freeDualFSSGTDCF(DualFSSGTDCFDealerKeys&);

DualFSSBatchCheckResult batchCheckDualFSS(
    const std::vector<DualFSSAuthShare>& party0,
    const std::vector<DualFSSAuthShare>& party1,
    GroupElement delta_share0, GroupElement delta_share1,
    int value_bits, int stat_bits, std::uint64_t coin_seed = 0);
