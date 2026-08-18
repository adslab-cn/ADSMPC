#pragma once
#include <cryptoTools/Common/Defines.h>
#include <FSS/group_element.h>
#include <FSS/keypack.h>

// 传统DPF接口
std::pair<DPFKeyPack, DPFKeyPack> keyGenDPF(int bin, int bout, GroupElement idx, GroupElement payload);
GroupElement evalDPF_EQ(int party, DPFKeyPack &key, GroupElement x);
GroupElement evalDPF_GT(int party, DPFKeyPack &key, GroupElement x);
GroupElement evalDPF_LT(int party, DPFKeyPack &key, GroupElement x);
void evalAll(int party, DPFKeyPack &key, GroupElement rightShift, GroupElement *out);
GroupElement evalAll_reduce(int party, DPFKeyPack &key, GroupElement rightShift, const std::vector<GroupElement> &tab);

// Grotto-DPF接口
std::pair<DPFETKeyPack, DPFETKeyPack> keyGenDPFET(int bin, GroupElement idx);
std::pair<GroupElement, GroupElement> evalAll_reduce_et(int party, DPFETKeyPack &key, GroupElement rightShift, const std::vector<GroupElement> &tab);
GroupElement evalDPFET_LT(int party, const DPFETKeyPack &key, GroupElement x);

// Grotto prefix/segment parity helpers. These return XOR shares of the
// corresponding predicate for the point encoded by key:
//   prefix:  [alpha < endpoint]
//   segment: [alpha in [start, end)) on the cyclic Z_(2^bin) domain.
GroupElement evalDPFETPrefixParity(int party, const DPFETKeyPack &key, GroupElement endpoint);
GroupElement evalDPFETSegmentParity(int party, const DPFETKeyPack &key,
                                   GroupElement start, GroupElement end);

// GTDPF接口
std::pair<DPFETKeyPack, DPFETKeyPack> keyGenGTDPF(int bin, GroupElement idx);
GroupElement evalGTDPF(int party, const DPFETKeyPack &key, GroupElement x);

// GTDCF
std::pair<GTDCFKeyPack, GTDCFKeyPack> keyGenGTDCF(
    int bin, int w, int groupSize, GroupElement idx, const GroupElement* beta);

void evalGTDCF(
    int party, const GTDCFKeyPack &key, GroupElement x, GroupElement* res);
