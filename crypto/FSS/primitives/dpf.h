#pragma once
#include <cryptoTools/Common/Defines.h>
#include "../aux_parameter/group_element.h"
#include "../aux_parameter/keypack.h"
#include <vector>
#include <map>
#include <cstdint>
#include <cassert>
std::pair<DPFKeyPack, DPFKeyPack> keyGenDPF(int bin, int bout, GroupElement idx, GroupElement payload);
GroupElement evalDPF_EQ(int party, DPFKeyPack &key, GroupElement x);
GroupElement evalDPF_GT(int party, DPFKeyPack &key, GroupElement x);
GroupElement evalDPF_LT(int party, DPFKeyPack &key, GroupElement x);
void evalAll(int party, DPFKeyPack &key, GroupElement rightShift, GroupElement *out);
GroupElement evalAll_reduce(int party, DPFKeyPack &key, GroupElement rightShift, const std::vector<GroupElement> &tab);

std::pair<DPFETKeyPack, DPFETKeyPack> keyGenDPFET(int bin, GroupElement idx);
std::pair<GroupElement, GroupElement> evalAll_reduce_et(int party, DPFETKeyPack &key, GroupElement rightShift, const std::vector<GroupElement> &tab);
GroupElement evalDPFET_LT(int party, const DPFETKeyPack &key, GroupElement x);

GroupElement evalDPF_with_payload(int party, DPFKeyPack &key, GroupElement x);

/**
 * @brief 使用 Grotto 的 prefix-parity 算法，高效地计算一批前缀区间的奇偶性。
 * 
 * @param party      当前参与方的 ID (SERVER=2, CLIENT=3)。
 * @param key        当前参与方持有的 DPF 密钥份额。
 * @param endpoints  一个C-style数组，包含所有需要计算奇偶性的前缀端点 K。
 * @param num_endpoints endpoints 数组的大小。
 * @return           一个 map，将每个端点 K 映射到其对应的前缀奇偶性 Parity(ē_i[0, K)) 的份额。
 *                   这个map由调用者负责管理内存。
 */
std::map<GroupElement, uint8_t>* compute_prefix_parities(
    int party,
    const DPFKeyPack& key,
    const GroupElement* endpoints,
    int num_endpoints
);
