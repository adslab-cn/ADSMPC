#pragma once
#include <crypto/protocol/util/keypack.h>

std::pair<MultKey, MultKey> MultGen(GroupElement rin1, GroupElement rin2, GroupElement rout);
GroupElement MultEval(int party, const MultKey &k, const GroupElement &l, const GroupElement &r);
GroupElement mult_helper(uint8_t party, GroupElement x, GroupElement y, GroupElement x_mask, GroupElement y_mask);

std::pair<SquareKey, SquareKey> keyGenSquare(GroupElement rin, GroupElement rout);
GroupElement evalSquare(int party, GroupElement x, const SquareKey &k);


// RSSMul