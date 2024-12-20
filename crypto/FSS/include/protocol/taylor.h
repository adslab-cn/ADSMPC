#pragma once
#include "msnzb.h"
#include "truncate.h"
#include <aux_parameter/keypack.h>

// struct PrivateScaleKeyPack
// {
//     GroupElement rin;
//     GroupElement rout;
// };

// struct SquareKey {
//     GroupElement a;
//     GroupElement b;
// };

// struct TaylorKeyPack {
//     MSNZBKeyPack msnzbKey;
//     SquareKey squareKey;
//     BulkyLRSKeyPack lrsKeys[3];
//     PrivateScaleKeyPack privateScaleKey;
// };

std::pair<PrivateScaleKeyPack, PrivateScaleKeyPack> keyGenPrivateScale(int bin, int bout, GroupElement rin, GroupElement rout);

GroupElement evalPrivateScale(int party, int bin, int bout, GroupElement x, const PrivateScaleKeyPack &key, uint64_t scalar);

std::pair<TaylorSqKey, TaylorSqKey> keyGenTaylorSq(int bin, int bout, GroupElement rin, GroupElement rout);

GroupElement evalTaylorSq(int party, int bin, int bout, GroupElement x, const TaylorSqKey &key);

std::pair<TaylorKeyPack, TaylorKeyPack> keyGenTaylor(int bin, int bout, double a, double b, double c, GroupElement rin, GroupElement rout, int sf, int logk);

std::pair<GroupElement, GroupElement> evalTaylor_round1(int party, int bin, int bout, double a, double b, double c, GroupElement x, const TaylorKeyPack &key, int sf, int logk);

std::pair<GroupElement, GroupElement> evalTaylor_round2(int party, int bin, int bout, double a, double b, double c, GroupElement x, const TaylorKeyPack &key, int sf, int logk, GroupElement alpha, GroupElement square);

GroupElement evalTaylor_round3(int party, int bin, int bout, double a, double b, double c, GroupElement x, const TaylorKeyPack &key, int sf, int logk, GroupElement alpha, GroupElement square, GroupElement ax2bx, uint64_t scalar = 1);

GroupElement evalTaylor_round4(int party, int bin, int bout, double a, double b, double c, GroupElement x, const TaylorKeyPack &key, int sf, int logk, GroupElement alpha, GroupElement ax2bxctrunc);
