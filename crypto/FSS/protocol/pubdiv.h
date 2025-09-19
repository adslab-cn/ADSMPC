

#pragma once
#include "../aux_parameter/keypack.h"

std::pair<ScmpKeyPack, ScmpKeyPack> keyGenSCMP(int Bin, int Bout, GroupElement rin1, GroupElement rin2,
                                GroupElement rout);

GroupElement evalSCMP(int party, ScmpKeyPack key, GroupElement x, GroupElement y);

std::pair<ARSKeyPack, ARSKeyPack> keyGenARS(int Bin, int Bout, uint64_t shift, GroupElement rin, GroupElement rout);

GroupElement evalARS(int party, GroupElement x, uint64_t shift, const ARSKeyPack &k);

std::pair<EdabitsPrTruncKeyPack, EdabitsPrTruncKeyPack> keyGenEdabitsPrTrunc(int bw, int shift, GroupElement rin, GroupElement rout);

std::pair<TruncateReduceKeyPack, TruncateReduceKeyPack> keyGenTruncateReduce(int bin, int shift, GroupElement rin, GroupElement rout);
GroupElement evalTruncateReduce(int party, GroupElement x, const TruncateReduceKeyPack &k);

std::pair<SlothLRSKeyPack, SlothLRSKeyPack> keyGenSlothLRS(int bin, int shift, GroupElement rin, GroupElement rinWrap, GroupElement rout);
GroupElement evalSlothLRS(int party, GroupElement x, GroupElement w, const SlothLRSKeyPack &k);
