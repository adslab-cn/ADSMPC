#pragma once

#include <utility>
#include <crypto/protocol/util/keypack.h>

// Definitions are provided by the FSS library, not by this header.

void freeMatMulKey(MatMulKey &key);
void freeMatMulKeyPair(std::pair<MatMulKey, MatMulKey> &keys);
void freeConv2dKey(Conv2DKey &key);
void freeConv3dKey(Conv3DKey &key);
void freeReluKeyPack(ReluKeyPack &key);
void freeReluKeyPackPair(std::pair<ReluKeyPack,ReluKeyPack> &keys);
void freeMaxpoolKeyPack(MaxpoolKeyPack &key);
void freeMaxpoolKeyPackPair(std::pair<MaxpoolKeyPack,MaxpoolKeyPack> &keys);
void freeARSKeyPack(ARSKeyPack &key);
void freeARSKeyPackPair(std::pair<ARSKeyPack, ARSKeyPack> &keys);
void freeReluTruncateKeyPack( ReluTruncateKeyPack &key);
void freeReluTruncateKeyPackPair(const std::pair<ReluTruncateKeyPack, ReluTruncateKeyPack> &keys);
void freeRelu2RoundKeyPack(Relu2RoundKeyPack &key);
void freeRelu2RoundKeyPackPair(const std::pair<Relu2RoundKeyPack, Relu2RoundKeyPack> &keys);
void freeSplineKey(SplineKeyPack &key);
void freeSplineKeyPair(std::pair<SplineKeyPack, SplineKeyPack> &keys);
void freeMICKeyPack(MICKeyPack &key);
void freeMSNZBKeyPack(MSNZBKeyPack &key);
void freeBulkyLRSKeyPack(BulkyLRSKeyPack &key, int m);
void freeTaylorKeyPack(TaylorKeyPack &key, int m);
void freeMaxpoolDoubleKeyPack(MaxpoolDoubleKeyPack &key);
void freeMaxpoolDoubleKeyPackPair(std::pair<MaxpoolDoubleKeyPack,MaxpoolDoubleKeyPack> &keys);
void freeFixToFloatKeyPack(FixToFloatKeyPack &key);
void freeFixToFloatKeyPackPair(std::pair<FixToFloatKeyPack, FixToFloatKeyPack> &keys);
void freeFloatToFixKeyPack(FloatToFixKeyPack &key);
void freeFloatToFixKeyPackPair(std::pair<FloatToFixKeyPack, FloatToFixKeyPack> &keys);
void freeReluExtendKeyPack(ReluExtendKeyPack &key);
void freeReluExtendKeyPackPair(std::pair<ReluExtendKeyPack, ReluExtendKeyPack> &keys);
void freeSignExtend2KeyPack(SignExtend2KeyPack &key);
void freeSignExtend2KeyPackPair(std::pair<SignExtend2KeyPack, SignExtend2KeyPack> &keys);
void freeTripleKey(TripleKeyPack &key);
void freeLUTKeyPack(LUTKeyPack &key);
void freeLUTKeyPackPair(std::pair<LUTKeyPack, LUTKeyPack> &keys);
void freeClipKeyPack(ClipKeyPack &key);
void freeClipKeyPackPair(std::pair<ClipKeyPack, ClipKeyPack> &keys);
void freeF2BF16KeyPack(F2BF16KeyPack &key);
void freeF2BF16KeyPackPair(std::pair<F2BF16KeyPack, F2BF16KeyPack> &keys);
void freeTruncateReduceKeyPack(TruncateReduceKeyPack &key);
void freeTruncateReduceKeyPackPair(std::pair<TruncateReduceKeyPack, TruncateReduceKeyPack> &keys);
void freeSlothDreluKeyPack(SlothDreluKeyPack &key);
void freeSlothDreluKeyPackPair(std::pair<SlothDreluKeyPack, SlothDreluKeyPack> &keys);
void freeLUTDPFETKeyPack(LUTDPFETKeyPack &key);
void freeLUTDPFETKeyPackPair(std::pair<LUTDPFETKeyPack, LUTDPFETKeyPack> &keys);
void freeWrapDPFKeyPack(WrapDPFKeyPack &key);
void freeWrapDPFKeyPackPair(std::pair<WrapDPFKeyPack, WrapDPFKeyPack> &keys);
