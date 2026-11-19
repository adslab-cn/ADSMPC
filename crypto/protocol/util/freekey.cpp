#include <crypto/protocol/util/freekey.h>
#include <crypto/primitives/FSS/freekey.h>
#include <crypto/commom/comms.h>
#include <crypto/commom/config.h>

void freeMatMulKey(MatMulKey &key){
    delete[] key.a;
    delete[] key.b;
    delete[] key.c;
}

void freeMatMulKeyPair(std::pair<MatMulKey, MatMulKey> &keys){
    delete[] keys.first.a;
    delete[] keys.first.b;
    delete[] keys.first.c;
    delete[] keys.second.a;
    delete[] keys.second.b;
    delete[] keys.second.c;
}

void freeConv2dKey(Conv2DKey &key){
    delete[] key.a;
    delete[] key.b;
    delete[] key.c;
}

void freeConv3dKey(Conv3DKey &key){
    delete[] key.a;
    delete[] key.b;
    delete[] key.c;
}

void freeReluKeyPack(ReluKeyPack &key)
{
    if (!Config::dealer->keyBuf->isMem()) {
        delete[] key.k;
    }
    delete[] key.g;
    if (!(Config::dealer->keyBuf->isMem() && (key.Bout > 32))) {
        delete[] key.v;
    }
}

void freeReluKeyPackPair(std::pair<ReluKeyPack,ReluKeyPack> &keys)
{
    delete[] keys.first.k;
    delete[] keys.second.k;
    delete[] keys.first.g;
    delete[] keys.first.v;
    // other key shares g and v, dont delete again
}

void freeMaxpoolKeyPack(MaxpoolKeyPack &key)
{
    freeReluKeyPack(key.reluKey);
}

void freeMaxpoolKeyPackPair(std::pair<MaxpoolKeyPack,MaxpoolKeyPack> &keys)
{
    delete[] keys.first.reluKey.k;
    delete[] keys.second.reluKey.k;
    delete[] keys.first.reluKey.g;
    delete[] keys.first.reluKey.v;
}

void freeARSKeyPack(ARSKeyPack &key)
{
    freeDCFKeyPack(key.dcfKey);
    if (key.Bout > key.Bin - key.shift) {
        freeDualDCFKeyPack(key.dualDcfKey);
    }
}

void freeARSKeyPackPair(std::pair<ARSKeyPack, ARSKeyPack> &keys)
{
    delete[] keys.first.dcfKey.k;
    delete[] keys.second.dcfKey.k;
    delete[] keys.first.dcfKey.g;
    delete[] keys.first.dcfKey.v;
    if (keys.first.Bout > keys.first.Bin - keys.first.shift) {
        delete[] keys.first.dualDcfKey.sb;
        delete[] keys.second.dualDcfKey.sb;
        delete[] keys.first.dualDcfKey.dcfKey.k;
        delete[] keys.second.dualDcfKey.dcfKey.k;
        delete[] keys.first.dualDcfKey.dcfKey.g;
        delete[] keys.first.dualDcfKey.dcfKey.v;
    }
}

void freeReluTruncateKeyPack( ReluTruncateKeyPack &key)
{
    freeDCFKeyPack(key.dcfKeyN);
    freeDCFKeyPack(key.dcfKeyS);
}

void freeReluTruncateKeyPackPair(const std::pair<ReluTruncateKeyPack, ReluTruncateKeyPack> &keys)
{
    delete[] keys.first.dcfKeyN.k;
    delete[] keys.second.dcfKeyN.k;
    delete[] keys.first.dcfKeyN.g;
    delete[] keys.first.dcfKeyN.v;

    delete[] keys.first.dcfKeyS.k;
    delete[] keys.second.dcfKeyS.k;
    delete[] keys.first.dcfKeyS.g;
    delete[] keys.first.dcfKeyS.v;
}

void freeRelu2RoundKeyPack(Relu2RoundKeyPack &key)
{
    freeDCFKeyPack(key.dcfKey);
}

void freeRelu2RoundKeyPackPair(const std::pair<Relu2RoundKeyPack, Relu2RoundKeyPack> &keys)
{
    delete[] keys.first.dcfKey.k;
    delete[] keys.second.dcfKey.k;
    delete[] keys.first.dcfKey.g;
    delete[] keys.first.dcfKey.v;
}

void freeSplineKey(SplineKeyPack &key)
{
    freeDCFKeyPack(key.dcfKey);
    key.p.clear();
    key.e_b.clear();
    key.beta_b.clear();
}

void freeSplineKeyPair(std::pair<SplineKeyPack, SplineKeyPack> &keys)
{
    delete[] keys.first.dcfKey.k;
    delete[] keys.second.dcfKey.k;
    delete[] keys.first.dcfKey.g;
    delete[] keys.first.dcfKey.v;
    keys.first.p.clear();
    keys.second.p.clear();
    keys.first.e_b.clear();
    keys.second.e_b.clear();
    keys.first.beta_b.clear();
    keys.second.beta_b.clear();
}

void freeMICKeyPack(MICKeyPack &key)
{
    freeDCFKeyPack(key.dcfKey);
    delete[] key.z;
}

void freeMSNZBKeyPack(MSNZBKeyPack &key)
{
    freeMICKeyPack(key.micKey);
}

void freeBulkyLRSKeyPack(BulkyLRSKeyPack &key, int m)
{
    freeDCFKeyPack(key.dcfKeyN);
    delete[] key.z;
    for(int i = 0; i < m; i++) {
        freeDCFKeyPack(key.dcfKeyS[i]);
    }
    delete[] key.dcfKeyS;
}

void freeTaylorKeyPack(TaylorKeyPack &key, int m)
{
    freeMSNZBKeyPack(key.msnzbKey);
    freeBulkyLRSKeyPack(key.lrsKeys[0], m);
    freeBulkyLRSKeyPack(key.lrsKeys[1], m);
}

void freeMaxpoolDoubleKeyPack(MaxpoolDoubleKeyPack &key)
{
    freeRelu2RoundKeyPack(key.reluKey);
}

void freeMaxpoolDoubleKeyPackPair(std::pair<MaxpoolDoubleKeyPack,MaxpoolDoubleKeyPack> &keys)
{
    delete[] keys.first.reluKey.dcfKey.k;
    delete[] keys.second.reluKey.dcfKey.k;
    delete[] keys.first.reluKey.dcfKey.g;
    delete[] keys.first.reluKey.dcfKey.v;
}

void freeFixToFloatKeyPack(FixToFloatKeyPack &key)
{
    freeMICKeyPack(key.micKey);
}

void freeFixToFloatKeyPackPair(std::pair<FixToFloatKeyPack, FixToFloatKeyPack> &keys)
{
    delete[] keys.first.micKey.dcfKey.k;
    delete[] keys.second.micKey.dcfKey.k;
    delete[] keys.first.micKey.dcfKey.g;
    delete[] keys.first.micKey.dcfKey.v;
}

void freeFloatToFixKeyPack(FloatToFixKeyPack &key)
{
    freeDCFKeyPack(key.dcfKey);
}

void freeFloatToFixKeyPackPair(std::pair<FloatToFixKeyPack, FloatToFixKeyPack> &keys)
{
    delete[] keys.first.dcfKey.k;
    delete[] keys.second.dcfKey.k;
    delete[] keys.first.dcfKey.g;
    delete[] keys.first.dcfKey.v;
}

void freeReluExtendKeyPack(ReluExtendKeyPack &key)
{
    freeDCFKeyPack(key.dcfKey);
}

void freeReluExtendKeyPackPair(std::pair<ReluExtendKeyPack, ReluExtendKeyPack> &keys)
{
    delete[] keys.first.dcfKey.k;
    delete[] keys.second.dcfKey.k;
    delete[] keys.first.dcfKey.g;
    delete[] keys.first.dcfKey.v;
}

void freeSignExtend2KeyPack(SignExtend2KeyPack &key)
{
    freeDCFKeyPack(key.dcfKey);
}

void freeSignExtend2KeyPackPair(std::pair<SignExtend2KeyPack, SignExtend2KeyPack> &keys)
{
    delete[] keys.first.dcfKey.k;
    delete[] keys.second.dcfKey.k;
    delete[] keys.first.dcfKey.g;
    delete[] keys.first.dcfKey.v;
}

void freeTripleKey(TripleKeyPack &key){
    delete[] key.a;
    delete[] key.b;
    delete[] key.c;
}

void freeLUTKeyPack(LUTKeyPack &key){
    freeDPFKeyPack(key.dpfKey);
}

void freeLUTKeyPackPair(std::pair<LUTKeyPack, LUTKeyPack> &keys){
    delete[] keys.first.dpfKey.s;
    delete[] keys.second.dpfKey.s;
}

void freeClipKeyPack(ClipKeyPack &key)
{
    freeDCFKeyPack(key.cmpKey.dcfKey);
}

void freeClipKeyPackPair(std::pair<ClipKeyPack, ClipKeyPack> &keys)
{
    delete[] keys.first.cmpKey.dcfKey.k;
    delete[] keys.second.cmpKey.dcfKey.k;
    delete[] keys.first.cmpKey.dcfKey.g;
    delete[] keys.first.cmpKey.dcfKey.v;
}

void freeF2BF16KeyPack(F2BF16KeyPack &key)
{
    freeDCFKeyPack(key.dcfKey);
}

void freeF2BF16KeyPackPair(std::pair<F2BF16KeyPack, F2BF16KeyPack> &keys)
{
    delete[] keys.first.dcfKey.k;
    delete[] keys.second.dcfKey.k;
    delete[] keys.first.dcfKey.g;
    delete[] keys.first.dcfKey.v;
}

void freeTruncateReduceKeyPack(TruncateReduceKeyPack &key)
{
    freeDCFKeyPack(key.dcfKey);
}

void freeTruncateReduceKeyPackPair(std::pair<TruncateReduceKeyPack, TruncateReduceKeyPack> &keys)
{
    delete[] keys.first.dcfKey.k;
    delete[] keys.second.dcfKey.k;
    delete[] keys.first.dcfKey.g;
    delete[] keys.first.dcfKey.v;
}

void freeSlothDreluKeyPack(SlothDreluKeyPack &key){
    freeDPFKeyPack(key.dpfKey);
}

void freeSlothDreluKeyPackPair(std::pair<SlothDreluKeyPack, SlothDreluKeyPack> &keys){
    delete[] keys.first.dpfKey.s;
    delete[] keys.second.dpfKey.s;
}

void freeLUTDPFETKeyPack(LUTDPFETKeyPack &key){
    freeDPFKeyPack(key.dpfKey);
}

void freeLUTDPFETKeyPackPair(std::pair<LUTDPFETKeyPack, LUTDPFETKeyPack> &keys){
    delete[] keys.first.dpfKey.s;
    delete[] keys.second.dpfKey.s;
}

void freeWrapDPFKeyPack(WrapDPFKeyPack &key){
    freeDPFKeyPack(key.dpfKey);
}

void freeWrapDPFKeyPackPair(std::pair<WrapDPFKeyPack, WrapDPFKeyPack> &keys){
    delete[] keys.first.dpfKey.s;
    delete[] keys.second.dpfKey.s;
}
