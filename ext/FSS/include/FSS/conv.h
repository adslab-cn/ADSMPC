#include <FSS/keypack.h>


std::pair<MatMulKey, MatMulKey> KeyGenMatMul(int Bin, int Bout, int s1, int s2, int s3, GroupElement *rin1, GroupElement *rin2, GroupElement *rout);

std::pair<Conv2DKey, Conv2DKey> KeyGenConv2D(
    int Bin, int Bout,
    int N, int H, int W, int CI, int FH, int FW, int CO,
    int zPadHLeft, int zPadHRight, 
    int zPadWLeft, int zPadWRight,
    int strideH, int strideW,
    GroupElement *rin1,  GroupElement * rin2, GroupElement * rout);

void EvalConv2D(int party, const Conv2DKey &key,
    int N, int H, int W, int CI, int FH, int FW, int CO,
    int zPadHLeft, int zPadHRight, 
    int zPadWLeft, int zPadWRight,
    int strideH, int strideW, GroupElement* input, GroupElement* filter, GroupElement* output);

std::pair<Conv3DKey, Conv3DKey> KeyGenConv3D(
    int Bin, int Bout,
    int N, int D, int H, int W, int CI, int FD, int FH, int FW, int CO,
    int zPadDLeft, int zPadDRight, 
    int zPadHLeft, int zPadHRight, 
    int zPadWLeft, int zPadWRight,
    int strideD, int strideH, int strideW,
    GroupElement *rin1,  GroupElement * rin2, GroupElement * rout);

void EvalConv3D(int party, const Conv3DKey &key,
    int N, int D, int H, int W, int CI, int FD, int FH, int FW, int CO,
    int zPadDLeft, int zPadDRight, 
    int zPadHLeft, int zPadHRight, 
    int zPadWLeft, int zPadWRight,
    int strideD, int strideH, int strideW, GroupElement* input, GroupElement* filter, GroupElement* output);

std::pair<TripleKeyPack, TripleKeyPack> KeyGenConvTranspose3D(
    int bw,
    int64_t N, 
    int64_t D, 
    int64_t H, 
    int64_t W, 
    int64_t CI, 
    int64_t FD, 
    int64_t FH, 
    int64_t FW, 
    int64_t CO, 
    int64_t zPadDLeft, 
    int64_t zPadDRight, 
    int64_t zPadHLeft, 
    int64_t zPadHRight, 
    int64_t zPadWLeft, 
    int64_t zPadWRight, 
    int64_t strideD, 
    int64_t strideH, 
    int64_t strideW, 
    int64_t outD, 
    int64_t outH, 
    int64_t outW, 
    GroupElement* inputArr, 
    GroupElement* filterArr, 
    GroupElement* outArr);

void EvalConvTranspose3D(int party, const TripleKeyPack &key,
    int64_t N, 
    int64_t D, 
    int64_t H, 
    int64_t W, 
    int64_t CI, 
    int64_t FD, 
    int64_t FH, 
    int64_t FW, 
    int64_t CO, 
    int64_t zPadDLeft, 
    int64_t zPadDRight, 
    int64_t zPadHLeft, 
    int64_t zPadHRight, 
    int64_t zPadWLeft, 
    int64_t zPadWRight, 
    int64_t strideD, 
    int64_t strideH, 
    int64_t strideW, 
    int64_t outD, 
    int64_t outH, 
    int64_t outW, 
    GroupElement* inputArr, 
    GroupElement* filterArr, 
    GroupElement* outArr);

