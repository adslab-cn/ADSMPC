/* MatMul(矩阵乘法)和Conv2D层密钥生成和评估接口 */
#include <aux_parameter/keypack.h>


std::pair<MatMulKey, MatMulKey> KeyGenMatMul(int Bin, int Bout, int s1, int s2, int s3, GroupElement *rin1, GroupElement *rin2, GroupElement *rout);

// Bin, Bout：输入和输出分割的基数（可能是秘密分享的份额数量）。
// N, H, W, CI, FH, FW, CO：分别代表批次大小、输入高度、输入宽度、输入通道数、滤波器高度、滤波器宽度和输出通道数。
// zPadHLeft, zPadHRight, zPadWLeft, zPadWRight：在高度和宽度上的左右填充量。
// strideH, strideW：在高度和宽度上的步长。
// rin1, rin2, rout：指向输入数据、卷积核数据和输出数据的指针。
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