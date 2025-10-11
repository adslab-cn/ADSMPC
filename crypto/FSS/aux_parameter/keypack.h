/*
    密钥管理
*/
#pragma once

#include <cryptoTools/Common/Defines.h>
#include "group_element.h"

struct DCFKeyPack{
    int Bin, Bout, groupSize;
    osuCrypto::block *k;   // size Bin+1
    GroupElement *g;    // bitsize Bout, size groupSize
    GroupElement *v;   // bitsize Bout, size Bin x groupSize
    DCFKeyPack(int Bin, int Bout, int groupSize,
                osuCrypto::block *k,
                GroupElement *g,
                GroupElement *v) : Bin(Bin), Bout(Bout), groupSize(groupSize), k(k), g(g), v(v){}
    DCFKeyPack() {
        Bin = Bout = groupSize = 0;
        k = nullptr;
        g = nullptr;
        v = nullptr;
    }
};

struct DualDCFKeyPack{  
    int Bin, Bout, groupSize;
    DCFKeyPack dcfKey;
    GroupElement *sb;   // size: groupSize
    DualDCFKeyPack() {}
};

struct AddKey{
    int Bin, Bout;
    GroupElement rb;
};

struct MultKey{
    int Bin, Bout;
    GroupElement a, b, c;
};

struct MatMulKey{
    int Bin, Bout;
    int s1, s2, s3;
    GroupElement *a, *b, *c;    
};

struct MultKeyNew {
    GroupElement a, b, c;
    DCFKeyPack k1, k2, k3, k4;
};

struct Conv2DKey{
    int Bin, Bout;
    int N, H, W, CI, FH, FW, CO,
        zPadHLeft, zPadHRight, 
        zPadWLeft, zPadWRight,
        strideH, strideW;
    GroupElement *a, *b, *c;    
};

struct Conv3DKey{
    int Bin, Bout;
    int N, D, H, W, CI, FD, FH, FW, CO,
        zPadDLeft, zPadDRight, 
        zPadHLeft, zPadHRight, 
        zPadWLeft, zPadWRight,
        strideD, strideH, strideW;
    GroupElement *a, *b, *c;    
};

struct TripleKeyPack {
    int bw;
    int64_t na, nb, nc;
    GroupElement *a, *b, *c;
};

struct ScmpKeyPack
{
    int Bin, Bout;
    DualDCFKeyPack dualDcfKey;
    GroupElement rb;
};

struct PublicICKeyPack
{
    int Bin, Bout;
    DCFKeyPack dcfKey;
    GroupElement zb;
};

struct PublicDivKeyPack
{
    int Bin, Bout;
    DualDCFKeyPack dualDcfKey;
    ScmpKeyPack scmpKey;
    GroupElement zb;
};

struct SignedPublicDivKeyPack
{
    int Bin, Bout;
    GroupElement d;     // divisor
    DCFKeyPack dcfKey;
    PublicICKeyPack publicICkey;
    ScmpKeyPack scmpKey;
    GroupElement A_share, corr_share, B_share, rdiv_share;
    GroupElement rout_temp_share, rout_share;
};

struct ReluKeyPack
{
    int Bin, Bout; // 输入/输出位宽
    osuCrypto::block *k; // DCF 密钥块
    GroupElement *g, *v; // DCF 辅助数据
    GroupElement e_b0, e_b1;		 // size: degree+1 (same as beta) // 线性校正项份额
    GroupElement beta_b0, beta_b1;	 // size: degree+1 (shares of beta, which is set of poly coeffs) (beta: highest to lowest power left to right)
    GroupElement r_b; // 输出掩码份额
    GroupElement drelu; // ReLU导数掩码
};

struct MaxpoolKeyPack
{
    int Bin, Bout;
    ReluKeyPack reluKey;
    GroupElement rb;
};

struct ARSKeyPack
{
    // arithmetic right shift
    int Bin, Bout, shift;
    DCFKeyPack dcfKey;
    DualDCFKeyPack dualDcfKey;      // groupSize = 2 for payload
    GroupElement rb;
    ARSKeyPack() {}
};

struct ReluTruncateKeyPack {
    int Bin, Bout, shift;
    DCFKeyPack dcfKeyN;
    DCFKeyPack dcfKeyS;
    GroupElement zTruncate;
    GroupElement a, b, c, d1, d2;
};

struct Relu2RoundKeyPack {
    int effectiveBin, Bin;
    DCFKeyPack dcfKey;
    GroupElement a, b, c, d1, d2;
};

/*
struct SplineOneKeyPack
{
    int Bin, Bout;
    int degree; // degree of poly in payload beta
    DCFKeyPack dcfKey;
    std::vector<GroupElement> e_b;		 // size: degree+1 (same as beta)
    std::vector<GroupElement> beta_b;	 // size: degree+1 (shares of beta, which is set of poly coeffs) (beta: highest to lowest power left to right)
    GroupElement r_b;
};
*/
struct SplineKeyPack
{
    int Bin, Bout;
    int numPoly, degree;
    DCFKeyPack dcfKey;
    std::vector<GroupElement> p;        // spline breakpoints, size: numPoly + 1; p[0] = 0 and p[numPoly] = N-1
    std::vector<std::vector<GroupElement>> e_b; // 2d array dim: numPoly x (degree+1) (size is same as beta)
    std::vector<GroupElement> beta_b;           // 1d array size: numPoly * (degree+1) (shares of beta, which is set of poly coeffs) (beta: highest to lowest power left to right)
    GroupElement r_b;
};

struct PrivateScaleKeyPack
{
    GroupElement rin;
    GroupElement rout;
};

struct SquareKey {
    GroupElement b;
    GroupElement c;
};

struct TaylorSqKey {
    GroupElement a;
    GroupElement b;
};

struct MICKeyPack {
    DCFKeyPack dcfKey;
    GroupElement *z;
};

struct MSNZBKeyPack {
    MICKeyPack micKey;
    GroupElement r;
};

struct BulkyLRSKeyPack
{
    DCFKeyPack dcfKeyN;
    DCFKeyPack *dcfKeyS;
    GroupElement *z;
    GroupElement out;
};

struct TaylorKeyPack {
    MSNZBKeyPack msnzbKey;
    TaylorSqKey squareKey;
    BulkyLRSKeyPack lrsKeys[2];
    PrivateScaleKeyPack privateScaleKey;
};

struct SelectKeyPack {
    int Bin;
    GroupElement a, b, c, d1, d2;
};

struct MaxpoolDoubleKeyPack
{
    int Bin, Bout;
    Relu2RoundKeyPack reluKey;
    GroupElement rb;
};

struct BitwiseAndKeyPack
{
    GroupElement t[4];
};

struct FixToFloatKeyPack
{
    MICKeyPack micKey;
    GroupElement rs, rpow, ry, rm;
    SelectKeyPack selectKey;
};

struct FloatToFixKeyPack
{
    GroupElement rm, re, rw, /*rt,*/ rh;
    DCFKeyPack dcfKey;
    SelectKeyPack selectKey;
    ARSKeyPack arsKey;
    GroupElement p[1024];
    GroupElement q[1024];
};

struct ReluExtendKeyPack
{
    DCFKeyPack dcfKey;
    GroupElement rd, rw;
    GroupElement p[4];
    GroupElement q[2];
};

struct SignExtend2KeyPack
{
    DCFKeyPack dcfKey;
    GroupElement rw;
    GroupElement p[2];
};

struct EdabitsPrTruncKeyPack
{
    GroupElement a, b;
};



class DPFKeyPack
{
public:
    int bin, bout;
    osuCrypto::block *s;
    // 通过 union 与 tcw[2] 共享内存，tcw[0] 对应 tLcw，tcw[1] 对应 tRcw
    union {
        struct {
            uint64_t tLcw;
            uint64_t tRcw;
        };
        uint64_t tcw[2]; // 各为 64 位整数
    };
    GroupElement payload; // 当输入 x 等于目标点 idx 时，双方密钥评估结果的​​异或值​​为 payload；否则为 0

    DPFKeyPack(int bin, int bout) : bin(bin), bout(bout)
    {
        s = new osuCrypto::block[bin+1]; // 为每层种子预留空间
        tLcw = 0;
        tRcw = 0;
    }

    DPFKeyPack()
    {
        s = nullptr;
        tLcw = 0;
        tRcw = 0;
    }
};

class DPFETKeyPack
{
public:
    int bin;
    osuCrypto::block *s;
    union {
        struct {
            uint64_t tLcw;
            uint64_t tRcw;
        };
        uint64_t tcw[2];
    };
    osuCrypto::block leaf;

    DPFETKeyPack(int bin) : bin(bin)
    {
        s = new osuCrypto::block[bin+1-7];
        tLcw = 0;
        tRcw = 0;
    }

    DPFETKeyPack()
    {
        s = nullptr;
        tLcw = 0;
        tRcw = 0;
    }
};

struct PubCmpKeyPack {
    int bin;
    DCFKeyPack dcfKey;
    GroupElement rout;
};

struct ClipKeyPack {
    int bin;
    PubCmpKeyPack cmpKey;
    GroupElement a, b, c, d1, d2;
};

struct LUTKeyPack {
    int bin, bout;
    DPFKeyPack dpfKey;
    GroupElement rout;
};

struct F2BF16KeyPack {
    int bin;
    DCFKeyPack dcfKey, dcfTruncate;
    GroupElement rout_k, rout_m, rin, prod, rout, rProd;
};

struct TruncateReduceKeyPack {
    int bin, shift;
    DCFKeyPack dcfKey;
    GroupElement rout;
};

struct LUTSSKeyPack {
    int bin, bout;
    GroupElement b0, b1, b2, b3;
    GroupElement routRes, routCorr;
    GroupElement rout;
};

struct LUTDPFETKeyPack {
    int bin, bout;
    DPFETKeyPack dpfKey;
    GroupElement routRes, routCorr;
};

struct SlothDreluKeyPack {
    int bin;
    DPFETKeyPack dpfKey;
    GroupElement r;
};

struct WrapSSKeyPack {
    int bin;
    uint64_t b0, b1;
};

struct WrapDPFKeyPack {
    int bin;
    DPFETKeyPack dpfKey;
    GroupElement r;
};

struct SlothLRSKeyPack {
    int bin, shift;
    GroupElement msb;
    GroupElement rout;
    GroupElement select;
};

struct SlothTRKeyPack {
    int bin, shift;
    GroupElement rout;
    GroupElement select;
};

struct SlothSignExtendKeyPack {
    int bin, bout;
    GroupElement rout;
    GroupElement select;
};



// ==== BigStateDMPF ============================================================
// class BigStateDMPFKeyPack
// {
// public:
//     int bin, bout, t; // 输入长度，输出长度，非零点数量
//     osuCrypto::block *seed; // 种子状态 (每个节点)
//     std::vector<std::vector<u8>> signs; //每层长度为t的向量，用于保存多个控制位
//     std::vector<std::vector<uint64_t>> CW; //每层校正字

//     // uint64_t CW;
//     // uint64_t convCW;
//     GroupElement payloads;

//     // union 使struct和tcw[4]共享内存
//     // union {
//     //     struct {
//     //         uint64_t tLcw; // (BigStateDMPF 新增):[层数][方向][t维符号]
//     //         uint64_t tRcw;
//     //         uint64_t tLconvCW; // 转换层校正字 (DMPF 新增)
//     //         uint64_t tRconvCW; // 转换层校正字 (DMPF 新增)
//     //         uint64_t signCW;
//     //     };
//     //     uint64_t tcw[5];
//     // };

//     // 构造函数
//     BigStateDMPFKeyPack(int bin, int bout, int t) : bin(bin), bout(bout), t(t)
//     {
//         s = new osuCrypto::block[bin+1];
//         std::vector<u8> sign(t, 0); 

//         tLcw = 0;
//         tRcw = 0;
//         tLconvCW = 0;
//         tRconvCW = 0;
//         signCW = 0;
//     }

//     BigStateDMPFKeyPack()
//     {
//         s = nullptr;
//         tLcw = 0;
//         tRcw = 0;
//         tLconvCW = 0;
//         tRconvCW = 0;
//         signCW = 0;
//     }
// };
struct DpfRouteKeyPack {
    int size;
    int data_bin;
    int rank_bin;
    DPFKeyPack *routing_keys;
    GroupElement *r_shares;
    GroupElement *s_shares;

    // 构造函数
    DpfRouteKeyPack(int _size, int _data_bin, int _rank_bin) {
        size = _size;
        data_bin = _data_bin;
        rank_bin = _rank_bin;
        routing_keys = new DPFKeyPack[size];
        r_shares = new GroupElement[size];
        s_shares = new GroupElement[size];
    }
        DpfRouteKeyPack() : size(0), data_bin(0), rank_bin(0), 
                    routing_keys(nullptr), r_shares(nullptr), s_shares(nullptr) {}
};

struct ElemWiseMulKeyPack {
    int32_t size;
    GroupElement *a, *b, *c;
};

struct GraphUpdateKeyPack {
    int n, c;
    int A_bw, F_bw, A_data_bw, F_data_bw;

    std::vector<DPFKeyPack> keys_A; 
    std::vector<DPFKeyPack> keys_F; 

    GraphUpdateKeyPack() : n(0), c(0) {} 

    GraphUpdateKeyPack(int _n, int _c, int _A_bw, int _F_bw, int _A_data_bw, int _F_data_bw)
        : n(_n), c(_c), A_bw(_A_bw), F_bw(_F_bw), A_data_bw(_A_data_bw), F_data_bw(_F_data_bw)
    {
        keys_A.resize(n);
        keys_F.resize(c);
    }

};