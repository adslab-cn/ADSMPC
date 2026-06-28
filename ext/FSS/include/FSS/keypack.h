#pragma once

#include <cryptoTools/Common/Defines.h>
#include <FSS/group_element.h>

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
    int Bin, Bout;
    osuCrypto::block *k;
    GroupElement *g, *v;
    GroupElement e_b0, e_b1;		 // size: degree+1 (same as beta)
    GroupElement beta_b0, beta_b1;	 // size: degree+1 (shares of beta, which is set of poly coeffs) (beta: highest to lowest power left to right)
    GroupElement r_b;
    GroupElement drelu;
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
    union {
        struct {
            uint64_t tLcw;
            uint64_t tRcw;
        };
        uint64_t tcw[2];
    };
    GroupElement payload;

    DPFKeyPack(int bin, int bout) : bin(bin), bout(bout)
    {
        s = new osuCrypto::block[bin+1];
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

// =========================================================================
// GTDCF KeyPack
// =========================================================================
struct GTDCFKeyPack {
    int bin, w, d, groupSize;
    osuCrypto::block seed;       
    osuCrypto::block *scw;       
    uint8_t *tcw;                
    GroupElement *vcw;           
    GroupElement *leaf_vcw;      
    GroupElement rout_share; // 输出的掩码份额

    GTDCFKeyPack(int bin, int w, int groupSize = 2) 
        : bin(bin), w(w), groupSize(groupSize) {
        d = bin - w;
        int B = 1 << w; 
        scw = new osuCrypto::block[d];
        tcw = new uint8_t[2 * d];
        vcw = new GroupElement[d * groupSize];
        leaf_vcw = new GroupElement[B * groupSize];
        rout_share = 0;
    }

    GTDCFKeyPack() {
        bin = w = d = groupSize = 0;
        scw = nullptr; tcw = nullptr; vcw = nullptr; leaf_vcw = nullptr;
        rout_share = 0;
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


// ==== 请添加到 keypack.h 的末尾 ====
class QuadDPFKeyPack
{
public:
    int bin;        // 输入位宽
    int bout;       // 输出位宽
    int depth;      // 树深 ceil(bin / 2)
    osuCrypto::block s0_initial; // 初始种子
    osuCrypto::block *scw;       // Correction Words for seeds (size: depth * 4)
    uint8_t *tcw;                // Correction Words for bits (size: depth) 每一位打包了4个分支的t_cw
    GroupElement payload;        // 叶子节点的 Payload

    QuadDPFKeyPack(int bin, int bout) : bin(bin), bout(bout)
    {
        depth = (bin + 1) / 2;
        scw = new osuCrypto::block[depth * 4];
        tcw = new uint8_t[depth];
    }

    QuadDPFKeyPack()
    {
        scw = nullptr;
        tcw = nullptr;
        bin = bout = depth = 0;
    }
};

class OctDPFKeyPack
{
public:
    int bin;        // 输入位宽
    int bout;       // 输出位宽
    int depth;      // 树深 ceil(bin / 3)
    osuCrypto::block s0_initial; // 初始种子
    osuCrypto::block *scw;       // Correction Words for seeds (size: depth * 8)
    uint8_t *tcw;                // Correction Words for bits (size: depth) 每个字节存8位
    GroupElement payload;        // 叶子节点的 Payload

    OctDPFKeyPack(int bin, int bout) : bin(bin), bout(bout)
    {
        depth = (bin + 2) / 3;
        scw = new osuCrypto::block[depth * 8];
        tcw = new uint8_t[depth];
    }

    OctDPFKeyPack()
    {
        scw = nullptr;
        tcw = nullptr;
        bin = bout = depth = 0;
    }
};

// 可验证DPF——VDPF
// ==========================================
// Verifiable DPF KeyPack
// Based on Castro-Polychroniadou VerDPF.
// Compatible with the existing DPF style:
//   s[0]       : initial seed
//   s[i+1]     : seed correction word scw_i
//   tcw[0/1]   : left/right control-bit correction words
//   cs[4]      : final 4λ-bit correction seed
//   ocw        : output correction word
// ==========================================
struct VerDPFKeyPack
{
    int bin, bout;

    osuCrypto::block *s; // size: bin + 1

    union {
        struct {
            uint64_t tLcw;
            uint64_t tRcw;
        };
        uint64_t tcw[2];
    };

    osuCrypto::block cs[4]; // 4λ-bit final correction seed
    GroupElement ocw;

    VerDPFKeyPack(int bin, int bout) : bin(bin), bout(bout)
    {
        s = new osuCrypto::block[bin + 1];
        tcw[0] = 0;
        tcw[1] = 0;
        ocw = 0;
        for (int i = 0; i < 4; ++i) {
            cs[i] = osuCrypto::ZeroBlock;
        }
    }

    VerDPFKeyPack()
    {
        bin = bout = 0;
        s = nullptr;
        tcw[0] = 0;
        tcw[1] = 0;
        ocw = 0;
        for (int i = 0; i < 4; ++i) {
            cs[i] = osuCrypto::ZeroBlock;
        }
    }
};


// ============================================================
// IFSS authenticated share and MAC key
// ============================================================

struct IFSSAuthShare
{
    GroupElement value;  // value share
    GroupElement tag;    // MAC/tag share
};

struct IFSSGlobalMACKey
{
    GroupElement deltaA0;
    GroupElement deltaA1;
    GroupElement deltaA;
};

// ============================================================
// IFSS_DPF
//
// Since current DPF supports only single payload,
// IFSS_DPF is implemented by two ordinary DPF keys:
//
//   valKey : DPF(alpha, beta)
//   macKey : DPF(alpha, deltaA * beta)
// ============================================================

struct IFSS_DPFKeyPack
{
    int bin, bout;

    DPFKeyPack valKey;
    DPFKeyPack macKey;

    IFSS_DPFKeyPack()
    {
        bin = bout = 0;
    }
};

// ============================================================
// IFSS_DCF_TwoDCF
//
// Two-DCF implementation:
//
//   valKey : DCF(alpha, beta)
//   macKey : DCF(alpha, deltaA * beta)
// ============================================================

struct IFSS_DCF_TwoDCFKeyPack
{
    int bin, bout;

    DCFKeyPack valKey;
    DCFKeyPack macKey;

    IFSS_DCF_TwoDCFKeyPack()
    {
        bin = bout = 0;
    }
};

// ============================================================
// IFSS_DCF
//
// Vector-payload implementation:
//
//   payload[0] = beta
//   payload[1] = deltaA * beta
//
// One DCF key with groupSize = 2.
// ============================================================

struct IFSS_DCFKeyPack
{
    int bin, bout;

    DCFKeyPack dcfKey; // groupSize = 2

    IFSS_DCFKeyPack()
    {
        bin = bout = 0;
    }
};



// ============================================================
// DIFKeyPack
//
// Distributed Interval Function:
//   DIF([a,b], beta)(x) = beta if a <= x <= b, else 0.
//
// Implementation using existing DCF:
//   beta * {a <= x <= b}
// = beta * {x < b + 1} - beta * {x < a}
//
// For b = 2^bin - 1, the first term is a constant beta.
// For a = 0, the second term is omitted.
// ============================================================

struct DIFKeyPack
{
    int bin, bout;

    bool hasUpperDcf;   // beta * {x < b+1}
    bool hasLowerDcf;   // -beta * {x < a}

    GroupElement constShare; // used when b is max: share of beta

    DCFKeyPack upperKey;
    DCFKeyPack lowerKey;

    DIFKeyPack()
    {
        bin = 0;
        bout = 0;
        hasUpperDcf = false;
        hasLowerDcf = false;
        constShare = 0;
    }
};

