#pragma once

#include <cstdint>
#include <cryptoTools/Common/Defines.h>
#include <crypto/primitives/ASS/group_element.h>


// FSS primitive keys and authenticated-share support types.

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
