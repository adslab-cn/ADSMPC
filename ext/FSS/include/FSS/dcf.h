#pragma once
#include <array>
#include <vector>
#include <utility>
#include <cryptoTools/Common/Defines.h>
#include <cryptoTools/Crypto/AES.h>
#include <cryptoTools/Crypto/PRNG.h>
#include <cryptoTools/gsl/span>
#include <FSS/keypack.h>

void clearAESevals();
inline osuCrypto::u8 lsb(const osuCrypto::block &b)
{
    return _mm_cvtsi128_si64x(b) & 1;
}

std::pair<DCFKeyPack, DCFKeyPack> keyGenDCF(int Bin, int Bout, int groupSize,
                GroupElement idx, GroupElement* payload);

std::pair<DCFKeyPack, DCFKeyPack> keyGenDCF(int Bin, int Bout,
                GroupElement idx, GroupElement payload);

void evalDCF(int party, GroupElement *res, GroupElement idx, const DCFKeyPack &key);
void evalDCF(int Bin, int Bout, int groupSize, 
                GroupElement *out, // groupSize
                int party, GroupElement idx, 
                osuCrypto::block *k, // bin + 1
                GroupElement *g , // groupSize
                GroupElement *v, // bin * groupSize
                bool geq = false, int evalGroupIdxStart = 0,
                int evalGroupIdxLen = -1);

void evalDCFPartial(int party, GroupElement *res, GroupElement idx, const DCFKeyPack &key, int start, int len);

std::pair<DualDCFKeyPack, DualDCFKeyPack> keyGenDualDCF(int Bin, int Bout, int groupSize, GroupElement idx, GroupElement *payload1, GroupElement *payload2);

std::pair<DualDCFKeyPack, DualDCFKeyPack> keyGenDualDCF(int Bin, int Bout, GroupElement idx, GroupElement payload1, GroupElement payload2);

void evalDualDCF(int party, GroupElement* res, GroupElement idx, const DualDCFKeyPack &key);

// ============================================================
// IFSS_DCF implemented by two ordinary DCF keys
// ============================================================

std::pair<IFSS_DCF_TwoDCFKeyPack, IFSS_DCF_TwoDCFKeyPack>
keyGenIFSS_DCF_TwoDCF(int bin,
                      int bout,
                      GroupElement alpha,
                      GroupElement beta,
                      GroupElement deltaA);

IFSSAuthShare evalIFSS_DCF_TwoDCF(int party,
                                  IFSS_DCF_TwoDCFKeyPack &key,
                                  GroupElement x);

// ============================================================
// IFSS_DCF implemented by one vector-payload DCF key
// groupSize = 2
// ============================================================

std::pair<IFSS_DCFKeyPack, IFSS_DCFKeyPack>
keyGenIFSS_DCF(int bin,
               int bout,
               GroupElement alpha,
               GroupElement beta,
               GroupElement deltaA);

IFSSAuthShare evalIFSS_DCF(int party,
                           IFSS_DCFKeyPack &key,
                           GroupElement x);



// ============================================================
// DIF from DCF
// ============================================================

std::pair<DIFKeyPack, DIFKeyPack>
keyGenDIF(int bin,
          int bout,
          GroupElement a,
          GroupElement b,
          GroupElement beta);

GroupElement evalDIF(int party,
                     DIFKeyPack &key,
                     GroupElement x);

