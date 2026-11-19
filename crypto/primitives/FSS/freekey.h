#pragma once

#include <utility>
#include <crypto/primitives/FSS/keypack.h>

// Definitions are provided by the FSS library, not by this header.

void freeDCFKeyPack(DCFKeyPack &key);
void freeDCFKeyPackPair(std::pair<DCFKeyPack, DCFKeyPack> &keys);
void freeDualDCFKeyPack(DualDCFKeyPack &key);
void freeDualDCFKeyPackPair(std::pair<DualDCFKeyPack, DualDCFKeyPack> &keys);
void freeDPFKeyPack(DPFKeyPack &key);
void freeDPFKeyPackPair(std::pair<DPFKeyPack, DPFKeyPack> &keys);
void freeDPFKeyPack(DPFETKeyPack &key);
void freeDPFKeyPackPair(std::pair<DPFETKeyPack, DPFETKeyPack> &keys);
void freeGTDCFKeyPack(GTDCFKeyPack &key);
void freeGTDCFKeyPackPair(std::pair<GTDCFKeyPack, GTDCFKeyPack> &keys);
void freeQuadDPFKeyPack(QuadDPFKeyPack &key);
void freeQuadDPFKeyPackPair(std::pair<QuadDPFKeyPack, QuadDPFKeyPack> &keys);
void freeOctDPFKeyPack(OctDPFKeyPack &key);
void freeOctDPFKeyPackPair(std::pair<OctDPFKeyPack, OctDPFKeyPack> &keys);
void freeVerDPFKeyPack(VerDPFKeyPack &key);
void freeVerDPFKeyPackPair(std::pair<VerDPFKeyPack, VerDPFKeyPack> &keys);
void freeIFSS_DPFKeyPackPair(std::pair<IFSS_DPFKeyPack, IFSS_DPFKeyPack> &keys);
void freeIFSS_DCF_TwoDCFKeyPackPair(std::pair<IFSS_DCF_TwoDCFKeyPack, IFSS_DCF_TwoDCFKeyPack> &keys);
void freeIFSS_DCFKeyPackPair(std::pair<IFSS_DCFKeyPack, IFSS_DCFKeyPack> &keys);
void freeDIFKeyPackPair(std::pair<DIFKeyPack, DIFKeyPack> &keys);
