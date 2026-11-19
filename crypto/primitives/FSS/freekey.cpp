#include <crypto/primitives/FSS/freekey.h>
#include <crypto/commom/comms.h>
#include <crypto/commom/config.h>

void freeDCFKeyPack(DCFKeyPack &key){
    if (!Config::dealer->keyBuf->isMem()) {
        delete[] key.k;
    }
    delete[] key.g;
    delete[] key.v;
}

void freeDCFKeyPackPair(std::pair<DCFKeyPack, DCFKeyPack> &keys){
    delete[] keys.first.k;
    delete[] keys.second.k;
    delete[] keys.first.g;
    delete[] keys.first.v;
}

void freeDualDCFKeyPack(DualDCFKeyPack &key){
    freeDCFKeyPack(key.dcfKey);
    delete[] key.sb;
}

void freeDualDCFKeyPackPair(std::pair<DualDCFKeyPack, DualDCFKeyPack> &keys){
    delete[] keys.first.dcfKey.k;
    delete[] keys.second.dcfKey.k;
    delete[] keys.first.dcfKey.g;
    delete[] keys.first.dcfKey.v;
    delete[] keys.first.sb;
    delete[] keys.second.sb;
}

void freeDPFKeyPack(DPFKeyPack &key){
    if (!Config::dealer->keyBuf->isMem()) {
        delete key.s;
    }
}

void freeDPFKeyPackPair(std::pair<DPFKeyPack, DPFKeyPack> &keys){
    delete[] keys.first.s;
    delete[] keys.second.s;
}

void freeDPFKeyPack(DPFETKeyPack &key){
    if (!Config::dealer->keyBuf->isMem()) {
        delete key.s;
    }
}

void freeDPFKeyPackPair(std::pair<DPFETKeyPack, DPFETKeyPack> &keys){
    delete[] keys.first.s;
    delete[] keys.second.s;
}

// freeGTDCFKeyPack==============================================================
void freeGTDCFKeyPack(GTDCFKeyPack &key) {
    if (key.scw) { delete[] key.scw; key.scw = nullptr; }
    if (key.tcw) { delete[] key.tcw; key.tcw = nullptr; }
    if (key.vcw) { delete[] key.vcw; key.vcw = nullptr; }
    if (key.leaf_vcw) { delete[] key.leaf_vcw; key.leaf_vcw = nullptr; }
}

void freeGTDCFKeyPackPair(std::pair<GTDCFKeyPack, GTDCFKeyPack> &keys) {
    freeGTDCFKeyPack(keys.first);
    freeGTDCFKeyPack(keys.second);
}

void freeQuadDPFKeyPack(QuadDPFKeyPack &key){
    if (!Config::dealer->keyBuf->isMem()) {
        delete[] key.scw;
        delete[] key.tcw;
    }
}

void freeQuadDPFKeyPackPair(std::pair<QuadDPFKeyPack, QuadDPFKeyPack> &keys){
    delete[] keys.first.scw;
    delete[] keys.second.scw;
    delete[] keys.first.tcw;
    delete[] keys.second.tcw;
}

void freeOctDPFKeyPack(OctDPFKeyPack &key){
    if (!Config::dealer->keyBuf->isMem()) {
        delete[] key.scw;
        delete[] key.tcw;
    }
}

void freeOctDPFKeyPackPair(std::pair<OctDPFKeyPack, OctDPFKeyPack> &keys){
    delete[] keys.first.scw;
    delete[] keys.second.scw;
    delete[] keys.first.tcw;
    delete[] keys.second.tcw;
}

// 可验证DPF——VDPF
void freeVerDPFKeyPack(VerDPFKeyPack &key)
{
    delete[] key.s;
    key.s = nullptr;
}

void freeVerDPFKeyPackPair(std::pair<VerDPFKeyPack, VerDPFKeyPack> &keys)
{
    delete[] keys.first.s;
    delete[] keys.second.s;
    keys.first.s = nullptr;
    keys.second.s = nullptr;
}

// ============================================================
// Free IFSS_DPF keys
// ============================================================
void freeIFSS_DPFKeyPackPair(std::pair<IFSS_DPFKeyPack, IFSS_DPFKeyPack> &keys)
{
    delete[] keys.first.valKey.s;
    delete[] keys.second.valKey.s;

    delete[] keys.first.macKey.s;
    delete[] keys.second.macKey.s;
}

// ============================================================
// Free IFSS_DCF_TwoDCF keys
// ============================================================
void freeIFSS_DCF_TwoDCFKeyPackPair(std::pair<IFSS_DCF_TwoDCFKeyPack, IFSS_DCF_TwoDCFKeyPack> &keys)
{
    // value DCF pair
    delete[] keys.first.valKey.k;
    delete[] keys.second.valKey.k;
    delete[] keys.first.valKey.g;
    delete[] keys.first.valKey.v;

    // MAC DCF pair
    delete[] keys.first.macKey.k;
    delete[] keys.second.macKey.k;
    delete[] keys.first.macKey.g;
    delete[] keys.first.macKey.v;
}

// ============================================================
// Free IFSS_DCF vector-payload keys
// ============================================================
void freeIFSS_DCFKeyPackPair(std::pair<IFSS_DCFKeyPack, IFSS_DCFKeyPack> &keys)
{
    delete[] keys.first.dcfKey.k;
    delete[] keys.second.dcfKey.k;
    delete[] keys.first.dcfKey.g;
    delete[] keys.first.dcfKey.v;
}

// ============================================================
// Free DIF key pair
// ============================================================
void freeDIFKeyPackPair(std::pair<DIFKeyPack, DIFKeyPack> &keys)
{
    if (keys.first.hasUpperDcf)
    {
        delete[] keys.first.upperKey.k;
        delete[] keys.second.upperKey.k;

        delete[] keys.first.upperKey.g;
        delete[] keys.first.upperKey.v;

        keys.first.upperKey.k = nullptr;
        keys.second.upperKey.k = nullptr;
        keys.first.upperKey.g = nullptr;
        keys.first.upperKey.v = nullptr;
    }

    if (keys.first.hasLowerDcf)
    {
        delete[] keys.first.lowerKey.k;
        delete[] keys.second.lowerKey.k;

        delete[] keys.first.lowerKey.g;
        delete[] keys.first.lowerKey.v;

        keys.first.lowerKey.k = nullptr;
        keys.second.lowerKey.k = nullptr;
        keys.first.lowerKey.g = nullptr;
        keys.first.lowerKey.v = nullptr;
    }
}
