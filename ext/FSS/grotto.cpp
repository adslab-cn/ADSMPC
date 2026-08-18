#include <FSS/grotto.h>

#include <FSS/assert.h>
#include <FSS/dpf.h>
#include <FSS/utils.h>
#include "mult.h"

namespace {

void assignShare(GroupElement value, int bin,
                 GroupElement &serverShare, GroupElement &clientShare)
{
    auto shares = splitShare(value, bin);
    serverShare = shares.first;
    clientShare = shares.second;
}

} // namespace

std::pair<GrottoReLUKeyPack, GrottoReLUKeyPack> keyGenGrottoReLU(
    int bin, GroupElement inputMask, GroupElement outputMask)
{
    always_assert(bin >= 8 && bin <= 64);

    GroupElement point = random_ge(bin);
    auto dpfKeys = keyGenDPFET(bin, point);

    GrottoReLUKeyPack serverKey{};
    GrottoReLUKeyPack clientKey{};
    serverKey.dpfKey = dpfKeys.first;
    clientKey.dpfKey = dpfKeys.second;

    GroupElement shiftedMask = inputMask + point;
    mod(shiftedMask, bin);
    assignShare(point, bin, serverKey.iShare, clientKey.iShare);
    assignShare(shiftedMask, bin, serverKey.shiftedMaskShare,
                clientKey.shiftedMaskShare);

    auto productKeys = TernaryMultGen(bin);
    serverKey.productKey = productKeys.first;
    clientKey.productKey = productKeys.second;
    assignShare(outputMask, bin, serverKey.routShare, clientKey.routShare);

    return {serverKey, clientKey};
}

GrottoReLULocalShares evalGrottoReLULocalShares(
    int evaluatorParty, const GrottoReLUKeyPack &key,
    GroupElement maskedInput, GroupElement openedDelta)
{
    always_assert(evaluatorParty == 0 || evaluatorParty == 1);

    const int bin = key.dpfKey.bin;
    const GroupElement nonnegativeEnd = GroupElement{1} << (bin - 1);
    GroupElement segmentStart = GroupElement{0} - openedDelta;
    GroupElement segmentEnd = nonnegativeEnd - openedDelta;
    mod(segmentStart, bin);
    mod(segmentEnd, bin);

    GroupElement positiveBit = evalDPFETSegmentParity(
        evaluatorParty, key.dpfKey, segmentStart, segmentEnd);
    GroupElement negativeBit = positiveBit ^ (evaluatorParty == 0 ? 1 : 0);

    // Grotto Eq. (3): lift an XOR share as +bit for P0 and -bit for P1.
    GroupElement slopeShare = evaluatorParty == 0
        ? positiveBit
        : GroupElement{0} - positiveBit;
    GroupElement correctionSignShare = evaluatorParty == 0
        ? positiveBit + negativeBit
        : GroupElement{0} - positiveBit - negativeBit;

    // Convert the repository's public masked value x+r into an additive
    // share of x, using rShare = shiftedMaskShare - iShare.
    GroupElement inputShare = evaluatorParty * maskedInput
                            - key.shiftedMaskShare + key.iShare;
    mod(slopeShare, bin);
    mod(correctionSignShare, bin);
    mod(inputShare, bin);
    return {correctionSignShare, slopeShare, inputShare};
}

void freeGrottoReLUKeyPair(
    std::pair<GrottoReLUKeyPack, GrottoReLUKeyPack> &keys)
{
    delete[] keys.first.dpfKey.s;
    delete[] keys.second.dpfKey.s;
    keys.first.dpfKey.s = nullptr;
    keys.second.dpfKey.s = nullptr;
}

void freeGrottoReLUKey(GrottoReLUKeyPack &key, bool ownsDPFStorage)
{
    if (ownsDPFStorage)
        delete[] key.dpfKey.s;
    key.dpfKey.s = nullptr;
}
