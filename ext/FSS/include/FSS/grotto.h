#pragma once

#include <FSS/group_element.h>
#include <FSS/keypack.h>

#include <utility>

struct GrottoReLULocalShares {
    GroupElement correctionSign;
    GroupElement slope;
    GroupElement input;
};

std::pair<GrottoReLUKeyPack, GrottoReLUKeyPack> keyGenGrottoReLU(
    int bin, GroupElement inputMask, GroupElement outputMask);

// Local Grotto evaluation after the parties have opened x-i. No communication
// or global party state is used by these primitives.
GrottoReLULocalShares evalGrottoReLULocalShares(
    int evaluatorParty, const GrottoReLUKeyPack &key,
    GroupElement maskedInput, GroupElement openedDelta);

void freeGrottoReLUKeyPair(
    std::pair<GrottoReLUKeyPack, GrottoReLUKeyPack> &keys);
void freeGrottoReLUKey(GrottoReLUKeyPack &key, bool ownsDPFStorage);
