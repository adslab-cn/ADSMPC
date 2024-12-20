#pragma once
#include <aux_parameter/keypack.h>

std::pair<SelectKeyPack, SelectKeyPack> keyGenSelect(int Bin, GroupElement s, GroupElement y, GroupElement out);
GroupElement evalSelect(int party, GroupElement s, GroupElement x, const SelectKeyPack &key);
