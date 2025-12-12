#include <utility> // <--- 添加这一行
#include "../aux_parameter/keypack.h"

std::pair<FastReluKeyPack, FastReluKeyPack> keyGenFastRelu(int Bin, int Bout);
std::pair<FastReluDPFETKeyPack, FastReluDPFETKeyPack> keyGenFastRelu_DPFET(int Bin, int Bout);