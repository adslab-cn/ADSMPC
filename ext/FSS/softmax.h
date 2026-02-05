#pragma once
#include <FSS/keypack.h>
#include <vector>


// 3. OblivSoftmax
// L = s2 (number of classes)
// scale_in: 输入的定点数缩放因子
// scale_out: 输出的定点数缩放因子
std::pair<OblivSoftmaxKeyPack, OblivSoftmaxKeyPack> keyGenOblivSoftmax(
    int s1, int s2, int Bin, int Bout, 
    GroupElement* rin, GroupElement* rout, 
    int sf, int logk);