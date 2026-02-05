#pragma once
#include <FSS/keypack.h>

// 基础ReLU协议
std::pair<ReluKeyPack, ReluKeyPack> keyGenRelu(int Bin, int Bout,
                        GroupElement rin, GroupElement rout, GroupElement routDrelu = 0);

// GroupElement evalRelu(int party, GroupElement x, const ReluKeyPack &k);
GroupElement evalRelu(int party, GroupElement x, const ReluKeyPack &k, GroupElement *drelu = nullptr);

std::pair<MaxpoolKeyPack, MaxpoolKeyPack> keyGenMaxpool(int Bin, int Bout, GroupElement rin1, GroupElement rin2, GroupElement rout, GroupElement routBit);
GroupElement evalMaxpool(int party, GroupElement x, GroupElement y, const MaxpoolKeyPack &k, GroupElement &bit);

// 2轮ReLU协议
std::pair<Relu2RoundKeyPack, Relu2RoundKeyPack> keyGenRelu2Round(int effectiveBw, int bin, GroupElement rin, GroupElement routRelu, GroupElement rout);
GroupElement evalRelu2_drelu(int party, GroupElement x, const Relu2RoundKeyPack &key);
GroupElement evalRelu2_mult(int party, GroupElement x, GroupElement y, const Relu2RoundKeyPack &key);

std::pair<MaxpoolDoubleKeyPack, MaxpoolDoubleKeyPack> keyGenMaxpoolDouble(int Bin, int Bout, GroupElement rin1, GroupElement rin2, GroupElement routBit, GroupElement rout);
GroupElement evalMaxpoolDouble_1(int party, GroupElement x, GroupElement y, const MaxpoolDoubleKeyPack &k);
GroupElement evalMaxpoolDouble_2(int party, GroupElement x, GroupElement y, GroupElement s, const MaxpoolDoubleKeyPack &k);

// SIGMA的CPU版本ReLU协议
std::pair<SlothDreluKeyPack, SlothDreluKeyPack> keyGenSlothDrelu(int bin, GroupElement rin, GroupElement rout);
GroupElement evalSlothDrelu(int party, GroupElement x, const SlothDreluKeyPack &k);

// FastSecNet
std::pair<FastSecNetReluKeyPack, FastSecNetReluKeyPack> keyGenFastSecNetRelu(int Bin, int Bout,
                        GroupElement rin, GroupElement rout);

GroupElement evalFastSecNetRelu1(int party, GroupElement x_, const FastSecNetReluKeyPack &k);
GroupElement evalFastSecNetRelu2(int party, GroupElement x_shift, const FastSecNetReluKeyPack &k);

// 新的Drelu计算方案
std::pair<NewDreluKeyPack, NewDreluKeyPack> keyGenNewDrelu(int bin, GroupElement rin, GroupElement rout);
GroupElement evalNewDrelu(int party, GroupElement x, const NewDreluKeyPack &k);

std::pair<OblivReLUKeyPack, OblivReLUKeyPack> keyGenOblivReLU(int Bin, int Bout, GroupElement rin, GroupElement rout);
GroupElement evalOblivReLU(int party, GroupElement x, const OblivReLUKeyPack &key);
