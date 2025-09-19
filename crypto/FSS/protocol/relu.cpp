#include "../protocol/relu.h"
#include "../primitives/dcf.h"
#include "../primitives/dpf.h"
#include <assert.h>

std::pair<ReluKeyPack, ReluKeyPack> keyGenRelu(int Bin, int Bout,
                        GroupElement rin, GroupElement rout, GroupElement routDrelu)
{
    // represents offset poly p(x-rin)'s coefficients, where p(x)=x
    // p(x)=x, 构建偏移项p(x-rin) 
    GroupElement beta[2];
    beta[0] = 1;
    beta[1] = -rin; // 掩码偏移量
    mod(beta[1], Bout); // 模运算确保值域合法（将输入截断到指定位宽）
    ReluKeyPack k0, k1;
    
    k0.Bin = Bin; k1.Bin = Bin;
    k0.Bout = Bout; k1.Bout = Bout;

    GroupElement gamma = rin - 1;
    auto dcfKeys = keyGenDCF(Bin, Bout, 2, gamma, beta); //gamma
    // 这里两个 DCF 的 beta 是不同的！

    // 计算输入值在掩码空间中的位置关系，确定ReLU的输出区间
    GroupElement p = 0;
    GroupElement q = (((uint64_t)1 << (Bin-1)) - 1); // 最大值 2^{Bin-1}-1
    GroupElement q1 = q + 1, alpha_L = p + rin, alpha_R = q + rin, alpha_R1 = q + 1 + rin; // alpha_L输入下限（p=0），alpha_R 输入上限
    mod(q1, Bin);
    mod(alpha_L, Bin);
    mod(alpha_R, Bin);
    mod(alpha_R1, Bin);
    GroupElement neg1 = -1;
    mod(neg1, Bin);
    // cr：组合多个比较结果（如alpha_L > alpha_R 等），标识输入是否在有效激活区间。
    GroupElement cr = GroupElement((alpha_L > alpha_R) - (alpha_L > p) + (alpha_R1 > q1) + (alpha_R == neg1));
    mod(cr, Bin);

    // 将中间值（cr、beta、掩码）分割为两份，分发给两方
    GroupElement val;
    // 分割 cr 相关的线性项
    val = beta[0] * cr;
    auto val_split = splitShare(val, Bin);
    k0.e_b0 = val_split.first;
    k1.e_b0 = val_split.second;
    
    val = beta[1] * cr;
    val_split = splitShare(val, Bin);
    k0.e_b1 = val_split.first;
    k1.e_b1 = val_split.second;

    // 分割多项式系数 beta
    auto beta_split = splitShare(beta[0], Bin);
    k0.beta_b0 = beta_split.first;
    k1.beta_b0 = beta_split.second;

    beta_split = splitShare(beta[1], Bin);
    k0.beta_b1 = beta_split.first;
    k1.beta_b1 = beta_split.second;

    // 分割输出掩码
    auto rout_split = splitShare(rout, Bout);
    k0.r_b = rout_split.first; k1.r_b = rout_split.second;
    auto drelu_split = splitShare(routDrelu, 1);
    k0.drelu = drelu_split.first; k1.drelu = drelu_split.second;
    k0.k = dcfKeys.first.k;
    k0.g = dcfKeys.first.g;
    k0.v = dcfKeys.first.v;
    k1.k = dcfKeys.second.k;
    k1.g = dcfKeys.second.g;
    k1.v = dcfKeys.second.v;

    return std::make_pair(k0, k1);
}

GroupElement evalRelu(int party, GroupElement x, const ReluKeyPack &k, GroupElement *drelu)
{
    int Bout = k.Bout;
    int Bin = k.Bin;
    mod(x, Bin);

    GroupElement p = 0;
    GroupElement q = GroupElement((((uint64_t)1 << (Bin-1)) - 1));
    mod(q, Bin);
    GroupElement q1 = q + 1, xL = x - 1, xR1 = x - 1 - q1; // 计算xL和xR1
    mod(q1, Bin);
    mod(xL, Bin);
    mod(xR1, Bin);
    // DCF评估
    GroupElement share_L[2]; 
    evalDCF(Bin, Bout, 2, share_L, party, xL, k.k, k.g, k.v);
    GroupElement share_R1[2];
    evalDCF(Bin, Bout, 2, share_R1, party, xR1, k.k, k.g, k.v);

    // 计算cx(边界条件)
    GroupElement cx = GroupElement((x > 0) - (x > q1)); //标识输入是否在激活区间
    mod(cx, k.Bin);
    GroupElement sum = 0;
    
    // 组合计算结果
    ///////////////////////////////////////////////////////////////////////////
    // 在标准ReLU协议中，为了高效地同时计算ReLU和dReLU（ReLU的导数），并处理整数环的边界问题，协议采用了一种多项式近似的方法。具体来说，它使用两个DCF来评估一个线性多项式，这个多项式在掩码空间内近似ReLU函数。
    GroupElement w_b = cx * k.beta_b0 - share_L[0] + share_R1[0] + k.e_b0;
    mod(w_b, Bout);
    if (drelu != nullptr) {
        *drelu = (w_b + k.drelu); // 导数输出
        mod(*drelu, 1);
    }
    sum = sum + (w_b * x);

    w_b = cx * k.beta_b1 - share_L[1] + share_R1[1] + k.e_b1;
    sum = sum + w_b;

    // 最终输出
    GroupElement ub(k.r_b + sum);
    mod(ub, Bout);
    return ub;
}


std::pair<MaxpoolKeyPack, MaxpoolKeyPack> keyGenMaxpool(int Bin, int Bout, GroupElement rin1, GroupElement rin2, GroupElement rout, GroupElement routBit)
{
    // maxpool(x, y) = relu(x - y) + y
    // for correctness, ensure magnitude(x) + magnitude(y) in signed context < N/2
    MaxpoolKeyPack k0, k1;
    k0.Bin = Bin; k1.Bin = Bin;
    k0.Bout = Bout; k1.Bout = Bout;

    auto reluKeys = keyGenRelu(Bin, Bout, rin2 - rin1, 0, routBit);
    k0.reluKey = reluKeys.first; 
    k1.reluKey = reluKeys.second;

    auto rb_split = splitShare(-rin1 + rout, Bout);
    k0.rb = rb_split.first; k1.rb = rb_split.second;

    return std::make_pair(k0, k1);
}

GroupElement evalMaxpool(int party, GroupElement x, GroupElement y, const MaxpoolKeyPack &k, GroupElement &bit)
{
    // maxpool(x, y) = relu(x - y) + y
    // for correctness, ensure magnitude(x) + magnitude(y) in signed context < N/2
    GroupElement res = evalRelu(party, y - x, k.reluKey, &bit) + (party * x) + k.rb;
    return res;
}

std::pair<Relu2RoundKeyPack, Relu2RoundKeyPack> keyGenRelu2Round(int effectiveBw, int bin, GroupElement rin, GroupElement routRelu, GroupElement rout)
{
    mod(rin, bin);
    GroupElement r1, r0;
    
    auto dcfN = keyGenDCF(effectiveBw, 1, rin, 1); //1个DCF

    GroupElement a;
    GroupElement b;
    GroupElement c;
    GroupElement d1 = 0;
    GroupElement d2 = 0;
    a = routRelu & 1; // dReLU掩码的LSB
    b = rin; // 输入掩码
    c = (routRelu & 1) * rin + rout; // 组合掩码
    if ((routRelu & 1) == 1) { // 当routRelu=1时创建调整项
        d1 = 2;
        d2 = -2 * rin;
    }

    Relu2RoundKeyPack k0, k1;
    k0.Bin = bin; k1.Bin = bin;
    k0.effectiveBin = effectiveBw; k1.effectiveBin = effectiveBw;
    k0.dcfKey = dcfN.first; k1.dcfKey = dcfN.second;
    auto aSplit = splitShare(a, bin);
    k0.a = aSplit.first; k1.a = aSplit.second;
    auto bSplit = splitShare(b, bin);
    k0.b = bSplit.first; k1.b = bSplit.second;
    auto cSplit = splitShare(c, bin);
    k0.c = cSplit.first; k1.c = cSplit.second;
    auto d1Split = splitShare(d1, bin);
    k0.d1 = d1Split.first; k1.d1 = d1Split.second;
    auto d2Split = splitShare(d2, bin);
    k0.d2 = d2Split.first; k1.d2 = d2Split.second;
    return std::make_pair(k0, k1);
}

// Relu2Round在线评估第一轮：计算dReLU
GroupElement evalRelu2_drelu(int party, GroupElement x, const Relu2RoundKeyPack &key) {
    // 输入正规化
    GroupElement xp = x + (1ULL<<(key.effectiveBin - 1)); // 添加偏移防止负值
    mod(xp, key.effectiveBin);
    mod(x, key.effectiveBin);
    // 双通道DCF比较
    GroupElement t2 = 0;
    GroupElement t1 = 0;
    evalDCF(party, &t2, xp, key.dcfKey); // 偏移值比较
    evalDCF(party, &t1, x, key.dcfKey); // 原始值比较
    // 组合比较结果
    GroupElement res;
    res = t2 - t1 + key.a;
    mod(res, 1);
    if (party == 1) { // 处理特殊边界（仅dealer）
        if (xp >= (1ULL<<(key.effectiveBin - 1))) {
            res += 1;
        }
    }
    mod(res, 1); // 确保结果为0/1
    return res;
}
// Relu2Round在线评估第二轮：计算ReLU (evalRelu2_mult)
GroupElement evalRelu2_mult(int party, GroupElement x, GroupElement y, const Relu2RoundKeyPack &key) {
    // x: 第一轮得到的完整dReLU值; y: 需要计算ReLU的输入值
    GroupElement t1 = 0;
    GroupElement t2 = 0;
    mod(x, 1);
    if (x == 0) { // x即dReLU值
        t1 = key.d1;
        t2 = key.d2;
    }
    // 本地线性计算​​
    GroupElement res;
    res = -key.a * y - key.b * x + key.c + y * t1 + t2; // 符号位掩码调整 + 输入掩码调整 + 组合掩码 + dReLU=1时的线性调整 + 常数项调整
    if (party == 1) {
        res += x * y; // 添加交互项
    }
    mod(res, key.Bin); // 位宽约束
    return res; 
}

std::pair<MaxpoolDoubleKeyPack, MaxpoolDoubleKeyPack> keyGenMaxpoolDouble(int Bin, int Bout, GroupElement rin1, GroupElement rin2, GroupElement routBit, GroupElement rout)
{
    // maxpool(x, y) = relu(x - y) + y
    // for correctness, ensure magnitude(x) + magnitude(y) in signed context < N/2
    MaxpoolDoubleKeyPack k0, k1;
    k0.Bin = Bin; k1.Bin = Bin;
    k0.Bout = Bout; k1.Bout = Bout;

    auto reluKeys = keyGenRelu2Round(Bin, Bout, rin2 - rin1, routBit, 0);
    k0.reluKey = reluKeys.first; 
    k1.reluKey = reluKeys.second;

    auto rb_split = splitShare(-rin1 + rout, Bout);
    k0.rb = rb_split.first; k1.rb = rb_split.second;

    return std::make_pair(k0, k1);
}

GroupElement evalMaxpoolDouble_1(int party, GroupElement x, GroupElement y, const MaxpoolDoubleKeyPack &k)
{
    // maxpool(x, y) = relu(x - y) + y
    // for correctness, ensure magnitude(x) + magnitude(y) in signed context < N/2
    GroupElement res = evalRelu2_drelu(party, y - x, k.reluKey);// + (party * y) + k.rb;
    return res;
}

GroupElement evalMaxpoolDouble_2(int party, GroupElement x, GroupElement y, GroupElement s, const MaxpoolDoubleKeyPack &k)
{
    GroupElement res = evalRelu2_mult(party, s, y - x, k.reluKey) + (party * x) + k.rb;
    return res;
}

std::pair<SlothDreluKeyPack, SlothDreluKeyPack> keyGenSlothDrelu(int bin, GroupElement rin, GroupElement rout)
{
    GroupElement x_1 = -rin;
    mod(x_1, bin);
    GroupElement y_1 = x_1;
    mod(y_1, bin - 1);

    auto dpfKeys = keyGenDPFET(bin - 1, y_1);
    GroupElement r = rout ^ 1 ^ ((x_1 >> (bin - 1)) & 1);
    auto r_split = splitShare(r, 1);

    SlothDreluKeyPack k0, k1;
    k0.bin = bin; k1.bin = bin;
    k0.dpfKey = dpfKeys.first; k1.dpfKey = dpfKeys.second;
    k0.r = r_split.first; k1.r = r_split.second;
    return std::make_pair(k0, k1);
}

GroupElement evalSlothDrelu(int party, GroupElement x, const SlothDreluKeyPack &kp)
{
    GroupElement y_0 = - x - 1;
    mod(y_0, kp.bin - 1);
    GroupElement u_b = evalDPFET_LT(party, kp.dpfKey, y_0);
    GroupElement res = u_b ^ kp.r; // 通过异或操作组合结果，1轮交互
    if (party == 0)
    {
        res ^= ((x >> (kp.bin - 1)) & 1);
    }
    return res;
}



// FastSecNetReLU
// 调用DCF或DPF比较方案，将小于比较变为大于比较
std::pair<ReluKeyPack, ReluKeyPack> keyGenFastSecNetRelu(int Bin, int Bout,
                        GroupElement rin, GroupElement rout, GroupElement routDrelu)
{
    // represents offset poly p(x-rin)'s coefficients, where p(x)=x
    GroupElement beta[2];
    beta[0] = 1;
    beta[1] = -rin;
    mod(beta[1], Bout);
    ReluKeyPack k0, k1;
    
    k0.Bin = Bin; k1.Bin = Bin;
    k0.Bout = Bout; k1.Bout = Bout;

    // 调用DCF
    GroupElement gamma = rin - 1;
    auto dcfKeys = keyGenDCF(Bin, Bout, 2, gamma, beta);

    GroupElement p = 0;
    GroupElement q = (((uint64_t)1 << (Bin-1)) - 1);
    GroupElement q1 = q + 1, alpha_L = p + rin, alpha_R = q + rin, alpha_R1 = q + 1 + rin;
    mod(q1, Bin);
    mod(alpha_L, Bin);
    mod(alpha_R, Bin);
    mod(alpha_R1, Bin);
    GroupElement neg1 = -1;
    mod(neg1, Bin);
    GroupElement cr = GroupElement((alpha_L > alpha_R) - (alpha_L > p) + (alpha_R1 > q1) + (alpha_R == neg1));
    mod(cr, Bin);

    GroupElement val;
    val = beta[0] * cr;
    auto val_split = splitShare(val, Bin);
    k0.e_b0 = val_split.first;
    k1.e_b0 = val_split.second;

    val = beta[1] * cr;
    val_split = splitShare(val, Bin);
    k0.e_b1 = val_split.first;
    k1.e_b1 = val_split.second;

    auto beta_split = splitShare(beta[0], Bin);
    k0.beta_b0 = beta_split.first;
    k1.beta_b0 = beta_split.second;

    beta_split = splitShare(beta[1], Bin);
    k0.beta_b1 = beta_split.first;
    k1.beta_b1 = beta_split.second;


    auto rout_split = splitShare(rout, Bout);
    k0.r_b = rout_split.first; k1.r_b = rout_split.second;
    auto drelu_split = splitShare(routDrelu, 1);
    k0.drelu = drelu_split.first; k1.drelu = drelu_split.second;
    k0.k = dcfKeys.first.k;
    k0.g = dcfKeys.first.g;
    k0.v = dcfKeys.first.v;
    k1.k = dcfKeys.second.k;
    k1.g = dcfKeys.second.g;
    k1.v = dcfKeys.second.v;

    return std::make_pair(k0, k1);
}

GroupElement evalFastSecNetRelu(int party, GroupElement x, const ReluKeyPack &k, GroupElement *drelu)
{
    int Bout = k.Bout;
    int Bin = k.Bin;
    mod(x, Bin);

    GroupElement p = 0;
    GroupElement q = GroupElement((((uint64_t)1 << (Bin-1)) - 1));
    mod(q, Bin);
    GroupElement q1 = q + 1, xL = x - 1, xR1 = x - 1 - q1;
    mod(q1, Bin);
    mod(xL, Bin);
    mod(xR1, Bin);
    GroupElement share_L[2]; 
    evalDCF(Bin, Bout, 2, share_L, party, xL, k.k, k.g, k.v);
    GroupElement share_R1[2];
    evalDCF(Bin, Bout, 2, share_R1, party, xR1, k.k, k.g, k.v);

    GroupElement cx = GroupElement((x > 0) - (x > q1));
    mod(cx, k.Bin);
    GroupElement sum = 0;
    
    GroupElement w_b = cx * k.beta_b0 - share_L[0] + share_R1[0] + k.e_b0;
    mod(w_b, Bout);
    if (drelu != nullptr) {
        *drelu = (w_b + k.drelu);
        mod(*drelu, 1);
    }
    sum = sum + (w_b * x);

    w_b = cx * k.beta_b1 - share_L[1] + share_R1[1] + k.e_b1;
    sum = sum + w_b;

    GroupElement ub(k.r_b + sum);
    mod(ub, Bout);
    return ub;
}
