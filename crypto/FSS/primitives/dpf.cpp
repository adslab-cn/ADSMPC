#include "dpf.h"
#include "../aux_parameter/assert.h"
#include <cassert>
#include <algorithm> // for std::sort
#include <vector>

using namespace osuCrypto;

inline uint8_t lsb(const block &b)
{
    return _mm_cvtsi128_si64x(b) & 1;
}


/*
 * lambda = 127
 */
std::pair<DPFKeyPack, DPFKeyPack> keyGenDPF(int bin, int bout, GroupElement idx, GroupElement payload)
{
    always_assert(bin <= 64);
    always_assert(bout <= 64);
    static const block notOneBlock = toBlock(~0, ~1); // 创建一个128位掩码，DPF协议中常将数据块的最低位作为​​奇偶标记位​​（如t标记路径选择状态），这个操作可保留数据块的127位有效信息，同时强制最低位置零，确保后续加密或修正操作不受残留标记影响。
    const static block pt[2] = {ZeroBlock, OneBlock}; // 定义明文输入数组，用于​​AES加密的规范化输入​​，分别对应比特值0和1。通过预定义pt[0]和pt[1]，可​​直接调用AES加密函数​​生成对应子节点种子，避免动态构造输入块的开销

    DPFKeyPack key0(bin, bout);
    DPFKeyPack key1(bin, bout);

    // ============ 初始化 ============
    int tid = omp_get_thread_num();
    auto s = FSSConfig::prngs[tid].get<std::array<block, 2>>();
    // s0，s1：双方当前层种子（128 位），初始值来自 PRNG
    auto s0 = s[0];
    auto s1 = s[1];
    // 通过 notOneBlock 清除最低位（保留种子部分）。
    s0 = s0 & notOneBlock;
    s1 = s1 & notOneBlock;
    key0.s[0] = s0; // Party0 的根种子
    key1.s[0] = s1; // Party1 的根种子
    
    // 标志位，双方当前路径选择位，初始值：t0=0, t1=1（强制差异化）。
    u8 t0 = 0; // Party0 的初始 t 值
    u8 t1 = 1; // Party1 的初始 t 值

    block ct0[2];
    block ct1[2];

    for (int i = 0; i < bin; ++i)
    {
        // ​​路径选择
        const u8 keep = static_cast<uint8_t>(idx >> (bin - 1 - i)) & 1; // 当前树层目标路径的比特（0=左/1=右）
        const u8 loose = keep ^ 1; // 非目标路径比特

        // AES 伪随机生成器 (PRG) 扩展
        AES ak0(s0);
        AES ak1(s1);
        ak0.ecbEncTwoBlocks(pt, ct0); // 生成 s0 的子节点 (ct0[0], ct0[1])
        ak1.ecbEncTwoBlocks(pt, ct1); // 生成 s1 的子节点 (ct1[0], ct1[1])

        // 计算纠正字 (CW)
        auto scw = (ct0[loose] ^ ct1[loose]) & notOneBlock; // 种子纠正字
        u64 tcw[2];
        u64 &tLcw = tcw[0];
        u64 &tRcw = tcw[1];
        tLcw = lsb(ct0[0]) ^ lsb(ct1[0]) ^ keep ^ 1; // 左子树 t 纠正
        tRcw = lsb(ct0[1]) ^ lsb(ct1[1]) ^ keep; // 右子树 t 纠正

        key0.s[i+1] = scw; // 双方共享相同的种子 CW
        key1.s[i+1] = scw;
        key0.tLcw |= (tLcw << (bin - 1 - i)); // 按比特位置存储 tLcw
        key0.tRcw |= (tRcw << (bin - 1 - i)); // 按比特位置存储 tRcw

        /* Party0 更新 */
        if (t0 == 0)
        {
            s0 = ct0[keep] & notOneBlock;
            t0 = lsb(ct0[keep]); //更新
        }
        else
        {
            s0 = (ct0[keep] & notOneBlock) ^ scw;
            t0 = lsb(ct0[keep]) ^ tcw[keep];
        }
        /* Party1 更新 */
        if (t1 == 0)
        {
            s1 = ct1[keep] & notOneBlock;
            t1 = lsb(ct1[keep]);
        }
        else
        {
            s1 = (ct1[keep] & notOneBlock) ^ scw;
            t1 = lsb(ct1[keep]) ^ tcw[keep];
        }
    }

    // 1. 同步双方 t 纠正字
    key1.tLcw = key0.tLcw;
    key1.tRcw = key0.tRcw;
    // 2. 计算 payload
    key0.payload = payload - _mm_extract_epi64(s0, 0) + _mm_extract_epi64(s1, 0);
    if (t1 == 1) key0.payload = -key0.payload;
    key1.payload = key0.payload;

    return std::make_pair(key0, key1);
}

GroupElement evalDPF_EQ(int party, DPFKeyPack &key, GroupElement x)
{
    static const block notOneBlock = toBlock(~0, ~1);
    int bin = key.bin;
    int bout = key.bout;

    auto s = _mm_loadu_si128(key.s);
    u8 t = party;
    for (int i = 0; i < bin; ++i)
    {
        assert(lsb(s) == 0);
        const u8 x_i = static_cast<uint8_t>(x >> (bin - 1 - i)) & 1;
        
        AES ak(s);
        block ct = ak.ecbEncBlock(toBlock(0, x_i));
        s = ct & notOneBlock;
        u8 t_old = t;
        t = lsb(ct);

        if (t_old) {
            s = s ^ _mm_loadu_si128(key.s + i + 1);
            t = t ^ ((key.tcw[x_i] >> (bin - 1 - i)) & 1);
        }
    }

    return t;
}

GroupElement evalDPF_GT(int party, DPFKeyPack &key, GroupElement x)
{
    static const block notOneBlock = toBlock(~0, ~1);
    int bin = key.bin;
    int bout = key.bout;

    auto s = _mm_loadu_si128(key.s);
    u8 t = party;
    
    u8 x_prev = 0;
    u8 t_dcf = 0;

    for (int i = 0; i < bin; ++i)
    {
        assert(lsb(s) == 0);
        const u8 x_i = static_cast<uint8_t>(x >> (bin - 1 - i)) & 1;

        if (x_prev != x_i)
        {
            t_dcf = t_dcf ^ t;
        }
        x_prev = x_i;

        AES ak(s);
        block ct = ak.ecbEncBlock(toBlock(0, x_i));
        s = ct & notOneBlock;
        u8 t_old = t;
        t = lsb(ct);
        

        if (t_old) {
            s = s ^ _mm_loadu_si128(key.s + i + 1);
            t = t ^ ((key.tcw[x_i] >> (bin - 1 - i)) & 1);
        }
    }

    if (x_prev == 1)
    {
        t_dcf = t_dcf ^ t;
    }
    return t_dcf;
}

GroupElement evalDPF_LT(int party, DPFKeyPack &key, GroupElement x)
{
    static const block notOneBlock = toBlock(~0, ~1);
    int bin = key.bin;
    int bout = key.bout;

    auto s = _mm_loadu_si128(key.s);
    u8 t = party;
    
    u8 x_prev = 1;
    u8 t_dcf = 0;

    for (int i = 0; i < bin; ++i)
    {
        assert(lsb(s) == 0);
        const u8 x_i = static_cast<uint8_t>(x >> (bin - 1 - i)) & 1;

        if (x_prev != x_i)
        {
            t_dcf = t_dcf ^ t;
        }
        x_prev = x_i;

        AES ak(s);
        block ct = ak.ecbEncBlock(toBlock(0, x_i));
        s = ct & notOneBlock;
        u8 t_old = t;
        t = lsb(ct);
        

        if (t_old) {
            s = s ^ _mm_loadu_si128(key.s + i + 1);
            t = t ^ ((key.tcw[x_i] >> (bin - 1 - i)) & 1);
        }
    }

    if (x_prev == 0)
    {
        t_dcf = t_dcf ^ t;
    }
    return t_dcf;
}

void evalAll_helper(int party, DPFKeyPack &key, GroupElement rightShift, GroupElement *out, block s_prev, u8 t_prev, int i, GroupElement acc)
{
    if (i == key.bin)
    {
        GroupElement idx = acc + rightShift;
        mod(idx, key.bin);
        out[idx] = (1 - 2 * party) * (_mm_extract_epi64(s_prev, 0) + key.payload * t_prev);
        return;
    }

    const static block pt[2] = {ZeroBlock, OneBlock};
    static const block notOneBlock = toBlock(~0, ~1);

    AES ak(s_prev);
    block ct[2];
    ak.ecbEncTwoBlocks(pt, ct);

    for (int x_i = 0; x_i < 2; ++x_i)
    {
        block s = ct[x_i] & notOneBlock;
        u8 t = lsb(ct[x_i]);

        if (t_prev) {
            s = s ^ _mm_loadu_si128(key.s + i + 1);
            t = t ^ ((key.tcw[x_i] >> (key.bin - 1 - i)) & 1);
        }

        evalAll_helper(party, key, rightShift, out, s, t, i+1, 2 * acc + x_i);
    }
}

void evalAll(int party, DPFKeyPack &key, GroupElement rightShift, GroupElement *out)
{
    auto s = _mm_loadu_si128(key.s);
    u8 t = party;
    
    evalAll_helper(party, key, rightShift, out, s, t, 0, 0);
}

void evalAll_reduce_helper(int party, DPFKeyPack &key, GroupElement rightShift, const std::vector<GroupElement> &tab, GroupElement &out, block &s_prev, u8 t_prev, int i, GroupElement acc)
{
    if (i == key.bin)
    {
        GroupElement idx = acc + rightShift;
        mod(idx, key.bin);
        out = out + tab[idx] * ((1 - 2 * party) * (_mm_extract_epi64(s_prev, 0) + key.payload * t_prev));
        return;
    }

    const static block pt[2] = {ZeroBlock, OneBlock};
    static const block notOneBlock = toBlock(~0, ~1);

    AES ak(s_prev);
    block ct[2];
    ak.ecbEncTwoBlocks(pt, ct);

    for (int x_i = 0; x_i < 2; ++x_i)
    {
        block s = ct[x_i] & notOneBlock;
        u8 t = lsb(ct[x_i]);

        if (t_prev) {
            s = s ^ _mm_loadu_si128(key.s + i + 1);
            t = t ^ ((key.tcw[x_i] >> (key.bin - 1 - i)) & 1);
        }

        evalAll_reduce_helper(party, key, rightShift, tab, out, s, t, i+1, 2 * acc + x_i);
    }
}

GroupElement evalAll_reduce(int party, DPFKeyPack &key, GroupElement rightShift, const std::vector<GroupElement> &tab)
{
    auto s = _mm_loadu_si128(key.s);
    u8 t = party;
    GroupElement out = 0;
    
    evalAll_reduce_helper(party, key, rightShift, tab, out, s, t, 0, 0);
    return out;
}

std::pair<DPFETKeyPack, DPFETKeyPack> keyGenDPFET(int bin, GroupElement idx)
{
    always_assert(bin <= 64);
    always_assert(bin >= 8);
    static const block notOneBlock = toBlock(~0, ~1);
    const static block pt[2] = {ZeroBlock, OneBlock};

    DPFETKeyPack key0(bin);
    DPFETKeyPack key1(bin);

    int tid = omp_get_thread_num();
    auto s = FSSConfig::prngs[tid].get<std::array<block, 2>>();
    auto s0 = s[0];
    auto s1 = s[1];

    s0 = s0 & notOneBlock;
    s1 = s1 & notOneBlock;
    key0.s[0] = s0;
    key1.s[0] = s1;
    
    u8 t0 = 0;
    u8 t1 = 1;

    block ct0[2];
    block ct1[2];

    for (int i = 0; i < bin - 7; ++i)
    {
        const u8 keep = static_cast<uint8_t>(idx >> (bin - 1 - i)) & 1;
        const u8 loose = keep ^ 1;

        AES ak0(s0);
        AES ak1(s1);

        ak0.ecbEncTwoBlocks(pt, ct0);
        ak1.ecbEncTwoBlocks(pt, ct1);

        auto scw = (ct0[loose] ^ ct1[loose]) & notOneBlock;
        u64 tcw[2];
        u64 &tLcw = tcw[0];
        u64 &tRcw = tcw[1];
        tLcw = lsb(ct0[0]) ^ lsb(ct1[0]) ^ keep ^ 1;
        tRcw = lsb(ct0[1]) ^ lsb(ct1[1]) ^ keep;

        key0.s[i+1] = scw;
        key1.s[i+1] = scw;
        key0.tLcw |= (tLcw << (bin - 1 - i));
        key0.tRcw |= (tRcw << (bin - 1 - i));

        if (t0 == 0)
        {
            s0 = ct0[keep] & notOneBlock;
            t0 = lsb(ct0[keep]);
        }
        else
        {
            s0 = (ct0[keep] & notOneBlock) ^ scw;
            t0 = lsb(ct0[keep]) ^ tcw[keep];
        }

        if (t1 == 0)
        {
            s1 = ct1[keep] & notOneBlock;
            t1 = lsb(ct1[keep]);
        }
        else
        {
            s1 = (ct1[keep] & notOneBlock) ^ scw;
            t1 = lsb(ct1[keep]) ^ tcw[keep];
        }
    }

    key1.tLcw = key0.tLcw;
    key1.tRcw = key0.tRcw;

    if (t0 == 1) s0 = s0 ^ OneBlock;
    if (t1 == 1) s1 = s1 ^ OneBlock;
    uint64_t e0, e1;
    GroupElement ip = idx % 128;
    if (ip >= 64) {
        e0 = 0;
        e1 = 1ULL << (127 - ip);
    }
    else {
        e0 = 1ULL << (63 - ip);
        e1 = 0;
    }
    key0.leaf = s0 ^ s1 ^ osuCrypto::toBlock(e0, e1);
    key1.leaf = key0.leaf;

    return std::make_pair(key0, key1);
}

GroupElement evalDPFET_LT(int party, const DPFETKeyPack &key, GroupElement x)
{
    static const block notOneBlock = toBlock(~0, ~1);
    int bin = key.bin;

    auto s = _mm_loadu_si128(key.s);
    u8 t = party;
    
    u8 x_prev = 1;
    u8 t_dcf = 0;

    for (int i = 0; i < bin - 7; ++i)
    {
        assert(lsb(s) == 0);
        const u8 x_i = static_cast<uint8_t>(x >> (bin - 1 - i)) & 1;

        if (x_prev != x_i)
        {
            t_dcf = t_dcf ^ t;
        }
        x_prev = x_i;

        AES ak(s);
        block ct = ak.ecbEncBlock(toBlock(0, x_i));
        s = ct & notOneBlock;
        u8 t_old = t;
        t = lsb(ct);

        if (t_old) {
            s = s ^ _mm_loadu_si128(key.s + i + 1);
            t = t ^ ((key.tcw[x_i] >> (bin - 1 - i)) & 1);
        }
    }

    osuCrypto::block leaf = s;
    if (t) leaf = leaf ^ OneBlock ^ key.leaf;
    uint64_t b;

    {
        const u8 x_i = static_cast<uint8_t>(x >> 6) & 1;
        if (x_prev != x_i)
        {
            t_dcf = t_dcf ^ t;
        }
        x_prev = x_i;
        if (x_i) 
        {
            b = _mm_extract_epi64(leaf, 0);
        }
        else 
        {
            b = _mm_extract_epi64(leaf, 1);
        }
        t = __builtin_parityll(b);
    }

    GroupElement xp = x % 64;

    if (x_prev == 0)
    {
        for (int i = 0; i <= xp; ++i)
        {
            t_dcf = t_dcf ^ ((b >> (63 - i)) & 1);
        }
    }
    else
    {
        for (int i = xp + 1; i < 64; ++i)
        {
            t_dcf = t_dcf ^ ((b >> (63 - i)) & 1);
        }
    }

    return t_dcf;
}

void evalAll_reduce_helper_et(int party, DPFETKeyPack &key, GroupElement rightShift, const std::vector<GroupElement> &tab, GroupElement &out, GroupElement &corr, block &s_prev, u8 t_prev, int i, GroupElement acc)
{
    if (i == key.bin - 7)
    {
        osuCrypto::block leaf = s_prev;
        if (t_prev) leaf = leaf ^ OneBlock ^ key.leaf;
        
        uint64_t b = _mm_extract_epi64(leaf, 1);
        for (int j = 0; j < 64; ++j) {
            GroupElement idx = 128 * acc + j + rightShift;
            mod(idx, key.bin);
            GroupElement e = ((1 - 2 * party) * ((b >> (63 - j)) & 1));
            out = out + tab[idx] * e;
            corr = corr + e;
        }

        b = _mm_extract_epi64(leaf, 0);
        for (int j = 64; j < 128; ++j) {
            GroupElement idx = 128 * acc + j + rightShift;
            mod(idx, key.bin);
            GroupElement e = ((1 - 2 * party) * ((b >> (127 - j)) & 1));
            out = out + tab[idx] * e;
            corr = corr + e;
        }
        return;
    }

    const static block pt[2] = {ZeroBlock, OneBlock};
    static const block notOneBlock = toBlock(~0, ~1);

    AES ak(s_prev);
    block ct[2];
    ak.ecbEncTwoBlocks(pt, ct);

    for (int x_i = 0; x_i < 2; ++x_i)
    {
        block s = ct[x_i] & notOneBlock;
        u8 t = lsb(ct[x_i]);

        if (t_prev) {
            s = s ^ _mm_loadu_si128(key.s + i + 1);
            t = t ^ ((key.tcw[x_i] >> (key.bin - 1 - i)) & 1);
        }

        evalAll_reduce_helper_et(party, key, rightShift, tab, out, corr, s, t, i+1, 2 * acc + x_i);
    }
}

std::pair<GroupElement, GroupElement> evalAll_reduce_et(int party, DPFETKeyPack &key, GroupElement rightShift, const std::vector<GroupElement> &tab)
{
    auto s = _mm_loadu_si128(key.s);
    u8 t = party;
    GroupElement out = 0;
    GroupElement corr = 0;
    
    evalAll_reduce_helper_et(party, key, rightShift, tab, out, corr, s, t, 0, 0);
    return std::make_pair(out, corr);
}

// GroupElement evalDPF_with_payload(int party, DPFKeyPack &key, GroupElement x)
// {
//     static const block notOneBlock = toBlock(~0, ~1);
//     int bin = key.bin;

//     // 初始化 s 和 t
//     block s = _mm_loadu_si128(key.s);
//     u8 t = party;

//     // 沿树路径向下遍历
//     for (int i = 0; i < bin; ++i)
//     {
//         // 您的实现中断言 lsb(s) 是 0，因为控制位 t 是分开存储的
//         // assert(lsb(s) == 0); 
        
//         const u8 x_i = static_cast<uint8_t>(x >> (bin - 1 - i)) & 1;
        
//         AES ak(s);
//         block ct = ak.ecbEncBlock(toBlock(0, x_i));
        
//         u8 t_old = t;
//         s = ct & notOneBlock;
//         t = lsb(ct);

//         if (t_old) { // 如果在特殊路径上
//             s = s ^ _mm_loadu_si128(key.s + i + 1);
//             t = t ^ ((key.tcw[x_i] >> (bin - 1 - i)) & 1);
//         }
//     }

//     // --- 最终份额计算 ---
//     // 这是与 evalDPF_EQ 唯一不同的地方

//     // 从最终的种子 s 中提取数值部分
//     GroupElement final_s_val = _mm_extract_epi64(s, 0);

//     // 根据 FSS 输出公式计算份额
//     GroupElement result = final_s_val + key.payload * t;
    
//     // 乘以 (-1)^party
//     if (party == 1) {
//         return -result;
//     }
//     mod(result, key.bout); 
//     return result;
// }

GroupElement evalDPF_with_payload(int party, DPFKeyPack &key, GroupElement x)
{
    // --- 1. 初始化 ---
    
    // a. 定义常量，用于从128位的 AES 输出中分离出127位的种子和1位的控制位。
    //    notOneBlock 的最低位为0，其他位为1。
    static const block notOneBlock = toBlock(~0, ~1);
    
    // b. 从密钥中获取输入/输出的逻辑位宽。
    int bin = key.bin;
    int bout = key.bout;

    // c. 从密钥中加载根种子 s_root，并根据 party ID 初始化根控制位 t_root。
    //    这是协议的起点：双方从一个公共的随机状态（通过校正词关联）和一个
    //    初始差异（t=0 vs t=1）开始。
    block s = _mm_loadu_si128(key.s);
    u8 t = party; // party 必须是 0 或 1

    // --- 2. GGM 树遍历 ---
    
    // 循环遍历输入 x 的每一位，从最高位到最低位，模拟在GGM树中从根向叶子的移动。
    for (int i = 0; i < bin; ++i)
    {
        // a. 根据输入 x 的当前位，决定本层的路径（0=左，1=右）。
        //    这个路径选择对于两方是相同的，因为 x 是公开的。
        const u8 x_i = static_cast<uint8_t>(x >> (bin - 1 - i)) & 1;
        
        // b. 使用当前种子 s 作为密钥，通过 AES (模拟PRG) 生成下一层的子节点种子。
        //    我们只需要计算所选路径上的那个子节点。
        AES ak(s);
        block ct = ak.ecbEncBlock(toBlock(0, x_i));
        
        // c. 保存当前的 t 值，因为它决定了本层是否需要应用校正。
        u8 t_old = t;
        
        // d. 从 AES 输出中分离出新的种子 s_new 和新的控制位 t_new。
        s = ct & notOneBlock;
        t = lsb(ct);

        // e. 如果自己正处于“特殊路径”(t_old == 1)，则必须应用校正词(CW)
        //    来确保在非目标路径上，双方的状态能够恢复一致。
        if (t_old) {
            // 应用种子校正词 (s_cw)：通过异或操作，将自己的种子修正得和对方一样。
            s = s ^ _mm_loadu_si128(key.s + i + 1);
            
            // 应用控制位校正词 (t_cw)：提取对应路径和层级的 t_cw 比特，并进行异或。
            t = t ^ ((key.tcw[x_i] >> (bin - 1 - i)) & 1);
        }
    }

    // --- 3. 最终份额计算 ---
    
    // a. GGM树遍历结束，我们到达了叶子节点。's' 和 't' 是各自最终的状态。
    //    从128位的最终种子 s 中提取低64位作为数值部分。
    //    这个操作必须和 keyGenDPF 中计算 payload 的方式保持一致。
    GroupElement final_s_val = _mm_extract_epi64(s, 0);

    // b. 根据标准的 FSS 输出公式计算份额。
    //    这个计算是在 bitlength (64位) 的大环上进行的。
    //    t 的值 (0或1) 决定了 payload 校正词是否生效。
    GroupElement result = final_s_val + key.payload * t;
    
    // c. 应用 (-1)^party 因子。这是为了让两方的份额在相加时能够正确抵消或组合。
    //    对于无符号整数，取负等价于计算其在模 2^bitlength 下的加法逆元。
    if (party == 1) { // 对应 CLIENT
        result = -result;
    }
    
    
    // e. 返回这个干净的、在正确域内的最终份额。
    return result;
}

using namespace osuCrypto;

// 辅助函数：从打包的 t-CW 中提取特定层级的比特
inline uint8_t get_tcw_bit(const DPFKeyPack& key, int level, int direction) {
    // direction: 0 for left, 1 for right
    if (direction == 0) {
        return (key.tLcw >> (key.bin - 1 - level)) & 1;
    } else {
        return (key.tRcw >> (key.bin - 1 - level)) & 1;
    }
}

std::map<GroupElement, uint8_t>* compute_prefix_parities(
    int party,
    const DPFKeyPack& key,
    const GroupElement* endpoints,
    int num_endpoints)
{
    const block notOneBlock = toBlock(~0, ~1);
    const block pt[2] = {ZeroBlock, OneBlock};
    // --- 0. 准备工作 ---
    auto results = new std::map<GroupElement, uint8_t>();
    if (num_endpoints == 0) {
        return results;
    }

    // a. 为了排序和去重，我们使用 std::vector 作为临时容器
    std::vector<GroupElement> sorted_endpoints(endpoints, endpoints + num_endpoints);
    std::sort(sorted_endpoints.begin(), sorted_endpoints.end());
    sorted_endpoints.erase(std::unique(sorted_endpoints.begin(), sorted_endpoints.end()), sorted_endpoints.end());

    // --- 1. 初始化记忆化数据结构 (使用C-style数组) ---
    const int bin = key.bin;
    block* path = new block[bin + 1];
    uint8_t* parity = new uint8_t[bin + 1];

    // 初始化根节点状态
    path[0] = _mm_loadu_si128(key.s);
    parity[0] = (party - 2); // SERVER=0, CLIENT=1

    // --- 2. 遍历所有端点，利用记忆化进行计算 ---
    GroupElement prev_endpoint = -1; // 使用一个不可能的值初始化

    for (GroupElement current_endpoint : sorted_endpoints) {
        
        // a. 计算与上一个端点的最长公共前缀 (LCP) 长度
        int common_prefix_len = 0;
        if (prev_endpoint != -1) {
            // __builtin_clzll 是 GCC/Clang 内置函数，用于计算64位无符号整数前导零的数量
            // 64 - clz = 第一个不同比特的位置
            common_prefix_len = (bin > 63) ? 0 : 64 - __builtin_clzll(current_endpoint ^ prev_endpoint);
            if (common_prefix_len > bin) common_prefix_len = bin;
        }
        
        // b. 遍历非公共路径部分 (从分叉点到新的叶子节点)
        for (int i = common_prefix_len; i < bin; ++i) {
            
            block s_parent = path[i];
            uint8_t t_parent = parity[i];

            // i. GGM 扩展
            AES ak(s_parent);
            block children_ct[2];
            ak.ecbEncTwoBlocks(pt, children_ct);

            block s_left_raw = children_ct[0] & notOneBlock;
            uint8_t t_left_raw = lsb(children_ct[0]);
            block s_right_raw = children_ct[1] & notOneBlock;
            uint8_t t_right_raw = lsb(children_ct[1]);

            // ii. 应用校正词 (CW)
            if (t_parent == 1) {
                block scw = _mm_loadu_si128(key.s + i + 1);
                s_left_raw  ^= scw;
                s_right_raw ^= scw;
                t_left_raw  ^= get_tcw_bit(key, i, 0); // tLcw
                t_right_raw ^= get_tcw_bit(key, i, 1); // tRcw
            }
            
            // iii. 核心逻辑：根据当前端点的路径选择，并更新累积奇偶性
            const uint8_t current_direction = (current_endpoint >> (bin - 1 - i)) & 1;
            
            uint8_t running_parity = parity[i];
            if (current_direction == 1) { // 如果路径向右
                // 累加左兄弟子树的奇偶性
                running_parity ^= t_left_raw;
            }

            // iv. 记忆化：保存新路径节点的状态
            if (current_direction == 0) { // 向左走
                path[i + 1] = s_left_raw;
                parity[i + 1] = running_parity;
            } else { // 向右走
                path[i + 1] = s_right_raw;
                parity[i + 1] = running_parity;
            }
        }

        // c. 存储最终结果 (注意：Grotto的完整算法可能更复杂，这里是核心思想的实现)
        // 根据论文，前缀奇偶性是到达叶子节点前的累积值，不包括叶子本身
        // 但具体取决于叶子节点的处理方式。为简化，我们取遍历到最后一层的累积值。
        (*results)[current_endpoint] = parity[bin];

        prev_endpoint = current_endpoint;
    }

    // --- 3. 清理内存 ---
    delete[] path;
    delete[] parity;

    return results;
}