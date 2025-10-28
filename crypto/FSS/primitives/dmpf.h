#pragma once
#include <vector>
#include <map>
#include <algorithm>
#include "cryptoTools/Common/Defines.h"
#include "cryptoTools/Crypto/PRNG.h"
#include "cryptoTools/Crypto/AES.h"
#include "../aux_parameter/group_element.h" // 假设group_element.h已存在

// 前向声明
struct BigStateDMPFKeyPack; 

// ===================================
//      辅助数据结构 (Helper Structs)
// ===================================

// 用于高效管理稀疏输入点的二叉前缀树
// 对应 Rust 中的 BinaryTrie
class BinaryTrie {
public:
    int depth;
    std::vector<GroupElement> words;
    std::map<std::pair<GroupElement, int>, std::pair<bool, bool>> prefixes;

    BinaryTrie(const std::vector<std::pair<GroupElement, GroupElement>>& inputs, int d);

    // 获取在指定深度的所有不重复的前缀
    std::vector<GroupElement> get_prefixes_at_depth(int d);
    
    // 检查一个前缀节点是否有左/右子节点
    std::pair<bool, bool> has_son(GroupElement prefix, int prefix_len);
};


// 表示一个 t_points 长度的比特向量，是 Big-State 的核心
// 对应 Rust 中的 Signs
struct BigStateSigns {
    int t_points;
    std::vector<oc::block> data;

    BigStateSigns(int t);
    BigStateSigns() : t_points(0) {}

    void set_bit(int idx, bool val);
    bool get_bit(int idx) const;
    void fill_with_seed(const oc::block& seed, int direction, oc::AES& aes);
    void xor_into(const BigStateSigns& a, const BigStateSigns& b);
    void zero();
};

// 仅在密钥生成时使用，表示一个 t_points x t_points 的布尔矩阵
// 对应 Rust 中的 KeyGenSigns
struct BigStateKeyGenSigns {
    int t_points;
    int nodes_per_point;
    std::vector<oc::block> data;

    BigStateKeyGenSigns(int t);
    void set_signs(int point_idx, const BigStateSigns& signs);
    void get_signs(int point_idx, BigStateSigns& signs) const;
    void set_bit(int point_idx, int bit_idx, bool val);
};

// 每一层树的纠正字
// 对应 Rust 中的 SignsCW
struct BigStateSignsCW {
    int t_points;
    int nodes_per_point_per_direction;
    std::vector<oc::block> data;

    BigStateSignsCW(int t, int depth, oc::PRNG& prng);
    size_t coordinates(bool direction, size_t point_idx) const;
    void put_sign(const BigStateSigns& signs, bool direction, size_t point_idx);
    void flip_bit(bool direction, size_t point_idx, size_t bit_idx);
};

// 对应 Rust 中的 CW
struct CW_Level {
    int t_points;
    std::vector<oc::block> seeds; // 种子纠正
    BigStateSignsCW signs_cw;      // "大状态"纠正

    CW_Level(int t, int d, oc::PRNG& prng) : t_points(t), signs_cw(t, d, prng) {}

    // 计算并应用纠正
    oc::block correct(const BigStateKeyGenSigns& signs_in, int point_idx, 
                      bool has_left, bool has_right,
                      BigStateSigns& signs_out_left, BigStateSigns& signs_out_right) const;
    
    oc::block correct_single(const BigStateSigns& signs_in, bool direction, BigStateSigns& signs_out) const;
};


// 最后一层的转换层纠正字
// 对应 Rust 中的 ConvCW
struct ConvCW_Level {
    std::vector<GroupElement> corrections;

    ConvCW_Level() = default;
    ConvCW_Level(
        const std::vector<std::pair<GroupElement, GroupElement>>& inputs,
        const std::vector<oc::block>& seed0, 
        const std::vector<oc::block>& seed1,
        const BigStateKeyGenSigns& signs0);

    GroupElement conv_correct(const BigStateSigns& final_signs) const;
};

// ===================================
//     Big-State DMPF 密钥包
// ===================================

struct BigStateDMPFKeyPack {
    int bin;
    int bout;
    int t_points;
    int party;

    oc::block root_seed;
    std::vector<CW_Level> cws;
    ConvCW_Level conv_cw;
    
    BigStateDMPFKeyPack(int b_in, int b_out, int t) :
        bin(b_in), bout(b_out), t_points(t), party(0) {}
    BigStateDMPFKeyPack() : bin(0), bout(0), t_points(0), party(0) {}
};


// ===================================
//        核心功能函数 (Core Functions)
// ===================================

// 密钥生成
std::pair<BigStateDMPFKeyPack, BigStateDMPFKeyPack> keyGenBigStateDMPF(
    int bin, int bout,
    const std::vector<std::pair<GroupElement, GroupElement>>& inputs);

// 单点求值
GroupElement evalBigStateDMPF(int party, BigStateDMPFKeyPack& key, GroupElement x);

// 全域求值
void evalAllBigStateDMPF(int party, BigStateDMPFKeyPack& key, GroupElement* out);