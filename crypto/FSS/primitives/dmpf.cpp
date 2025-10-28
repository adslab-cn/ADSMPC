#include "dmpf.h"
#include <stdexcept>
#include <numeric>

using namespace osuCrypto;

// PRG 辅助函数
inline void expand_seed(AES& aes, const block& seed, block& child0, block& child1) {
    const static block pt[2] = { ZeroBlock, OneBlock };
    block cts[2];
    aes.setKey(seed);
    aes.ecbEncTwoBlocks(pt, cts);
    child0 = cts[0];
    child1 = cts[1];
}

// ===================================================================
//                  BinaryTrie 实现
// ===================================================================
BinaryTrie::BinaryTrie(const std::vector<std::pair<GroupElement, GroupElement>>& inputs, int d) : depth(d) {
    for(const auto& p : inputs) {
        words.push_back(p.first);
    }
    std::sort(words.begin(), words.end());

    for (const auto& w : words) {
        for (int i = 0; i < depth; ++i) {
            GroupElement prefix = w >> (depth - i);
            bool next_bit = (w >> (depth - i - 1)) & 1;
            auto& node = prefixes[{prefix, i}];
            if (next_bit) node.second = true;
            else node.first = true;
        }
    }
}

std::vector<GroupElement> BinaryTrie::get_prefixes_at_depth(int d) {
    std::vector<GroupElement> result;
    if (d == 0) {
        if (!words.empty()) result.push_back(0);
        return result;
    }
    for (const auto& w : words) {
        result.push_back(w >> (depth - d));
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::pair<bool, bool> BinaryTrie::has_son(GroupElement prefix, int prefix_len) {
    auto it = prefixes.find({prefix, prefix_len});
    if (it != prefixes.end()) {
        return it->second;
    }
    return {false, false};
}

// ===================================================================
//                  BigStateSigns 实现
// ===================================================================
BigStateSigns::BigStateSigns(int t) : t_points(t) {
    size_t num_blocks = (t + 127) / 128;
    data.resize(num_blocks, ZeroBlock);
}

void BigStateSigns::set_bit(int idx, bool val) {
    int block_idx = idx / 128;
    int bit_idx = idx % 128;
    if (val) {
        data[block_idx] = data[block_idx] | (OneBlock << bit_idx);
    } else {
        data[block_idx] = data[block_idx] & (~(OneBlock << bit_idx));
    }
}

bool BigStateSigns::get_bit(int idx) const {
    int block_idx = idx / 128;
    int bit_idx = idx % 128;
    return (_mm_cvtsi128_si64x(data[block_idx] >> bit_idx)) & 1;
}

// void BigStateSigns::fill_with_seed(const block& seed, int direction, AES& aes) {
//     // 使用不同的 IV 来为左右子节点生成不同的 signs
//     block iv = toBlock(direction);
//     aes.setKey(seed);
//     aes.ecbEncBlocks(iv, data.size(), data.data());
// }

// 修正后的 fill_with_seed 函数
void BigStateSigns::fill_with_seed(const block& seed, int direction, AES& aes) {
    aes.setKey(seed);

    // 为了生成伪随机流，我们需要加密一个计数器序列。
    // cryptoTools 的 ecbEncBlocks 需要一个明文块数组作为输入。
    std::vector<block> plaintexts(data.size());

    // 仿照 Rust 代码中的逻辑，为左右子节点 (direction=0/1) 
    // 使用不同的计数器起始点，以确保生成的伪随机流不重叠。
    // 一个安全的偏移量可以是 2 + data.size() * direction。
    // 这里的 '2' 是因为 DPF/DMPF 的种子扩展通常使用 0 和 1 作为基础 IV。
    u64 counter_start = 2 + data.size() * direction;
    
    for (size_t i = 0; i < data.size(); ++i) {
        plaintexts[i] = toBlock(counter_start + i);
    }

    // 使用明文数组作为输入，将加密结果（伪随机块）直接写入 data 缓冲区
    aes.ecbEncBlocks(plaintexts.data(), data.size(), data.data());
}

void BigStateSigns::xor_into(const BigStateSigns& a, const BigStateSigns& b) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = a.data[i] ^ b.data[i];
    }
}

void BigStateSigns::zero() {
    std::fill(data.begin(), data.end(), ZeroBlock);
}

// ===================================================================
//                  BigStateKeyGenSigns 实现
// ===================================================================
BigStateKeyGenSigns::BigStateKeyGenSigns(int t) : t_points(t) {
    nodes_per_point = (t + 127) / 128;
    data.resize(t * nodes_per_point, ZeroBlock);
}

void BigStateKeyGenSigns::set_signs(int point_idx, const BigStateSigns& signs) {
    size_t start = point_idx * nodes_per_point;
    for (int i = 0; i < nodes_per_point; ++i) {
        data[start + i] = signs.data[i];
    }
}

void BigStateKeyGenSigns::get_signs(int point_idx, BigStateSigns& signs) const {
    size_t start = point_idx * nodes_per_point;
    for (int i = 0; i < nodes_per_point; ++i) {
        signs.data[i] = data[start + i];
    }
}

void BigStateKeyGenSigns::set_bit(int point_idx, int bit_idx, bool val) {
    size_t start_node = point_idx * nodes_per_point;
    int block_idx = bit_idx / 128;
    int in_block_idx = bit_idx % 128;
    if (val) {
        data[start_node + block_idx] = data[start_node + block_idx] | (OneBlock << in_block_idx);
    } else {
        data[start_node + block_idx] = data[start_node + block_idx] & (~(OneBlock << in_block_idx));
    }
}


// ===================================================================
//                  BigStateSignsCW 实现
// ===================================================================
BigStateSignsCW::BigStateSignsCW(int t, int depth, PRNG& prng) : t_points(t) {
    nodes_per_point_per_direction = (t + 127) / 128;
    size_t points_at_depth = std::min((size_t)t, (size_t)1 << depth);
    data.resize(2 * points_at_depth * nodes_per_point_per_direction);
    prng.get(data.data(), data.size());
}

size_t BigStateSignsCW::coordinates(bool direction, size_t point_idx) const {
    return ((point_idx << 1) + (direction ? 1 : 0)) * nodes_per_point_per_direction;
}

void BigStateSignsCW::put_sign(const BigStateSigns& signs, bool direction, size_t point_idx) {
    size_t start = coordinates(direction, point_idx);
    for (int i = 0; i < nodes_per_point_per_direction; ++i) {
        data[start + i] = signs.data[i];
    }
}

void BigStateSignsCW::flip_bit(bool direction, size_t point_idx, size_t bit_idx) {
    size_t start_node = coordinates(direction, point_idx);
    int block_idx = bit_idx / 128;
    int in_block_idx = bit_idx % 128;
    data[start_node + block_idx] = data[start_node + block_idx] ^ (OneBlock << in_block_idx);
}


// ===================================================================
//                      CW_Level 实现
// ===================================================================
block CW_Level::correct(const BigStateKeyGenSigns& signs_in, int point_idx, 
                      bool has_left, bool has_right,
                      BigStateSigns& signs_out_left, BigStateSigns& signs_out_right) const 
{
    block s_cw = ZeroBlock;
    BigStateSigns temp_signs(t_points);
    signs_in.get_signs(point_idx, temp_signs);

    for (int i = 0; i < std::min(t_points, (int)seeds.size()); ++i) {
        if (temp_signs.get_bit(i)) {
            s_cw = s_cw ^ seeds[i];
            if (has_left) {
                size_t start = signs_cw.coordinates(false, i);
                for (int j = 0; j < signs_cw.nodes_per_point_per_direction; ++j) {
                    signs_out_left.data[j] = signs_out_left.data[j] ^ signs_cw.data[start + j];
                }
            }
            if (has_right) {
                size_t start = signs_cw.coordinates(true, i);
                 for (int j = 0; j < signs_cw.nodes_per_point_per_direction; ++j) {
                    signs_out_right.data[j] = signs_out_right.data[j] ^ signs_cw.data[start + j];
                }
            }
        }
    }
    return s_cw;
}

block CW_Level::correct_single(const BigStateSigns& signs_in, bool direction, BigStateSigns& signs_out) const {
    block s_cw = ZeroBlock;
    for (int i = 0; i < std::min(t_points, (int)seeds.size()); ++i) {
        if (signs_in.get_bit(i)) {
            s_cw = s_cw ^ seeds[i];
            size_t start = signs_cw.coordinates(direction, i);
            for (int j = 0; j < signs_cw.nodes_per_point_per_direction; ++j) {
                signs_out.data[j] = signs_out.data[j] ^ signs_cw.data[start + j];
            }
        }
    }
    return s_cw;
}

// ===================================================================
//                      ConvCW_Level 实现
// ===================================================================
ConvCW_Level::ConvCW_Level(
    const std::vector<std::pair<GroupElement, GroupElement>>& inputs,
    const std::vector<block>& seed0, 
    const std::vector<block>& seed1,
    const BigStateKeyGenSigns& signs0)
{
    int t = inputs.size();
    corrections.resize(t);
    BigStateSigns temp_signs(t);

    for(int i = 0; i < t; ++i) {
        GroupElement s0_val = _mm_extract_epi64(seed0[i], 0);
        GroupElement s1_val = _mm_extract_epi64(seed1[i], 0);
        GroupElement payload = inputs[i].second;

        GroupElement cw = s0_val - s1_val - payload;
        
        signs0.get_signs(i, temp_signs);
        if (temp_signs.get_bit(i)) {
            corrections[i] = -cw;
        } else {
            corrections[i] = cw;
        }
    }
}

GroupElement ConvCW_Level::conv_correct(const BigStateSigns& final_signs) const {
    GroupElement correction_sum = 0;
    for (int i = 0; i < corrections.size(); ++i) {
        if (final_signs.get_bit(i)) {
            correction_sum += corrections[i];
        }
    }
    return correction_sum;
}


// ===================================================================
//                      核心功能实现
// ===================================================================

std::pair<BigStateDMPFKeyPack, BigStateDMPFKeyPack> keyGenBigStateDMPF(
    int bin, int bout,
    const std::vector<std::pair<GroupElement, GroupElement>>& inputs)
{
    if (inputs.empty()) throw std::runtime_error("Inputs cannot be empty.");
    int t = inputs.size();
    PRNG prng(sysRandomSeed());
    AES aes;

    // 1. 初始化密钥包和状态
    BigStateDMPFKeyPack key0(bin, bout, t);
    BigStateDMPFKeyPack key1(bin, bout, t);
    key0.party = 0;
    key1.party = 1;

    key0.root_seed = prng.get<block>();
    key1.root_seed = prng.get<block>();

    BinaryTrie trie(inputs, bin);

    std::vector<block> seed0(t, ZeroBlock), seed1(t, ZeroBlock);
    std::vector<block> next_seed0(t), next_seed1(t);
    seed0[0] = key0.root_seed;
    seed1[0] = key1.root_seed;

    BigStateKeyGenSigns signs0(t), signs1(t);
    BigStateKeyGenSigns next_signs0(t), next_signs1(t);
    signs1.set_bit(0, 0, true); // 初始差异

    // 2. 逐层生成纠正字
    for (int depth = 0; depth < bin; ++depth) {
        auto prefixes = trie.get_prefixes_at_depth(depth);
        CW_Level cw(t, depth, prng);

        BigStateSigns new_signs0_left(t), new_signs0_right(t);
        BigStateSigns new_signs1_left(t), new_signs1_right(t);
        BigStateSigns delta_signs_left(t), delta_signs_right(t);
        
        std::vector<block> cw_seeds_temp;

        // 2.1. 为该层生成 CW
        size_t total_points_next = 0;
        for (size_t idx = 0; idx < prefixes.size(); ++idx) {
            auto prefix = prefixes[idx];
            auto sons = trie.has_son(prefix, depth);
            bool has_left = sons.first;
            bool has_right = sons.second;

            block s0_left, s0_right, s1_left, s1_right;
            expand_seed(aes, seed0[idx], s0_left, s0_right);
            expand_seed(aes, seed1[idx], s1_left, s1_right);
            
            new_signs0_left.fill_with_seed(seed0[idx], 0, aes);
            new_signs0_right.fill_with_seed(seed0[idx], 1, aes);
            new_signs1_left.fill_with_seed(seed1[idx], 0, aes);
            new_signs1_right.fill_with_seed(seed1[idx], 1, aes);

            delta_signs_left.xor_into(new_signs0_left, new_signs1_left);
            delta_signs_right.xor_into(new_signs0_right, new_signs1_right);
            
            cw.signs_cw.put_sign(delta_signs_left, false, idx);
            cw.signs_cw.put_sign(delta_signs_right, true, idx);

            if (has_left && has_right) cw_seeds_temp.push_back(prng.get<block>());
            else if (has_left) cw_seeds_temp.push_back(s0_right ^ s1_right);
            else cw_seeds_temp.push_back(s0_left ^ s1_left);

            if(has_left) cw.signs_cw.flip_bit(false, idx, total_points_next++);
            if(has_right) cw.signs_cw.flip_bit(true, idx, total_points_next++);
        }
        cw.seeds = cw_seeds_temp;
        
        // 2.2. 更新下一层状态
        size_t next_pos = 0;
        for (size_t idx = 0; idx < prefixes.size(); ++idx) {
             auto prefix = prefixes[idx];
            auto sons = trie.has_son(prefix, depth);
            bool has_left = sons.first;
            bool has_right = sons.second;

            block s0_left, s0_right, s1_left, s1_right;
            expand_seed(aes, seed0[idx], s0_left, s0_right);
            expand_seed(aes, seed1[idx], s1_left, s1_right);

            if (has_left) {
                 new_signs0_left.fill_with_seed(seed0[idx], 0, aes);
                 new_signs1_left.fill_with_seed(seed1[idx], 0, aes);
            }
            if(has_right){
                 new_signs0_right.fill_with_seed(seed0[idx], 1, aes);
                 new_signs1_right.fill_with_seed(seed1[idx], 1, aes);
            }

            block s_corr0 = cw.correct(signs0, idx, has_left, has_right, new_signs0_left, new_signs0_right);
            block s_corr1 = cw.correct(signs1, idx, has_left, has_right, new_signs1_left, new_signs1_right);

            if (has_left) {
                next_seed0[next_pos] = s0_left ^ s_corr0;
                next_seed1[next_pos] = s1_left ^ s_corr1;
                next_signs0.set_signs(next_pos, new_signs0_left);
                next_signs1.set_signs(next_pos, new_signs1_left);
                next_pos++;
            }
            if (has_right) {
                next_seed0[next_pos] = s0_right ^ s_corr0;
                next_seed1[next_pos] = s1_right ^ s_corr1;
                next_signs0.set_signs(next_pos, new_signs0_right);
                next_signs1.set_signs(next_pos, new_signs1_right);
                next_pos++;
            }
        }
        key0.cws.push_back(cw);
        key1.cws.push_back(cw);

        seed0 = next_seed0;
        seed1 = next_seed1;
        signs0 = next_signs0;
        signs1 = next_signs1;
    }

    // 3. 生成转换层CW
    key0.conv_cw = ConvCW_Level(inputs, seed0, seed1, signs0);
    key1.conv_cw = key0.conv_cw;

    return std::make_pair(key0, key1);
}


GroupElement evalBigStateDMPF(int party, BigStateDMPFKeyPack& key, GroupElement x) {
    AES aes;
    block seed = key.root_seed;
    BigStateSigns signs(key.t_points);
    BigStateSigns next_signs(key.t_points);
    signs.zero();
    if (party == 1) {
        signs.set_bit(0, true);
    }

    for (int i = 0; i < key.bin; ++i) {
        bool x_i = (x >> (key.bin - 1 - i)) & 1;
        
        block s_left, s_right;
        expand_seed(aes, seed, s_left, s_right);
        
        block next_seed_prg = x_i ? s_right : s_left;
        next_signs.fill_with_seed(seed, x_i, aes);
        
        block s_corr = key.cws[i].correct_single(signs, x_i, next_signs);
        
        seed = next_seed_prg ^ s_corr;
        signs = next_signs;
    }

    GroupElement final_val = _mm_extract_epi64(seed, 0);
    final_val += key.conv_cw.conv_correct(signs);

    if (party == 1) {
        return -final_val;
    }
    return final_val;
}

void evalAllBigStateDMPF_helper(
    int party, BigStateDMPFKeyPack& key, GroupElement* out,
    block seed, BigStateSigns& signs, int i, GroupElement acc, AES& aes)
{
    if (i == key.bin) {
        GroupElement val = _mm_extract_epi64(seed, 0);
        val += key.conv_cw.conv_correct(signs);
        if (party == 1) val = -val;
        out[acc] = val;
        return;
    }

    block s_left, s_right;
    expand_seed(aes, seed, s_left, s_right);

    BigStateSigns next_signs_left(key.t_points), next_signs_right(key.t_points);
    next_signs_left.fill_with_seed(seed, 0, aes);
    next_signs_right.fill_with_seed(seed, 1, aes);

    block s_corr_l = key.cws[i].correct_single(signs, false, next_signs_left);
    block s_corr_r = key.cws[i].correct_single(signs, true, next_signs_right);
    
    block next_seed_l = s_left ^ s_corr_l;
    block next_seed_r = s_right ^ s_corr_r;
    
    evalAllBigStateDMPF_helper(party, key, out, next_seed_l, next_signs_left, i + 1, acc << 1, aes);
    evalAllBigStateDMPF_helper(party, key, out, next_seed_r, next_signs_right, i + 1, (acc << 1) | 1, aes);
}


void evalAllBigStateDMPF(int party, BigStateDMPFKeyPack& key, GroupElement* out) {
    AES aes;
    block seed = key.root_seed;
    BigStateSigns signs(key.t_points);
    signs.zero();
    if (party == 1) {
        signs.set_bit(0, true);
    }
    evalAllBigStateDMPF_helper(party, key, out, seed, signs, 0, 0, aes);
}