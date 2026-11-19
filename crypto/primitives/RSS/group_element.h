#pragma once
#include <crypto/primitives/ASS/group_element.h>
#include <cryptoTools/Crypto/PRNG.h>

#include <array>
#include <stdexcept>

// RSS 使用独立的三方编号；与 FSS 角色编号的映射由调用方负责，当前尚未实现。
enum class RssParty { P0 = 0, P1 = 1, P2 = 2 };

// P_i 持有 (x_i, x_(i+1))，下标对 3 取模；秘密为三个分量之和（模 2^bw）。
struct RssShare {
    GroupElement first = 0;
    GroupElement second = 0;
};

// 同样的循环布局，但秘密为三个分量的按位 XOR，表示 bw 个布尔值。
// 与 RssShare 不隐式转换；算术与布尔分享之间的转换需要另行实现协议。
struct RssXorShare {
    GroupElement first = 0;
    GroupElement second = 0;
};

namespace rss_detail {

inline void validateBitwidth(int bw)
{
    if (bw < 1 || bw > 64) {
        throw std::invalid_argument("RSS bit width must be in [1, 64]");
    }
}

inline void validateParty(RssParty party)
{
    switch (party) {
    case RssParty::P0:
    case RssParty::P1:
    case RssParty::P2:
        return;
    }
    throw std::invalid_argument("RSS party must be P0, P1 or P2");
}

} // namespace rss_detail

// 仅由知道 value 的输入持有方调用；ownerPrng 必须由调用方安全初始化。
// 返回数组按 P0、P1、P2 排列；网络分发尚未实现，每方只应获得自己的数组项。
inline std::array<RssShare, 3> splitShareRss(GroupElement value, int bw,
                                           osuCrypto::PRNG& ownerPrng)
{
    rss_detail::validateBitwidth(bw);
    GroupElement x0 = ownerPrng.get<GroupElement>();
    GroupElement x1 = ownerPrng.get<GroupElement>();
    mod(x0, bw);
    mod(x1, bw);

    GroupElement x2 = value - x0 - x1;
    mod(x2, bw);

    return {RssShare{x0, x1}, RssShare{x1, x2}, RssShare{x2, x0}};
}

// 公开常量按 (x0, x1, x2) = (value, 0, 0) 编码；各方独立计算，无需通信。
inline RssShare rssFromPublic(GroupElement value, RssParty party, int bw)
{
    rss_detail::validateBitwidth(bw);
    rss_detail::validateParty(party);
    mod(value, bw);

    if (party == RssParty::P0) return {value, 0};
    if (party == RssParty::P2) return {0, value};
    return {0, 0}; // P1
}

// 秘密加秘密：对应分量相加。
inline RssShare rssAdd(RssShare x, RssShare y, int bw)
{
    rss_detail::validateBitwidth(bw);
    x.first += y.first;
    x.second += y.second;
    mod(x.first, bw);
    mod(x.second, bw);
    return x;
}

// 秘密减秘密：对应分量相减。
inline RssShare rssSub(RssShare x, RssShare y, int bw)
{
    rss_detail::validateBitwidth(bw);
    x.first -= y.first;
    x.second -= y.second;
    mod(x.first, bw);
    mod(x.second, bw);
    return x;
}

// 秘密乘公开环元素；这不是秘密乘秘密协议。
inline RssShare rssMulPublic(RssShare x, GroupElement scalar, int bw)
{
    rss_detail::validateBitwidth(bw);
    x.first *= scalar;
    x.second *= scalar;
    mod(x.first, bw);
    mod(x.second, bw);
    return x;
}

// 仅执行本地重构。missing 必须为本方缺少的 x_(i+2)；
// 网络接收及一致性验证由调用方负责，当前尚未实现打开协议。
inline GroupElement rssCombine(RssShare local, GroupElement missing, int bw)
{
    rss_detail::validateBitwidth(bw);
    GroupElement value = local.first + local.second + missing;
    mod(value, bw);
    return value;
}

inline RssShare rssNeg(RssShare x, int bw)
{
    return rssSub(RssShare{}, x, bw);
}

// 公开常量只注入 x0 的两个副本，不能给所有分量都加上常量。
inline RssShare rssAddPublic(RssShare x, GroupElement value, RssParty party, int bw)
{
    return rssAdd(x, rssFromPublic(value, party, bw), bw);
}

inline RssShare rssSubPublic(RssShare x, GroupElement value, RssParty party, int bw)
{
    return rssSub(x, rssFromPublic(value, party, bw), bw);
}

// 布尔输入拆分；输入持有方及分发要求与 splitShareRss 相同。
inline std::array<RssXorShare, 3> splitShareRssXor(GroupElement value, int bw,
                                                 osuCrypto::PRNG& ownerPrng)
{
    rss_detail::validateBitwidth(bw);
    GroupElement x0 = ownerPrng.get<GroupElement>();
    GroupElement x1 = ownerPrng.get<GroupElement>();
    mod(x0, bw);
    mod(x1, bw);

    GroupElement x2 = value ^ x0 ^ x1;
    mod(x2, bw);

    return {RssXorShare{x0, x1}, RssXorShare{x1, x2}, RssXorShare{x2, x0}};
}

// 与算术公开值采用相同的 (value, 0, 0) 布局，但返回布尔份额。
inline RssXorShare rssFromPublicXor(GroupElement value, RssParty party, int bw)
{
    rss_detail::validateBitwidth(bw);
    rss_detail::validateParty(party);
    mod(value, bw);

    if (party == RssParty::P0) return {value, 0};
    if (party == RssParty::P2) return {0, value};
    return {0, 0}; // P1
}

inline RssXorShare rssXor(RssXorShare x, RssXorShare y, int bw)
{
    rss_detail::validateBitwidth(bw);
    x.first ^= y.first;
    x.second ^= y.second;
    mod(x.first, bw);
    mod(x.second, bw);
    return x;
}

inline RssXorShare rssXorPublic(RssXorShare x, GroupElement value, RssParty party, int bw)
{
    return rssXor(x, rssFromPublicXor(value, party, bw), bw);
}

// 与公开掩码按位 AND 可逐分量计算；秘密 AND 需要另行实现协议。
inline RssXorShare rssAndPublic(RssXorShare x, GroupElement mask, int bw)
{
    rss_detail::validateBitwidth(bw);
    x.first &= mask;
    x.second &= mask;
    mod(x.first, bw);
    mod(x.second, bw);
    return x;
}

// 通过公开值注入异或低 bw 位全一掩码，仅翻转有效位。
inline RssXorShare rssNot(RssXorShare x, RssParty party, int bw)
{
    return rssXorPublic(x, ~GroupElement{0}, party, bw);
}

// 仅执行本地 XOR 重构；missing、通信及一致性验证要求与 rssCombine 相同。
inline GroupElement rssCombineXor(RssXorShare local, GroupElement missing, int bw)
{
    rss_detail::validateBitwidth(bw);
    GroupElement value = local.first ^ local.second ^ missing;
    mod(value, bw);
    return value;
}

// 生成未知随机秘密的本地方份额，不用于拆分调用方指定的 value。
// 对 P_i：prevPrng 与 P_(i-1) 共享，生成 first = x_i；
//          nextPrng 与 P_(i+1) 共享，生成 second = x_(i+1)。
// 三条边必须使用独立随机流；每条边两端持有独立的同种子 PRNG 实例，
// 并以相同顺序消费。调用方负责初始化、种子分发、用途隔离和同步；
// 本文件不实现这些通信机制。输入拆分、算术及布尔随机份额应使用独立流。
inline RssShare randomShareRss(int bw, osuCrypto::PRNG& prevPrng,
                              osuCrypto::PRNG& nextPrng)
{
    rss_detail::validateBitwidth(bw);
    GroupElement first = prevPrng.get<GroupElement>();
    GroupElement second = nextPrng.get<GroupElement>();
    mod(first, bw);
    mod(second, bw);
    return {first, second};
}

// 与 randomShareRss 使用相同的边与同步约定；使用专门的布尔随机流。
inline RssXorShare randomShareRssXor(int bw, osuCrypto::PRNG& prevPrng,
                                    osuCrypto::PRNG& nextPrng)
{
    rss_detail::validateBitwidth(bw);
    GroupElement first = prevPrng.get<GroupElement>();
    GroupElement second = nextPrng.get<GroupElement>();
    mod(first, bw);
    mod(second, bw);
    return {first, second};
}
