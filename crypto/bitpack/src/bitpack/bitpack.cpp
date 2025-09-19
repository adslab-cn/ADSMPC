// 提供比特级打包和解包的功能，用于高效存储和传输整数数据

#include <bitpack/bitpack.h>

namespace bitpack {
    /* 将 x 截断为 bw 位， 确保数值在 MPC 计算中不超出指定比特范围 */
    inline uint64_t mod(uint64_t x, int bw)
    {
        return x & ((1LL << bw) - 1); // 保留低 bw 位
    }

    /* 计算打包 n 个 bw 位整数所需的 64 位字数量，预先分配存储空间 */
    std::size_t packed_size(std::size_t n, int bw)
    {
        return (n * bw + 63) / 64;
    }

    /* bw = 1 时由pack函数调用 */
    std::size_t pack_1bit(uint64_t *dst, const uint64_t *src, std::size_t n)
    {
        for (int i = 0; i < n; ++i) // 遍历每个输入值
        {
            std::size_t dsti = i / 64;
            std::size_t dstj = i % 64;
            dst[dsti] |= ((src[i] & 1) << dstj);
        }
        return packed_size(n, 1);
    }

    /* 将 n 个 bw 位整数打包到连续的比特流中，packs `n` `bw` bit integers from `src` into `dst` */
    std::size_t pack(uint64_t *dst, const uint64_t *src, std::size_t n, int bw)
    {
        if (bw == 64) // bw=64：直接复制（无需压缩）
        {
            for (int i = 0; i < n; ++i)
            {
                dst[i] = src[i];
            }
            return n;
        }

        std::size_t ps = packed_size(n, bw);
        for (int i = 0; i < ps; ++i) // 初始化目标数组为 0
            dst[i] = 0; 

        if (bw == 1) // bw=1：调用 pack_1bit() 单比特特化版本
        {
            return pack_1bit(dst, src, n);
        }

        std::size_t dsti = 0;
        std::size_t dstj = 0;

        
            
        for (int i = 0; i < n; ++i)
        {
            uint64_t x = mod(src[i], bw);
            std::size_t rem = 64 - dstj;
            if (bw <= rem) // 若当前 64 位字有足够空间，写入低位并更新偏移
            {
                dst[dsti] |= x << dstj;
                dstj += bw;
                if (dstj == 64)
                {
                    dstj = 0;
                    ++dsti;
                }
            }
            else // 若空间不足，拆分值到两个连续字（跨字边界处理）
            {
                dst[dsti] |= x << dstj;
                dstj += bw;
                dst[dsti + 1] |= x >> rem;
                dstj -= 64;
                ++dsti;
            }
        }

        return dsti + (dstj > 0);
    }

    /* bw = 1 时由unpack函数调用 */
    std::size_t unpack_1bit(uint64_t *dst, const uint64_t *src, std::size_t n)
    {
        for (int i = 0; i < n; ++i)
        {
            std::size_t srci = i / 64;
            std::size_t srcj = i % 64;
            dst[i] = src[srci] >> srcj;
        }
        return packed_size(n, 1);
    }

    /* 从比特流恢复原始 n 个 bw 位整数，unpacks `n` `bw` bit integers from `src` into `dst` */
    std::size_t unpack(uint64_t *dst, const uint64_t *src, std::size_t n, int bw)
    {
        if (bw == 1) // bw=1：调用 unpack_1bit()
        {
            return unpack_1bit(dst, src, n);
        }
        
        if (bw == 64) // bw=64：直接复制
        {
            for (int i = 0; i < n; ++i)
            {
                dst[i] = src[i];
            }
            return n;
        }

        std::size_t srci = 0;
        std::size_t srcj = 0;
        uint64_t cache = src[0]; // 用 cache 缓存当前处理的 64 位字
        
        for (int i = 0; i < n; ++i) // 逐位输出
        {
            uint64_t x = cache >> srcj; // 从 cache 取可用低位部分
            std::size_t rem = 64 - srcj;
            if (bw <= rem) // 若剩余比特不足，加载新字取高位部分拼接
            {
                dst[i] = x;
                srcj += bw;
                if (srcj == 64)
                {
                    srcj = 0;
                    ++srci;
                    cache = src[srci];
                }
            }
            else 
            {
                ++srci;
                cache = src[srci];
                dst[i] = x | (cache << rem);
                srcj = srcj + bw - 64;
            }
        }

        return srci + (srcj > 0);
    }
};
