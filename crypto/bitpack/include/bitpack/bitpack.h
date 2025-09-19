// 提供比特级打包和解包的功能，用于高效存储和传输整数数据

#pragma once

#include <stdint.h>
#include <cstddef>

namespace bitpack {
    // 将x截断为bw位（取低bw位）
    uint64_t mod(uint64_t x, int bw); 
    // 计算打包n个bw位的整数需要多少个64位字
    std::size_t packed_size(std::size_t n, int bw); 
    // 一个内部辅助函数，专门处理1位整数的打包
    std::size_t pack(uint64_t *dst, const uint64_t *src, std::size_t n, int bw);
    // 将n个bw位的整数（在src数组中）打包到dst（64位数组）中。返回打包后的64位字数
    std::size_t unpack(uint64_t *dst, const uint64_t *src, std::size_t n, int bw);
};
