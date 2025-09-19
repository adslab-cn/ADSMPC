/*
    MPC框架中的输入处理模块
*/
#pragma once
#include "group_element.h"
#include <thread>
#include <chrono>

void input_prng_init(); // 伪随机数生成器初始化
void input_layer(GroupElement *x, GroupElement *x_mask, int size, int owner);// 输入数据秘密分享处理：处理输入数据（x）及其掩码（x_mask）

// 根据条件 condition 决定是否测量代码块 x 的执行时间，并将耗时累加到 accumulator（单位为微秒）
#define TIME_THIS_BLOCK_FOR_INPUT_IF(x, condition, accumulator) \
{\
    if (condition) {\
    auto start = std::chrono::high_resolution_clock::now();\
    x;\
    auto end = std::chrono::high_resolution_clock::now();\
    accumulator += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();\
    } else {\
        x;\
    }\
}
