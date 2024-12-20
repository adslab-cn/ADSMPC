#pragma once
#include <aux_parameter/group_element.h>
#include <thread>
#include <chrono>

void input_prng_init();
void input_layer(GroupElement *x, GroupElement *x_mask, int size, int owner);

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
