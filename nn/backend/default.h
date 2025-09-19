// 选择默认后端
// 定义了一个模板函数 defaultBackend，用于根据数据类型 T 自动选择并实例化合适的计算后端


#pragma once

#include "cleartext.h"
#include "float.h"

template <typename T>
Backend<T>* defaultBackend()
{
    if constexpr (std::is_floating_point<T>::value) {
        return new FloatClearText<T>(); // 若 T 是浮点类型，则实例化 FloatClearText<T> 后端
    } else {
        return new ClearText<T>(); // 否则实例化 ClearText<T> 后端（通用整数或定点数计算）
    }
}
