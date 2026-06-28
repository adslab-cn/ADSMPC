#pragma once

#include <backend/cleartext.h>
#include <backend/float.h>

template <typename T>
Backend<T>* defaultBackend()
{
    if constexpr (std::is_floating_point<T>::value) {
        return new FloatClearText<T>();
    } else {
        return new ClearText<T>();
    }
}
