// 这是安全多方计算（MPC）框架的核心抽象后端的接口，
// 定义了MPC协议在张量计算层面需要的各种操作

#pragma once

#include "../tensor.h"
#include "../../crypto/FSS/api/api.h"
#include "../../crypto/FSS/aux_parameter/assert.h"

// 使用宏 NOT_IMPLEMENTED 强制子类实现具体协议逻辑   
#define NOT_IMPLEMENTED { \
        throw std::runtime_error("not implemented");\
}

template <typename T>
class Backend {
public:
    // truncation API：截断操作 (Truncation)​​，处理定点数精度调整
    // shift：比特位移量（控制精度）
    // mode：处理模式（不同协议实现可能不同）
    virtual void truncate(T *in, T *out, u64 shift, u64 size, u8 mode = 0) NOT_IMPLEMENTED;
    
    void truncate(const Tensor4D<T> &in, const Tensor4D<T> &out, u64 shift, u8 mode = 0) {
        always_assert(in.d1 == out.d1);
        always_assert(in.d2 == out.d2);
        always_assert(in.d3 == out.d3);
        always_assert(in.d4 == out.d4);
        truncate(in.data, out.data, shift, in.d1 * in.d2 * in.d3 * in.d4, mode);
    }
    
    void truncate(const Tensor4D<T> &in, u64 shift, u8 mode = 0) {
        truncate(in.data, in.data, shift, in.d1 * in.d2 * in.d3 * in.d4, mode);
    }

    virtual void truncateForward(const Tensor4D<T> &in, u64 shift, u8 mode = 0) {
        truncate(in.data, in.data, shift, in.d1 * in.d2 * in.d3 * in.d4, mode);
    }
    
    void truncate(const Tensor2D<T> &in, u64 shift, u8 mode = 0) {
        truncate(in.data, in.data, shift, in.d1 * in.d2, mode);
    }
    
    void truncate(const Tensor<T> &in, u64 shift, u8 mode = 0) {
        truncate(in.data, in.data, shift, in.size, mode);
    }

    void truncate(T &in, u64 shift, u8 mode = 0) {
        truncate(&in, &in, shift, 1, mode);
    }

    // matmul API 矩阵乘法
    virtual void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) NOT_IMPLEMENTED; // 2D*2D
    virtual void matmul(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c) NOT_IMPLEMENTED; // 4D*2D
    virtual void matmul(const Tensor2D<T> &a, const Tensor4D<T> &b, Tensor4D<T> &c) NOT_IMPLEMENTED; // 2D*4D
    virtual void matmulTransposeA(const Tensor4D<T> &a, const Tensor4D<T> &b, Tensor2D<T> &c) NOT_IMPLEMENTED; 
    virtual void matmulTransposeB(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c) NOT_IMPLEMENTED;
    virtual void matmulTransposeB(const Tensor2D<T> &a, const Tensor4D<T> &b, Tensor4D<T> &c) NOT_IMPLEMENTED;
    // virtual void matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) NOT_IMPLEMENTED;

    // conv API 卷积
    virtual void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output) NOT_IMPLEMENTED;
    
    // relu API 
    virtual void relutruncate(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 shift) NOT_IMPLEMENTED;
    virtual void relu(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 scale, int mode) NOT_IMPLEMENTED;
    virtual void select(const Tensor4D<T> &in, const Tensor4D<T> &drelu, const Tensor4D<T> &out) NOT_IMPLEMENTED;

    // maxpool API
    virtual void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode) NOT_IMPLEMENTED;;
    virtual void maxPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<u64> &maxIdx) NOT_IMPLEMENTED;;

    virtual void signext(Tensor4D<T> &x, u64 scale) NOT_IMPLEMENTED;

    virtual void optimize(LayerGraphNode<T> *root)
    {
        
    }

};
