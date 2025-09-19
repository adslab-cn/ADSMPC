// 实现明文环境下的安全计算模拟后端的实现
// ClearText 为本文的实现
// 以这个为主

#pragma once
#include "backend.h"
#include "../utils.h"
#include <thread>

template <typename T>
class ClearText : public Backend<T> {
private:
public:
    void truncate(T *in, T *out, u64 shift, u64 size, u8 mode);
    static const u64 lr_fp = 1;
    static const u64 lr_scale = 6;
    static const u64 mom_fp = 29;
    static const u64 mom_scale = 5;
    static const bool probablistic = false;
    static const bool localTruncationEmulation = false;
    static const bool numThreads = 120;

    template <typename Functor>
    void fastfor(u64 size, Functor f)
    {
        if (numThreads == 1) {
            for (u64 i = 0; i < size; i++) {
                f(i);
            }
        }
        else {
            std::thread threads[numThreads];
            u64 chunkSize = size / numThreads;
            for (u64 i = 0; i < numThreads - 1; i++) {
                threads[i] = std::thread([=, &f]() {
                    for (u64 j = i * chunkSize; j < (i + 1) * chunkSize; j++) {
                        f(j);
                    }
                });
            }
            threads[numThreads-1] = std::thread([=, &f]() {
                for (u64 j = (numThreads - 1) * chunkSize; j < size; j++) {
                    f(j);
                }
            });
        }
    }



    void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c);
    void matmul(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c);
    void matmul(const Tensor2D<T> &a, const Tensor4D<T> &b, Tensor4D<T> &c); //添加
    void matmulTransposeA(const Tensor4D<T> &a, const Tensor4D<T> &b, Tensor2D<T> &c);
    void matmulTransposeB(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c);
    void matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c);
    // void matmulTransposeB(const Tensor2D<T> &a, const Tensor4D<T> &b, Tensor4D<T> &c); //添加

    void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output);

    void relutruncate(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 shift);
    void relu(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 scale, int mode);
    void select(const Tensor4D<T> &in, const Tensor4D<T> &drelu, const Tensor4D<T> &out);
    // void truncate(const Tensor4D<T> &in, const Tensor4D<T> &out, u64 shift);
    // void truncate(const Tensor4D<T> &in, u64 shift);
    // void truncate(const Tensor2D<T> &in, u64 shift);
    // void truncate(const Tensor<T> &in, u64 shift);
    void truncate(T &in, u64 shift);
    void div(const Tensor4D<T> &in, T divisor, u64 scale);
    void divPartial(const Tensor4D<T> &in, T divisor, u64 scale);
    u64 log2(u64 x);

    void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode);
    Tensor2D<T> channelReshape(const Tensor4D<T> &x);    

};
