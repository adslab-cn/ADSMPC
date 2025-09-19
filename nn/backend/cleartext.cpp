#include "cleartext.h"
#include <Eigen/Dense>

template <typename T>
void ClearText<T>::matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
    assert(a.d2 == b.d1);
    assert(c.d1 == a.d1);
    assert(c.d2 == b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
    eC = eA * eB;
}

template <typename T>
void ClearText<T>::matmul(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c) {
    assert(a.d2 == b.d1);
    assert(a.d3 == 1);
    assert(a.d4 == 1);
    assert(c.d1 == a.d1);
    assert(c.d2 == b.d2);
    assert(c.d3 == 1);
    assert(c.d4 == 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
    eC = eA * eB;
}

template <typename T>
void ClearText<T>::matmul(const Tensor2D<T> &a, const Tensor4D<T> &b, Tensor4D<T> &c) {
    assert(a.d2 == b.d1);
    assert(b.d3 == 1);
    assert(b.d4 == 1);
    assert(c.d1 == a.d1);
    assert(c.d2 == b.d2);
    assert(c.d3 == 1);
    assert(c.d4 == 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
    eC = eA * eB;
}

template <typename T>
void ClearText<T>::matmulTransposeA(const Tensor4D<T> &a, const Tensor4D<T> &b, Tensor2D<T> &c) {
    assert(a.d1 == b.d1);
    assert(a.d3 == 1);
    assert(a.d4 == 1);
    assert(b.d3 == 1);
    assert(b.d4 == 1);
    assert(c.d1 == a.d2);
    assert(c.d2 == b.d2);
//    c.zero();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eA(a.data, a.d2, a.d1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
    eC = eA * eB;
}

template <typename T>
void ClearText<T>::matmulTransposeB(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c) {
    assert(a.d2 == b.d2);
    assert(a.d3 == 1);
    assert(a.d4 == 1);
    assert(c.d1 == a.d1);
    assert(c.d2 == b.d1);
    assert(c.d3 == 1);
    assert(c.d4 == 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eB(b.data, b.d2, b.d1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
    eC = eA * eB;
}

// template <typename T>
// void ClearText<T>::matmulTransposeB(const Tensor2D<T> &a, const Tensor4D<T> &b, Tensor4D<T> &c) { // ....
//     assert(a.d2 == b.d2);
//     assert(b.d3 == 1);
//     assert(b.d4 == 1);
//     assert(c.d1 == a.d1);
//     assert(c.d2 == b.d1);
//     assert(c.d3 == 1);
//     assert(c.d4 == 1);
//     Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
//     Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eB(b.data, b.d2, b.d1);
//     Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
//     eC = eA * eB;
// }

template <typename T>
void ClearText<T>::matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
    assert(a.d2 == b.d2);
    assert(c.d1 == a.d1);
    assert(c.d2 == b.d1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eB(b.data, b.d2, b.d1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
    eC = eA * eB;
}

template <typename T>
void ClearText<T>::conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output)
{
    assert(input.d4 == ci);
    assert(filter.d1 == co);
    assert(filter.d2 == fh * fw * ci);
    u64 newH = (((input.d2 + 2*padding - fh)/stride) + 1);
    u64 newW = (((input.d3 + 2*padding - fw)/stride) + 1);
    assert(output.d1 == input.d1);
    assert(output.d2 == newH);
    assert(output.d3 == newW);
    assert(output.d4 == co);

    Tensor2D<T> reshapedInput = reshapeInputTransposed<T>(input, padding, stride, fh, fw);
    Tensor2D<T> tempOutput(filter.d1, reshapedInput.d1);
    matmulTransposeB(filter, reshapedInput, tempOutput);
    reshapeOutput<T>(tempOutput, input.d1, (((input.d2 + 2*padding - fh)/stride) + 1), (((input.d3 + 2*padding - fw)/stride) + 1), co, output);
}


template <typename T>
void ClearText<T>::relutruncate(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 shift) {
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(in.d3 == out.d3);
    assert(in.d4 == out.d4);
    assert(in.d1 == drelu.d1);
    assert(in.d2 == drelu.d2);
    assert(in.d3 == drelu.d3);
    assert(in.d4 == drelu.d4);
    fastfor(in.d1, [&] (u64 i) {
        for (u64 j = 0; j < in.d2; j++) {
            for (u64 k = 0; k < in.d3; k++) {
                for (u64 l = 0; l < in.d4; l++) {
                    drelu(i, j, k, l) = (T)(in(i, j, k, l) > 0);
                    if(localTruncationEmulation) {
                        if (drelu(i, j, k, l) == 0) {
                            out(i, j, k, l) = 0;
                        }
                        else {
                            u64 a = prngStr.get<T>();
                            u64 b = ((u64)in(i, j, k, l)) - a;
                            a = a >> shift;
                            b = -((-b) >> shift);
                            out(i, j, k, l) = a + b;
                        }
                        continue;
                    }
                    out(i, j, k, l) = (drelu(i, j, k, l) == 1) ? (in(i, j, k, l) / (1LL << shift)) : 0;
                    if (probablistic) {
                        u64 r = rand() % (1ULL << shift);
                        u64 x0 = ((u64)in(i, j, k, l)) % (1ULL << shift);
                        out(i, j, k, l) += (x0 < r ? 0 : 1); 
                    }
                }
            }
        }
    });
}

template <typename T>
void ClearText<T>::relu(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 scale, int mode) {
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(in.d3 == out.d3);
    assert(in.d4 == out.d4);
    assert(in.d1 == drelu.d1);
    assert(in.d2 == drelu.d2);
    assert(in.d3 == drelu.d3);
    assert(in.d4 == drelu.d4);
    fastfor(in.d1, [&] (u64 i) {
        for (u64 j = 0; j < in.d2; j++) {
            for (u64 k = 0; k < in.d3; k++) {
                for (u64 l = 0; l < in.d4; l++) {
                    drelu(i, j, k, l) = (T)(in(i, j, k, l) > 0);
                    assert(drelu(i, j, k, l) == 1 || drelu(i, j, k, l) == 0);
                    out(i, j, k, l) = (drelu(i, j, k, l) == 1) ? in(i, j, k, l) : 0;
                }
            }
        }
    });
}

template <typename T>
void ClearText<T>::select(const Tensor4D<T> &in, const Tensor4D<T> &drelu, const Tensor4D<T> &out) {
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(in.d3 == out.d3);
    assert(in.d4 == out.d4);
    assert(in.d1 == drelu.d1);
    assert(in.d2 == drelu.d2);
    assert(in.d3 == drelu.d3);
    assert(in.d4 == drelu.d4);
    fastfor(in.d1, [&] (u64 i) {
        for (u64 j = 0; j < in.d2; j++) {
            for (u64 k = 0; k < in.d3; k++) {
                for (u64 l = 0; l < in.d4; l++) {
                    assert(drelu(i, j, k, l) == 0 || drelu(i, j, k, l) == 1);
                    out(i, j, k, l) = (drelu(i, j, k, l) == 1) ? in(i, j, k, l) : 0;
                }
            }
        }
    });
}

template <typename T>
void ClearText<T>::truncate(T *in, T *out, u64 shift, u64 size, u8 mode) {
    fastfor(size, [&] (u64 i) {
        if constexpr (std::is_floating_point<T>::value) {
            out[i] = in[i] / ((T)(1ULL << shift));
        } else {
            if(localTruncationEmulation) {
                u64 a = prngStr.get<u64>();
                u64 b = ((u64)in[i]) - a;
                a = a >> shift;
                b = -((-b) >> shift);
                out[i] = a + b;
                return;
            }
            u64 x0 = ((u64)in[i]) % (1ULL << shift);
            in[i] = in[i] >> shift;
            if (probablistic) {
                u64 r = rand() % (1ULL << shift);
                out[i] += (x0 < r ? 0 : 1); 
            }
        }
    });
}

// template <typename T>
// void ClearText<T>::truncate(const Tensor4D<T> &in, const Tensor4D<T> &out, u64 shift) {
//     assert(in.d1 == out.d1);
//     assert(in.d2 == out.d2);
//     assert(in.d3 == out.d3);
//     assert(in.d4 == out.d4);
//     truncate(in.data, out.data, shift, in.d1 * in.d2 * in.d3 * in.d4);
// }

// template <typename T>
// void ClearText<T>::truncate(const Tensor4D<T> &in, u64 shift) {
//     // Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2 * in.d3 * in.d4);
//     // eA = eA / ((T)(1LL << shift));
//     truncate(in.data, in.data, shift, in.d1 * in.d2 * in.d3 * in.d4);
// }

// template <typename T>
// void ClearText<T>::truncate(const Tensor2D<T> &in, u64 shift) {
// //    Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2);
// //    eA = eA / ((T)(1LL << shift)); // this gives bad accuracy, why?
//     truncate(in.data, in.data, shift, in.d1 * in.d2);
// }

// template <typename T>
// void ClearText<T>::truncate(const Tensor<T> &in, u64 shift) {
// //    Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2);
// //    eA = eA / ((T)(1LL << shift)); // this gives bad accuracy, why?
//     truncate(in.data, in.data, shift, in.size);
// }

template <typename T>
void ClearText<T>::truncate(T &in, u64 shift) {
    if constexpr (std::is_floating_point<T>::value) {
        in = in / ((T)(1ULL << shift));
    } else {
        if(localTruncationEmulation) {
            u64 a = prngStr.get<T>();
            u64 b = ((u64)in) - a;
            a = a >> shift;
            b = -((-b) >> shift);
            in = a + b;
            return;
        }
        u64 x0 = ((u64)in) % (1ULL << shift);
        in = in >> shift;
        if (probablistic) {
            u64 r = rand() % (1ULL << shift);
            in += (x0 < r ? 0 : 1); 
        }
    }
}

template <typename T>
void ClearText<T>::div(const Tensor4D<T> &in, T divisor, u64 scale) {
    divPartial(in, divisor, scale);
    Backend<T>::truncate(in, scale);
}

template <typename T>
void ClearText<T>::divPartial(const Tensor4D<T> &in, T divisor, u64 scale) {
    T inv = (1LL << scale) / divisor;
    fastfor(in.d1, [&] (u64 i) {
        for (u64 j = 0; j < in.d2; j++) {
            for (u64 k = 0; k < in.d3; k++) {
                for (u64 l = 0; l < in.d4; l++) {
                    in(i, j, k, l) = in(i, j, k, l) * inv;
                }
            }
        }
    });
}


template <typename T>
u64 ClearText<T>::log2(u64 x) {
    u64 y = 0;
    while (x >>= 1) y++;
    return y;
}


template <typename T>
void ClearText<T>::maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode) {
    assert(in.d1 == out.d1);
    assert(in.d4 == out.d4);
    u64 newH = (in.d2 + 2*padding - ks)/stride + 1;
    u64 newW = (in.d3 + 2*padding - ks)/stride + 1;
    assert(out.d2 == newH);
    assert(out.d3 == newW);
    fastfor(in.d1, [&](int i) {
        for(int j = 0; j < newH; j++) {
            for(int k = 0; k < newW; k++) {
                for(int l = 0; l < in.d4; l++) {
                    T max = std::numeric_limits<T>::lowest();
                    u64 maxIdxI = 0;
                    u64 maxIdxJ = 0;
                    for(int m = 0; m < ks; m++) {
                        for(int n = 0; n < ks; n++) {
                            auto h2 = j*stride+m-padding;
                            auto w2 = k*stride+n-padding;
                            T val = 0;
                            if (h2 < in.d2 && w2 < in.d3 && h2 >= 0 && w2 >= 0)
                                val = in(i, h2, w2, l);
                            if(val > max) {
                                max = val;
                                maxIdxI = m;
                                maxIdxJ = n;
                            }
                        }
                    }
                    out(i, j, k, l) = max;
                    maxIdx(i, j, k, l) = maxIdxI * ks + maxIdxJ;
                }
            }
        }
    });
}


template <typename T>
Tensor2D<T> ClearText<T>::channelReshape(const Tensor4D<T> &x) {
    Tensor2D<T> res(x.d4, x.d1 * x.d2 * x.d3);
    for(int i = 0; i < x.d1; i++) {
        for(int j = 0; j < x.d2; j++) {
            for(int k = 0; k < x.d3; k++) {
                for(int l = 0; l < x.d4; l++) {
                    res(l, i*x.d2*x.d3 + j*x.d3 + k) = x(i, j, k, l);
                }
            }
        }
    }
    return res;
}



template class ClearText<i64>;
template class ClearText<i32>;
template class ClearText<u64>;
template class ClearText<double>;
template class ClearText<float>;
