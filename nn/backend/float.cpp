#include "float.h"
#include <Eigen/Dense>

template <typename T>
void FloatClearText<T>::matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
    assert(a.d2 == b.d1);
    assert(c.d1 == a.d1);
    assert(c.d2 == b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
    eC = eA * eB;
}

template <typename T>
void FloatClearText<T>::matmul_triangular(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
    assert(a.d2 == b.d1);
    assert(c.d1 == a.d1);
    assert(c.d2 == b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
    eC = (eA * eB).template triangularView<Eigen::Lower>();
}

template <typename T>
void FloatClearText<T>::matmulTransposeA(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
    assert(a.d1 == b.d1);
    assert(c.d1 == a.d2);
    assert(c.d2 == b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eA(a.data, a.d2, a.d1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
    eC = eA * eB;
}

template <typename T>
void FloatClearText<T>::matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
    assert(a.d2 == b.d2);
    assert(c.d1 == a.d1);
    assert(c.d2 == b.d1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eB(b.data, b.d2, b.d1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
    eC = eA * eB;
}

template <typename T>
void FloatClearText<T>::conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output, bool isFirst)
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
void FloatClearText<T>::relu(const Tensor<T> &in, const Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode) {
    assert(in.is_same_shape(out));
    assert(in.is_same_shape(drelu));
    fastfor(in.size(), [&] (u64 i) {
        drelu.data[i] = (T)(in.data[i] > 0);
        out.data[i] = (in.data[i] > 0) ? in.data[i] : 0;
    });
}

template <typename T>
void FloatClearText<T>::truncate(T *in, T *out, u64 shift, u64 size, u8 mode) {
    always_assert(shift == 0);
    fastfor(size, [&] (u64 i) {
        out[i] = in[i];
    });
}

template <typename T>
void FloatClearText<T>::div(Tensor<T> &in, T divisor, u64 scale) {
    always_assert(scale == 0);

    fastfor(in.size(), [&] (u64 i) {
        in.data[i] /= divisor;
    });
}

template <typename T>
void FloatClearText<T>::div(T &in, T divisor, u64 scale) {
    in = in / divisor;
}


template <typename T>
void FloatClearText<T>::maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode) {
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
void FloatClearText<T>::add(const std::vector<Tensor<T> *> &in, Tensor<T> &out)
{
    always_assert(in.size() > 0);
    always_assert(out.size() == in[0]->size());
    for (int i = 0; i < in.size(); i++) {
        always_assert(out.size() == in[i]->size());
    }
    fastfor(out.size(), [&](int i) {
        T sum = 0;
        for (int j = 0; j < in.size(); j++) {
            sum += in[j]->data[i];
        }
        out.data[i] = sum;
    });
}


template <typename T>
void FloatClearText<T>::softmax(Tensor<T> &_in, Tensor<T> &_out, u64 scale, u64 mode)
{
    always_assert(_in.shape.size() == 2);
    always_assert(_out.shape.size() == 2);
    always_assert(_in.shape[0] == _out.shape[0]);
    always_assert(_in.shape[1] == _out.shape[1]);
    always_assert((scale == 0));

    auto in = _in.as_2d();
    auto out = _out.as_2d();

    auto batchSize = in.d1;
    auto numClasses = in.d2;
    for(int b = 0; b < batchSize; ++b) {
        T max = in(b, 0);
        for(u64 j = 1; j < numClasses; ++j) {
            if(in(b, j) > max) {
                max = in(b, j);
            }
        }

        double den = 0.0;
        double exps[numClasses];
        for(u64 j = 0; j < numClasses; ++j) {
            double x = in(b, j) - max;
            exps[j] = std::exp(x);
            den += exps[j];
        }

        for(u64 j = 0; j < numClasses; ++j) {
            out(b, j) = exps[j] / den;
        }
    }
}


template <typename T>
void FloatClearText<T>::layernorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale)
{
    always_assert(A.d1 == B.d1);
    always_assert(A.d1 == x.shape.back());
    always_assert(x.is_same_shape(y));
    
    u64 channels = x.shape.back();

    fastfor(x.size() / channels, [&](u64 i) {
        T mean = 0;
        T var = 0;
        for (u64 j = 0; j < channels; j++) {
            mean += x.data[i * channels + j];
        }
        mean = mean / T(channels);

        for (u64 j = 0; j < channels; j++) {
            var += (x.data[i * channels + j] - mean) * (x.data[i * channels + j] - mean);
        }

        var = var / T(channels);

        for (u64 j = 0; j < channels; j++) {
            y.data[i * channels + j] = (x.data[i * channels + j] - mean) / std::sqrt(var);
        }
    });

    fastfor(x.size(), [&](u64 i) {
        y.data[i] = y.data[i] * A(i % channels) + B(i % channels);
    });
}

template <typename T>
void FloatClearText<T>::addbias(Tensor<T> &x, const Tensor1D<T> &bias)
{
    always_assert(x.shape.back() == bias.d1);
    fastfor(x.size(), [&](u64 i) {
        x.data[i] += bias(i % bias.d1);
    });
}

template <typename T>
void FloatClearText<T>::scalarmul(Tensor<T> &x, T scalar, Tensor<T> &y)
{
    always_assert(x.is_same_shape(y));
    fastfor(x.size(), [&](u64 i) {
        y.data[i] = x.data[i] * scalar;
    });
}




template <typename T>
void FloatClearText<T>::scalardiv(Tensor<T> &in, double scalar, Tensor<T> &out, u64 scale, u64 mode)
{
    always_assert(scale == 0);
    fastfor(in.size(), [&](u64 i) {
        out.data[i] = in.data[i] / scalar;
    });
}

template class FloatClearText<double>;
template class FloatClearText<float>;
