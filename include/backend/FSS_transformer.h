#pragma once
#include <backend/FSS_base.h>

template <typename T>
class FSSTransformer : public FSSBase<T>
{
public:
    void truncate(T *in, T *out, u64 shift, u64 size, u8 mode)
    {
        // ARS(size, in, in, out, out, shift);
        // SlothARS(size, in, out, shift);
        if (mode == 0)
        {
            SlothFaithfulARS(size, FSSConfig::bitlength, in, out, shift, "Linear::");
        }
        else if (mode == 1)
        {
            SlothARS(size, in, out, shift);
        }
        else
        {
            assert(0 && "Unknown truncate type");
        }
    }

    // ================== 新增 ReLU 实现 ==================
    void relu(Tensor<T> &in, Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode) override
    {
        // 1. 检查维度
        always_assert(in.size() == out.size());
        
        // SIGMA ReLU==============================================
        // 2. 确定有效位宽 (Bitwidth)
        // mode 0: 使用全位宽 (64)
        // mode 1: 使用缩减位宽 (64 - scale)
        int effective_bin = FSSConfig::bitlength;
        if (mode == 1) {
            effective_bin -= scale;
        }

        // 3. 调用框架自带的 SlothRelu
        // SlothRelu 实现了 FastSecNet 论文中的 ReLU 逻辑 (Approximate ReLU)
        SlothRelu(in.size(), effective_bin, in.data, out.data, "SIGMAReLU::");

        // FastSecNet ReLU==============================================
        // 2. 调用 FastSecNetRelu 协议
        // FastSecNetRelu(in.size(), // size: 元素总数
        //                in.data,   // Client/Server Input Share，inArr:  Client/Server 的输入份额 (in.data)
        //                out.data,  // Client/Server Output Share，outArr: Client/Server 的输出份额 (out.data)
        //                in.data,   // Dealer Input Mask，Dealer 的输入掩码 (in.data) (虽然 FastSecNet 内部生成 r，但为了接口兼容传入)
        //                out.data,  // Dealer Output Mask，outArr_mask: Dealer 的输出掩码 (FastSecNet 会将其设为 0)
        //                "FastSecNetReLU::"); // 统计前缀
        // 解释：
        // 在该框架中，Tensor.data 指针在不同 Party 下指向不同含义的内存：
        // - Party=Dealer: data 指向掩码 (Mask)
        // - Party=Server/Client: data 指向份额 (Share)
        // 因此，我们将 in.data 和 out.data 同时传给 Share 参数和 Mask 参数是安全的。
        // api.cpp 内部会根据 if(party == DEALER) 来决定使用哪个参数。
        // 注意：FastSecNetRelu 目前只负责计算 ReLU(x)，
        // 不计算导数 drelu。如果是推理任务，这没问题。


        // GT_ReLU==============================================
        // GT_ReLU(in.size(), in.data, out.data, "ReLU::");
        // GT_ReLU(in.size(), // size: 元素总数
        //                in.data,   // Client/Server Input Share，inArr:  Client/Server 的输入份额 (in.data)
        //                out.data,  // Client/Server Output Share，outArr: Client/Server 的输出份额 (out.data)
        //                in.data,   // Dealer Input Mask，Dealer 的输入掩码 (in.data) (虽然 FastSecNet 内部生成 r，但为了接口兼容传入)
        //                out.data,  // Dealer Output Mask，outArr_mask: Dealer 的输出掩码 (FastSecNet 会将其设为 0)
        //                "GTReLU::"); // 统计前缀
    }

    void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode) override
    {
        // 1. 检查 Tensor 维度一致性
        always_assert(in.d1 == out.d1); // Batch
        always_assert(in.d4 == out.d4); // Channel
        
        // 2. 准备参数
        u64 N = out.d1;
        u64 H_out = out.d2;
        u64 W_out = out.d3;
        u64 C = out.d4;
        
        // 3. 计算临时空间 oneHot 的大小
        u64 oneHotSize = (ks * ks - 1) * N * C * H_out * W_out;

        // 【修改点】：加上花括号 {}，将 oneHotSize 包装成 std::initializer_list
        Tensor<T> oneHot({oneHotSize});

        // 4. 调用 api.cpp 中的 MaxPool 协议
        MaxPool(N, H_out, W_out, C, 
                ks, ks, 
                padding, padding, padding, padding, 
                stride, stride, 
                in.d1, in.d2, in.d3, in.d4, 
                in.data, in.data,       // MASK_PAIR 逻辑：传入份额和掩码指针
                out.data, out.data,     // 同上
                oneHot.data,            // 比较结果缓冲区
                "MaxPool2D::");
    }

    void gelu(const Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode = 0)
    {
        u64 sz = in.size();
        always_assert(sz == out.size());
        if (mode == 0)
        {
            SlothGelu(sz, FSSConfig::bitlength, in.data, out.data, scale);
        }
        else if (mode == 1)
        {
            SlothGelu(sz, FSSConfig::bitlength - scale, in.data, out.data, scale);
        }
    }

    void silu(const Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode = 0)
    {
        u64 sz = in.size();
        always_assert(sz == out.size());
        if (mode == 0)
        {
            SlothSilu(sz, FSSConfig::bitlength, in.data, out.data, scale);
        }
        else if (mode == 1)
        {
            SlothSilu(sz, FSSConfig::bitlength - scale, in.data, out.data, scale);
        }
    }

    void softmax(Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode)
    {
        in.is_same_shape(out);
        if (mode == 0){
            Softmax(in.shape[0], in.shape[1], FSSConfig::bitlength, in.data, out.data, scale);
            // BPGCNSoftmax(in.shape[0], in.shape[1], 
            //      in.data, out.data, 
            //      in.data, out.data, // Dealer 掩码数组直接同传
            //      scale, "BPGCN_Softmax::");
        }
        else if (mode == 1)
            Softmax(in.shape[0], in.shape[1], FSSConfig::bitlength - scale + 1, in.data, out.data, scale);
    }

    void layernorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale)
    {
        always_assert(A.d1 == B.d1);
        always_assert(A.d1 == x.shape.back());
        always_assert(x.is_same_shape(y));
        u64 s2 = x.shape.back();
        u64 s1 = x.size() / s2;
        SlothLayerNorm(s1, s2, x.data, A.data, B.data, y.data, scale);
    }

    void rmsnorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale)
    {
        always_assert(A.d1 == B.d1);
        always_assert(A.d1 == x.shape.back());
        always_assert(x.is_same_shape(y));
        u64 s2 = x.shape.back();
        u64 s1 = x.size() / s2;
        SlothRMSNorm(s1, s2, x.data, A.data, B.data, y.data, scale);
    }

    void attention_mask(Tensor<T> &x, T scalar, Tensor<T> &y)
    {
        always_assert(x.is_same_shape(y));
        always_assert(x.shape.size() == 2);
        always_assert(x.shape[0] == x.shape[1]);

        if (FSSConfig::party == DEALER)
        {
            y.copy(x, false);
        }
        else
        {
            u64 n_seq = x.shape[0];
            auto y_2d = y.as_2d();
            auto x_2d = x.as_2d();

            for (u64 j = 0; j < n_seq; ++j)
            {
                for (u64 k = 0; k < j + 1; ++k)
                {
                    y_2d(j, k) = x_2d(j, k);
                }
                for (u64 k = j + 1; k < n_seq; ++k)
                {
                    y_2d(j, k) = x_2d(j, k) - scalar;
                }
            }
        }
    }

    void softmax_triangular(Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode)
    {
        in.is_same_shape(out);
        if (mode == 0)
            SoftmaxTriangular(in.shape[0], in.shape[1], FSSConfig::bitlength, in.data, out.data, scale);
        else if (mode == 1)
            SoftmaxTriangular(in.shape[0], in.shape[1], FSSConfig::bitlength - scale, in.data, out.data, scale);
    }

    void tanh(const Tensor<T> &in, const Tensor<T> &out, u64 scale)
    {
        Tanh(in.size(), in.data, out.data, scale);
    }

    void mul(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out)
    {
        always_assert(a.is_same_shape(b));
        always_assert(a.is_same_shape(out));
        // Mul(a.size(), a.data, b.data, out.data);
        ElemWiseMul(a.size(), a.data, b.data, out.data);
    }

    void doOptimizeGelu(LayerGraphNode<T> *node, LayerGraphNode<T> *root)
    {
        if (node->layer->doTruncationForward)
        {
            if (node->children.size() == 1)
            {
                LayerGraphNode<T> *child = node->children[0];
                if (child->layer->name == "GeLU")
                {
                    child->layer->mode = 1;
                }
            }
        }
    }

    void doOptimizeDiv(LayerGraphNode<T> *node, LayerGraphNode<T> *root)
    {
        if (node->layer->doTruncationForward)
        {
            if (node->children.size() == 1)
            {
                LayerGraphNode<T> *child = node->children[0];
                if (child->layer->name == "_ScalarDiv")
                {
                    auto layer_sd = (_ScalarDiv<T> *)child->layer;
                    T d = T(double(1LL << (layer_sd->scale)) / layer_sd->scalar);
                    // if d is power of two
                    if ((d & (d - 1)) == 0)
                    {
                        // seems very hacky
                        node->layer->scale += (layer_sd->scale - log2(d));
                        child->layer->mode = 1;
                    }
                }
            }
        }
    }

    void attention_triangular(Tensor2D<T> &q, Tensor2D<T> &k, Tensor2D<T> &v, Tensor2D<T> &out, u64 scale, u64 n_heads)
    {
        u64 n_seq = q.d1;
        u64 n_embd = q.d2;
        SlothAttentionTriangular(n_seq, n_embd, n_heads, q.data, k.data, v.data, out.data, scale);
    }

    void doOptimizeSoftmax(LayerGraphNode<T> *node, LayerGraphNode<T> *root)
    {
        if (node->layer->doTruncationForward || node->layer->name == "_ScalarDiv")
        {
            if (node->children.size() == 1)
            {
                LayerGraphNode<T> *child = node->children[0];
                if (child->layer->name == "SoftMax" || child->layer->name == "SoftMaxTriangular")
                {
                    child->layer->mode = 1;
                }
            }
        }
    }

    void optimize(LayerGraphNode<T> *root)
    {
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { doOptimizeGelu(n, r); });
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { doOptimizeSoftmax(n, r); });
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { doOptimizeDiv(n, r); });
    }

    void scalardiv(Tensor<T> &x, double scalar, Tensor<T> &y, u64 scale, u64 mode)
    {
        if (mode == 1)
        {
            y.copy(x, false);
        }
        else
        {
            T d = T(double(1LL << (scale)) / scalar);
            if ((d & (d - 1)) == 0)
            {
                SlothFaithfulARS(x.size(), FSSConfig::bitlength, x.data, y.data, scale - log2(d), "Linear::");
            }
            else
            {
                this->scalarmul(x, d, y);
                SlothFaithfulARS(y.size(), FSSConfig::bitlength, y.data, y.data, scale, "Linear::");
            }
        }
    }
};
