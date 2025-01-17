/*
    GCNConv层无法继承自layer类，只能重新设计
*/

#pragma once
#include <utils.h>
#include <aux_parameter/assert.h>
#include <backend/cleartext.h>
#include <string>

struct layer_dims_GNN {
    u64 h, w;

    u64 size() {
        return h * w;
    }
};


// GNN网络需要两个输入矩阵，按照 Layer 的方式重构 Maggage(消息传递函数)抽象类
// 矩阵都是Tensor2D, 但为了和框架统一，特征矩阵在输入时为Tensor4D
template <typename T>
class MessagePassing {
public:
    std::string name;
    Tensor4D<T> activation; // 计算结果(激活值)
    // Tensor2D<T> inputDerivative1; // 邻接矩阵
    Tensor4D<T> inputDerivative; // 用于存储当前层对输入的梯度
    bool doTruncationForward = false;
    bool doPreSignExtension = false;
    bool doPostSignExtension = false;
    bool isFirst = false;
    u64 scale = 0;
    Backend<T> *backend = nullptr;
    static bool fakeExecution;
    int mode = 0; // only used in ReLU in llama improved to decide between relu and reluext, might need something cleaner?
    int forwardTruncationMode = 0;
    LayerGraphNode<T> *node = nullptr;
    bool useBias = true;

    MessagePassing(const std::string &id) : activation(0,0,0,0), inputDerivative(0,0,0,0), name(id) {
        backend = new ClearText<T>();
    }
    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale) {
        initScale(scale);
        resize(d1, d2, d3, d4);
    }
    virtual void initScale(u64 scale) {};
    virtual void resize(u64 d1, u64 d2, u64 d3, u64 d4) = 0;
    virtual void forward_internal(Tensor2D<T> &a, Tensor4D<T> &b, bool train = true) = 0;
    Tensor4D<T>& forward(Tensor2D<T> &a, Tensor4D<T> &b, bool train = true) { 
        if (fakeExecution) {
            activation.graphNode = new LayerGraphNode<T>();
            node = activation.graphNode;
            activation.graphNode->layer = this;
            activation.graphNode->parents.push_back(a.graphNode);
            activation.graphNode->allNodesInExecutionOrderRef = a.graphNode->allNodesInExecutionOrderRef;
            activation.graphNode->allNodesInExecutionOrderRef->push_back(activation.graphNode);
            a.graphNode->children.push_back(activation.graphNode);
            layer_dims_GNN indims = {b.d1, b.d2};
            layer_dims_GNN outdims = this->get_output_dims(indims);
            activation.resize(outdims.h, outdims.w);
            inputDerivative.resize(b.d1, b.d2);
            return activation;
        }
        if (b.d1 != inputDerivative.d1 || b.d2 != inputDerivative.d2) {
            resize(b.d1, b.d2);
        }
        if (node != nullptr) {
            node->currTensor = &activation;
            activation.graphNode = node;
        }
        if (doPreSignExtension) {
            this->backend->signext(b, scale);
        }
        forward_internal(a, b, train);
        if (doTruncationForward) {
            this->backend->truncateForward(activation, scale, forwardTruncationMode);
        }
        if (doPostSignExtension) {
            this->backend->signext(activation, scale);
        }
        if (a.graphNode != nullptr) {
            bool gcHappened = b.graphNode->incrementAndGc();
            // if (gcHappened) {
            //     std::cerr << "Output of " << a.graphNode->layer->name << " cleared" << std::endl;
            // }
        }
        return activation;
    }
    virtual void backward(const Tensor4D<T> &e) = 0;
    virtual Tensor2D<T>& getweights() { throw std::runtime_error("not implemented"); };
    virtual Tensor<T>& getbias() { throw std::runtime_error("not implemented"); };
    virtual struct layer_dims_GNN get_output_dims(struct layer_dims_GNN &in) = 0;
    virtual void setBackend(Backend<T> *b) {
        backend = b;
    }
};



// 
template <typename T>
class GCNConv : public MessagePassing<T> {
public:
    Tensor2D<T> A_hat; // A_hat: 邻接矩阵 (num_nodes, num_nodes)
    Tensor2D<T> X; //X: 特征矩阵 (num_nodes, in)
    Tensor2D<T> weight; // weight: 权重张量 (in x out)
    Tensor2D<T> weightGrad; // 与权重张量相同维度的梯度张量weightGrad，用于存储权重的梯度
    Tensor2D<T> Vw; // 权重的动量（momentum）或速度（velocity），用于优化算法（如SGD）
    Tensor<T> bias; // 存储偏置项
    Tensor<T> Vb; // 偏置项的动量或速度
    u64 in, out; // in: 输入节点特征数, out: 输出节点特征数

    // 构造函数 
    GCNConv(u64 in, u64 out, bool useBias = false) : MessagePassing<T>("GCNConv"), 
        in(in), out(out), weight(in, out), bias(out), A_hat(0,0), X(0,0), weightGrad(in,out), Vw(in,out), Vb(out) {
        this->doTruncationForward = true; // 指示在前向传播中进行截断操作
        this->useBias = useBias; // 不使用偏置项
    }

    void initScale(u64 scale) {
        always_assert(std::is_integral<T>::value || scale == 0);
        this->scale = scale;  // 一个整数，缩放因子
        double xavier = 1.0 / sqrt(in);
        weight.randomize(xavier * (1ULL<<scale)); // 初始化和FC一样，只初始化权重weight
        if (this->useBias)
        bias.randomize(xavier * (1ULL<<(2*scale)));
        Vw.fill(0);
        Vb.fill(0);
    }

    //调整输入张量 X 和相关导数张量的大小
    void resize(u64 d1, u64 d2) {
        always_assert(d2 == in); 
        X.resize(d1, in);
        // this->inputDerivative.resize(d1, d1);
        this->inputDerivative.resize(d1, in);
        this->activation.resize(d1, out);
    }

    // 执行向前传播
    // 首先将输入 a 复制到 X，然后使用矩阵乘法计算输出激活值，并根据是否使用偏置项添加偏置。
    void forward_internal(Tensor2D<T> &A_hat, Tensor2D<T> &X, bool train = true) {
        this->X.copy(X); //将输入X复制到X，保存输入的副本，以便在后续的反向传播中使用
        Tensor2D<T> AX(A_hat.d1, X.d2);  // 用于存储 A_hat * X 的结果
        this->backend->matmul(A_hat, X, AX); // 先执行矩阵乘法：A_hat * X
        this->backend->matmul(AX, weight, this->activation); // 然后执行矩阵乘法：AX*W
        if (this->useBias) //不使用偏置项
        this->activation.addBias2D(bias);
    }

    // 因为邻接矩阵A从头到尾是不变的，因此不需要计算A的梯度，保持原来不变
    void backward(const Tensor2D<T> &e) {
        if (!(this->isFirst)) { // 如果当前层不是网络中的第一层，则需要计算对前一个层的梯度
            this->backend->matmulTransposeB(e, weight, this->inputDerivative);
            this->backend->truncate(this->inputDerivative, this->scale);
        }
        this->backend->matmulTransposeA(X, e, weightGrad);
        // this->backend->truncate(weightGrad, scale);
        Vw.resize(weightGrad.d1, weightGrad.d2);
        this->backend->updateWeight(weight, weightGrad, Vw, this->scale);
        if (this->useBias) //不使用偏置项
        this->backend->updateBias(bias, e, Vb, this->scale);
    }
    // void backward(const Tensor2D<T> &e) {
    //     Tensor2D<T> A_transpose_e(A_hat.d1, e.d2);  // 用于存储 A.T * e 的结果
    //     this->backend->matmulTranspose(A_hat, e, A_transpose_e);  // 假设 e 返回 (d1, out) 的二维张量

    //     // 如果当前层不是网络中的第一层（即，它有前一个层），则需要计算对前一个层的梯度（即输入导数）
    //     if (!(this->isFirst)) { // ===================================================================
    //         this->backend->matmulTransposeB(A_transpose_e, weight, this->inputDerivative); //再改一下
    //         this->backend->truncate(this->inputDerivative, this->scale);
    //     }

    //     Tensor2D<T> AX_transpose_e(A_hat.d2, e.d2);  // 临时变量用于存储 (A * X).T * e 的结果，但实际上我们不需要显式计算它
    //     // 直接计算 weightGrad，因为 (A * X).T * e = A.T * (X.T * e) 在这里 X.T * e 就是 e 的原始形状（已经是二维）
    //     this->backend->matmulTransposeA(A_hat, e, weightGrad);
    //     // this->backend->truncate(weightGrad, scale);
    //     Vw.resize(weightGrad.d1, weightGrad.d2);
    //     this->backend->updateWeight(weight, weightGrad, Vw, this->scale);

    //     if (this->useBias) //不使用偏置项
    //     this->backend->updateBias(bias, e, Vb, this->scale);
    // }
    
    Tensor2D<T>& getweights() { return weight; }
    Tensor<T>& getbias() { return bias; }

    // 根据输入维度计算并返回输出维度
    struct layer_dims_GNN get_output_dims(struct layer_dims_GNN &in) {
        return {in.h, out};
    }
};