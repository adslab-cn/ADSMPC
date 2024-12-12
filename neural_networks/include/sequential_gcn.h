#pragma once
/*
    GCN的Sequential，需要输入两个矩阵
*/
#include <layers/layer_gcn.h>
#include <aux_parameter/assert.h>

template <typename T>
class Sequential_GCN {
public:
    std::vector<MessagePassing<T>*> layers; // 存储 MessagePassing<T>* 的向量，表示图网络模型中所有的层
    Tensor2D<T> activation; // 存储模型的最终激活输出 Tensor2D<T> 
    Backend<T> *backend = new ClearText<T>(); // 指向 Backend<T> 的指针，用于执行特定的后端操作
    LayerGraphNode<T> *root = nullptr; // 指向 LayerGraphNode<T> 的指针，代表图结构的根节点，用于表示层的执行顺序和依赖关系。
    std::vector<LayerGraphNode<T> *> allNodesInExecutionOrder; // 存储 LayerGraphNode<T>* 的向量，包含所有层节点，按照执行顺序排列

    // 初始化scale
    void initScale(u64 scale) {
        always_assert(std::is_integral<T>::value || scale == 0);
        for(auto &l : layers) {
            l->initScale(scale);
        }
    }

    // 改变矩阵形状，(bs, num_nodes, features, num_classes)
    void resize(u64 d1, u64 d2) {
        for(auto &l : layers) {
            l->resize(d1, d2);
            d1 = l->activation.d1;
            d2 = l->activation.d2;
        }
        this->activation.resize(d1, d2);
    }


    void genGraph()
    {
        Tensor2D<T> ip(0, 0); // （Input Placeholder）
        Tensor2D<T> A(0,0);
        ip.graphNode = new LayerGraphNode<T>();
        ip.graphNode->layer = new PlaceHolderLayer<T>("Input");
        ip.graphNode->allNodesInExecutionOrderRef = &allNodesInExecutionOrder;
        Layer<T>::fakeExecution = true;
        auto &res = this->_forward(A, ip); //这里需要两个输入
        Layer<T>::fakeExecution = false;
        root = ip.graphNode;
    }

    // 初始化
    void init(u64 d1, u64 d2, u64 scale)
    {
        initScale(scale);
        resize(d1, d2);
        genGraph();
    }

    void optimize()
    {
        backend->optimize(root);
    }

    Sequential_GCN(std::vector<MessagePassing<T>*> _layers) : layers(_layers), activation(0, 0) {
        int s = layers.size();
        // Set isFirst
        for(int i = 0; i < s; ++i) {
            if (layers[i]->name == "GCNConv" || layers[i]->name == "FC") {
                layers[i]->isFirst = true;
                break;
            }
        }
        
        // Optimization: ReLU-MaxPool
        for(int i = 0; i < s - 1; i++) {
            if (layers[i+1]->name == "MaxPool2D") {
                auto &n = layers[i]->name;
                if (n == "ReLU") {
                    std::swap(layers[i], layers[i+1]);
                }
            }
        }
    }
    
    // _forward负责向前传播
    Tensor2D<T>& _forward(Tensor2D<T> &A_hat, Tensor2D<T> &X, bool train = true) {
        layers[0]->forward(A_hat, X, train);
        u64 size = layers.size();
        for(u64 i = 1; i < size; i++) {
            layers[i]->forward(A_hat, layers[i-1]->activation, train);
        }
        return A_hat, layers[size-1]->activation; // 需不需要返回两个矩阵
    }

    // 调用_forward, 并将结果保存在activation
    Tensor2D<T>& forward(Tensor2D<T> &A_hat, Tensor2D<T> &X, bool train = true) {
        auto& res = this->_forward(A_hat, X, train);
        this->activation.resize(res.d1, res.d2);
        this->activation.copy(res); 
        return A_hat, this->activation;  // 需不需要返回两个矩阵
    }

    // 反向传播, A_hat不影响反向传播
    void backward(const Tensor2D<T> &e) {
        int size = layers.size();
        layers[size-1]->backward(e);
        for (int i = size - 2; i >= 0; i--) {
            layers[i]->backward(layers[i+1]->inputDerivative);
        }
    }

    // 获取输出维度
    struct layer_dims_GNN get_output_dims(struct layer_dims_GNN &in) {
        struct layer_dims_GNN res = in;
        u64 size = layers.size();
        for(u64 i = 0; i < size; i++) {
            res = layers[i]->get_output_dims(res);
        }
        return res;
    }

    virtual void setBackend(Backend<T> *b) {
        for(auto &l : layers) {
            l->setBackend(b);
        }
        this->backend = b;
    }
};