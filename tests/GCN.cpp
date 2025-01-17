#define USE_CLEARTEXT
/*
GCN训练
    fptraining_init()：float point 初始化
    microbenchmark_conv(int party)：GCNConv层测试
    cifar10_fill_images(Tensor4D<T>& trainImages, Tensor<u64> &trainLabels, int datasetOffset = 0)：cifar10数据集处理
    FSS_test_3layer(int party)：三层的CNN网络的端到端训练
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <layers/layers.h>
#include <layers/softmax.h>
#include <networks.h>
#include <datasets/cifar10.h>
#include <datasets/graph_data.h>
#include <filesystem>
#include <Eigen/Dense>
#include <backend/FSS_extended.h>
#include <backend/FSS_improved.h>
#include <sequential.h>




void fptraining_init() {
    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));
    // set floating point precision
    // std::cout << std::fixed  << std::setprecision(1);
#ifdef NDEBUG
    std::cerr << "> Release Build" << std::endl;
#else
    std::cerr << "> Debug Build" << std::endl;
#endif
    std::cerr << "> Eigen will use " << Eigen::nbThreads() << " threads" << std::endl;
}


// 只用来测试GCNConv层能否运行以及正确性
void microbenchmark_GCNConv(int party) {
    // MPC参数定义
    using FSSVersion = FSSExtended<u64>;
    FSSVersion *FSS = new FSSVersion();
    srand(time(NULL));
    const u64 scale = 24;
    FSSConfig::bitlength = 64;
    FSSConfig::party = party;
    FSSConfig::stochasticT = true;
    FSSConfig::stochasticRT = true;
    FSSConfig::num_threads = 4;
    std::string ip = "127.0.0.1";
    FSS->init(ip, true); //初始化FSS设置，目前只支持两方，SERVER和CLIENT，以及一个第三方服务器DEALER
    const u64 bs = 1; // batch_size 在CNN结构中需要，在GCN中暂时不用，但为了统一结构仍然保留
    // bs是不是可以相当于节点数

    auto gcnconv = new GCNConv<u64>(3703, 64);
    auto fc1 = new FC<u64>(3703, 64);
    gcnconv->doTruncationForward = false; // 向前传播不进行截断
    auto model = Sequential<u64>({
        gcnconv,
        // fc1, 
    });

    model.setBackend(FSS); // 设置后端，还不太清楚代码细节

    FSS->initializeWeights(model); // dealer初始化权重
    FSS::start(); //api中的内容，初始化FSS各方
    u64 features_in = 3703;
    u64 features_out = 64;
    u64 num_nodes = 3327;
    u64 num_classes = 6;
    model.init(num_nodes, features_in, 1, 1, scale);
    Tensor4D<u64> X(num_nodes, features_in, 1, 1);
    // model.init(features_in, features_out, 1, 1, scale); // num_nodes, num_node_features, num_classes
    // Tensor4D<u64> X(num_nodes, features_in, 1, 1); // 给特征矩阵准备空间
    Tensor2D<u64> A_hat(num_nodes, num_nodes); // 给邻接矩阵准备空间
    gcnconv->setA(A_hat);
    // FSS->inputA(trainImages);
    model.forward(X);
    FSS::end(); //api中的内容，输出开销

    // FSS->outputA(gcnconv->activation); 
    // if (FSSConfig::party != 1) {
    //     gcnconv->activation.print<i64>();
    // }

    FSS->finalize(); // 结束，关闭连接
}

template <typename T, u64 scale>
void get_dataset(Tensor4D<T>& X, Tensor2D<T>& A, Tensor<u64> &labels) {
    // X: (num_nodes, features_in, 1, 1)
    int num_nodes = X.d1;
    assert(X.d2 == 3703); // features_in
    assert(X.d3 == 1);
    assert(X.d4 == 1);
    // 读取dataset
    auto dataset_X = readCSV("./CiteSeer/CiteSeer_X.csv"); // 特征矩阵X
    auto dataset_labels = readCSV("./CiteSeer/CiteSeer_labels.csv");  // 标签
    // 填充特征矩阵X: (num_nodes, features_in) 和 label: (num_nodes, 1)
    for(int b = 0; b < num_nodes; ++b) {
        for(u64 j = 0; j < 3703; ++j) {
            for(u64 k = 0; k < 1; ++k) {
                for(u64 l = 0; l < 1; ++l) {
                    X(b, j, k, l) = (T)((dataset_X[b][j]) * (1LL << (scale)));// 将读取的数据填入
                }
            }
        }
        labels(b) = dataset_labels[b][0];
    }
    auto dataset_A_hat = readCSV("./CiteSeer/CiteSeer_A_hat.csv"); // 邻接矩阵A_hat
    // 填充邻接矩阵A: (num_nodes, num_nodes)
    for(int b = 0; b < num_nodes; ++b) {
        for(u64 j = 0; j < num_nodes; ++j) {
            A(b, j) = (T)((dataset_A_hat[b][j]) * (1LL << (scale)));
        }
    }
}

void FSS_2layer_GCN(int party) {
    using FSSVersion = FSSExtended<u64>;
    FSSVersion *FSS = new FSSVersion();
    srand(time(NULL));
    const u64 scale = 24;
    FSSConfig::bitlength = 64;
    FSSConfig::party = party;
    FSSConfig::stochasticT = true;
    FSSConfig::stochasticRT = true;
    FSSConfig::num_threads = 4;
    std::string ip = "127.0.0.1";
    // std::string ip = "172.31.45.174";
    FSS->init(ip, true);
    if (party != 1) {
        secfloat_init(party - 1, ip);
    }
    const u64 bs = 2;
    
    auto gcnconv1 = new GCNConv<u64>(3703, 32);
    auto gcnconv2 = new GCNConv<u64>(32, 6);
    // auto fc1 = new FC<u64>(64, 10);
    auto model = Sequential<u64>({
        gcnconv1,
        new ReLU<u64>(),
        gcnconv2,
        new ReLU<u64>(), 
    });
    u64 features_in = 3703;
    u64 num_nodes = 3327;
    u64 num_classes = 6;
    model.init(num_nodes, features_in, 1, 1, scale);
    model.setBackend(FSS);
    model.optimize();

    Tensor4D<u64> X(3327, 3703, 1, 1);
    Tensor2D<u64> A_hat(3327, 3327);
    Tensor<u64> labels(3327);
    Tensor4D<u64> e(num_nodes, 6, 1, 1);
    // trainImage.fill(1);

    FSS->initializeWeights(model); // dealer initializes the weights and sends to the parties
    FSS::start();

    int numIterations = 1; //迭代次数
    for(int i = 0; i < numIterations; ++i) {
        get_dataset<u64, scale>(X, A_hat, labels);
        FSS->inputA(X);
        model.forward(X);
        softmax_secfloat(model.activation, e, scale, party);
        if (party != 1) {
            for(int b = 0; b < num_nodes; ++b) {
                e(b, labels(b), 0, 0) -= (((1LL<<scale))/num_nodes);
            }
        }
        model.backward(e);
    }
    FSS::end();
    
    FSS->finalize();
}



int main(int argc, char** argv) {
    fptraining_init();

    int party = 0;
    if (argc > 1) {
        party = atoi(argv[1]);
    }

    // microbenchmark_GCNConv(party);
    FSS_2layer_GCN(party);

    // std::string edgesFile = "./CiteSeer/CiteSeer_A_hat.csv";
    // auto edges = readCSV(edgesFile);
    // std::cout << edges[1][1] << std::endl;

}