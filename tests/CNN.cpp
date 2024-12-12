#define USE_CLEARTEXT
/*
CNN训练
    fptraining_init()：float point 初始化
    microbenchmark_conv(int party)：conv层测试
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
#include <filesystem>
// #include <cassert>
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



void microbenchmark_conv(int party) {
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
    const u64 bs = 20; //batch size

    auto conv = new Conv2D<u64>(64, 64, 5, 1);
    conv->doTruncationForward = false;
    auto model = Sequential<u64>({
        conv,
    });
    model.setBackend(FSS);

    FSS->initializeWeights(model); // dealer 初始化权重
    FSS::start();
    u64 imgSize = 64;
    model.init(bs, imgSize, imgSize, 64, scale);
    Tensor4D<u64> trainImages(bs, imgSize, imgSize, 64);
    // FSS->inputA(trainImages);
    model.forward(trainImages);
    FSS::end();

    FSS->outputA(conv->filter); 
    if (FSSConfig::party != 1) {
        conv->filter.print<i64>();
    }

    FSS->finalize();
}



// 用于将 CIFAR-10 数据集中的图像和标签填充到给定的张量（Tensor）中
// 调用函数前先分配好空间，函数的作用是将数据填到分配的空间里
template <typename T, u64 scale>
void cifar10_fill_images(Tensor4D<T>& trainImages, Tensor<u64> &trainLabels, int datasetOffset = 0) {
    int numImages = trainImages.d1; //图片数量
    assert(trainImages.d2 == 32);
    assert(trainImages.d3 == 32);
    assert(trainImages.d4 == 3);
    // 使用cifar::read_dataset函数读取CIFAR-10数据集。这个函数返回一个包含训练图像和训练标签的结构体
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(); 
    // 遍历指定数量的图像
    for(int b = 0; b < numImages; ++b) {
        for(u64 j = 0; j < 32; ++j) {
            for(u64 k = 0; k < 32; ++k) {
                for(u64 l = 0; l < 3; ++l) {
                    // 将读取的数据填入
                    // 将图像数据从[0, 255]的范围归一化到[0, 1]，然后乘以2^scale（通过左移scale位实现）
                    trainImages(b, j, k, l) = (T)((dataset.training_images[datasetOffset+b][j * 32 + k + l * 32 * 32] / 255.0) * (1LL << (scale)));
                }
            }
        }
        trainLabels(b) = dataset.training_labels[datasetOffset+b]; // 将每张图像的标签存储到trainLabels中
    }
}


void FSS_test_3layer(int party) {
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
    
    auto conv1 = new Conv2D<u64>(3, 64, 5, 1);
    auto conv2 = new Conv2D<u64>(64, 64, 5, 1);
    auto conv3 = new Conv2D<u64>(64, 64, 5, 1);
    auto fc1 = new FC<u64>(64, 10);
    auto model = Sequential<u64>({
        conv1,
        new ReLU<u64>(),
        new MaxPool2D<u64>(3, 0, 2),
        conv2,
        new ReLU<u64>(),
        new MaxPool2D<u64>(3, 0, 2),
        conv3,
        new ReLU<u64>(),
        new MaxPool2D<u64>(3, 0, 2),
        new Flatten<u64>(),
        fc1,
    });
    model.init(bs, 32, 32, 3, scale);
    model.setBackend(FSS);
    model.optimize();

    Tensor4D<u64> trainImages(bs, 32, 32, 3);
    Tensor<u64> trainLabels(bs);
    Tensor4D<u64> e(bs, 10, 1, 1);
    // trainImage.fill(1);

    FSS->initializeWeights(model); // dealer initializes the weights and sends to the parties
    FSS::start();

    int numIterations = 1; //迭代次数
    for(int i = 0; i < numIterations; ++i) {
        cifar10_fill_images<u64, scale>(trainImages, trainLabels, i * bs);
        FSS->inputA(trainImages);
        model.forward(trainImages);
        softmax_secfloat(model.activation, e, scale, party);
        if (party != 1) {
            for(int b = 0; b < bs; ++b) {
                e(b, trainLabels(b), 0, 0) -= (((1LL<<scale))/bs);
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

    microbenchmark_conv(party);
    // FSS_test_3layer(party);
}