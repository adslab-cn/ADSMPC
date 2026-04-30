#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>

#include <backend/FSS_transformer.h>
// #include <layers/layers.h>
#include <module.h>
#include <utils.h>

// 定义两层 CNN 模块
template <typename T>
class SimpleCNN : public SytorchModule<T> {
public:
    Conv2D<T> *conv1;
    ReLU<T> *relu1;
    MaxPool2D<T> *pool1;
    
    Conv2D<T> *conv2;
    ReLU<T> *relu2;
    MaxPool2D<T> *pool2;
    
    Flatten<T> *flatten;
    FC<T> *fc1;

    SimpleCNN() {
        // Layer 1: 28x28x1 -> 12x12x4
        conv1 = new Conv2D<T>(1, 4, 5, 0, 1, true); // useBias=true
        relu1 = new ReLU<T>();
        pool1 = new MaxPool2D<T>(2, 0, 2);

        // Layer 2: 12x12x4 -> 4x4x8
        conv2 = new Conv2D<T>(4, 8, 5, 0, 1, true);
        relu2 = new ReLU<T>();
        pool2 = new MaxPool2D<T>(2, 0, 2);

        // Layer 3: 4x4x8 -> 10
        flatten = new Flatten<T>();
        fc1 = new FC<T>(128, 10, true);
    }

    Tensor<T>& _forward(Tensor<T> &input) {
        auto &x1 = conv1->forward(input);
        auto &x2 = relu1->forward(x1);
        auto &x3 = pool1->forward(x2);
        
        auto &x4 = conv2->forward(x3);
        auto &x5 = relu2->forward(x4);
        auto &x6 = pool2->forward(x5);
        
        auto &x7 = flatten->forward(x6);
        auto &x8 = fc1->forward(x7);
        return x8;
    }
};

int main(int argc, char** argv) {
    // 基础初始化
    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));
    sytorch_init();

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <party>" << std::endl;
        return 1;
    }
    int party = atoi(argv[1]);
    std::string ip = "127.0.0.1";
    if (argc > 2) ip = argv[2];

    // 【修改点2】使用 FSSTransformer
    using FSSVersion = FSSTransformer<u64>;
    FSSVersion *backend = new FSSVersion();
    
    const u64 scale = 12;
    FSSConfig::bitlength = 64;
    FSSConfig::party = party;
    FSSConfig::num_threads = 4;
    
    // 初始化网络
    backend->init(ip, true);

    SimpleCNN<u64> net;
    net.init(scale);
    net.setBackend(backend);
    
    // 【修改点3】Optimize 对于 FSSTransformer 是可选的，但建议保留以兼容未来扩展
    // 对于 FSSImproved 则是必须的
    net.optimize();

    // 权重处理
    if (party == SERVER) {
        // net.load("weights.dat"); // 实际使用加载权重
    } else if (party == DEALER) {
        // Dealer 随机化权重在 initScale 中已经完成，或者调用 net.zero()
    }

    // 预处理 (Dealer/Server)
    backend->initializeInferencePartyA(net.root);

    // 输入处理
    // 使用通用的 Tensor<T> 构造函数
    Tensor<u64> input({1, 28, 28, 1}); 
    
    if (party == CLIENT) {
        // 模拟 Client 输入数据
        input.fill(1LL << scale); // 全 1
    }

    // 预处理 (Client 输入掩码)
    backend->initializeInferencePartyB(input);

    // 执行推理
    std::cout << ">> Starting Inference..." << std::endl;
    FSS::start();
    net.forward(input);
    FSS::end();
    std::cout << ">> Inference Done." << std::endl;

    // 输出结果
    auto &output = net.activation;
    backend->outputA(output);

    if (party == CLIENT) {
        std::cout << "Result:" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        for(int i = 0; i < 10; ++i) {
            double val = (double)((i64)output.data[i]) / (double)(1ULL << scale);
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    backend->finalize();
    delete backend;
    return 0;
}