#include <sytorch/backend/FSS_extended.h>
#include <sytorch/backend/FSS_transformer.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <FSS/utils.h>
#include <FSS/api.h>

int main(int __argc, char** __argv) {
    if (__argc < 2) {
        std::cerr << "Usage: ./mp_gcnconv <party_id> [ip]" << std::endl;
        return 1;
    }

    sytorch_init();
    int party = atoi(__argv[1]);
    std::string ip = (__argc > 2) ? __argv[2] : "127.0.0.1";

    using FSSVersion = FSSTransformer<u64>;
    FSSVersion *FSS = new FSSVersion();
    srand(time(NULL));

    FSSConfig::bitlength = 64;
    FSSConfig::party = party;
    FSSConfig::num_threads = 4;

    FSS->init(ip, true);

    // 图规模定义
    int numNodes = 19717;   // |V|
    int numEdges = 44338;   // |E|
    int inDim = 500;      // 原始节点特征长度
    int outDim = 256;       // 模拟第一层 GCN 后的隐藏层维度 (降维目标)
    // int numNodes = 19717;   // |V|
    // int numEdges = 44338;   // |E|
    // int inDim = 256;      // 原始节点特征长度
    // int outDim = 3;       // 模拟第一层 GCN 后的隐藏层维度 (降维目标)


    // int numNodes = 3327;   // |V|
    // int numEdges = 4732;   // |E|
    // int inDim = 3703;      // 原始节点特征长度
    // int outDim = 256;       // 模拟第一层 GCN 后的隐藏层维度 (降维目标)

    // int numNodes = 2708;   // |V|
    // int numEdges = 5429;   // |E|
    // int inDim = 1433;      // 原始节点特征长度
    // int outDim = 256;       // 模拟第一层 GCN 后的隐藏层维度 (降维目标)

    // int numNodes = 2708;   // |V|
    // int numEdges = 5429;   // |E|
    // int inDim = 256;      // 原始节点特征长度
    // int outDim = 7;       // 模拟第一层 GCN 后的隐藏层维度 (降维目标)

    Tensor<u64> A({(u64)numNodes, (u64)numNodes});
    Tensor<u64> F({(u64)numNodes, (u64)inDim});
    Tensor<u64> W({(u64)inDim, (u64)outDim});
    Tensor<u64> outF1({(u64)numNodes, (u64)inDim});
    Tensor<u64> outF2({(u64)numNodes, (u64)outDim});

    if(party == 3) { // CLIENT 端提供数据
        F.fill(1); 
        W.fill(1);
    }

    // 初始化掩码
    FSS->initializeInferencePartyB(F);
    FSS->initializeInferencePartyB(W);

    // 执行推理
    FSS::start();
    // GraphitiGCNConv(numNodes, numEdges, inDim, outDim, 
    //                 F.data, F.data,   // 数据及其掩码
    //                 W.data, W.data, 
    //                 outF.data, outF.data);
    MatMul2D(numNodes, numNodes, outDim, A.data, A.data, F.data, F.data, outF1.data, outF1.data, false);
    // MatMul2D(numNodes, inDim, outDim, F.data, F.data, W.data, W.data, outF2.data, outF2.data, false);
    FSS::end();

    // FSS->outputA(outF2);
    // if (party == 3) {
    //     std::cout << "SUCCESS: GCN output feature [0,0] = " << outF.data[0] << std::endl;
    // }

    FSS->finalize();
    return 0;
}