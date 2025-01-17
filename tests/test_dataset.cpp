/*
    读取GCN数据集的测试
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <layers/layers.h>
#include <layers/softmax.h>
#include <networks.h>
#include <datasets/graph_data.h>
#include <filesystem>
// #include <cassert>
#include <Eigen/Dense>
#include <backend/FSS_extended.h>
#include <backend/FSS_improved.h>
#include <sequential.h>


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>



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

int main() {
    // 读取边信息
    std::string edgesFile = "./CiteSeer/CiteSeer_A_hat.csv";
    auto edges = readCSV(edgesFile);

    const u64 scale = 24;
    Tensor4D<u64> X(3327, 3703, 1, 1);
    Tensor2D<u64> A_hat(3327, 3327);
    Tensor<u64> labels(3327);
    get_dataset<u64, scale>(X, A_hat, labels);
    // 加个assert，检查是否正确读入数据
    std::cout << edges[1][1] << std::endl;
    // 打印边信息
    // for (const auto& row : edges) {
    //     for (const auto& cell : row) {
    //         std::cout << cell << " ";
    //     }
    //     std::cout << std::endl;
    // }
    return 0;
}