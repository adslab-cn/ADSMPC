/*
Takafumi Hoiruchi. 2018.
https://github.com/takafumihoriuchi/MNIST_for_C
*/


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <utility>

// 用于存储稀疏矩阵的非零元素
struct SparseMatrixElement {
    int row;
    int col;
    float value;
};

// 读取ind.cora.allx文件并返回稀疏矩阵的元素
std::vector<SparseMatrixElement> readSparseMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        exit(1);
    }

    std::vector<SparseMatrixElement> matrixElements;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int row, col;
        float value;
        if (iss >> row >> col >> value) {
            matrixElements.push_back({row, col, value});
        } else {
            std::cerr << "Error parsing line: " << line << std::endl;
        }
    }

    file.close();
    return matrixElements;
}

// 打印稀疏矩阵的元素（用于验证）
void printSparseMatrix(const std::vector<SparseMatrixElement>& matrixElements) {
    for (const auto& elem : matrixElements) {
        std::cout << "(" << elem.row << ", " << elem.col << ", " << elem.value << ")" << std::endl;
    }
}

int main() {
    std::string filename = "ind.cora.allx";
    auto matrixElements = readSparseMatrix(filename);

    // 打印稀疏矩阵的元素（可选）
    printSparseMatrix(matrixElements);

    // 根据需要处理稀疏矩阵元素，例如将其转换为CSR或CSC格式

    return 0;
}



// 这部分改为在预处理做：
//  计算规范化的邻接矩阵
//  创建一个全零矩阵作为邻接矩阵的初始状态
//  根据edge_index填充邻接矩阵，无向图因此两个方向都要填充
//  计算每个节点的度
//  计算度矩阵的逆平方根，用于后续的归一化
//  计算对称归一化的邻接矩阵：norm_adj