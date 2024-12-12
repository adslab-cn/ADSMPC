/*
    Cora, CiteSeer, PubMed
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// 这部分相当于在预处理做：
//  计算规范化的邻接矩阵
//  创建一个全零矩阵作为邻接矩阵的初始状态
//  根据edge_index填充邻接矩阵，无向图因此两个方向都要填充
//  计算每个节点的度
//  计算度矩阵的逆平方根，用于后续的归一化
//  计算对称归一化的邻接矩阵：norm_adj

// 读取 CSV 文件并返回二维数组
std::vector<std::vector<double>> readCSV(const std::string& filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            // Convert string cell to double
            double cell_value = std::stod(cell);
            row.push_back(cell_value);
        }
        data.push_back(row);
    }
    return data;
}



