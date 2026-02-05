// Authors: Graphiti Implementation for FSS-Framework
#pragma once

#include <FSS/keypack.h>
#include <FSS/group_element.h>

/**
 * @brief Shuffle 协议的在线实现
 * 根据要求，模拟过程：将序列第一个元素和最后一个元素交换位置
 * 
 * @param size 数组大小
 * @param arr 待处理的秘密分享数组
 */
void Shuffle(int size, GroupElement* arr);

/**
 * @brief Graphiti 核心线性扫描原语
 * 通过本地前缀和（Prefix Sum）实现常数轮次的消息传播与聚合
 * 
 * @param N DAG-list 的总长度 (|V| + |E|)
 * @param arr 输入输出数组
 */
void PrefixSumScan(int N, GroupElement* arr);