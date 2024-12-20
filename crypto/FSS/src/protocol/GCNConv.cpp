// /*
//     GCNConv层密钥生成和评估细节
// */

// #include "protocol/GCNConv.h"
// #include "protocol/conv.h" //引入矩阵乘法
// #include <aux_parameter/array.h>
// #include <aux_parameter/comms.h>
// #include <aux_parameter/utils.h>
// #include <assert.h>


// // 这部分改为在预处理做：
// //  计算规范化的邻接矩阵
// //  创建一个全零矩阵作为邻接矩阵的初始状态
// //  根据edge_index填充邻接矩阵，无向图因此两个方向都要填充
// //  计算每个节点的度
// //  计算度矩阵的逆平方根，用于后续的归一化
// //  计算对称归一化的邻接矩阵：norm_adj

// // 调用矩阵乘法的密钥生成函数
// std::pair<GCNConvKey, GCNConvKey> KeyGenGCNConv2D(
//     int Bin, int Bout, //输入输出
//     int num_nodes, int in_features, int num_edges, //N:节点数，f:特征数，e:边数
//     int out_features, //输出特征维度
//     GroupElement *rin1,  GroupElement * rin2, GroupElement * rin3, GroupElement * rout 
//     //输入数据-节点特征矩阵、输入数据-边矩阵、输出数据-节点特征矩阵
//     )
// {
//     // 初始化两个密钥对象，用于分享
//     GCNConvKey k0;
//     GCNConvKey k1;
//     k0.Bin = Bin;                               k1.Bin = Bin;
//     k0.Bout = Bout;                             k1.Bout = Bout;
//     k0.num_node = num_nodes;                    k1.num_node = num_nodes;
//     k0.in_node_features = in_features;          k1.in_node_features = in_features;
//     k0.num_edge = num_edges;                    k1.num_edge = num_edges;
//     k0.out_node_features = out_features;        k1.out_node_features = out_features;

//     // 计算输出维度d1和d2: x特征维度改变, edge_index不变
//     int d0 = num_nodes;
//     int d1 = in_features;
//     int d2 = out_features;
//     // 给输入和输出分配空间
//     // k0.a = make_array<GroupElement>(d0, d1); //节点特征矩阵x: 大小为[num_nodes, in_features]
//     // k1.a = make_array<GroupElement>(d0, d1); 
//     // k0.b = make_array<GroupElement>(num_nodes, num_nodes); //边索引edge_index: 大小为[2, num_edge]
//     // k1.b = make_array<GroupElement>(num_nodes, num_nodes);
//     // k0.c = make_array<GroupElement>(d0, d2); //输出节点特征矩阵x: 大小为[num_nodes, out_features]
//     // k1.c = make_array<GroupElement>(d0, d2);

//     // 需要两次矩阵乘法
//     auto keys = KeyGenMatMul(bitlength, bitlength, s1, s2, s3, A_mask, B_mask, C_mask);
// }



// void EvalGCNConv(
//     int party, const GCNConvKey &key,
//     int num_nodes, int in_features, int num_edges, //N:节点数，f:特征数，e:边数
//     int out_features, //输出特征维度
//     GroupElement* input1, GroupElement* input2, GroupElement* output)
// {
//     int d0 = num_nodes;
//     int d1 = in_features;
//     int d2 = out_features;

//     // 为GCNConv操作分配缓存
//     GroupElement* cache = make_array<GroupElement>(d0, d2);

//     GroupElement* weight = make_array<GroupElement>(in_features, out_features); 
//     // 需要临时数组temp，因为matmul不能在本地完成
//     GroupElement* c = make_array<GroupElement>(num_nodes * out_features);

//     // 复制向量
//     MatCopy4(d0, d1, d2, d3, key.c, output);

//     // SERVER和CLIENT的操作
//     if (party == SERVER)
//     {
//         GroupElement *tempFilter = make_array<GroupElement>(FH, FW, CI, CO);
//         MatSub(num_nodes, in_features, input1, key.a, tempFilter); //减去 key.b

//         MatMul(num_nodes, in_features, out_features, input1, weight, c);
//         MatMul(num_nodes, num_nodes, out_features, input2, c, output);

//         MatAdd(num_nodes, out_features, cache, output, output); //结果与输出数据相加
//         delete[] tempFilter;
//     }
//     else
//     {
//         MatMul(num_nodes, in_features, out_features, input1, weight, c);
//         MatMul(num_nodes, num_nodes, out_features, input2, c, output);
//         MatSub(num_nodes, out_features, output, cache, output); //从输出数据中减去卷积结果
//     }

//     MatMul(num_nodes, in_features, out_features, input1, weight, c);
//     MatMul(num_nodes, num_nodes, out_features, input2, c, output);

//     MatSub4(num_nodes, out_features, output, cache, output);

//     delete[] cache; //释放缓存
// }







// // std::pair<GCNConvKey, GCNConvKey> KeyGenGCNConv2D(
// //     int Bin, int Bout, //输入输出
// //     // int N, //batch size
// //     int num_nodes, int in_features, int num_edges, //N:节点数，f:特征数，e:边数
// //     int out_features, //输出特征维度
// //     // 输入部分可能还需要一个weight
// //     GroupElement *rin1,  GroupElement * rin2, GroupElement * rout)
// // {
// //     // 初始化两个密钥对象，用于分享
// //     GCNConvKey k0;
// //     GCNConvKey k1;
// //     k0.Bin = Bin;                               k1.Bin = Bin;
// //     k0.Bout = Bout;                             k1.Bout = Bout;
// //     k0.num_node = num_nodes;                    k1.num_node = num_nodes;
// //     k0.in_node_features = in_features;          k1.in_node_features = in_features;
// //     k0.num_edge = num_edges;                    k1.num_edge = num_edges;
// //     k0.out_node_features = out_features;        k1.out_node_features = out_features;

// //     // 计算输出维度d1和d2: x特征维度改变, edge_index不变
// //     // int d0 = N;
// //     int d0 = num_nodes;
// //     int d1 = in_features;
// //     int d2 = out_features;
// //     // 给输入和输出分配空间
// //     k0.a = make_array<GroupElement>(d0, d1); //节点特征矩阵x: 大小为[num_nodes, in_features]
// //     k1.a = make_array<GroupElement>(d0, d1); 
// //     k0.b = make_array<GroupElement>(num_nodes, num_nodes); //边索引edge_index: 大小为[2, num_edge]
// //     k1.b = make_array<GroupElement>(num_nodes, num_nodes);
// //     k0.c = make_array<GroupElement>(d0, d2); //输出节点特征矩阵x: 大小为[num_nodes, out_features]
// //     k1.c = make_array<GroupElement>(d0, d2);

// //     // 这部分改为在预处理做：
// //     //  计算规范化的邻接矩阵
// //     //  创建一个全零矩阵作为邻接矩阵的初始状态
// //     //  根据edge_index填充邻接矩阵，无向图因此两个方向都要填充
// //     //  计算每个节点的度
// //     //  计算度矩阵的逆平方根，用于后续的归一化
// //     //  计算对称归一化的邻接矩阵：norm_adj

// //     // Massage：节点特征*权重矩阵：矩阵乘法： X * W，W需要初始化随机一个// support = torch.matmul(x, self.weight)
// //     // 初始化一个weight
// //     GroupElement* weight = make_array<GroupElement>(in_features, out_features); 
// //     // 需要临时数组temp，因为matmul不能在本地完成
// //     GroupElement* c = make_array<GroupElement>(num_nodes * out_features);
// //     // 矩阵A的行数、矩阵A的列数、矩阵B的列数，输入矩阵A，输入矩阵B，输出矩阵C
// //     MatMul(num_nodes, in_features, out_features, rin1, weight, c);

// //     // 消息传递：通过归一化的邻接矩阵传播特征（矩阵乘法）# D^{-1/2} * A * D^{-1/2} * (X * W) + b
// //     MatMul(num_nodes, num_nodes, out_features, rin2, c, rout);

// //     delete[] c;

// //     // 秘密分享
// //     for(int h = 0; h < num_nodes; ++h) {
// //         for(int w = 0; w < in_features; ++w) {
// //                 auto rin1_split = splitShareCommonPRNG(Arr2DIdx(rin1, num_nodes, in_features, h, w), Bin);
// //                 Arr2DIdx(k0.a, num_nodes, in_features, h, w) = rin1_split.first;
// //                 Arr2DIdx(k1.a, num_nodes, in_features, h, w) = rin1_split.second;
// //         }
// //     }

// //     for(int h = 0; h < num_nodes; ++h) {
// //         for(int w = 0; w < in_features; ++w) {
// //                 auto rin2_split = splitShareCommonPRNG(Arr2DIdx(rin2, num_nodes, in_features, h, w), Bin);
// //                 Arr2DIdx(k0.a, num_nodes, in_features, h, w) = rin2_split.first;
// //                 Arr2DIdx(k1.a, num_nodes, in_features, h, w) = rin2_split.second;
// //         }
// //     }

// //     for(int h = 0; h < num_nodes; ++h) {
// //         for(int w = 0; w < out_features; ++w) {
// //                 auto rout_split = splitShareCommonPRNG(Arr2DIdx(rout, num_nodes, out_features, h, w), Bout);
// //                 Arr2DIdx(k0.c, num_nodes, out_features, h, w) = rout_split.first;
// //                 Arr2DIdx(k1.c, num_nodes, out_features, h, w) = rout_split.second;
// //         }
// //     }

// //     delete[] c;

// //     return std::make_pair(k0, k1);

// // }