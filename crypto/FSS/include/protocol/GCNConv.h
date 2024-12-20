// /* GCNConv层密钥生成和评估接口 */

// #include <aux_parameter/keypack.h>



// std::pair<GCNConvKey, GCNConvKey> KeyGenGCNConv2D(
//     int Bin, int Bout, //输入输出
//     int num_nodes, int in_features, int num_edges, //N:节点数，f:特征数，e:边数
//     int out_features, //输出特征维度
//     GroupElement *rin1,  GroupElement * rin2, GroupElement * rin3, GroupElement * rout 
//     //考虑自身特征的邻接矩阵A-bar、特征矩阵X、权重矩阵W
// );

// void EvalGCNConv2D(int party, const GCNConvKey &key,
//     int num_nodes, int in_features, int num_edges, //N:节点数，f:特征数，e:边数
//     int out_features, //输出特征维度
//     GroupElement* A_bar, GroupElement* X, GroupElement* W, GroupElement* out
//     //考虑自身特征的邻接矩阵A-bar、特征矩阵X、权重矩阵W
//     );