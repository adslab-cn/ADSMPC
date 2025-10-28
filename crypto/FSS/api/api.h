// FSS方法的接口

#pragma once

#include "../aux_parameter/group_element.h"

#define MASK_PAIR(x) x, x##_mask
/* 
    测试调用方式
    FSS::start();  // 初始化通信、同步、计时器
    调用隐私计算算子（如Conv2D, Relu）
    FSS::end();    // 输出性能统计（通信量、耗时、密钥读取时间
*/

namespace FSS {
    void start();
    void end();
}

// #################################################################################################
/* 重构结果 */
void reconstruct(int32_t size, GroupElement *arr, int bw);

/* 定点转浮点 */
void FixToFloat(int size, GroupElement *inp, GroupElement *out, int scale);

/* 浮点转定点 */
void FloatToFix(int size, GroupElement *inp, GroupElement *out, int scale);

/* 逻辑左移 */
void ScaleUp(int32_t size, MASK_PAIR(GroupElement *inArr), int32_t sf);

/* 算术右移 */
void ARS(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), int32_t shift);

/* 逻辑右移 */
void ScaleDown(int32_t size, MASK_PAIR(GroupElement *inArr), int32_t sf);

// #################################################################################################
/* 矩阵乘法 */
void MatMul2D(int32_t s1, int32_t s2, int32_t s3, MASK_PAIR(GroupElement *A),
            MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C), bool modelIsA);

/* 2D卷积 */
void Conv2DWrapper(int32_t N, int32_t H, int32_t W,
                   int32_t CI, int32_t FH, int32_t FW,
                   int32_t CO, int32_t zPadHLeft,
                   int32_t zPadHRight, int32_t zPadWLeft,
                   int32_t zPadWRight, int32_t strideH,
                   int32_t strideW, MASK_PAIR(GroupElement *inputArr), MASK_PAIR(GroupElement *filterArr),
                   MASK_PAIR(GroupElement *outArr));

// =========================== 非线性计算 ===========================
/* 条件选择 */
void Select(int32_t size, GroupElement *s, GroupElement *x, GroupElement *out, std::string prefix = "", bool doReconstruct = true);

/* 条件选择（多一个参数 bin） */
void Select(int32_t size, int bin, GroupElement *s, GroupElement *x, GroupElement *out, std::string prefix = "", bool doReconstruct = true);

/* 标准ReLU协议： FSS'21 DCF的复用 */
void Relu(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *drelu, std::string prefix = "");

/* 两阶段relu协议，Relu2Round */
void Relu2Round(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *drelu_cache, int effectiveInputBw);

/* 最大池化 */
void MaxPool(int32_t N, int32_t H, int32_t W, int32_t C, int32_t ksizeH,
             int32_t ksizeW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *oneHot, std::string prefix = "");

/* 将最大池化过程中记录的maxBits转换为one-hot编码（指示最大值位置） */
void MaxPoolOneHot(int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH, int32_t FW, GroupElement *maxBits, GroupElement *oneHot);

void MaxPoolBackward(int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH,
             int32_t FW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *oneHot);


/* Softmax近似, 求最大值、减去最大值、指数运算（通过ReLU截断近似）、求和、除法等步骤 */
void PiranhaSoftmax(int32_t s1, int32_t s2, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), int32_t sf);

/* 高效的ReLU导数计算 */
void SlothDrelu(int size, int bin, GroupElement *x, GroupElement *y, std::string prefix = "");

/* 高效的ReLU计算 */
void SlothRelu(int size, int bin, GroupElement *x, GroupElement *y, std::string prefix = "");

/* Softmax函数（与PiranhaSoftmax不同） */
void Softmax(int32_t s1, int32_t s2, int bin, GroupElement *x, GroupElement *y, int32_t scale);

void SlothMax(int size, int bin, GroupElement *x, GroupElement *y, GroupElement *out, std::string prefix);
void ElemWiseMul(int32_t size, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C));
void DpfRoute(
    int32_t size,
    MASK_PAIR(GroupElement *y_in),
    int rank_bw,
    MASK_PAIR(GroupElement *z_in),
    int data_bw, 
    MASK_PAIR(GroupElement *z_out));
void prng_shared_init();
void print_array(const std::string& title, int party, int size, const GroupElement* arr, int limit = 10);
void SecretShare(int32_t size, const GroupElement *plain_in, GroupElement *share_out, int owner);
void obliviousGraphUpdate(
    int party,
    int target_node_v_star,
    int n, int c,
    // 明文数据只在 Dealer 端需要，Server/Client 端可以传入空矩阵
    GroupElement ** A_old, GroupElement ** A_new, int A_bw, int A_data_bw,
    GroupElement ** F_old, GroupElement ** F_new, int F_bw, int F_data_bw,
    // 份额数据只在 Server/Client 端需要
    GroupElement ** A_share,
    GroupElement ** F_share
);
struct OneHotShares {
    GroupElement s0; // 份额 for 区间 1
    GroupElement s1; // 份额 for 区间 2
    GroupElement s2; // 份额 for 区间 3
};

void three_interval_check(
    uint8_t party,
    GroupElement x_share,
    GroupElement a,
    GroupElement b,
    uint8_t bin,
    OneHotShares& result_shares
);

void FastRelu(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), std::string prefix = "");

void SoftmaxODE(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), int iter_num, bool clip);

GroupElement double_to_fixed(double val, int scale);

double fixed_to_double(GroupElement val, int scale);

void print_double_array(const std::string& title, int party, int size, GroupElement* arr, int limit);

void ARS_CrypTen_Style(int32_t size, 
                       GroupElement* inArr, 
                       GroupElement* outArr, 
                       int32_t shift);
/* Sloth的算数左移 */
void SlothLRS(int size, GroupElement *x, GroupElement *y, int scale, std::string prefix = "");
/* Sloth的算数右移 */
void SlothARS(int size, GroupElement *x, GroupElement *y, int scale, std::string prefix = "");