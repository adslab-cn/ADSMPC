// 在 keys.cpp
#include "relufastsecnet.h" // 确保包含了头文件
#include "../primitives/dcf.h"

std::pair<FastReluKeyPack, FastReluKeyPack> keyGenFastRelu(int Bin, int Bout) {
    // 1. Dealer 选择随机偏移量 r
    GroupElement r = random_ge(10);

    // 2. 定义分段线性函数的参数
    // FastSecNet 定义 AReLU_r(x) = (x-r) if x>=r, else 0.
    // 这可以表示为 b0*x + b1。
    // 如果 x >= r, (b0, b1) = (1, -r)
    // 如果 x < r, (b0, b1) = (0, 0)
    // 我们要生成的 FSS 函数是在 x>=r 时 payload 为 (1, -r)。
    
    GroupElement payload[2];
    payload[0] = -1;      // 这是 b0 的值
    payload[1] = r;     // 这是 b1 的值
    mod(payload[1], Bout); // 确保在正确的环上

    // 3. 调用现有的 keyGenDCF，但 groupSize=2
    // 这个DCF将用于安全比较 "input > r"
    // 注意：EzPC的keyGenDCF是小于比较，而FastSecNet需要大于等于。
    // a > b <=> b < a。所以我们需要比较 r < input。
    // 因此，特殊点 alpha 就是 r。
    auto dcf_keys = keyGenDCF(Bin, Bout, 2, r, payload);

    // 4. 生成 r 的秘密份额
    auto r_shares = splitShare(r, Bin);

    // 5. 打包成 FastReluKeyPack
    FastReluKeyPack k0, k1;
    k0.Bin = k1.Bin = Bin;
    k0.Bout = k1.Bout = Bout;

    k0.dcfKey = dcf_keys.first;
    k1.dcfKey = dcf_keys.second;

    k0.r_sh = r_shares.first;
    k1.r_sh = r_shares.second;

    auto b_0 = splitShare(-1 * payload[0], Bin);
    auto b_1 = splitShare(-1 * payload[1], Bin);
    
    k0.b_sh[0] = b_0.first;
    k1.b_sh[0] = b_0.second;
    
    k0.b_sh[1] = b_1.first;
    k1.b_sh[1] = b_1.second;






    // //std::cout << "\n--- [DEALER SIDE KEY VERIFICATION FOR ELEMENT " << i << "] ---" << std::endl;
    // std::cout << "Secret comparison point r = " << r << std::endl;
    
    // // 定义测试点: 一个小于r, 一个大于等于r
    // GroupElement x_less = (r > 0) ? r - 1 : r; // 取 r-1，除非 r=0
    // GroupElement x_ge = r;
    // std::vector<GroupElement> test_points = {x_less, x_ge};

    // for (GroupElement x : test_points) {
    //     std::cout << "  Testing with x = " << x << std::endl;

    //     // 准备输出数组
    //     GroupElement* b_shares_0 = new GroupElement[2]; // Party 0 (Server) 的份额
    //     GroupElement* b_shares_1 = new GroupElement[2]; // Party 1 (Client) 的份额

    //     // 2. Dealer 模拟双方执行 evalDCF
    //     // evalDCF(party, res_array, input_x, key)
    //     // 修正1: evalDCF 的 party 参数应该是 SERVER0 (0) 和 SERVER1 (1)
    //     // 修正2: evalDCF 是 void 函数, 它直接修改传入的数组，没有返回值
    //     evalDCF(0, b_shares_0, x, k0.dcfKey);
    //     evalDCF(1, b_shares_1, x, k1.dcfKey);

    //     // 3. Dealer 在本地重构结果
    //     GroupElement* res = new GroupElement[2];
    //     res[0] = b_shares_0[0] + b_shares_1[0];
    //     res[1] = b_shares_0[1] + b_shares_1[1];
    //     mod_array(res,2, 127);

        
    //     // 4. 打印重构后的明文结果
    //     std::cout << "    Reconstructed result (b0, b1): (" << res[0] << ", " << res[1] << ")" << std::endl;

    //     delete[] b_shares_0;
    //     delete[] b_shares_1;
    //     delete[] res;
    // }





    return std::make_pair(k0, k1);
}