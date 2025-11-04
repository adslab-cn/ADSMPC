/* DPF评估 */
#include "../../crypto/FSS/primitives/dpf.h"
#include <iostream>
#include "../../nn/backend/FSS_base.h"

void DPFIC_TEST_3_INTERVALS()
{
    // ... (种子和PRNG设置，与之前相同) ...
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int samples = 10;
    int bin = 64;

    // --- 预处理阶段: Dealer 为随机点 alpha 生成 DPF 密钥 ---
    GroupElement alpha = 5000; // 对应论文中的随机点 i
    auto keys = keyGenDPFET(bin, alpha);

    GroupElement a = -2;
    GroupElement b = 6;

    std::cout << "Testing 3-interval check for x in [" << samples - 1 << "]" << std::endl;
    std::cout << "Intervals: (-inf, " << a << "), [" << a << ", " << b << "), [" << b << ", +inf)" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = -4; i < samples; i++)
    {
        GroupElement x = i; // 秘密输入 x

        // --- 在线阶段 ---
        
        // 1. 计算公开的移位量 d = x - alpha
        GroupElement d = x - alpha;
        
        // 2. 旋转边界点 a 和 b
        GroupElement a_prime = a - d;
        GroupElement b_prime = b - d;
        
        // 3. 执行两次安全比较，得到基础判断结果的明文
        // res1 = [x < a] <=> [alpha < a']
        auto res1 = (evalDPFET_LT(0, keys.first, a_prime) ^ evalDPFET_LT(1, keys.second, a_prime));
        // res2 = [x < b] <=> [alpha < b']
        auto res2 = (evalDPFET_LT(0, keys.first, b_prime) ^ evalDPFET_LT(1, keys.second, b_prime));
        
        // --- 结果组合 ---

        // s0: 判断 x < a
        uint8_t s0 = res1;

        // s1: 判断 a <= x < b
        uint8_t s1 = res1 ^ res2;

        // s2: 判断 x >= b
        uint8_t s2 = 1 ^ res2;

        // 打印结果
        std::cout << "x = " << x 
                  << " -> (" 
                  << "x < " << a << ": " << (int)s0 << ", "
                  << a << " <= x < " << b << ": " << (int)s1 << ", "
                  << "x >= " << b << ": " << (int)s2
                  << ")" << std::endl;
        
        // 验证独热编码属性 (可选)
        always_assert(s0 + s1 + s2 == 1);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Total time for " << samples << " checks: " << duration << " ns" << std::endl;
    std::cout << "Average time per check: " << duration / samples << " ns" << std::endl;
}

GroupElement double_to_fixed(double val, int scale) {
    return static_cast<GroupElement>(round(val * (1LL << scale)));
}

// 将定点数 GroupElement 转换回 double
double fixed_to_double(GroupElement val, int scale) {
    // 处理负数 (补码)
    int bitlength = FSSConfig::bitlength;
    if (val & (1ULL << (bitlength - 1))) {
        int64_t signed_val = val - (1ULL << bitlength);
        return static_cast<double>(signed_val) / (1LL << scale);
    }
    return static_cast<double>(val) / (1LL << scale);
}


void run_interval_test()
{
    // --- 0. 环境设置 ---
    u64 seedKey = 0xdeadbeefbadc0ffe;
    FSSConfig::prngs[0].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    const int bin = 64;
    const int scale = 16;

    // --- 1. 预处理 ---
    GroupElement alpha = 0;
    auto keys = keyGenDPFET(bin, alpha);

    // --- 2. 测试设置 (使用原始值) ---
    const double x_start = -4.0;
    const double x_end = 10.0;
    const double lower_bound_plain = -2.0;
    const double upper_bound_plain = 6.0;

    // 转换为定点数
    GroupElement lower_fixed_orig = double_to_fixed(lower_bound_plain, scale);
    GroupElement upper_fixed_orig = double_to_fixed(upper_bound_plain, scale);
    
    // 平移以确保所有计算都在正数域
    GroupElement shift = double_to_fixed(1000.0, scale);
    GroupElement lower_shifted = lower_fixed_orig + shift;
    GroupElement upper_shifted = upper_fixed_orig + shift;

    std::cout << "Testing 3-interval check for x from " << x_start << " to " << x_end-1 << std::endl;
    std::cout << "Intervals: (-inf, " << lower_bound_plain << "), [" << lower_bound_plain << ", " << upper_bound_plain << "), [" << upper_bound_plain << ", +inf)" << std::endl;
    std::cout << "Hypothesis: evalDPFET_LT(C) computes [alpha > C-1] which is equivalent to [alpha >= C]" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    for (double i = x_start; i < x_end; i += 1.0)
    {
        GroupElement x_fixed_orig = double_to_fixed(i, scale);
        
        // --- 3. 在线阶段 (模拟) ---
        GroupElement x_shifted = x_fixed_orig + shift;
        GroupElement d = x_shifted - alpha;
        
        // 我们要计算 [x >= lower] 和 [x >= upper]
        // 我们的函数 eval(C) 计算的是 [alpha >= C]
        // 所以我们想让 eval(C) 计算 [x >= lower]
        // [x >= lower] <=> [x-alpha >= lower-alpha] <=> [d >= lower-alpha]
        // ... 这个转换很复杂，让我们回到我们破译的规则
        
        // 破译的规则: res_ge_a = [x >= a]
        // 这是通过调用 eval(a-1-d) 实现的。我们直接用这个。
        GroupElement C1 = lower_shifted - 1;
        GroupElement a_prime = C1 - d;
        auto res_ge_a = (evalDPFET_LT(0, keys.first, a_prime) ^ evalDPFET_LT(1, keys.second, a_prime));

        GroupElement C2 = upper_shifted - 1;
        GroupElement b_prime = C2 - d;
        auto res_ge_b = (evalDPFET_LT(0, keys.first, b_prime) ^ evalDPFET_LT(1, keys.second, b_prime));

        // --- 4. 结果组合 ---
        uint8_t s0 = 1 ^ res_ge_a; // s0 = NOT [x >= lower] = [x < lower]
        uint8_t s1 = res_ge_a ^ res_ge_b; // s1 = [x >= lower] XOR [x >= upper] = [lower <= x < upper]
        uint8_t s2 = res_ge_b;     // s2 = [x >= upper]

        // --- 5. 打印结果 ---
        std::cout << "x = " << i << " -> (" 
                  << "s0=" << (int)s0 << ", "
                  << "s1=" << (int)s1 << ", "
                  << "s2=" << (int)s2
                  << ")"
                  << "  |  raw_evals: ([x>=" << lower_bound_plain << "]=" << (int)res_ge_a 
                  << ", [x>=" << upper_bound_plain << "]=" << (int)res_ge_b << ")" << std::endl;
        
        // 在循环内部进行断言，以便立即知道哪一行出错了
        assert(s0 + s1 + s2 == 1);
    }
    
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "All tests passed!" << std::endl;
}

int main() {
    run_interval_test();
    return 0;
}