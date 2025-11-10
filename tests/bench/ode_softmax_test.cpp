#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cassert>
#include <algorithm>
#include "../../nn/backend/FSS_extended.h"
#include "../../crypto/FSS/api/api.h"
#include <random>
#include <iomanip> // 为了 std::fixed 和 std::setprecision

// --- 辅助函数 ---
std::vector<double> generate_random_double_array(size_t size, double min_val, double max_val) {
    std::vector<double> data;
    data.reserve(size); // 预分配内存以提高效率

    // 1. 初始化随机数生成器
    // 使用 std::random_device 获取一个真实的硬件随机种子
    std::random_device rd;  
    // 使用 Mersenne Twister 算法作为随机数引擎，并用硬件种子初始化
    std::mt19937 gen(rd()); 

    // 2. 定义一个均匀分布
    // std::uniform_real_distribution 会在 [min_val, max_val] 区间内生成均匀分布的浮点数
    std::uniform_real_distribution<> distrib(min_val, max_val);

    // 3. 循环生成数据
    for (size_t i = 0; i < size; ++i) {
        data.push_back(distrib(gen));
    }

    return data;
}
// 打印 double 数组
void print_double_array(const std::string& title, const std::vector<double>& arr,int limit = 10) {
    std::cout << "\n--- " << title << " ---" << std::endl;
    for (size_t i = 0; i < arr.size()&&i<limit; ++i) {
        std::cout << "  [" << i << "]: " << arr[i] << std::endl;
    }
}

// 在明文 double 数组上执行 SHAFT 的 ODE Softmax 近似算法
void plaintext_softmax_ode(
    const std::vector<double>& input, 
    std::vector<double>& output,
    int iter_num, 
    bool clip,
    double lower_bound,
    double upper_bound
) {
    int size = input.size();
    std::vector<double> x = input;

    // 1. 裁剪
    if (clip) {
        for (int i = 0; i < size; ++i) {
            x[i] = std::max(lower_bound, std::min(x[i], upper_bound));
        }
    }

    // 2. 缩放
    for (int i = 0; i < size; ++i) {
        x[i] /= iter_num;
    }

    // 3. 初始化 g
    std::vector<double> g(size, 1.0 / size);
    //print_double_array("g0",g);
    // 4. 迭代
    for (int k = 0; k < iter_num; ++k) {
        double dot_prod = 0.0;
        for (int i = 0; i < size; ++i) {
            dot_prod += g[i] * x[i];
            //std::cout<<g[i]<<"*"<<x[i]<<"="<< g[i] * x[i]<<" dotprod= "<<dot_prod<<std::endl;
        }
        
        for (int i = 0; i < size; ++i) {
            g[i] += (x[i] - dot_prod) * g[i];
        }
        //std::cout<<"g"<<k<<std::endl;
        //print_double_array("g",g);
    }
    output = g;
}




// --- 主测试函数 ---

void test_softmax_ode(int party) {
    std::cout << "\n\n>> SHAFT Softmax ODE Protocol Test - Start" << std::endl;

    using FSSVersion = FSSExtended<u64>;
    FSSVersion *FSS = new FSSVersion();
    FSSConfig::bitlength = 64;
    FSSConfig::party = party;
    FSSConfig::num_threads = 4;
    std::string ip = "127.0.0.1";
    FSS->init(ip, true);
    

    // --- 2. 准备数据 ---
    const int size =2048;
    //const int size =8;
    const int scale = 16;
    const int iter_num = 16; // 必须是2的幂
    const bool clip = true;
    const double lower_bound = -4.0;
    const double upper_bound = 12.0;

    std::vector<double> plain_input_double(size);
    std::vector<GroupElement> plain_input_fixed(size);

    if (party ==SERVER) {
        double min_value = -20.0;
        double max_value = 20.0;
        // 创建一些包含极端值的数据
        plain_input_double = {-20.0, -3.0, 0.0, 1.0, 2.5, 10.0, 15.0, 0.5};
        //plain_input_double = {-2.0, -10.0, 5.0, 8.0, 12, 16.0, 15.0, 0.5, 2.3, 2.6, -3.3333, 8.56, 12.444, 16.556, 15.4789, 0.55654,-2.0, -10.0, 5.0, 8.0, 12, 16.0, 15.0, 0.5, 2.3, 2.6, -3.3333, 8.56, 12.444, 16.556, 15.4789, 0.55654,-2.0, -10.0, 5.0, 8.0, 12, 16.0, 15.0, 0.5, 2.3, 2.6, -3.3333, 8.56, 12.444, 16.556, 15.4789, 0.55654,-2.0, -10.0, 5.0, 8.0, 12, 16.0, 15.0, 0.5, 2.3, 2.6, -3.3333, 8.56, 12.444, 16.556, 15.4789, 0.55654};
        plain_input_double = generate_random_double_array(size, min_value, max_value);
        //-4 -3 0 1 2.5 10 12 0.5
        //0.125
        //-0.5 -0.375 0 0.125 0.3125 1.25 1.5 0.0625
        for (int i = 0; i < size; ++i) {
            plain_input_fixed[i] = double_to_fixed(plain_input_double[i], scale);
        }
    }

    // --- 3. 计算期望结果 ---
    std::vector<double> expected_output_double(size);
    if (party == SERVER) {
        plaintext_softmax_ode(plain_input_double, expected_output_double, iter_num, clip, lower_bound, upper_bound);
        print_double_array("Plaintext Input (double)", plain_input_double);
        print_double_array("Expected Plaintext Output (double, after ODE)", expected_output_double);
    }

    // --- 4. 秘密分享 ---
    print_double_array("input_restored",party,size,plain_input_fixed.data(),10);
    GroupElement* input_shares = new GroupElement[size]();
    if(party != DEALER)
        SecretShare(size, plain_input_fixed.data(), input_shares, SERVER);

    GroupElement * mask1 = new GroupElement[size];
    GroupElement * mask2 = new GroupElement[size];
    // --- 5. 执行协议 ---
    GroupElement* output_shares = new GroupElement[size]();
    FSS::start();
    auto start_time = std::chrono::high_resolution_clock::now();
    SoftmaxODE(size, input_shares, mask1, output_shares, mask2, iter_num, clip);

    FSS::end();

    auto end_time = std::chrono::high_resolution_clock::now();

    // 4. 计算时间差并打印
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // 为了防止多个参与方都打印时间，可以只让一个 party (例如 party 0) 打印
        std::cout << "================================================" << std::endl;
        std::cout << "Total execution time: " << duration.count() << " milliseconds" << std::endl;
        std::cout << "Total execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
        std::cout << "================================================" << std::endl;
    // --- 6. 重构与验证 ---
    if(party!=DEALER){
        std::cout << "\n   Party " << party << ": Reconstructing final output..." << std::endl;
        reconstruct(size, output_shares, FSSConfig::bitlength);
    }
    if (party == SERVER) {
        
        std::vector<double> mpc_output_double(size);
        for (int i = 0; i < size; ++i) {
            mpc_output_double[i] = fixed_to_double(output_shares[i], scale);
        }
        print_double_array("MPC Reconstructed Output (double)", mpc_output_double);

        // 验证
        bool all_ok = true;
        double max_error = 0.0;
        // 允许一个小的误差容限，因为定点数截断会引入误差
        const double tolerance = 1.0 / (1LL << (scale - 4)); // 允许约1/4096的误差

        for (int i = 0; i < size; i++) {
            double error = std::abs(mpc_output_double[i] - expected_output_double[i]);
            if (error > max_error) {
                max_error = error;
            }
            if (error > tolerance) {
                all_ok = false;
                std::cerr << "  [FAILURE] at index " << i << ": Expected " << expected_output_double[i] 
                          << ", but MPC got " << mpc_output_double[i] << " (Error: " << error << ")" << std::endl;
            }
        }

        std::cout << "\n--- [Party " << party << "] Verification ---" << std::endl;
        std::cout << "  Max error found: " << max_error << std::endl;
        std::cout << "  Acceptable tolerance: " << tolerance << std::endl;
        if (all_ok) {
            std::cout << "  SUCCESS: All elements are within the acceptable error tolerance." << std::endl;
        } else {
            std::cout << "  FAILURE: One or more elements exceeded the error tolerance." << std::endl;
        }
    }

    // --- 7. 清理 ---
    delete[] input_shares;
    delete[] output_shares;
    FSS->finalize();
    delete FSS;

    std::cout << ">> SHAFT Softmax ODE Protocol Test - End\n" << std::endl;
}

// main函数
int main(int argc, char** argv) {
    int party = 0;
    if (argc > 1) {
        party = atoi(argv[1]);
        if (party < 1 || party > 3) {
            std::cerr << "Error: Party ID must be 1, 2, or 3." << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Usage: " << argv[0] << " <party_id> (1=Dealer, 2=Server, 3=Client)" << std::endl;
        return 1;
    }

    test_softmax_ode(party);

    return 0;
}