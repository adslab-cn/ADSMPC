// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <map>
// #include <random> // For std::mt19937

// // 包含 osuCrypto 和您项目的头文件
// #include "cryptoTools/Common/Defines.h"
// #include "cryptoTools/Common/Log.h"
// #include "cryptoTools/Crypto/PRNG.h"
// #include "../../crypto/FSS/aux_parameter/assert.h" // 假设 assert.h 在这个路径

// // 包含 DPF 和 Big-State DMPF 的头文件
// #include "../../crypto/FSS/primitives/dpf.h"
// #include "../../crypto/FSS/primitives/dmpf.h"

// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <map>
// #include <random> // For std::mt19937
// #include <set>    // <--- 解决方案 1：添加此头文件
// #include <algorithm> // for std::sort

// // 包含 osuCrypto 和您项目的头文件
// #include "cryptoTools/Common/Defines.h"
// #include "cryptoTools/Common/Log.h"
// #include "cryptoTools/Crypto/PRNG.h"
// #include "../../crypto/FSS/aux_parameter/assert.h" // 假设 assert.h 在这个路径

// // 包含 DPF 和 Big-State DMPF 的头文件
// #include "../../crypto/FSS/primitives/dpf.h"
// #include "../../crypto/FSS/primitives/dmpf.h"
// // 引入命名空间
// using namespace osuCrypto;

#include <iostream>
#include <vector>
#include <chrono>
#include <map>
#include <random>
#include <set>
#include <algorithm>

// 包含 osuCrypto 和您项目的头文件
#include "cryptoTools/Common/Defines.h"
#include "cryptoTools/Common/Log.h"
#include "cryptoTools/Crypto/PRNG.h"
#include "../../crypto/FSS/aux_parameter/assert.h" // 假设 assert.h 在这个路径

// 包含 DPF 和 Big-State DMPF 的头文件
#include "../../crypto/FSS/primitives/dpf.h"
#include "../../crypto/FSS/primitives/dmpf.h"

// 引入命名空间
using namespace osuCrypto;

// Big-State DMPF 的测试函数
void BIG_STATE_DMPF_TEST()
{
    std::cout << "\n=========================================\n";
    std::cout << "     Running Big-State DMPF Test\n";
    std::cout << "=========================================\n";

    // 1. 初始化 PRNG
    u64 seedKey = 0xdeadbeefbadc0ffe;

    // FSSConfig::prngs 是一个静态数组，直接循环初始化即可。
    // 无需检查 empty() 或调用 resize()。
    for(int i = 0; i < 256; ++i) {
        FSSConfig::prngs[i].SetSeed(toBlock(time(NULL), seedKey));
    }
    PRNG prng(toBlock(12345));

    // 2. 设置测试参数
    int bin = 12;      
    int bout = 64;     
    int t_points = 10; 

    std::cout << "Parameters:\n"
              << "  - Input Domain Bits (bin): " << bin << " (Domain size: " << (1 << bin) << ")\n"
              << "  - Non-zero Points (t):     " << t_points << "\n\n";

    // 3. 生成测试数据
    std::vector<std::pair<GroupElement, GroupElement>> inputs;
    std::map<GroupElement, GroupElement> expected_outputs; 

    std::set<GroupElement> distinct_indices;
    GroupElement domain_mask = (1ULL << bin) - 1;

    while (distinct_indices.size() < t_points) {
        distinct_indices.insert(prng.get<u64>() & domain_mask);
    }

    for (const auto& idx : distinct_indices) {
        GroupElement payload = prng.get<u64>() % (1ULL << 30); 
        if (payload == 0) payload = 1;
        
        inputs.push_back(std::make_pair(idx, payload));
        expected_outputs[idx] = payload;
    }
    
    std::sort(inputs.begin(), inputs.end(), 
        [](const auto& a, const auto& b){ return a.first < b.first; });


    // 4. 测试密钥生成 (KeyGen)
    std::cout << "--- Testing Key Generation ---\n";
    auto start_gen = std::chrono::high_resolution_clock::now();
    
    auto keys = keyGenBigStateDMPF(bin, bout, inputs);
    auto key0 = keys.first;
    auto key1 = keys.second;

    auto end_gen = std::chrono::high_resolution_clock::now();
    auto duration_gen = std::chrono::duration_cast<std::chrono::microseconds>(end_gen - start_gen).count();
    std::cout << "Key generation time: " << duration_gen << " us\n\n";


    // 5. 测试单点求值 (Eval)
    std::cout << "--- Testing Single-Point Evaluation (eval) ---\n";
    int eval_samples = 1000;
    bool eval_correct = true;

    auto start_eval = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < eval_samples; ++i) {
        GroupElement x = prng.get<u64>() & domain_mask;

        GroupElement res0 = evalBigStateDMPF(0, key0, x);
        GroupElement res1 = evalBigStateDMPF(1, key1, x);
        GroupElement combined_res = res0 + res1;

        GroupElement expected_res = 0;
        if (expected_outputs.count(x)) {
            expected_res = expected_outputs[x];
        }

        if (combined_res != expected_res) {
            std::cerr << "Eval failed at x = " << x << "! Expected " << expected_res << ", Got " << combined_res << std::endl;
            eval_correct = false;
        }
    }
    auto end_eval = std::chrono::high_resolution_clock::now();
    auto duration_eval_total = std::chrono::duration_cast<std::chrono::microseconds>(end_eval - start_eval).count();
    
    if (eval_correct) {
        std::cout << "Correctness check PASSED for " << eval_samples << " random points.\n";
    } else {
        std::cout << "Correctness check FAILED.\n";
    }
    std::cout << "Total eval time (" << eval_samples << " points): " << duration_eval_total << " us\n";
    std::cout << "Average eval time per point: " << (double)duration_eval_total / eval_samples << " us\n\n";
    

    // 6. 测试全域求值 (EvalAll)
    std::cout << "--- Testing Full-Domain Evaluation (evalAll) ---\n";
    u64 domain_size = 1ULL << bin;
    GroupElement* out0 = new GroupElement[domain_size];
    GroupElement* out1 = new GroupElement[domain_size];

    auto start_eval_all = std::chrono::high_resolution_clock::now();
    evalAllBigStateDMPF(0, key0, out0);
    evalAllBigStateDMPF(1, key1, out1);
    auto end_eval_all = std::chrono::high_resolution_clock::now();
    auto duration_eval_all = std::chrono::duration_cast<std::chrono::milliseconds>(end_eval_all - start_eval_all).count();
    
    std::cout << "Full domain evaluation time: " << duration_eval_all << " ms\n";
    
    bool eval_all_correct = true;
    for (u64 i = 0; i < domain_size; ++i) {
        GroupElement combined_res = out0[i] + out1[i];
        
        GroupElement expected_res = 0;
        if (expected_outputs.count(i)) {
            expected_res = expected_outputs[i];
        }

        if (combined_res != expected_res) {
            std::cerr << "EvalAll failed at index " << i << "! Expected " << expected_res << ", Got " << combined_res << std::endl;
            eval_all_correct = false;
        }
    }
    
    if (eval_all_correct) {
        std::cout << "Correctness check PASSED for the entire domain.\n";
    } else {
        std::cout << "Correctness check FAILED.\n";
    }
    
    delete[] out0;
    delete[] out1;
    
    std::cout << "\nBig-State DMPF Test Finished.\n";
    std::cout << "=========================================\n\n";
}

int main()
{
    // 您可以在这里选择运行哪个测试
    
    // DPF_TEST(); // 运行原始 DPF 测试
    BIG_STATE_DMPF_TEST(); // 运行新的 Big-State DMPF 测试

    return 0;
}