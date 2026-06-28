// GTDCF 和DCF,Grotto的对比
// Table 4: Comparison cost of comparison primitives
#include <FSS/dcf.h>
#include <FSS/dpf.h>
#include <iostream>
#include <backend/FSS_base.h>

void DPF_TEST()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int bin = 4;
    int bout = 64;

    for (int idx = 0; idx < 16; ++idx)
    {
        auto keys = keyGenDPF(bin, bout, idx, 1);
        auto& key0 = keys.first;
        auto& key1 = keys.second;

        for (int i = 0; i < 16; ++i)
        {
            auto y = (evalDPF_EQ(0, key0, i) ^ evalDPF_EQ(1, key1, i));
            if (i == idx)
            {
                always_assert(y == 1);
            }
            else
            {
                always_assert(y == 0);
            }
        }

        for (int i = 0; i < 16; ++i)
        {
            auto y = (evalDPF_GT(0, key0, i) ^ evalDPF_GT(1, key1, i));
            if (i > idx)
            {
                always_assert(y == 1);
            }
            else
            {
                always_assert(y == 0);
            }
        }

        for (int i = 0; i < 16; ++i)
        {
            auto y = (evalDPF_LT(0, key0, i) ^ evalDPF_LT(1, key1, i));
            if (i < idx)
            {
                always_assert(y == 1);
            }
            else
            {
                always_assert(y == 0);
            }
        }

        GroupElement out0[16];
        GroupElement out1[16];
        evalAll(0, key0, 0, out0);
        evalAll(1, key1, 0, out1);
        for (int i = 0; i < 16; ++i)
        {
            auto y = (out0[i] + out1[i]);
            if (i == idx)
            {
                always_assert(y == 1);
            }
            else
            {
                always_assert(y == 0);
            }
        }

        evalAll(0, key0, 7, out0);
        evalAll(1, key1, 7, out1);
        for (int i = 0; i < 16; ++i)
        {
            auto y = (out0[i] + out1[i]);
            if (i == ((idx+7)%16))
            {
                always_assert(y == 1);
            }
            else
            {
                always_assert(y == 0);
            }
        }

        GroupElement res0, res1;
        std::vector<GroupElement> tab(16);
        for (int i = 0; i < 16; ++i)
        {
            tab[i] = rand();
        }
        res0 = evalAll_reduce(0, key0, 0, tab);
        res1 = evalAll_reduce(1, key1, 0, tab);
        always_assert(res0 + res1 == tab[idx]);

    }
}



void DCF_TEST()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for (int i = 0; i < 256; ++i)
    {
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int samples = 100000;
    int bin = 64;
    int bout = 64;

    // auto start = std::chrono::high_resolution_clock::now();
    
    std::pair<DCFKeyPack, DCFKeyPack> *keys = new std::pair<DCFKeyPack, DCFKeyPack>[samples];
    GroupElement *r = new GroupElement[samples];
    GroupElement *op = new GroupElement[samples];

    for (int j = 0; j < samples; ++j)
    {
        r[j] = rand();
        // GroupElement idx = rand();
        //  % (1LL << bin);
        keys[j] = keyGenDCF(bin, bout, r[j], 1);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < samples; ++j)
    {
        auto &key0 = keys[j].first;
        auto &key1 = keys[j].second;
        GroupElement inp = rand();
        GroupElement res0;
        evalDCF(0, &res0, inp, key0);
        GroupElement res1;
        evalDCF(1, &res1, inp, key1);
        auto res = res0 + res1;
        mod(res, bout);
        // std::cout << "x=" << inp << ", alpha=" << r[j] << ", res=" << res << std::endl;

        if (inp < r[j])
        {
            always_assert(res == 1);
        }
        else
        {
            always_assert(res == 0);
        }
    }
    // std::cout << "Op=" << op[rand() % samples] << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "DCF Eval Time=" << elapsed << " ns" << std::endl;
}

void DPFET_TEST()
{
    std::cout << "=======================================================\n" << std::endl;
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int samples = 100000;
    int bin = 64;
    int bout = 64;

    std::pair<DPFETKeyPack, DPFETKeyPack> *keys = new std::pair<DPFETKeyPack, DPFETKeyPack>[samples];

    GroupElement *alpha = new GroupElement[samples];
    GroupElement *r = new GroupElement[samples];
    GroupElement *op = new GroupElement[samples];

    

    for (int idx = 0; idx < samples; ++idx)
    {
        // r[idx] = rand();
        alpha[idx] = 5;
        // keys[idx] = keyGenDPF(bin, bout, r[idx], 1);
        keys[idx] = keyGenDPFET(bin, alpha[idx]);
        // key0[idx] = keys[idx].first;
        // key1 = keys.second;
    }

    auto startGT = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; i++)
    {   
        auto y = (evalDPFET_LT(0, keys[i].first, i) ^ evalDPFET_LT(1, keys[i].second, i));
        // std::cout << "x=" << i << ", alpha=" << alpha[i] << ", res=" << y << std::endl;
        if (i < alpha[i])
        {
            always_assert(y == 1);
        }
        else
        {
            always_assert(y == 0);
        }
    }
    // 计时结束
    auto endGT = std::chrono::high_resolution_clock::now();
    auto durationGT = std::chrono::duration_cast<std::chrono::nanoseconds>(endGT - startGT).count();
    std::cout << "GROTTO-DPF Eval Time=" << durationGT << " ns" << std::endl;
    std::cout << "=======================================================\n" << std::endl;

}

void DCF_Array_TEST()
{
    std::cout << "Starting DCF_Array_TEST (groupSize=2)..." << std::endl;
    
    // 1. 初始化随机种子
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for (int i = 0; i < 256; ++i) {
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int samples = 100000;
    int bin = 64;
    int bout = 64;
    int groupSize = 2; // !!! 关键修改：设置 Payload 大小为 2 !!!

    // 2. 分配内存
    auto *keys = new std::pair<DCFKeyPack, DCFKeyPack>[samples];
    auto *r = new GroupElement[samples]; // 阈值
    
    // 用于存储预期的 Payload (每个样本有两个元素)
    // payload_expect[i][0] 和 payload_expect[i][1]
    auto *payload_expect = new GroupElement[samples * groupSize]; 

    // 3. KeyGen 阶段
    for (int j = 0; j < samples; ++j)
    {
        r[j] = rand();
        
        // 生成两个随机 Payload
        GroupElement current_payload[2];
        current_payload[0] = rand();
        current_payload[1] = rand();

        // 保存以便后续验证
        payload_expect[j * groupSize + 0] = current_payload[0];
        payload_expect[j * groupSize + 1] = current_payload[1];

        // 生成密钥，传入数组指针
        keys[j] = keyGenDCF(bin, bout, groupSize, r[j], current_payload);
    }

    // 4. Eval 阶段
    auto start = std::chrono::high_resolution_clock::now();
    
    int correct = 0;
    for (int j = 0; j < samples; ++j)
    {
        // 随机生成查询值 x
        GroupElement x = rand();

        // 准备输出数组 (大小为 2)
        GroupElement res0[2] = {0, 0};
        GroupElement res1[2] = {0, 0};
        GroupElement final_res[2] = {0, 0};

        // 评估
        // 注意：这里需要调用支持 GroupSize 的 evalDCF 接口，或者 wrapper 会自动读取 key.groupSize
        evalDCF(0, res0, x, keys[j].first);
        evalDCF(1, res1, x, keys[j].second);

        // 重构结果 (向量加法)
        final_res[0] = res0[0] + res1[0];
        final_res[1] = res0[1] + res1[1];
        
        // 验证逻辑
        // DCF 逻辑: if x < r then payload else 0
        bool condition = (x < r[j]);
        
        GroupElement expect0 = condition ? payload_expect[j * groupSize + 0] : 0;
        GroupElement expect1 = condition ? payload_expect[j * groupSize + 1] : 0;

        if (final_res[0] == expect0 && final_res[1] == expect1)
        {
            correct++;
        }
        else
        {
            std::cout << "Fail at " << j << ": x=" << x << ", r=" << r[j] << std::endl;
            std::cout << "  Elem0: Expect " << expect0 << ", Got " << final_res[0] << std::endl;
            std::cout << "  Elem1: Expect " << expect1 << ", Got " << final_res[1] << std::endl;
            exit(1);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    // 修正单位为 nanoseconds
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    std::cout << "DCF_Array_Time=" << elapsed << " ns" << std::endl;
    std::cout << "Avg Time per Eval=" << (double)elapsed / samples << " ns" << std::endl;
    std::cout << "Passed " << correct << "/" << samples << " cases." << std::endl;

    // 清理内存
    delete[] keys;
    delete[] r;
    delete[] payload_expect;
}

void GTDCF_TEST()
{
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "Starting GTDCF_TEST (w=8, groupSize=2)..." << std::endl;
    
    // 1. 初始化 PRNG 种子
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int samples = 100000;
    int bin = 64;
    int w = 8;          // 提前 8 位终止，相当于砍掉树的最后 8 层
    int groupSize = 2;  // 携带二维向量负载 (例如 ReLU 需要的 {1, -r})

    // 2. 分配内存
    auto *keys = new std::pair<GTDCFKeyPack, GTDCFKeyPack>[samples];
    auto *r = new GroupElement[samples];
    auto *beta = new GroupElement[samples * groupSize]; // 扁平化存储二维 Payload

    // 3. 离线生成阶段 (KeyGen)
    for (int i = 0; i < samples; ++i) {
        r[i] = rand();
        
        // 随机生成需要秘密传输的 Payload (2维)
        beta[i * groupSize] = rand();
        beta[i * groupSize + 1] = rand();

        // 传入的是 r[i] 作为比较阈值，以及对应的 beta 指针
        keys[i] = keyGenGTDCF(bin, w, groupSize, r[i], &beta[i * groupSize]);
    }

    int correct = 0;
    
    // 4. 在线评估阶段 (Eval) - 我们只统计这段极速代码的时间
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < samples; ++i) {
        GroupElement x = rand(); // 随机公共输入 (相当于 x+r)

        // 预分配结果数组
        GroupElement res0[2] = {0, 0};
        GroupElement res1[2] = {0, 0};

        // 双方分别本地求值
        evalGTDCF(0, keys[i].first, x, res0);
        evalGTDCF(1, keys[i].second, x, res1);

        // 重构结果
        GroupElement final_res0 = res0[0] + res1[0];
        GroupElement final_res1 = res0[1] + res1[1];

        // ================= 验证数学逻辑 =================
        // SparseVGT 的底层逻辑是大于等于 (>=) 比较
        bool condition = (x >= r[i]); 
        
        GroupElement exp0 = condition ? beta[i * groupSize] : 0;
        GroupElement exp1 = condition ? beta[i * groupSize + 1] : 0;

        if (final_res0 == exp0 && final_res1 == exp1) {
            correct++;
        } else {
            std::cout << "Fail at " << i << ": x=" << x << " r=" << r[i] << std::endl;
            std::cout << "  Elem0: Expect " << exp0 << ", Got " << final_res0 << std::endl;
            std::cout << "  Elem1: Expect " << exp1 << ", Got " << final_res1 << std::endl;
            exit(1);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << "GTDCF Eval Time   =" << duration << " ns" << std::endl;
    std::cout << "Avg Time per Eval =" << (double)duration / samples << " ns" << std::endl;
    std::cout << "Passed " << correct << "/" << samples << " cases." << std::endl;
    std::cout << "=======================================================\n" << std::endl;

    // 5. 清理内存 (极其重要，因为包里有 new 的动态数组)
    // for (int i = 0; i < samples; ++i) {
    //     freeGTDCFTKeyPackPair(keys[i]);
    // }
    delete[] keys;
    delete[] r;
    delete[] beta;
}


int main(){
    // DPF_TEST();
    DPFET_TEST();
    DCF_TEST();
    GTDCF_TEST();
}