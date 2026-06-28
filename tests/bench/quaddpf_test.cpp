#include <FSS/dcf.h>
#include <FSS/dpf.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <backend/FSS_base.h>
#include <FSS/freekey.h>

void QuadDPF_vs_DPF_TEST()
{
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "Starting Quad-Tree DPF vs Standard DPF Benchmarks..." << std::endl;

    // 1. 初始化随机种子
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int samples = 50000;
    int bin = 64;   // 64位输入
    int bout = 64;  // 64位输出

    std::pair<QuadDPFKeyPack, QuadDPFKeyPack> *quad_keys = new std::pair<QuadDPFKeyPack, QuadDPFKeyPack>[samples];
    std::pair<DPFKeyPack, DPFKeyPack> *std_keys = new std::pair<DPFKeyPack, DPFKeyPack>[samples];

    GroupElement *alpha = new GroupElement[samples];
    GroupElement *payload = new GroupElement[samples];
    GroupElement *query_x = new GroupElement[samples];

    for (int i = 0; i < samples; ++i) {
        alpha[i] = rand();
        payload[i] = rand();
        // 50%查询命中目标，50%查询其他位置
        query_x[i] = (rand() % 2 == 0) ? alpha[i] : rand();
    }

    // ================= 1. Standard DPF KeyGen =================
    auto startStdGen = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; ++i) {
        std_keys[i] = keyGenDPF(bin, bout, alpha[i], payload[i]);
    }
    auto endStdGen = std::chrono::high_resolution_clock::now();
    auto durStdGen = std::chrono::duration_cast<std::chrono::nanoseconds>(endStdGen - startStdGen).count();

    // ================= 2. Quad-Tree DPF KeyGen =================
    auto startQuadGen = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; ++i) {
        quad_keys[i] = keyGenQuadDPF(bin, bout, alpha[i], payload[i]);
    }
    auto endQuadGen = std::chrono::high_resolution_clock::now();
    auto durQuadGen = std::chrono::duration_cast<std::chrono::nanoseconds>(endQuadGen - startQuadGen).count();

    // ================= 3. Standard DPF Eval =================
    // 使用 evalDPF_EQ 来测试一条路径检索的底层开销
    auto startStdEval = std::chrono::high_resolution_clock::now();
    int std_correct = 0;
    for (int i = 0; i < samples; ++i) {
        GroupElement t0 = evalDPF_EQ(0, std_keys[i].first, query_x[i]);
        GroupElement t1 = evalDPF_EQ(1, std_keys[i].second, query_x[i]);
        GroupElement res_t = t0 ^ t1; // 二叉DPF的EQ输出判定
        GroupElement expected_t = (query_x[i] == alpha[i]) ? 1 : 0;
        if (res_t == expected_t) std_correct++;
    }
    auto endStdEval = std::chrono::high_resolution_clock::now();
    auto durStdEval = std::chrono::duration_cast<std::chrono::nanoseconds>(endStdEval - startStdEval).count();

    // ================= 4. Quad-Tree DPF Eval =================
    auto startQuadEval = std::chrono::high_resolution_clock::now();
    int quad_correct = 0;
    for (int i = 0; i < samples; ++i) {
        GroupElement y0 = evalQuadDPF(0, quad_keys[i].first, query_x[i]);
        GroupElement y1 = evalQuadDPF(1, quad_keys[i].second, query_x[i]);
        GroupElement res = y0 + y1;
        mod(res, bout);
        GroupElement expected = (query_x[i] == alpha[i]) ? payload[i] : 0;
        if (res == expected) quad_correct++;
    }
    auto endQuadEval = std::chrono::high_resolution_clock::now();
    auto durQuadEval = std::chrono::duration_cast<std::chrono::nanoseconds>(endQuadEval - startQuadEval).count();

    // ================= 释放内存 =================
    for (int i = 0; i < samples; ++i) {
        freeQuadDPFKeyPackPair(quad_keys[i]);
        freeDPFKeyPackPair(std_keys[i]);
    }
    delete[] quad_keys;
    delete[] std_keys;
    delete[] alpha;
    delete[] payload;
    delete[] query_x;

    // ================= 结果输出 =================
    std::cout << "\n[Correctness]" << std::endl;
    std::cout << "  Standard DPF EQ Accuracy: " << std_correct << " / " << samples << std::endl;
    std::cout << "  Quad-Tree DPF Accuracy:   " << quad_correct << " / " << samples << std::endl;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n[Evaluation Time] (Single Query)" << std::endl;
    std::cout << "  Standard DPF Eval : " << durStdEval / 1e6 << " ms (Avg: " << durStdEval / samples << " ns/op)" << std::endl;
    std::cout << "  Quad-Tree DPF Eval: " << durQuadEval / 1e6 << " ms (Avg: " << durQuadEval / samples << " ns/op)" << std::endl;
    std::cout << "  -> Eval Speedup   : " << (double)durStdEval / durQuadEval << "x" << std::endl;

    std::cout << "\n[Key Generation Time]" << std::endl;
    std::cout << "  Standard DPF Gen  : " << durStdGen / 1e6 << " ms (Avg: " << durStdGen / samples << " ns/op)" << std::endl;
    std::cout << "  Quad-Tree DPF Gen : " << durQuadGen / 1e6 << " ms (Avg: " << durQuadGen / samples << " ns/op)" << std::endl;

    double totalStd = durStdGen + durStdEval;
    double totalQuad = durQuadGen + durQuadEval;

    std::cout << "\n[Total Time (KeyGen + Eval)]" << std::endl;
    std::cout << "  Standard DPF Total: " << totalStd / 1e6 << " ms" << std::endl;
    std::cout << "  Quad-Tree Total   : " << totalQuad / 1e6 << " ms" << std::endl;
    std::cout << "  -> Overall Speedup: " << totalStd / totalQuad << "x" << std::endl;
    std::cout << "=======================================================\n" << std::endl;
}

int main(){
    QuadDPF_vs_DPF_TEST();
    return 0;
}