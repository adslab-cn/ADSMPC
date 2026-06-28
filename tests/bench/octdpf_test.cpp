// 二叉树、四叉树、八叉树的对比，加入AES-IN的工程优化
// 减少密钥生成时间
#include <FSS/dcf.h>
#include <FSS/dpf.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <backend/FSS_base.h>
#include <FSS/freekey.h>

void OctDPF_Benchmark_TEST()
{
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "Starting DPF Benchmarks: Binary vs Quad-Tree vs Oct-Tree..." << std::endl;

    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int samples = 50000;
    int bin = 64;   
    int bout = 64;  

    std::pair<DPFKeyPack, DPFKeyPack> *std_keys = new std::pair<DPFKeyPack, DPFKeyPack>[samples];
    std::pair<QuadDPFKeyPack, QuadDPFKeyPack> *quad_keys = new std::pair<QuadDPFKeyPack, QuadDPFKeyPack>[samples];
    std::pair<OctDPFKeyPack, OctDPFKeyPack> *oct_keys = new std::pair<OctDPFKeyPack, OctDPFKeyPack>[samples];

    GroupElement *alpha = new GroupElement[samples];
    GroupElement *payload = new GroupElement[samples];
    GroupElement *query_x = new GroupElement[samples];

    for (int i = 0; i < samples; ++i) {
        alpha[i] = rand();
        payload[i] = rand();
        query_x[i] = (rand() % 2 == 0) ? alpha[i] : rand();
    }

    // ================= KeyGen 阶段 =================
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; ++i) std_keys[i] = keyGenDPF(bin, bout, alpha[i], payload[i]);
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; ++i) quad_keys[i] = keyGenQuadDPF(bin, bout, alpha[i], payload[i]);
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; ++i) oct_keys[i] = keyGenOctDPF(bin, bout, alpha[i], payload[i]);
    auto t4 = std::chrono::high_resolution_clock::now();

    double durStdGen  = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    double durQuadGen = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    double durOctGen  = std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();

    // ================= Eval 阶段 =================
    int std_correct = 0, quad_correct = 0, oct_correct = 0;

    auto e1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; ++i) {
        // 使用之前的严格版 evalDPF_EQ2 或现有的实现
        GroupElement t0 = evalDPF_EQ(0, std_keys[i].first, query_x[i]);
        GroupElement t1 = evalDPF_EQ(1, std_keys[i].second, query_x[i]);
        if ((t0 ^ t1) == ((query_x[i] == alpha[i]) ? 1 : 0)) std_correct++;
    }
    auto e2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; ++i) {
        GroupElement res = evalQuadDPF(0, quad_keys[i].first, query_x[i]) + evalQuadDPF(1, quad_keys[i].second, query_x[i]);
        mod(res, bout);
        if (res == ((query_x[i] == alpha[i]) ? payload[i] : 0)) quad_correct++;
    }
    auto e3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < samples; ++i) {
        GroupElement res = evalOctDPF(0, oct_keys[i].first, query_x[i]) + evalOctDPF(1, oct_keys[i].second, query_x[i]);
        mod(res, bout);
        if (res == ((query_x[i] == alpha[i]) ? payload[i] : 0)) oct_correct++;
    }
    auto e4 = std::chrono::high_resolution_clock::now();

    double durStdEval  = std::chrono::duration_cast<std::chrono::nanoseconds>(e2 - e1).count();
    double durQuadEval = std::chrono::duration_cast<std::chrono::nanoseconds>(e3 - e2).count();
    double durOctEval  = std::chrono::duration_cast<std::chrono::nanoseconds>(e4 - e3).count();

    // ================= 释放内存 =================
    for (int i = 0; i < samples; ++i) {
        freeDPFKeyPackPair(std_keys[i]);
        freeQuadDPFKeyPackPair(quad_keys[i]);
        freeOctDPFKeyPackPair(oct_keys[i]);
    }
    delete[] std_keys; delete[] quad_keys; delete[] oct_keys;
    delete[] alpha; delete[] payload; delete[] query_x;

    // ================= 输出结果 =================
    std::cout << "\n[Correctness Check]" << std::endl;
    std::cout << "  Standard DPF: " << std_correct << " / " << samples << std::endl;
    std::cout << "  Quad-Tree DPF: " << quad_correct << " / " << samples << std::endl;
    std::cout << "  Oct-Tree DPF:  " << oct_correct << " / " << samples << std::endl;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n[Evaluation Time (Online)]" << std::endl;
    std::cout << "  Binary (Base) : " << durStdEval / 1e6 << " ms" << std::endl;
    std::cout << "  Quad-Tree     : " << durQuadEval / 1e6 << " ms (Speedup: " << durStdEval / durQuadEval << "x)" << std::endl;
    std::cout << "  Oct-Tree      : " << durOctEval / 1e6 << " ms (Speedup: " << durStdEval / durOctEval << "x)" << std::endl;

    std::cout << "\n[KeyGen Time (Offline)]" << std::endl;
    std::cout << "  Binary (Base) : " << durStdGen / 1e6 << " ms" << std::endl;
    std::cout << "  Quad-Tree     : " << durQuadGen / 1e6 << " ms" << std::endl;
    std::cout << "  Oct-Tree      : " << durOctGen / 1e6 << " ms" << std::endl;
    
    std::cout << "\n[Key Size (Communication)]" << std::endl;
    std::cout << "  Binary (Base) : " << (16 + 64 * 16 + 64 * 2 + 8) << " Bytes" << std::endl;
    std::cout << "  Quad-Tree     : " << (16 + 32 * 4 * 16 + 32 * 1 + 8) << " Bytes" << std::endl;
    std::cout << "  Oct-Tree      : " << (16 + 22 * 8 * 16 + 22 * 1 + 8) << " Bytes" << std::endl;

    std::cout << "=======================================================\n" << std::endl;
}

int main(){
    OctDPF_Benchmark_TEST();
    return 0;
}