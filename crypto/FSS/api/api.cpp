#include "../aux_parameter/comms.h"
#include "../aux_parameter/utils.h"
#include "../aux_parameter/array.h"
#include "../aux_parameter/config.h"
#include "../aux_parameter/stats.h"
#include "../aux_parameter/assert.h"
#include "../aux_parameter/freekey.h"
#include "api.h"

#include "../protocol/conv.h"
#include "../protocol/and.h"
// #include "../protocol/mult.h"
#include "../protocol/pubdiv.h"
#include "../protocol/relu.h"
// #include "../protocol/reluextend.h"
// #include "../protocol/signextend.h"
// #include "../protocol/clip.h"
#include "../primitives/dcf.h"
#include "../protocol/lut.h"
#include "../protocol/select.h"
// #include "../protocol/fixtobfloat16.h"
// #include "../protocol/wrap.h"
#include "../primitives/dpf.h"
// #include "../protocol/taylor.h"
#include "../protocol/float.h"
#include "../protocol/dpfsort.h"

#include <cassert>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <thread>
#include <Eigen/Dense>
#include <bitpack/bitpack.h>


template <typename T>
using pair = std::pair<T, T>;

bool localTruncation = false;

using namespace FSSConfig;

/* 记录运行时间 */
template <typename Functor>
uint64_t time_this_block(Functor f)
{
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

/* 记录通信开销 */
template <typename Functor>
auto time_comm_this_block(Functor f)
{
    uint64_t comm_start = peer->bytesReceived() + peer->bytesSent();
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    uint64_t comm_end = peer->bytesReceived() + peer->bytesSent();
    return std::make_pair((uint64_t)(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()), comm_end - comm_start);
}
void prng_shared_init() {
    if (party == DEALER) {
        osuCrypto::AES aesSeed(prngs[0].get<osuCrypto::block>());
        auto commonSeed = aesSeed.ecbEncBlock(osuCrypto::ZeroBlock);
        
        server->send_block(commonSeed);
        client->send_block(commonSeed); 
        
        prngShared.SetSeed(commonSeed);
    } else { 
        auto commonSeed = dealer->recv_block();
        prngShared.SetSeed(commonSeed);
    }
}
/* 初始化通信、同步、计时器 */
void FSS::start()
{
    FSS::stats.clear();
    // std::cerr << "=== COMPUTATION START ===\n\n";
    if (party != DEALER)
        peer->sync();

    startTime = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    if (party != DEALER)
    {
        if (party == SERVER)
        {
            inputOfflineComm = peer->bytesSent();
            inputOnlineComm = peer->bytesReceived();
        }
        else
        {
            inputOfflineComm = peer->bytesReceived();
            inputOnlineComm = peer->bytesSent();
        }
        peer->zeroBytesSent();
        peer->zeroBytesReceived();
    }
    else
    {
        // always_assert(server->bytesSent() == 16);
        // always_assert(server->bytesSent() == 16);
        server->zeroBytesSent();
        client->zeroBytesSent();
    }

    // if (party == DEALER)
    // {
    //     osuCrypto::AES aesSeed(prngs[0].get<osuCrypto::block>());
    //     auto commonSeed = aesSeed.ecbEncBlock(osuCrypto::ZeroBlock);
    //     server->send_block(commonSeed);
    //     prngShared.SetSeed(commonSeed);
    // }
    // else if (party == SERVER)
    // {
    //     auto commonSeed = dealer->recv_block();
    //     prngShared.SetSeed(commonSeed);
    // }
    sendTime = 0;
    recvTime = 0;
    packTime = 0;
    unpackTime = 0;
}

/* 输出性能统计（通信量、耗时、密钥读取时间 */
void FSS::end()
{
    // std::cerr << "\n=== COMPUTATION END ===\n\n";
    if (party != DEALER)
    {
        uint64_t agg_time = 0;
        uint64_t recons_time = 0;
        uint64_t keyread_time = 0;
        for (auto &func : FSS::stats)
        {
            uint64_t online_time = func.second.compute_time + func.second.reconstruct_time;
            agg_time += online_time;
            recons_time += func.second.reconstruct_time;
            keyread_time += func.second.keyread_time;
        }
        std::cerr << "Offline Communication = " << inputOfflineComm << " bytes\n";
        std::cerr << "Offline Time = " << accumulatedInputTimeOffline / 1000.0 << " milliseconds\n";
        std::cerr << "Online Rounds = " << numRounds << "\n";
        std::cerr << "Online Communication = " << (peer->bytesSent() + peer->bytesReceived() + inputOnlineComm + secFloatComm) /*/ (1024.0 * 1024.0)*/ << " B\n";
        std::cerr << "Input Online Communication = " << (inputOnlineComm) << " B\n";
        std::cerr << "Secfloat Online Communication = " << (secFloatComm) /*/ (1024.0 * 1024.0)*/ << " B\n";

        std::cerr << "Online Time = " << (evalMicroseconds + accumulatedInputTimeOnline + agg_time) / 1000.0 << " milliseconds\n";
        std::cerr << "Key Read Time = " << keyread_time / 1000.0 << " milliseconds\n";
        std::cerr << "Total Eigen Time = " << eigenMicroseconds / 1000.0 << " milliseconds\n";
        auto endTime = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::cerr << "Key Read Time = " << keyread_time / 1000.0 << " milliseconds\n";
        std::cerr << "Total Time (including Key Read) = " << (endTime - startTime) / 1000000.0 << " milliseconds\n";

        std::cerr << "packTime = " << packTime / 1000.0 << " miliseconds\n";
        std::cerr << "sendTime = " << sendTime / 1000.0 << " miliseconds\n";
        std::cerr << "recvTime = " << recvTime / 1000.0 << " miliseconds\n";
        std::cerr << "unpackTime = " << unpackTime / 1000.0 << " miliseconds\n";
        std::cerr << "reconsTime = " << recons_time / 1000.0 << " miliseconds\n";
        std::cerr << "accumulatedInputTimeOnline = " << accumulatedInputTimeOnline / 1000.0 << " miliseconds\n";

        if (convEvalMicroseconds > 0)
            std::cerr << "Conv Time = " << convEvalMicroseconds / 1000.0 << " milliseconds\n";
        if (arsEvalMicroseconds > 0)
            std::cerr << "ARS Time = " << arsEvalMicroseconds / 1000.0 << " milliseconds\n";

        if (convOnlineComm > 0)
            std::cerr << "Conv Online Communication = " << convOnlineComm << " bytes\n";
        if (arsOnlineComm > 0)
            std::cerr << "ARS Online Communication = " << arsOnlineComm << " bytes\n";

        FSS::dump_stats_csv("FSS" + std::to_string(party) + ".csv");
    }
    std::cerr << "=========\n";
}

const bool parallel_reconstruct = true;
const bool doPack = false;

/* 高效压缩整数数组，显著减少传输数据量 */
inline void pack_wrapper(GroupElement *dst, const GroupElement *src, std::size_t n, int bw)
{
    auto start = std::chrono::high_resolution_clock::now();
    bitpack::pack(dst, src, n, bw);
    auto end = std::chrono::high_resolution_clock::now();
    packTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

inline void unpack_wrapper(GroupElement *dst, const GroupElement *src, std::size_t n, int bw)
{
    auto start = std::chrono::high_resolution_clock::now();
    bitpack::unpack(dst, src, n, bw);
    auto end = std::chrono::high_resolution_clock::now();
    unpackTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

/* 位打包压缩，​减少通信量 */
void packed_reconstruct(int32_t size, GroupElement *arr, int bw)
{
    auto psize = bitpack::packed_size(size, bw);
    GroupElement *packedArr = new GroupElement[psize];
    GroupElement *packedTmp = new GroupElement[psize];
    pack_wrapper(packedArr, arr, size, bw);

    if (parallel_reconstruct)
    {
#pragma omp parallel sections
        {
#pragma omp section
            {
                peer->send_batched_input(packedArr, psize, 64);
            }

#pragma omp section
            {
                peer->recv_batched_input(packedTmp, psize, 64);
            }
        }
    }
    else
    {
        peer->send_batched_input(packedArr, psize, 64);
        peer->recv_batched_input(packedTmp, psize, 64);
    }

    GroupElement *tmp = new GroupElement[size];
    unpack_wrapper(tmp, packedTmp, size, bw);

    for (int i = 0; i < size; i++)
    {
        arr[i] = arr[i] + tmp[i];
    }
    delete[] tmp;
    delete[] packedArr;
    delete[] packedTmp;
    numRounds += 1;
}

void reconstruct(int32_t size, GroupElement *arr, int bw)
{
    if (doPack)
    {
        return packed_reconstruct(size, arr, bw);
    }
    uint64_t *tmp = new uint64_t[size];

    if (parallel_reconstruct)
    {
#pragma omp parallel sections
        {
#pragma omp section
            {
                // 发送本地片段
                peer->send_batched_input(arr, size, bw);
            }

#pragma omp section
            {
                // 接收对方片段
                peer->recv_batched_input(tmp, size, bw);
            }
        }
    }
    else
    {
        peer->send_batched_input(arr, size, bw);
        peer->recv_batched_input(tmp, size, bw);
    }
    for (int i = 0; i < size; i++)
    {
        // 本地合并
        arr[i] = arr[i] + tmp[i];
    }
    delete[] tmp;
    numRounds += 1;
}

void serverReconstruct(int32_t size, GroupElement *arr, int bw)
{
    // TODO: do packing
    if (party == CLIENT)
    {
        peer->send_batched_input(arr, size, bw);
    }
    else
    {
        uint64_t *tmp = new uint64_t[size];
        peer->recv_batched_input(tmp, size, bw);
        for (int i = 0; i < size; i++)
        {
            arr[i] = arr[i] + tmp[i];
        }
        delete[] tmp;
    }
    numRounds += 1;
}

void serverToClient(int32_t size, GroupElement *arr, int bw)
{
    // TODO: do packing
    if (party == SERVER)
    {
        peer->send_batched_input(arr, size, bw);
    }
    else
    {
        peer->recv_batched_input(arr, size, bw);
    }
    numRounds += 1;
}

/* 用在LUT_dpfet，相比reconstruct，​​reconstructRT额外处理了一个布尔位*/
void reconstructRT(int32_t size, GroupElement *arr, int bw)
{
    int bitarraySize = size % 8 == 0 ? size / 8 : size / 8 + 1;

    uint8_t *tmp2 = new uint8_t[bitarraySize];
    uint8_t *tmp3 = new uint8_t[bitarraySize];

    packBitArray(arr + size, size, tmp2);

    uint64_t *tmp = new uint64_t[size];
    GroupElement *sendArr, *recvArr;
    int sendSize;
    if (doPack)
    {
        auto psize = bitpack::packed_size(size, bw);
        sendArr = new GroupElement[psize];
        recvArr = new GroupElement[psize];
        pack_wrapper(sendArr, arr, size, bw);
        sendSize = psize;
    }
    else
    {
        sendArr = arr;
        recvArr = tmp;
        sendSize = size;
    }

    if (parallel_reconstruct)
    {
#pragma omp parallel sections
        {
#pragma omp section
            {
                peer->send_batched_input(sendArr, sendSize, (doPack ? 64 : bw));
                peer->send_uint8_array(tmp2, bitarraySize);
            }

#pragma omp section
            {
                peer->recv_batched_input(recvArr, sendSize, (doPack ? 64 : bw));
                peer->recv_uint8_array(tmp3, bitarraySize);
            }
        }
    }
    else
    {
        peer->send_batched_input(sendArr, sendSize, (doPack ? 64 : bw));
        peer->send_uint8_array(tmp2, bitarraySize);
        peer->recv_batched_input(recvArr, sendSize, (doPack ? 64 : bw));
        peer->recv_uint8_array(tmp3, bitarraySize);
    }

    if (doPack)
    {
        unpack_wrapper(tmp, recvArr, size, bw);
        delete[] sendArr;
        delete[] recvArr;
    }

    for (int i = 0; i < size; i++)
    {
        arr[i] = arr[i] + tmp[i];
        arr[i + size] = arr[i + size] + ((tmp3[i / 8] >> (i % 8)) & 1);
    }

    delete[] tmp;
    delete[] tmp2;
    delete[] tmp3;
    numRounds += 1;
}

/* 将数据分块由不同线程处理，加速本地计算 */
inline std::pair<int32_t, int32_t> get_start_end(int32_t size, int32_t thread_idx)
{
    int32_t chunk_size = size / num_threads;
    if (thread_idx == num_threads - 1)
    {
        return std::make_pair(thread_idx * chunk_size, size);
    }
    else
    {
        return std::make_pair(thread_idx * chunk_size, (thread_idx + 1) * chunk_size);
    }
}


// ===========================矩阵乘法==================================
// ===========================矩阵乘法==================================
// ===========================矩阵乘法==================================
// ===========================矩阵乘法==================================
// ===========================矩阵乘法==================================

/* 矩阵乘法辅助函数 */
inline void matmul2d_server_helper(int thread_idx, int s1, int s2, int s3, GroupElement *A, GroupElement *B, GroupElement *C, GroupElement *a, GroupElement *b, GroupElement *c)
{
    auto p = get_start_end(s1 * s3, thread_idx);
    for (int ik = p.first; ik < p.second; ik += 1)
    {
        int i = ik / s3;
        int k = ik % s3;
        Arr2DIdx(C, s1, s3, i, k) = Arr2DIdx(c, s1, s3, i, k);
        for (int j = 0; j < s2; j++)
        {
            Arr2DIdx(C, s1, s3, i, k) = Arr2DIdx(C, s1, s3, i, k) - Arr2DIdx(A, s1, s2, i, j) * Arr2DIdx(b, s2, s3, j, k) - Arr2DIdx(a, s1, s2, i, j) * Arr2DIdx(B, s2, s3, j, k) + Arr2DIdx(A, s1, s2, i, j) * Arr2DIdx(B, s2, s3, j, k);
        }
        // mod(Arr2DIdx(C, s1, s3, i, k));
    }
}

inline void matmul2d_client_helper(int thread_idx, int s1, int s2, int s3, GroupElement *A, GroupElement *B, GroupElement *C, GroupElement *a, GroupElement *b, GroupElement *c)
{
    auto p = get_start_end(s1 * s3, thread_idx);
    for (int ik = p.first; ik < p.second; ik += 1)
    {
        int i = ik / s3;
        int k = ik % s3;
        Arr2DIdx(C, s1, s3, i, k) = Arr2DIdx(c, s1, s3, i, k);
        for (int j = 0; j < s2; j++)
        {
            Arr2DIdx(C, s1, s3, i, k) = Arr2DIdx(C, s1, s3, i, k) - Arr2DIdx(A, s1, s2, i, j) * Arr2DIdx(b, s2, s3, j, k) - Arr2DIdx(a, s1, s2, i, j) * Arr2DIdx(B, s2, s3, j, k);
        }
        // mod(Arr2DIdx(C, s1, s3, i, k));
    }
}

/* 矩阵乘法 */
void MatMul2D(int32_t s1, int32_t s2, int32_t s3, MASK_PAIR(GroupElement *A),
              MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C), bool modelIsA)
{
    if (party == DEALER)
    {

        for (int i = 0; i < s1; ++i)
        {
            for (int j = 0; j < s3; ++j)
            {
                Arr2DIdx(C_mask, s1, s3, i, j) = random_ge(bitlength);
            }
        }

        auto keys = KeyGenMatMul(bitlength, bitlength, s1, s2, s3, A_mask, B_mask, C_mask);

        // server->send_matmul_key(keys.first);
        freeMatMulKey(keys.first);
        client->send_matmul_key(keys.second);
        freeMatMulKey(keys.second);
    }
    else
    {
        MatMulKey key;
        uint64_t keysize_start = dealer->bytesReceived();
        auto keyread_time = time_this_block([&]()
                                            { key = dealer->recv_matmul_key(bitlength, bitlength, s1, s2, s3); });

        peer->sync();

        auto compute_time = time_this_block([&]()
                                            { matmul_eval_helper(party, s1, s2, s3, A, B, C, key.a, key.b, key.c); });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(s1 * s3, C, bitlength); });

        FSS::stat_t stat = {"Linear::MatMul", keyread_time, compute_time, reconstruction_stats.first, reconstruction_stats.second, dealer->bytesReceived() - keysize_start};
        stat.print();
        FSS::push_stats(stat);

        freeMatMulKey(key);
    }
}

void Conv2DWrapper(int32_t N, int32_t H, int32_t W,
                   int32_t CI, int32_t FH, int32_t FW,
                   int32_t CO, int32_t zPadHLeft,
                   int32_t zPadHRight, int32_t zPadWLeft,
                   int32_t zPadWRight, int32_t strideH,
                   int32_t strideW, MASK_PAIR(GroupElement *inputArr), MASK_PAIR(GroupElement *filterArr),
                   MASK_PAIR(GroupElement *outArr))
{
    std::cerr << ">> Conv2D - Start" << std::endl;
    int d0 = N;
    int d1 = ((H - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    int d2 = ((W - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    int d3 = CO;

    if (party == DEALER)
    {
        auto local_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < d0; ++i)
        {
            for (int j = 0; j < d1; ++j)
            {
                for (int k = 0; k < d2; ++k)
                {
                    for (int l = 0; l < d3; ++l)
                    {
                        Arr4DIdx(outArr_mask, d0, d1, d2, d3, i, j, k, l) = random_ge(bitlength);
                    }
                }
            }
        }

        auto keys = KeyGenConv2D(bitlength, bitlength, N, H, W, CI, FH, FW, CO,
                                 zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW,
                                 inputArr_mask, filterArr_mask, outArr_mask);

        auto local_end = std::chrono::high_resolution_clock::now();

        // server->send_conv2d_key(keys.first);
        freeConv2dKey(keys.first);
        client->send_conv2d_key(keys.second);
        freeConv2dKey(keys.second);
        auto local_time_taken = std::chrono::duration_cast<std::chrono::microseconds>(local_end -
                                                                                      local_start)
                                    .count();
        dealerMicroseconds += local_time_taken;
        std::cerr << "   Dealer Time = " << local_time_taken / 1000.0 << " milliseconds\n";
    }
    else
    {

        auto keyread_start = std::chrono::high_resolution_clock::now();
        auto key = dealer->recv_conv2d_key(bitlength, bitlength, N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW);
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end - keyread_start).count();
        
        peer->sync();
        uint64_t eigen_start = eigenMicroseconds;
        auto local_start = std::chrono::high_resolution_clock::now();
        EvalConv2D(party, key, N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, outArr);
        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t onlineComm0 = peer->bytesReceived() + peer->bytesSent();
        reconstruct(d0 * d1 * d2 * d3, outArr, bitlength);
        uint64_t onlineComm1 = peer->bytesReceived() + peer->bytesSent();
        convOnlineComm += (onlineComm1 - onlineComm0);
        auto local_end = std::chrono::high_resolution_clock::now();

        freeConv2dKey(key);
        auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 -
                                                                                  local_start)
                                .count();
        auto reconstruct_time = std::chrono::duration_cast<std::chrono::microseconds>(local_end -
                                                                                      t1)
                                    .count();
        convEvalMicroseconds += (reconstruct_time + compute_time);
        evalMicroseconds += (reconstruct_time + compute_time);
        std::cerr << "   Key Read Time = " << keyread_time_taken << " milliseconds\n";
        std::cerr << "   Compute Time = " << compute_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Eigen Time = " << (eigenMicroseconds - eigen_start) / 1000.0 << " milliseconds\n";
        std::cerr << "   Reconstruct Time = " << reconstruct_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Time = " << (reconstruct_time + compute_time) / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Comm = " << (onlineComm1 - onlineComm0) << " bytes\n";
    }

    std::cerr << ">> Conv2D - End" << std::endl;
}

void fixtofloat_threads_helper(int thread_idx, int32_t size, int scale, GroupElement *inp, GroupElement *out, GroupElement *pl, GroupElement *q,
                               GroupElement *pow, GroupElement *sm, FixToFloatKeyPack *keys)
{
    auto p = get_start_end(size, thread_idx);
    for (int i = p.first; i < p.second; i += 1)
    {
        evalFixToFloat_1(party - 2, bitlength, scale, inp[i], keys[i], pl, q,
                         out[i * 4 + 0], out[i * 4 + 1], out[i * 4 + 2], out[i * 4 + 3], pow[i], sm[i]);
    }
}

void FixToFloat(int size, GroupElement *inp, GroupElement *out, int scale)
{
    // std::cerr << ">> FixToFloat - Start" << std::endl;
    GroupElement *p = new GroupElement[2 * bitlength];
    GroupElement *q = new GroupElement[2 * bitlength];
    fill_pq(p, q, bitlength);

    if (party == DEALER)
    {
        pair<FixToFloatKeyPack> *keys = new pair<FixToFloatKeyPack>[size];
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            keys[i] = keyGenFixToFloat(bitlength, scale, inp[i], p, q);
            out[4 * i] = 0;
            out[4 * i + 1] = 0;
            out[4 * i + 2] = 0;
            out[4 * i + 3] = 0;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_fix_to_float_key(keys[i].first, bitlength);
            client->send_fix_to_float_key(keys[i].second, bitlength);
            freeFixToFloatKeyPackPair(keys[i]);
        }
        delete[] keys;
    }
    else
    {
        auto keyread_start = std::chrono::high_resolution_clock::now();
        FixToFloatKeyPack *keys = new FixToFloatKeyPack[size];

        for (int i = 0; i < size; ++i)
        {
            keys[i] = dealer->recv_fix_to_float_key(bitlength);
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                                                        keyread_start)
                                      .count();
        GroupElement *pow = new GroupElement[size];
        GroupElement *sm = new GroupElement[size];
        GroupElement *ym = new GroupElement[size];

        peer->sync();
        auto eval_start = std::chrono::high_resolution_clock::now();

        std::thread thread_pool[num_threads];
        for (int i = 0; i < num_threads; ++i)
        {
            thread_pool[i] = std::thread(fixtofloat_threads_helper, i, size, scale, inp, out, p, q, pow, sm, keys);
        }

        for (int i = 0; i < num_threads; ++i)
        {
            thread_pool[i].join();
        }

        reconstruct(size, sm, 1);
        reconstruct(size, pow, bitlength);

        for (int i = 0; i < size; ++i)
        {
            ym[i] = 2 * evalSelect(party - 2, 1 ^ sm[i], inp[i], keys[i].selectKey);
            if (party == 2)
            {
                ym[i] = ym[i] - inp[i];
            }
        }

        reconstruct(size, ym, bitlength);

        for (int i = 0; i < size; ++i)
        {
            out[i * 4 + 0] = -keys[i].ry * pow[i] - keys[i].rpow * ym[i] + keys[i].rm;
            if (party == 2)
            {
                out[i * 4 + 0] = out[i * 4 + 0] + ym[i] * pow[i];
                out[i * 4 + 0] = -((-out[i * 4 + 0]) >> (bitlength - scale));
            }
            else
            {
                out[i * 4 + 0] = out[i * 4 + 0] >> (bitlength - scale);
            }
        }

        auto eval_end = std::chrono::high_resolution_clock::now();
        auto eval_time_taken = std::chrono::duration_cast<std::chrono::microseconds>(eval_end -
                                                                                     eval_start)
                                   .count();
        // std::cerr << "   Key Read Time = " << keyread_time_taken << " miliseconds" << std::endl;
        // std::cerr << "   Online Time = " << eval_time_taken / 1000.0 << " miliseconds" << std::endl;
        evalMicroseconds += eval_time_taken;
        delete[] sm;
        delete[] pow;
        delete[] ym;
        for (int i = 0; i < size; ++i)
        {
            freeFixToFloatKeyPack(keys[i]);
        }
        delete[] keys;
    }
    // std::cerr << ">> FixToFloat - End" << std::endl;
}

void FloatToFixCt(int size, GroupElement *inp, GroupElement *out, int scale)
{
    if (party == DEALER)
    {
        memset(out, 0, size * sizeof(GroupElement));
    }
    else
    {
        GroupElement *m = new GroupElement[2 * size];
        GroupElement *e = m + size;

        for (int i = 0; i < size; ++i)
        {
            m[i] = inp[4 * i + 0];
            e[i] = inp[4 * i + 1];
            // if (party == 2)
            // {
            //     e[i] += scale;
            //     e[i] -= 127; // fp32 bias
            // }
        }
        // now have m and e in the clear
        // reconstruct(2 * size, m, 64);
        for (int i = 0; i < size; ++i)
        {
            mod(m[i], 24);
            mod(e[i], 10);
            assert(e[i] < 256);

            // int eAsInt = e[i] < 512 ? e[i] : -1 * (1024 - e[i]);
            // assert(eAsInt <= 126 && eAsInt >= -127);
            // if(i < 10) printf("%d=%ld, %ld\n", i, m[i], e[i]);
            int ePrime = e[i] - 127 + scale;
            // if(i < 10) printf("%d=%ld, %ld, %d\n", i, m[i], e[i], ePrime);
            GroupElement x = 0;
            if (ePrime >= 0 && ePrime <= scale)
            {
                x = m[i] * (1ULL << ePrime);
                assert(x < (1ULL << 63));
                x >>= 23;
                // auto xf = x;
                // mod(xf, scale);
                // auto s = random_ge(scale);
                // if(s < xf) x += 1;
                // if(i < 10) printf("%d=%ld, %ld, %ld\n", i, m[i], ePrime, x);
            }
            out[i] = x;
        }
        delete[] m;
    }
}

void FloatToFix(int size, GroupElement *inp, GroupElement *out, int scale)
{
    // std::cerr << ">> FloatToFix - Start" << std::endl;

    if (party == DEALER)
    {
        pair<FloatToFixKeyPack> *keys = new pair<FloatToFixKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            auto rout = random_ge(bitlength);
            keys[i] = keyGenFloatToFix(bitlength, scale, rout);
            out[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_float_to_fix_key(keys[i].first, bitlength);
            client->send_float_to_fix_key(keys[i].second, bitlength);
            freeFloatToFixKeyPackPair(keys[i]);
        }
        delete[] keys;
    }
    else
    {
        auto keyread_start = std::chrono::high_resolution_clock::now();
        FloatToFixKeyPack *keys = new FloatToFixKeyPack[size];
        for (int i = 0; i < size; ++i)
        {
            keys[i] = dealer->recv_float_to_fix_key(bitlength);
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                                                        keyread_start)
                                      .count();

        GroupElement *m = new GroupElement[2 * size];
        GroupElement *e = m + size;
        GroupElement *w = new GroupElement[2 * size];
        GroupElement *t = new GroupElement[size];
        GroupElement *h = w + size;
        GroupElement *d = new GroupElement[size];

        peer->sync();
        auto eval_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; ++i)
        {
            m[i] = inp[4 * i + 0] + keys[i].rm;
            e[i] = inp[4 * i + 1] + keys[i].re;
            if (party == 2)
            {
                e[i] += (scale);
                e[i] -= 127; // fp32 bias
            }
        }

        // m and e are in a single array. m is the first half and e is the second half
        reconstruct(2 * size, m, 24);

        for (int i = 0; i < size; ++i)
        {
            mod(m[i], 24);
            evalDCF(party - 2, &w[i], m[i], keys[i].dcfKey);
            w[i] = w[i] + keys[i].rw;
        }

        for (int i = 0; i < size; i++)
        {
            mod(e[i], 10);
            d[i] = 0;
            for (int j = 0; j < 1024; j++)
            {
                d[i] = d[i] + (pow_helper(scale, j) * keys[i].p[(j - e[i]) % 1024]);
            }
            h[i] = keys[i].rh + (pow((GroupElement)2, 24) * d[i]);
        }

        // w and h are in a single array w. w is the first half and h is the second half
        reconstruct(2 * size, w, bitlength);

        for (int i = 0; i < size; ++i)
        {
            t[i] = evalSelect(party - 2, w[i], h[i], keys[i].selectKey);
            t[i] = t[i] + keys[i].q[e[i]];
            t[i] = t[i] + (m[i] * d[i]);
        }

        reconstruct(size, t, bitlength);

        for (int i = 0; i < size; ++i)
        {
            out[i] = evalARS(party - 2, t[i], 23, keys[i].arsKey);
        }

        // reconstruct(size, out, bitlength);

        auto eval_end = std::chrono::high_resolution_clock::now();
        auto eval_time_taken = std::chrono::duration_cast<std::chrono::microseconds>(eval_end -
                                                                                     eval_start)
                                   .count();
        // std::cerr << "   Key Read Time = " << keyread_time_taken << " miliseconds" << std::endl;
        // std::cerr << "   Online Time = " << eval_time_taken / 1000.0 << " miliseconds" << std::endl;
        evalMicroseconds += eval_time_taken;
        delete[] m;
        delete[] w;
        delete[] t;
        for (int i = 0; i < size; ++i)
        {
            freeFloatToFixKeyPack(keys[i]);
        }
        delete[] keys;
    }
    // std::cerr << ">> FloatToFix - End" << std::endl;
}

void Select(int32_t size, int bin, GroupElement *s, GroupElement *x, GroupElement *out, std::string prefix, bool doReconstruct)
{

    if (party == DEALER)
    {
        pair<SelectKeyPack> *keys = new pair<SelectKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            auto rout = random_ge(bin);
            keys[i] = keyGenSelect(bin, s[i], x[i], rout);
            out[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_select_key(keys[i].first);
            client->send_select_key(keys[i].second);
        }
    }
    else
    {
        SelectKeyPack *keys = new SelectKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        uint64_t keyread_time = time_this_block([&]()
                                                {
            for(int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_select_key(bin);
            } });

        peer->sync();

        uint64_t compute_time = time_this_block([&]()
                                                {
#pragma omp parallel for
            for(int i = 0; i < size; ++i) {
                out[i] = evalSelect(party - 2, s[i], x[i], keys[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         {
            if (doReconstruct)
                reconstruct(size, out, bin); });

        FSS::stat_t stat = {
            prefix + "Select",
            keyread_time,
            compute_time,
            reconstruction_stats.first,
            reconstruction_stats.second,
            dealer->bytesReceived() - keysize_start};
        stat.print();
        FSS::push_stats(stat);

        delete[] keys;
    }
}

void Select(int32_t size, GroupElement *s, GroupElement *x, GroupElement *out, std::string prefix, bool doReconstruct)
{
    Select(size, bitlength, s, x, out, prefix, doReconstruct);
}

/* 算数左移 */
void ScaleUp(int32_t size, MASK_PAIR(GroupElement *inArr), int32_t sf)
{
    if (party == DEALER)
    {
        for (int i = 0; i < size; ++i)
        {
            inArr_mask[i] = inArr_mask[i] << sf;
        }
    }
    else
    {
        for (int i = 0; i < size; ++i)
        {
            inArr[i] = inArr[i] << sf;
        }
    }
}

void ars_threads_helper(int thread_idx, int32_t size, GroupElement *inArr, GroupElement *outArr, ARSKeyPack *keys)
{
    auto p = get_start_end(size, thread_idx);
    for (int i = p.first; i < p.second; i += 1)
    {
        outArr[i] = evalARS(party - 2, inArr[i], keys[i].shift, keys[i]);
        freeARSKeyPack(keys[i]);
    }
}

void ARS(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), int32_t shift)
{
    std::cerr << ">> Truncate" << (FSSConfig::stochasticT ? " (stochastic)" : "") << " - Start" << std::endl;
    if (party == DEALER)
    {
        pair<ARSKeyPack> *keys = new pair<ARSKeyPack>[size];
        auto dealer_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            GroupElement rout = random_ge(bitlength);
            keys[i] = keyGenARS(bitlength, bitlength, shift, inArr_mask[i], rout);
            outArr_mask[i] = rout;
        }
        auto dealer_end = std::chrono::high_resolution_clock::now();
        auto dealer_time_taken = std::chrono::duration_cast<std::chrono::microseconds>(dealer_end -
                                                                                       dealer_start)
                                     .count();

        for (int i = 0; i < size; i++)
        {
            server->send_ars_key(keys[i].first);
            client->send_ars_key(keys[i].second);
            freeARSKeyPackPair(keys[i]);
        }
        dealerMicroseconds += dealer_time_taken;
        delete[] keys;
        std::cerr << "   Dealer Time = " << dealer_time_taken / 1000.0 << " milliseconds\n";
    }
    else
    {
        ARSKeyPack *keys = new ARSKeyPack[size];
        auto keyread_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; i++)
        {
            keys[i] = dealer->recv_ars_key(bitlength, bitlength, shift);
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                                                        keyread_start)
                                      .count();

        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];
        for (int i = 0; i < num_threads; ++i)
        {
            thread_pool[i] = std::thread(ars_threads_helper, i, size, inArr, outArr, keys);
        }

        for (int i = 0; i < num_threads; ++i)
        {
            thread_pool[i].join();
        }
        auto mid = std::chrono::high_resolution_clock::now();

        uint64_t onlineComm0 = peer->bytesReceived() + peer->bytesSent();
        reconstruct(size, outArr, bitlength);
        uint64_t onlineComm1 = peer->bytesReceived() + peer->bytesSent();
        arsOnlineComm += (onlineComm1 - onlineComm0);

        auto end = std::chrono::high_resolution_clock::now();
        auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start).count();
        auto reconstruct_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();
        std::cerr << "   Key Read Time = " << keyread_time_taken << " milliseconds\n";
        std::cerr << "   Compute Time = " << compute_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Reconstruct Time = " << reconstruct_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Time = " << (reconstruct_time + compute_time) / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Comm = " << (onlineComm1 - onlineComm0) << " bytes\n";
        evalMicroseconds += (reconstruct_time + compute_time);
        arsEvalMicroseconds += (reconstruct_time + compute_time);
        delete[] keys;
    }
    std::cerr << ">> Truncate - End" << std::endl;
}

void ScaleDown(int32_t size, MASK_PAIR(GroupElement *inArr), int32_t sf)
{
    std::cerr << ">> ScaleDown - Start " << std::endl;

    if (localTruncation)
    {
        uint64_t m = ((1L << sf) - 1) << (bitlength - sf);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; i++)
        {
            if (party == DEALER)
            {
                auto x_msb = msb(inArr_mask[i], bitlength);
                inArr_mask[i] = x_msb ? (inArr_mask[i] >> sf) | m : inArr_mask[i] >> sf;
                mod(inArr_mask[i], bitlength);
            }
            else
            {
                auto x_msb = msb(inArr[i], bitlength);
                inArr[i] = x_msb ? (inArr[i] >> sf) | m : inArr[i] >> sf;
                mod(inArr[i], bitlength);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto timeMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        if (party == DEALER)
        {
            dealerMicroseconds += timeMicroseconds;
        }
        else
        {
            evalMicroseconds += timeMicroseconds;
            arsEvalMicroseconds += timeMicroseconds;
            std::cerr << "   Eval Time = " << timeMicroseconds / 1000.0 << " milliseconds\n";
        }
    }
    else
    {
        ARS(size, inArr, inArr_mask, inArr, inArr_mask, sf);
    }
    std::cerr << ">> ScaleDown - End " << std::endl;
}


void Relu(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *drelu, std::string prefix)
{
    if (party == DEALER)
    {

        pair<ReluKeyPack> *keys = new pair<ReluKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; i += 1)
        {
            auto rout = random_ge(bitlength);
            drelu[i] = random_ge(1);
            keys[i] = keyGenRelu(bitlength, bitlength, inArr_mask[i], rout, drelu[i]);
            outArr_mask[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_relu_key(keys[i].first);
            client->send_relu_key(keys[i].second);
            freeReluKeyPackPair(keys[i]);
        }

        delete[] keys;
    }
    else
    {
        ReluKeyPack *keys = new ReluKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        auto keyread_time = time_this_block([&]()
                                            {
            for(int i = 0; i < size; i++){
                keys[i] = dealer->recv_relu_key(bitlength, bitlength);
            } });

        peer->sync();

        auto compute_time = time_this_block([&]()
                                            {
#pragma omp parallel for
            for(int i = 0; i < size; i++)
            {
                outArr[i] = evalRelu(party - 2, inArr[i], keys[i], &drelu[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         {
            reconstruct(size, outArr, bitlength);
            reconstruct(size, drelu, 1); });

        FSS::stat_t stat = {prefix + "ReLU-Spline", keyread_time, compute_time, reconstruction_stats.first, reconstruction_stats.second, dealer->bytesReceived() - keysize_start};
        stat.print();
        FSS::push_stats(stat);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            freeReluKeyPack(keys[i]);
        }
        delete[] keys;
    }
}

#define BIG_LOOPY(e)                        \
    for (int n = 0; n < N; ++n)             \
    {                                       \
        for (int h = 0; h < H; ++h)         \
        {                                   \
            for (int w = 0; w < W; ++w)     \
            {                               \
                for (int c = 0; c < C; ++c) \
                {                           \
                    e;                      \
                }                           \
            }                               \
        }                                   \
    }



void MaxPool(int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH,
             int32_t FW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *oneHot, std::string prefix)
{
    std::cerr << ">> MaxPool - Start" << std::endl;
    int d1 = ((imgH - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    int d2 = ((imgW - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    always_assert(d1 == H);
    always_assert(d2 == W);
    always_assert(N1 == N);
    always_assert(C1 == C);

    GroupElement *maxUntilNow = outArr;
    GroupElement *maxUntilNow_mask = outArr_mask;

    if (party == DEALER)
    {
        uint64_t dealer_file_read_time = 0;
        auto dealer_start = std::chrono::high_resolution_clock::now();
        for (int fh = 0; fh < FH; fh++)
        {
            for (int fw = 0; fw < FW; fw++)
            {
                for (int n = 0; n < N; n++)
                {
                    for (int c = 0; c < C; c++)
                    {
                        for (int ctH = 0; ctH < H; ctH++)
                        {
                            for (int ctW = 0; ctW < W; ctW++)
                            {
                                int leftTopCornerH = ctH * strideH - zPadHLeft;
                                int leftTopCornerW = ctW * strideW - zPadWLeft;

                                if (fh == 0 && fw == 0)
                                {
                                    if (leftTopCornerH < 0 || leftTopCornerW < 0 || leftTopCornerH >= imgH || leftTopCornerW >= imgW)
                                    {
                                        Arr4DIdx(maxUntilNow_mask, N, H, W, C, n, ctH, ctW, c) = GroupElement(0);
                                    }
                                    else
                                    {
                                        Arr4DIdx(maxUntilNow_mask, N, H, W, C, n, ctH, ctW, c) = Arr4DIdx(inArr_mask, N1, imgH, imgW, C1, n, leftTopCornerH, leftTopCornerW, c);
                                    }
                                }
                                else
                                {
                                    int curPosH = leftTopCornerH + fh;
                                    int curPosW = leftTopCornerW + fw;

                                    GroupElement maxi_mask = Arr4DIdx(maxUntilNow_mask, N, H, W, C, n, ctH, ctW, c);
                                    GroupElement temp_mask;
                                    if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW))))
                                    {
                                        temp_mask = GroupElement(0);
                                    }
                                    else
                                    {
                                        temp_mask = Arr4DIdx(inArr_mask, N1, imgH, imgW, C1, n, curPosH, curPosW, c);
                                    }
                                    GroupElement rout = random_ge(bitlength);
                                    GroupElement routBit = random_ge(1);
                                    auto keys = keyGenMaxpool(bitlength, bitlength, maxi_mask, temp_mask, rout, routBit);
                                    Arr5DIdx(oneHot, FH * FW - 1, N, H, W, C, fh * FW + fw - 1, n, ctH, ctW, c) = routBit;
                                    Arr4DIdx(maxUntilNow_mask, N, H, W, C, n, ctH, ctW, c) = rout;

                                    auto read_start = std::chrono::high_resolution_clock::now();
                                    server->send_maxpool_key(keys.first);
                                    client->send_maxpool_key(keys.second);
                                    freeMaxpoolKeyPackPair(keys);
                                    auto read_end = std::chrono::high_resolution_clock::now();
                                    auto read_time = std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start).count();
                                    dealer_file_read_time += read_time;
                                }
                            }
                        }
                    }
                }
            }
        }
        auto dealer_end = std::chrono::high_resolution_clock::now();
        auto dealer_time = std::chrono::duration_cast<std::chrono::microseconds>(dealer_end - dealer_start).count() - dealer_file_read_time;
        dealerMicroseconds += dealer_time;
        std::cerr << "   Dealer time: " << dealer_time / 1000.0 << " milliseconds" << std::endl;
    }
    else
    {
        MaxpoolKeyPack *keys = new MaxpoolKeyPack[(FH * FW - 1) * N * C * H * W];
        int kidx = 0;
        uint64_t keysize_start = dealer->bytesReceived();
        auto keyread_start = std::chrono::high_resolution_clock::now();
        for (int fh = 0; fh < FH; fh++)
        {
            for (int fw = 0; fw < FW; fw++)
            {
                if (fh == 0 && fw == 0)
                {
                    continue;
                }
                for (int n = 0; n < N; n++)
                {
                    for (int c = 0; c < C; c++)
                    {
                        for (int ctH = 0; ctH < H; ctH++)
                        {
                            for (int ctW = 0; ctW < W; ctW++)
                            {
                                keys[kidx] = dealer->recv_maxpool_key(bitlength, bitlength);
                                kidx++;
                            }
                        }
                    }
                }
            }
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        uint64_t keyread_time = std::chrono::duration_cast<std::chrono::microseconds>(keyread_end - keyread_start).count();

        peer->sync();
        uint64_t timeCompute = 0;
        uint64_t timeReconstruct = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int n = 0; n < N; n++)
        {
            for (int c = 0; c < C; c++)
            {
                for (int ctH = 0; ctH < H; ctH++)
                {
                    for (int ctW = 0; ctW < W; ctW++)
                    {
                        int leftTopCornerH = ctH * strideH - zPadHLeft;
                        int leftTopCornerW = ctW * strideW - zPadWLeft;
                        if (leftTopCornerH < 0 || leftTopCornerW < 0 || leftTopCornerH >= imgH || leftTopCornerW >= imgW)
                        {
                            Arr4DIdx(maxUntilNow, N, H, W, C, n, ctH, ctW, c) = 0;
                        }
                        else
                        {
                            Arr4DIdx(maxUntilNow, N, H, W, C, n, ctH, ctW, c) = Arr4DIdx(inArr, N1, imgH, imgW, C1, n, leftTopCornerH, leftTopCornerW, c);
                        }
                    }
                }
            }
        }
        uint64_t onlineComm0 = peer->bytesReceived() + peer->bytesSent();
        auto t0 = std::chrono::high_resolution_clock::now();
        timeCompute += std::chrono::duration_cast<std::chrono::microseconds>(t0 - start).count();

        for (int fh = 0; fh < FH; fh++)
        {
            for (int fw = 0; fw < FW; fw++)
            {
                if (fh == 0 && fw == 0)
                {
                    continue;
                }

                auto t1 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
                for (int i = 0; i < N * C * H * W; i += 1)
                {
                    int curr = i;
                    int ctW = curr % W;
                    curr = curr / W;
                    int ctH = curr % H;
                    curr = curr / H;
                    int c = curr % C;
                    curr = curr / C;
                    int n = curr % N;
                    curr = curr / N;

                    int leftTopCornerH = ctH * strideH - zPadHLeft;
                    int leftTopCornerW = ctW * strideW - zPadWLeft;
                    int curPosH = leftTopCornerH + fh;
                    int curPosW = leftTopCornerW + fw;

                    GroupElement maxi = Arr4DIdx(maxUntilNow, N, H, W, C, n, ctH, ctW, c);
                    GroupElement temp;
                    if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW))))
                    {
                        temp = GroupElement(0);
                    }
                    else
                    {
                        temp = Arr4DIdx(inArr, N1, imgH, imgW, C1, n, curPosH, curPosW, c);
                    }
                    int kidx = (fh * FW + fw - 1) * (N * C * H * W) + i;
                    Arr4DIdx(maxUntilNow, N, H, W, C, n, ctH, ctW, c) = evalMaxpool(party - 2, maxi, temp, keys[kidx], Arr5DIdx(oneHot, FH * FW - 1, N, H, W, C, fh * FW + fw - 1, n, ctH, ctW, c));
                    freeMaxpoolKeyPack(keys[kidx]);
                }

                auto t2 = std::chrono::high_resolution_clock::now();

                if (!(fh == 0 && fw == 0))
                {
                    reconstruct(N * C * H * W, maxUntilNow, bitlength);
                }
                auto t3 = std::chrono::high_resolution_clock::now();
                timeCompute += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
                timeReconstruct += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
            }
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        reconstruct(N * C * H * W * (FH * FW - 1), oneHot, 1);
        auto end = std::chrono::high_resolution_clock::now();

        uint64_t onlineComm1 = peer->bytesReceived() + peer->bytesSent();
        timeReconstruct += std::chrono::duration_cast<std::chrono::microseconds>(end - t4).count();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        delete[] keys;

        FSS::stat_t stat = {
            prefix + "MaxPool",
            keyread_time,
            timeCompute,
            timeReconstruct,
            onlineComm1 - onlineComm0,
            dealer->bytesReceived() - keysize_start};
        stat.print();
        FSS::push_stats(stat);
    }

    std::cerr << ">> MaxPool - End" << std::endl;
}

void MaxPoolOneHot(int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH, int32_t FW, GroupElement *maxBits, GroupElement *oneHot)
{
    std::cerr << ">> MaxPoolOneHot - Start" << std::endl;
    GroupElement *curr = make_array<GroupElement>(N * H * W * C);
    if (party == DEALER)
    {
        BIG_LOOPY(
            auto m = Arr5DIdx(maxBits, FH * FW - 1, N, H, W, C, FH * FW - 2, n, h, w, c);
            Arr4DIdx(curr, N, H, W, C, n, h, w, c) = m;
            Arr5DIdx(oneHot, FH * FW, N, H, W, C, FH * FW - 1, n, h, w, c) = m;)

        for (int f = FH * FW - 2; f >= 1; --f)
        {
            // out[f] = max[f - 1] ^ !curr
            BIG_LOOPY(
                auto max = Arr5DIdx(maxBits, FH * FW - 1, N, H, W, C, f - 1, n, h, w, c);
                auto c1 = Arr4DIdx(curr, N, H, W, C, n, h, w, c);
                auto rout = random_ge(1);
                auto keys = keyGenBitwiseAnd(max, c1, rout);
                server->send_bitwise_and_key(keys.first);
                client->send_bitwise_and_key(keys.second);
                Arr5DIdx(oneHot, FH * FW, N, H, W, C, f, n, h, w, c) = rout;)

            BIG_LOOPY(
                Arr4DIdx(curr, N, H, W, C, n, h, w, c) = Arr4DIdx(curr, N, H, W, C, n, h, w, c) ^ Arr5DIdx(oneHot, FH * FW, N, H, W, C, f, n, h, w, c);)
        }

        BIG_LOOPY(
            Arr5DIdx(oneHot, FH * FW, N, H, W, C, 0, n, h, w, c) = Arr4DIdx(curr, N, H, W, C, n, h, w, c);)
    }
    else
    {
        BitwiseAndKeyPack *keys = new BitwiseAndKeyPack[(FH * FW - 2) * N * H * W * C];
        auto keyread_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < (FH * FW - 2) * N * H * W * C; ++i)
        {
            keys[i] = dealer->recv_bitwise_and_key();
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time = std::chrono::duration_cast<std::chrono::microseconds>(keyread_end - keyread_start).count();

        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        BIG_LOOPY(
            auto m = Arr5DIdx(maxBits, FH * FW - 1, N, H, W, C, FH * FW - 2, n, h, w, c);
            Arr4DIdx(curr, N, H, W, C, n, h, w, c) = m;
            Arr5DIdx(oneHot, FH * FW, N, H, W, C, FH * FW - 1, n, h, w, c) = m;)

        for (int f = FH * FW - 2; f >= 1; --f)
        {

            // out[f] = max[f - 1] ^ !curr
            BIG_LOOPY(
                auto max = Arr5DIdx(maxBits, FH * FW - 1, N, H, W, C, f - 1, n, h, w, c);
                auto c1 = Arr4DIdx(curr, N, H, W, C, n, h, w, c);
                auto key = keys[(FH * FW - 2 - f) * N * H * W * C + n * H * W * C + h * W * C + w * C + c];
                Arr5DIdx(oneHot, FH * FW, N, H, W, C, f, n, h, w, c) = evalAnd(party - 2, max, 1 ^ c1, key);
                mod(Arr5DIdx(oneHot, FH * FW, N, H, W, C, f, n, h, w, c), 1);)

            // std::thread thread_pool[num_threads];
            // for(int i = 0; i < num_threads; ++i) {
            //     thread_pool[i] = std::thread(maxpool_onehot_threads_helper, i, f, N, H, W, C, FH, FW, maxBits, curr, oneHot, keys);
            // }

            // for(int i = 0; i < num_threads; ++i) {
            //     thread_pool[i].join();
            // }

            reconstruct(N * H * W * C, oneHot + f * N * H * W * C, 1);

            BIG_LOOPY(
                Arr4DIdx(curr, N, H, W, C, n, h, w, c) = Arr4DIdx(curr, N, H, W, C, n, h, w, c) ^ Arr5DIdx(oneHot, FH * FW, N, H, W, C, f, n, h, w, c);)
        }

        BIG_LOOPY(
            Arr5DIdx(oneHot, FH * FW, N, H, W, C, 0, n, h, w, c) = Arr4DIdx(curr, N, H, W, C, n, h, w, c) ^ 1;)
        auto end = std::chrono::high_resolution_clock::now();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        evalMicroseconds += eval_time;
        selectEvalMicroseconds += eval_time;
        std::cerr << "   Key Read Time = " << keyread_time / 1000.0 << " miliseconds" << std::endl;
        std::cerr << "   Online Time = " << eval_time / 1000.0 << " miliseconds" << std::endl;
        delete[] keys;
    }
    delete[] curr;
    std::cerr << ">> MaxPoolOneHot - End" << std::endl;
}

void MaxPoolBackward(int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH,
             int32_t FW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *oneHot)
{
    std::cerr << ">> MaxPoolBackward - Start" << std::endl;
    // currently not very confident about maxpool with padding, hence this assert
    always_assert((zPadHLeft == 0) && (zPadHRight == 0) && (zPadWLeft == 0) && (zPadWRight == 0));

    if (party == DEALER)
    {
        for(int n = 0; n < N; ++n) {
            for(int h = 0; h < imgH; ++h) {
                for(int w = 0; w < imgW; ++w) {
                    for(int c = 0; c < C; ++c) {
                        Arr4DIdx(inArr, N1, imgH, imgW, C1, n, h, w, c) = 0;
                    }
                }
            }
        }

        BIG_LOOPY(
            int leftTopCornerH = h * strideH - zPadHLeft;
            int leftTopCornerW = w * strideW - zPadWLeft;
            auto src = Arr4DIdx(outArr, N, H, W, C, n, h, w, c);
            for(int fh = 0; fh < FH; ++fh) {
                for(int fw = 0; fw < FW; ++fw) {
                    if ((leftTopCornerH + fh >= 0) && (leftTopCornerH + fh < imgH) && (leftTopCornerW + fw >= 0) && (leftTopCornerW + fw < imgW)) {
                        auto s = Arr5DIdx(oneHot, FH * FW, N, H, W, C, fh * FW + fw, n, h, w, c);
                        auto dst = Arr4DIdx(inArr, N1, imgH, imgW, C1, n, leftTopCornerH + fh, leftTopCornerW + fw, c);
                        // dst = dst + select(s, src)
                        auto rout = random_ge(bitlength);
                        auto keys = keyGenSelect(bitlength, s, src, rout);
                        Arr4DIdx(inArr, N1, imgH, imgW, C1, n, leftTopCornerH + fh, leftTopCornerW + fw, c) = dst + rout;
                        server->send_select_key(keys.first);
                        client->send_select_key(keys.second);
                    }
                }
            }
        )
    }
    else 
    {
        SelectKeyPack *keys = new SelectKeyPack[FH * FW * N * H * W * C];
        auto keyread_start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < FH * FW * N * H * W * C; ++i) {
            keys[i] = dealer->recv_select_key(bitlength);
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time = std::chrono::duration_cast<std::chrono::microseconds>(keyread_end - keyread_start).count();

        peer->sync();

        auto start = std::chrono::high_resolution_clock::now();
        for(int n = 0; n < N; ++n) {
            for(int h = 0; h < imgH; ++h) {
                for(int w = 0; w < imgW; ++w) {
                    for(int c = 0; c < C; ++c) {
                        Arr4DIdx(inArr, N1, imgH, imgW, C1, n, h, w, c) = 0;
                    }
                }
            }

            for(int h = 0; h < H; ++h) {
                for(int w = 0; w < W; ++w) {
                    for(int c = 0; c < C; ++c) {
                        int leftTopCornerH = h * strideH - zPadHLeft;
                        int leftTopCornerW = w * strideW - zPadWLeft;
                        auto src = Arr4DIdx(outArr, N, H, W, C, n, h, w, c);
                        for(int fh = 0; fh < FH; ++fh) {
                            for(int fw = 0; fw < FW; ++fw) {
                                auto s = Arr5DIdx(oneHot, FH * FW, N, H, W, C, fh * FW + fw, n, h, w, c);
                                auto dst = Arr4DIdx(inArr, N1, imgH, imgW, C1, n, leftTopCornerH + fh, leftTopCornerW + fw, c);
                                // dst = dst + select(s, src)
                                auto key = keys[n * H * W * C * FH * FW + h * W * C * FH * FW + w * C * FH * FW + c * FH * FW + fh * FW + fw];
                                // auto key = dealer->recv_select_key(bitlength);
                                Arr4DIdx(inArr, N1, imgH, imgW, C1, n, leftTopCornerH + fh, leftTopCornerW + fw, c) = dst + evalSelect(party - 2, s, src, key);
                            }
                        }
                    }
                }
            }
        }

        auto mid = std::chrono::high_resolution_clock::now();
        reconstruct(N1 * imgH * imgW * C1, inArr, bitlength);
        auto end = std::chrono::high_resolution_clock::now();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        auto reconstruct_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();
        auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start).count();
        evalMicroseconds += eval_time;
        selectEvalMicroseconds += eval_time;
        std::cerr << "   Key Read Time = " << keyread_time / 1000.0 << " miliseconds" << std::endl;
        std::cerr << "   Compute Time = " << compute_time / 1000.0 << " miliseconds" << std::endl;
        std::cerr << "   Reconstruct Time = " << reconstruct_time / 1000.0 << " miliseconds" << std::endl;
        std::cerr << "   Online Time = " << eval_time / 1000.0 << " miliseconds" << std::endl;

        delete[] keys;
    }
    std::cerr << ">> MaxPoolBackward - End" << std::endl;
}
std::pair<ElemWiseMulKeyPack, ElemWiseMulKeyPack> keyGenElemWiseMul(int32_t size)
{
    ElemWiseMulKeyPack k0, k1;
    k0.size = k1.size = size;
    k0.a = new GroupElement[size];
    k1.a = new GroupElement[size];
    k0.b = new GroupElement[size];
    k1.b = new GroupElement[size];
    k0.c = new GroupElement[size];
    k1.c = new GroupElement[size];

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        GroupElement a = random_ge(bitlength);
        GroupElement b = random_ge(bitlength);
        GroupElement c = a * b;

        auto a_split = splitShare(a, bitlength);
        k0.a[i] = a_split.first;
        k1.a[i] = a_split.second;

        auto b_split = splitShare(b, bitlength);
        k0.b[i] = b_split.first;
        k1.b[i] = b_split.second;

        auto c_split = splitShare(c, bitlength);
        k0.c[i] = c_split.first;
        k1.c[i] = c_split.second;
    }

    return std::make_pair(k0, k1);
}
void evalElemWiseMul(int party, int32_t size, 
                     const GroupElement* x, const GroupElement* y, 
                     GroupElement* z, const ElemWiseMulKeyPack &key)
{
    GroupElement* d_shares = new GroupElement[size];
    GroupElement* e_shares = new GroupElement[size];

    // 1. 计算 d = x - a 和 e = y - b 的份额
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        d_shares[i] = x[i] - key.a[i];
        e_shares[i] = y[i] - key.b[i];
    }

    // 2. 重构以获得公开的 d 和 e
    // 注意: reconstruct 会修改传入的数组，所以我们用 d_shares 的副本来重构 e
    GroupElement* e_reconstruct_buffer = new GroupElement[size];
    memcpy(e_reconstruct_buffer, e_shares, size * sizeof(GroupElement));


    
    reconstruct(size, d_shares, bitlength); // d_shares 现在是公开的 d
    reconstruct(size, e_reconstruct_buffer, bitlength); // e_reconstruct_buffer 现在是公开的 e


    GroupElement* d_public = d_shares;
    GroupElement* e_public = e_reconstruct_buffer;

    // 3. 根据 Beaver Triple 协议计算最终的输出份额 z
    // party 2 (SERVER) 的 party_bit 是 0, party 3 (CLIENT) 的 party_bit 是 1
    int party_bit = (party - 2); 

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        z[i] = (party_bit * d_public[i] * e_public[i]) + 
               (d_public[i] * key.b[i]) + 
               (e_public[i] * key.a[i]) + 
               key.c[i];
    }
    GroupElement* temp = new GroupElement[size];

    delete[] d_shares;
    delete[] e_shares;
    delete[] e_reconstruct_buffer;
}

void ElemWiseMul(int32_t size, 
                 MASK_PAIR(GroupElement *A), // 展开为 A, A_mask
                 MASK_PAIR(GroupElement *B), // 展开为 B, B_mask
                 MASK_PAIR(GroupElement *C)) // 展开为 C, C_mask
{
    if (party == DEALER) {
        auto keys = keyGenElemWiseMul(size);
        server->send_elemwisemul_key(keys.first);
        client->send_elemwisemul_key(keys.second);
        
        // #pragma omp parallel for
        // for (int i = 0; i < size; i++) {
        //     GroupElement beaver_a = keys.first.a[i] + keys.second.a[i];
        //     GroupElement beaver_b = keys.first.b[i] + keys.second.b[i];
        //     GroupElement beaver_c = keys.first.c[i] + keys.second.c[i];
        //     // 输出掩码 C_mask = c - a*B - b*A + a*b
        //     C_mask[i] = beaver_c - beaver_a * B_mask[i] - beaver_b * A_mask[i];
        // }

        // 释放密钥内存
        delete[] keys.first.a; delete[] keys.first.b; delete[] keys.first.c;
        delete[] keys.second.a; delete[] keys.second.b; delete[] keys.second.c;
    } else {
        auto key = dealer->recv_elemwisemul_key(size);
        std::cerr << "ElemWiseMul - Start Eval" << std::endl;
        evalElemWiseMul(party, size, A, B, C, key);
        GroupElement* temp = new GroupElement[size];
        memcpy(temp, C, size * sizeof(GroupElement));
        reconstruct(size, temp, bitlength); // d_shares 现在是公开的 d
        print_array("C", party, size, temp, 10);
        delete[] key.a; delete[] key.b; delete[] key.c;
    }
}
void SecretShare(int32_t size, const GroupElement *plain_in, GroupElement *share_out, int owner)
{
    if (size == 0) return;
    
    if (party == DEALER) {
        for (int i = 0; i < size; ++i) prngShared.get<uint64_t>();
        return;
    }

    if (party == owner) {
        //生成掩码的阶段必须是串行的
        std::vector<GroupElement> peer_share(size);
        for (int i = 0; i < size; ++i) {
            peer_share[i] = prngShared.get<uint64_t>();
            mod(peer_share[i], bitlength);
        }

        peer->send_ge_array(peer_share.data(), size);
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            share_out[i] = plain_in[i] - peer_share[i];
            mod(share_out[i], bitlength);
        }

    } else {
        peer->recv_ge_array(share_out, size);
        for (int i = 0; i < size; ++i) {
            prngShared.get<uint64_t>();
        }
    }
}
void print_array(const std::string& title, int party, int size, const GroupElement* arr, int limit) {
    //if (party == DEALER) return; // Dealer 不打印

    std::cout << "\n--- [Party " << party << "] " << title << " ---" << std::endl;
    for (int i = 0; i < size && i < limit; ++i) {
        // 为了可读性，我们可以将 uint64_t 转换为 int64_t 来打印
        // 这样负数（在模运算下的大正数）会更容易看懂
        std::cout << "  [" << i << "]: " << (int64_t)arr[i] << std::endl;
    }
    if (size > limit) {
        std::cout << "  ..." << std::endl;
    }
}
void DpfRoute(
    int32_t size,
    MASK_PAIR(GroupElement *y_in),  // 展开为 y_in, y_in_mask
    int rank_bw,
    MASK_PAIR(GroupElement *z_in),  // 展开为 z_in, z_in_mask
    int data_bw, 
    MASK_PAIR(GroupElement *z_out)) // 展开为 z_out, z_out_mask
{
    std::string prefix = "DPF_route::";
    if (party == DEALER) {
        auto keys = keyGenDpfRoute(size, data_bw, rank_bw);
        server->send_dpf_route_key(keys.first);
        client->send_dpf_route_key(keys.second);
        GroupElement* r = new GroupElement[size];
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            r[i] = keys.first.r_shares[i] + keys.second.r_shares[i];
        }
        print_array("Original Plaintext 'r'", party, size, r);
        print_array("Share 'r_1'", party, size, keys.first.r_shares);
        print_array("SHare 'r_2'", party, size, keys.second.r_shares);
        GroupElement* s = new GroupElement[size];
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            s[i] = keys.first.s_shares[i] + keys.second.s_shares[i];
        }
        print_array("Original Plaintext 's'", party, size, s);
        print_array("Share 's_1'", party, size, keys.first.s_shares);
        print_array("SHare 's_2'", party, size, keys.second.s_shares);
        GroupElement* s_shares_complete = new GroupElement[size];
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            s_shares_complete[i] = keys.first.s_shares[i] + keys.second.s_shares[i];
            mod(s_shares_complete[i], data_bw);
        }
        
        GroupElement* z_tilde_mask = new GroupElement[size];


        ElemWiseMul(size, 
                    z_in_mask, z_in_mask, 
                    s_shares_complete, s_shares_complete, 
                    z_tilde_mask, z_tilde_mask); 

        delete[] s_shares_complete;
        delete[] z_tilde_mask;

    } else { 
        DpfRouteKeyPack key;
        
        std::cerr << "\n1\n" << std::endl;
        std::cerr << "\n... peer->sync();  start...\n" << std::endl;
        peer->sync();
        std::cerr << "\n... peer->sync();  end...\n" << std::endl;
        std::cerr << "\n3\n" << std::endl;
        
        uint64_t keysize_start = dealer->bytesReceived();
        std::cerr << "\n4\n" << std::endl;
        key = dealer->recv_dpf_route_key(size, data_bw, rank_bw);
        std::cerr << "\n5\n" << std::endl;
        GroupElement* y_plus_r_shares = new GroupElement[size];
        GroupElement* z_mul_s_shares = new GroupElement[size];
        std::cerr << "\n6\n" << std::endl;

        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            y_plus_r_shares[i] = y_in[i] + key.r_shares[i];
        }
        
        
        // GroupElement* r_temp = new GroupElement[size];
        // memcpy(r_temp, key.r_shares, size * sizeof(GroupElement));
        // print_array("Share 'r_share'", party, size, r_temp);
        // reconstruct(size, r_temp, FSSConfig::bitlength);
        // print_array("Original Plaintext 'r'", party, size, r_temp);


        // GroupElement* yr_temp = new GroupElement[size];
        // memcpy(yr_temp, y_plus_r_shares.data(), size * sizeof(GroupElement));
        // print_array("Original Plaintext 'y_r_share'", party, size, yr_temp);
        // reconstruct(size, yr_temp, FSSConfig::bitlength);
        // print_array("Original Plaintext 'y_r'", party, size, yr_temp);

        std::cerr << "\n7\n" << std::endl;
        peer->sync();
        std::cerr << "\n7\n" << std::endl;
        ElemWiseMul(size, 
                    z_in, z_in, 
                    key.s_shares, key.s_shares,
                    z_mul_s_shares, z_mul_s_shares);
        std::cerr << "\n8\n" << std::endl;
        reconstruct(size, y_plus_r_shares, key.rank_bin); 
        GroupElement* y_hat_public = y_plus_r_shares; 
        std::cerr << "\n9\n" << std::endl;
        reconstruct(size, z_mul_s_shares, bitlength);
        GroupElement* z_tilde_public = z_mul_s_shares;
        print_array("Original Plaintext 'z*s'", party, size, z_tilde_public);
        std::cerr << "\n10\n" << std::endl;
        online_round2_compute(party, key, y_hat_public, z_tilde_public, z_out);
        reconstruct(size, z_out, data_bw);
        std::cerr << "\n11\n" << std::endl;
    }
}