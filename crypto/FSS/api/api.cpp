#include <bitset>
#include <iomanip>
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
#include "../protocol/signextend.h"
// #include "../protocol/reluextend.h"
// #include "../protocol/signextend.h"
// #include "../protocol/clip.h"
#include "../primitives/dcf.h"
#include "../protocol/lut.h"
#include "../protocol/select.h"
// #include "../protocol/fixtobfloat16.h"
// #include "../protocol/wrap.h"
#include "../primitives/dpf.h"
#include "../protocol/mult.h"
// #include "../protocol/taylor.h"
#include "../protocol/float.h"
#include "../protocol/dpfsort.h"
#include "../protocol/graphupdate.h"
#include "../protocol/relufastsecnet.h"
#include "../protocol/ars.h"
#include "../protocol/wrap.h"
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

void SlothLRSfromWrap(int size, GroupElement *x, GroupElement *w, GroupElement *y, int scale, std::string parent)
{
    if (party == DEALER)
    {
        pair<SlothLRSKeyPack> *keys = new pair<SlothLRSKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            GroupElement rout = random_ge(1);
            keys[i] = keyGenSlothLRS(bitlength, scale, x[i], w[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_sloth_lrs_key(keys[i].first);
            client->send_sloth_lrs_key(keys[i].second);
        }

        delete[] keys;
    }
    else
    {
        SlothLRSKeyPack *keys = new SlothLRSKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        uint64_t keyread_time = time_this_block([&]()
                                                {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_sloth_lrs_key(bitlength, scale);
            } });

        peer->sync();

        uint64_t compute_time = time_this_block([&]()
                                                {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                y[i] = evalSlothLRS(party - 2, x[i], w[i], keys[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, y, bitlength); });

        FSS::stat_t stat = {
            parent,
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




void SlothWrap_dpf(int size, int bin, GroupElement *x, GroupElement *y, std::string parent)
{
    if (party == DEALER)
    {
        pair<WrapDPFKeyPack> *keys = new pair<WrapDPFKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            GroupElement rout = random_ge(1);
            keys[i] = keyGenWrapDPF(bin, x[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_wrap_dpf_key(keys[i].first);
            client->send_wrap_dpf_key(keys[i].second);
            freeWrapDPFKeyPackPair(keys[i]);
        }

        delete[] keys;
    }
    else
    {
        WrapDPFKeyPack *keys = new WrapDPFKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        uint64_t keyread_time = time_this_block([&]()
                                                {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_wrap_dpf_key(bin);
            } });

        peer->sync();

        uint64_t compute_time = time_this_block([&]()
                                                {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                y[i] = evalWrapDPF(party - 2, x[i], keys[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, y, 1); });

        FSS::stat_t stat = {
            parent,
            keyread_time,
            compute_time,
            reconstruction_stats.first,
            reconstruction_stats.second,
            dealer->bytesReceived() - keysize_start};

        stat.print();
        FSS::push_stats(stat);

        for (int i = 0; i < size; ++i)
        {
            freeWrapDPFKeyPack(keys[i]);
        }
        delete[] keys;
    }
}


/* 检测整数运算的​​溢出行为(好像是) */
void SlothWrap_ss(int size, int bin, GroupElement *x, GroupElement *y, std::string parent)
{
    if (party == DEALER)
    {
        pair<WrapSSKeyPack> *keys = new pair<WrapSSKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            GroupElement rout = random_ge(1);
            keys[i] = keyGenWrapSS(bin, x[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_wrap_ss_key(keys[i].first);
            client->send_wrap_ss_key(keys[i].second);
        }

        delete[] keys;
    }
    else
    {
        WrapSSKeyPack *keys = new WrapSSKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        uint64_t keyread_time = time_this_block([&]()
                                                {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_wrap_ss_key(bin);
            } });

        peer->sync();

        uint64_t compute_time = time_this_block([&]()
                                                {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                y[i] = evalWrapSS(party - 2, x[i], keys[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, y, 1); });

        FSS::stat_t stat = {
            parent,
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

void SlothWrap(int size, int bin, GroupElement *x, GroupElement *w, std::string parent)
{
    if (bin <= 7)
    {
        SlothWrap_ss(size, bin, x, w, parent);
    }
    else
    {
        SlothWrap_dpf(size, bin, x, w, parent);
    }
}

void SlothLRS(int size, GroupElement *x, GroupElement *y, int scale, std::string prefix)
{
    GroupElement *w = new GroupElement[size];
    GroupElement *x0 = w;

    auto t = time_this_block([&]()
                             {
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        x0[i] = x[i];
        mod(x0[i], scale);
    } });

    SlothWrap(size, scale, x0, w, prefix + "Truncation");
    SlothLRSfromWrap(size, x, w, y, scale, prefix + "Truncation");

    if (party != DEALER)
        FSS::push_stats({prefix + "Truncation::Misc", 0, t, 0, 0, 0});

    delete[] w;
}

void SlothARS(int size, GroupElement *x, GroupElement *y, int scale, std::string prefix)
{
    GroupElement *z = new GroupElement[size];

    if (party == DEALER)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            z[i] = x[i];
        }
    }
    else
    {
        auto t = time_this_block([&]()
                                 {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            z[i] = x[i] + (1LL << (bitlength - 2));
        } });
        FSS::stat_t stat = {prefix + "Truncation::Misc", 0, t, 0, 0, 0};
        stat.print();
        FSS::push_stats(stat);
    }

    SlothLRS(size, z, z, scale, prefix);

    if (party == DEALER)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            y[i] = z[i];
        }
    }
    else
    {
        auto t = time_this_block([&]()
                                 {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            y[i] = z[i] - (1LL << (bitlength - scale - 2));
        } });
        FSS::stat_t stat = {prefix + "Truncation::Misc", 0, t, 0, 0, 0};
        stat.print();
        FSS::push_stats(stat);
    }
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
void debug_reconstruct_and_print(const std::string& tag, int32_t size, const GroupElement* shares, int scale) {
    if (party == DEALER) return; // Dealer does not participate in reconstruction

    // 1. Copy shares to prevent reconstruct from modifying the original data during calculation
    GroupElement* temp = new GroupElement[size];
    memcpy(temp, shares, size * sizeof(GroupElement));

    // 2. Sync and Reconstruct
    peer->sync();
    reconstruct(size, temp, FSSConfig::bitlength);

    // 3. Print (Only SERVER prints to avoid duplicate logs)
    if (party == SERVER) {
        std::cout << "\n========================================================" << std::endl;
        std::cout << "[DEBUG] >>> " << tag << " <<<" << std::endl;
        std::cout << "========================================================" << std::endl;
        
        // Limit output to first 10 elements to prevent console flooding
        int limit = (size > 10) ? 10 : size; 
        
        for (int i = 0; i < limit; ++i) {
            double val = fixed_to_double(temp[i], scale);
            uint64_t raw_unsigned = (uint64_t)temp[i]; // Treat as unsigned for Hex/Bin
            int64_t raw_signed = (int64_t)temp[i];     // Treat as signed for Decimal interpretation

            std::cout << "  Element [" << i << "]: " << val << "\n"
                      << "    Dec (s): " << raw_signed << "\n"
                      << "    Hex    : 0x" << std::hex << std::setw(16) << std::setfill('0') << raw_unsigned << std::dec << "\n"
                      << "    Bin    : " << std::bitset<64>(raw_unsigned) << "\n"
                      << "--------------------------------------------------------" << std::endl;
        }
        
        if (size > limit) {
            std::cout << "  ... (total " << size << " items, showing first " << limit << ")" << std::endl;
        }
        std::cout << std::endl;
    }
    
    delete[] temp;
    peer->sync(); // Sync again to ensure logs are flushed before proceeding
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

void SlothMax(int size, int bin, GroupElement *x, GroupElement *y, GroupElement *out, std::string prefix)
{
    auto t1 = time_this_block([&]()
                              {
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        out[i] = x[i] - y[i];
    } });
    FastRelu(size, out, out, out, out, prefix + "Max::");
    auto t2 = time_this_block([&]()
                              {
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        out[i] = out[i] + y[i];
    } });

    if (party != DEALER)
        FSS::push_stats({prefix + "Max::Misc", 0, t1 + t2, 0, 0, 0});
}

// x is [s1 x s2]
// y is [s1]
void SlothMaxpool(int s1, int s2, int bin, GroupElement *x, GroupElement *y, std::string prefix)
{
    GroupElement *left = new GroupElement[s1 * s2];  // more elements than required but whatever
    GroupElement *right = new GroupElement[s1 * s2]; // more elements than required but whatever
    GroupElement *res = new GroupElement[s1 * s2];
    GroupElement *tmp = new GroupElement[s1];

    auto t1 = time_this_block([&]()
                              {
#pragma omp parallel for
    for (int i = 0; i < s1 * s2; ++i)
    {
        res[i] = x[i];
    } });

    // do in log rounds
    int curr = s2;
    while (curr != 1)
    {
        int curr2 = curr / 2;

        auto t2 = time_this_block([&]()
                                  {
#pragma omp parallel for
        for (int i = 0; i < s1; ++i)
        {
            for (int j = 0; j < curr2; ++j)
            {
                Arr2DIdx(left, s1, curr2, i, j) = Arr2DIdx(res, s1, curr, i, 2 * j);
                Arr2DIdx(right, s1, curr2, i, j) = Arr2DIdx(res, s1, curr, i, 2 * j + 1);
            }
        } });

        SlothMax(s1 * curr2, bin, left, right, left, prefix + "Maxpool::");

        int currNext;
        auto t3 = time_this_block([&]()
                                  {
        if ((curr % 2) == 0)
        {
            currNext = curr / 2;
        }
        else
        {
            currNext = curr / 2 + 1;
#pragma omp parallel for
            for (int i = 0; i < s1; ++i)
            {
                tmp[i] = Arr2DIdx(res, s1, curr, i, curr - 1);
            }
#pragma omp parallel for
            for (int i = 0; i < s1; ++i)
            {
                Arr2DIdx(res, s1, currNext, i, currNext - 1) = tmp[i];
            }
        }

#pragma omp parallel for
        for (int i = 0; i < s1; ++i)
        {
            for (int j = 0; j < curr2; ++j)
            {
                Arr2DIdx(res, s1, currNext, i, j) = Arr2DIdx(left, s1, curr2, i, j);
            }
        }
        curr = currNext; });

        if (party != DEALER)
            FSS::push_stats({prefix + "Maxpool::Misc", 0, t2 + t3, 0, 0, 0});
    }

    auto t4 = time_this_block([&]()
                              {
#pragma omp parallel for
    for (int i = 0; i < s1; ++i)
    {
        y[i] = Arr2DIdx(res, s1, 1, i, 0);
    } });

    if (party != DEALER)
        FSS::push_stats({prefix + "Maxpool::Misc", 0, t1 + t4, 0, 0, 0});

    delete[] left;
    delete[] right;
    delete[] res;
    delete[] tmp;
}

void SlothDrelu(int size, int bin, GroupElement *x, GroupElement *y, std::string prefix)
{
    if (party == DEALER)
    {
        pair<SlothDreluKeyPack> *keys = new pair<SlothDreluKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            GroupElement rout = random_ge(1);
            keys[i] = keyGenSlothDrelu(bin, x[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_sloth_drelu_key(keys[i].first);
            client->send_sloth_drelu_key(keys[i].second);
            freeSlothDreluKeyPackPair(keys[i]);
        }

        delete[] keys;
    }
    else
    {
        SlothDreluKeyPack *keys = new SlothDreluKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        uint64_t keyread_time = time_this_block([&]()
                                                {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_slothdrelu_key(bin);
            } });

        peer->sync();

        uint64_t compute_time = time_this_block([&]()
                                                {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                y[i] = evalSlothDrelu(party - 2, x[i], keys[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, y, 1); });

        FSS::stat_t stat = {
            prefix + "Drelu",
            keyread_time,
            compute_time,
            reconstruction_stats.first,
            reconstruction_stats.second,
            dealer->bytesReceived() - keysize_start};

        stat.print();
        FSS::push_stats(stat);

        for (int i = 0; i < size; ++i)
        {
            freeSlothDreluKeyPack(keys[i]);
        }
        delete[] keys;
    }
}

// in .../FSS/api/api.cpp

void SecureSquare(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr)) {
    if (party == DEALER) {
        pair<SquareKey> *keys = new pair<SquareKey>[size];
        
        // Dealer 生成 Beaver Triples
        // 注意：inArr_mask 最好是随机数，如果传入的是未初始化的 dummy，正好作为随机源
        for (int i = 0; i < size; ++i) {
            keys[i] = keyGenSquare(inArr_mask[i], outArr_mask[i]);
        }

        for (int i = 0; i < size; ++i) {
            server->send_square_key(keys[i].first);
            client->send_square_key(keys[i].second);
        }
        delete[] keys;
    } else {
        SquareKey *keys = new SquareKey[size];
        for (int i = 0; i < size; ++i) {
            keys[i] = dealer->recv_square_key();
        }

        peer->sync();

        // --- Beaver Triple 在线阶段 ---
        
        // 1. 本地计算差值份额: [e] = [x] - [a]
        // (回忆：我们将 [a] 存在了 key.b 中)
        GroupElement *e_shares = new GroupElement[size];
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            e_shares[i] = inArr[i] - keys[i].b; 
        }

        // 2. 公开 e (重构)
        // 这是必要的通信步骤！
        reconstruct(size, e_shares, FSSConfig::bitlength);
        GroupElement *e_public = e_shares; // 重构后 e_shares 变成了明文 e

        // 3. 计算最终结果: [y] = e^2 + 2e[a] + [c]
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            outArr[i] = evalSquare(party - SERVER, e_public[i], keys[i]);
        }
        
        delete[] keys;
        delete[] e_shares;
    }
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
   

    delete[] d_shares;
    delete[] e_shares;
    delete[] e_reconstruct_buffer;
}

/**
 * @brief 使用 Crypten 风格的 "Mask-Reconstruct-Compute" 协议将布尔份额转换为算术份额。
 * 
 * @param size 批量转换的大小。
 * @param x_shares 输入的布尔(XOR)份额数组。
 * @param y_shares 输出的算术(加性)份额数组。
 */
void B2A_Crypten(int32_t size, const uint8_t* x_shares, GroupElement* y_shares)
{
    if (party == DEALER) {
        // Dealer 调用对应的密钥生成函数
        B2A_Crypten_KeyPack* server_keys = new B2A_Crypten_KeyPack[size];
        B2A_Crypten_KeyPack* client_keys = new B2A_Crypten_KeyPack[size];


        for (int i = 0; i < size; ++i) {
            // 1. Dealer 生成一个随机比特 r
            uint8_t r_plain = prngs[0].get<uint8_t>() & 1;

            // 2. 创建 r 的 XOR 份额
            auto r_xor_split = splitShareXor(r_plain,1); 
            
            // 3. 创建 r 的加性份额 (值为 0 或 1)
            auto r_add_split = splitShare((GroupElement)r_plain, FSSConfig::bitlength);

            // 4. 填充密钥包
            server_keys[i].r_xor_share = static_cast<uint8_t>(r_xor_split.first);
            server_keys[i].r_add_share = r_add_split.first;

            client_keys[i].r_xor_share = static_cast<uint8_t>(r_xor_split.second);
            client_keys[i].r_add_share = r_add_split.second;
        }

        // 5. 发送密钥包
        server->send_b2a_crypten_keys(server_keys, size);
        client->send_b2a_crypten_keys(client_keys, size);

        delete[] server_keys;
        delete[] client_keys;
        return;
    }

    // --- 1. 接收来自 Dealer 的随机数份额 ---
    std::vector<B2A_Crypten_KeyPack> keys(size);
    dealer->recv_b2a_crypten_keys(keys.data(), size);
    peer->sync(); // 确保双方都接收完毕

    // --- 2. 本地计算被屏蔽后的值 d 的 XOR 份额 ---
    std::vector<uint8_t> d_shares(size);
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        d_shares[i] = x_shares[i] ^ keys[i].r_xor_share;
    }

    // --- 3. 交互一次，重构以公开 d ---
    // reconstruct 函数会交换份额并相加/异或。我们需要一个XOR版本的reconstruct。
    // 如果你的 reconstruct 是加性的，我们需要修改。假设 reconstruct_bool 存在。
    // 为了简单，我们手动实现。
    std::vector<uint8_t> d_other_shares(size);
    if (party == SERVER) {
        peer->send_uint8_array(d_shares.data(), size);
        peer->recv_uint8_array(d_other_shares.data(), size);
    } else { // CLIENT
        peer->recv_uint8_array(d_other_shares.data(), size);
        peer->send_uint8_array(d_shares.data(), size);
    }
    
    uint8_t* d_public = new uint8_t[size];
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        d_public[i] = d_shares[i] ^ d_other_shares[i];
    }
    
    // --- 4. 本地计算最终的加性份额 ---
    // [y] = d + [r] - 2*d*[r]
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        GroupElement d_val = d_public[i];
        GroupElement r_add_share = keys[i].r_add_share;
        
        // 实现 [d] 的份额: 一方持有d，另一方持有0
        GroupElement d_add_share = (party == CLIENT) ? d_val : 0;
        
        // 计算 -2*d*[r] 的份额 (本地操作)
        GroupElement term3_share = -2 * d_val * r_add_share;

        y_shares[i] = d_add_share + r_add_share + term3_share;
    }
}

void reconstruct_bool(int32_t size, uint8_t* arr)
{
    if (party == DEALER) return;

    uint8_t* other_shares = new uint8_t[size];
    if (party == SERVER) {
        peer->send_uint8_array(arr, size);
        peer->recv_uint8_array(other_shares, size);
    } else { // CLIENT
        peer->recv_uint8_array(other_shares, size);
        peer->send_uint8_array(arr, size);
    }

    // 将份额异或起来得到明文
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        arr[i] = arr[i] ^ other_shares[i];
    }
    
    delete[] other_shares;
    
    // 增加轮次计数，因为发生了一次交互
    numRounds += 1; 
}

void reconstruct_big_bool(int32_t size, GroupElement* arr)
{
    if (party == DEALER) return;

    GroupElement* other_shares = new GroupElement[size];
    if (party == SERVER) {
        peer->send_uint64_array(arr, size);
        peer->recv_uint64_array(other_shares, size);
    } else { // CLIENT
        peer->recv_uint64_array(other_shares, size);
        peer->send_uint64_array(arr, size);
    }

    // 将份额异或起来得到明文
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        arr[i] = arr[i] ^ other_shares[i];
    }
    
    delete[] other_shares;
    
    // 增加轮次计数，因为发生了一次交互
    numRounds += 1; 
}


void ElemWiseMul(int32_t size, 
                 MASK_PAIR(const GroupElement *A),
                 MASK_PAIR(const GroupElement *B),
                 MASK_PAIR(GroupElement *C))
{
    const int scale = 16; // 假设的定点数小数位数

    if (party == DEALER) {
        // Dealer 需要为 Beaver Triple 和 ARS 生成密钥
        auto keys = keyGenElemWiseMul(size);
        server->send_elemwisemul_key(keys.first);
        client->send_elemwisemul_key(keys.second);
        
        GroupElement * x1 = new GroupElement[size];
        GroupElement * x2 = new GroupElement[size];
        // Dealer 还需要为 ARS 生成密钥
        //ARS(size, x1, x1, x2, x2, scale);
        //ARS_CrypTen_Style(size, x1, x1, scale);
            ARS_CrypTen_Style(size, x1, x1, scale);
        // Dealer 端的掩码逻辑保持不变
        // for (int i=0; i<size; ++i) C_mask[i] = 0;

        // 释放 Beaver Triple 密钥内存
        delete[] keys.first.a; delete[] keys.first.b; delete[] keys.first.c;
        delete[] keys.second.a; delete[] keys.second.b; delete[] keys.second.c;
    } else {
        // --- 计算方逻辑 ---

        // 1. 接收 Beaver Triple 密钥
        auto key = dealer->recv_elemwisemul_key(size);
        //debug_reconstruct_and_print("ElemWiseMul: A", size, A, scale);
        //debug_reconstruct_and_print("ElemWiseMul: B", size, B, scale);
        // 2. 执行整数乘法协议
        // 注意：我们把结果存储在一个临时数组中，因为它的小数位数是 2*scale
        GroupElement* z_full_precision = new GroupElement[size];
        evalElemWiseMul(party, size, A, B, z_full_precision, key);
        //debug_reconstruct_and_print("ElemWiseMul: z_full_precision", size, z_full_precision, 2*scale);
        // 3. 执行截断 (算术右移)
        // ARS 会接收 z_full_precision 的份额，计算截断后的份额，并存入 C
        //ARS(size, z_full_precision, nullptr, C, nullptr, scale);
       
        uint64_t scale_down_time = time_this_block([&]() {
             ARS_CrypTen_Style(size, z_full_precision, C, scale);
        });
        debug_reconstruct_and_print("ElemWiseMul: C", size, C, scale);
        FSS::push_stats({"ARS", 0, scale_down_time, 0, 0, 0});
        // 4. 清理内存
        delete[] key.a; delete[] key.b; delete[] key.c;
        delete[] z_full_precision;
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

std::pair<SecureANDKeyPack, SecureANDKeyPack> keyGenSecureAND(int32_t size) {
    SecureANDKeyPack k0, k1;
    k0.size = k1.size = size;
    k0.a_share = new GroupElement[size];
    k1.a_share = new GroupElement[size];
    k0.b_share = new GroupElement[size];
    k1.b_share = new GroupElement[size];
    k0.c_share = new GroupElement[size];
    k1.c_share = new GroupElement[size];

    // prngs[0] 是 Dealer 的随机数生成器
    for (int i = 0; i < size; ++i) {
        // 1. 生成随机布尔比特 a 和 b
        uint8_t a = prngs[0].get<uint8_t>() & 1;
        uint8_t b = prngs[0].get<uint8_t>() & 1;
        // 2. 计算 c = a AND b
        uint8_t c = a & b;

        // 3. 将 a, b, c 拆分成异或份额
        auto a_split = splitShareXor(a, 1); // 假设 splitShareXor 返回 std::pair<uint64_t, uint64_t>
        k0.a_share[i] = (uint8_t)a_split.first;
        k1.a_share[i] = (uint8_t)a_split.second;
        
        auto b_split = splitShareXor(b, 1);
        k0.b_share[i] = (uint8_t)b_split.first;
        k1.b_share[i] = (uint8_t)b_split.second;

        auto c_split = splitShareXor(c, 1);
        k0.c_share[i] = (uint8_t)c_split.first;
        k1.c_share[i] = (uint8_t)c_split.second;
    }
    
    return std::make_pair(k0, k1);
}

/**
 * @brief 对两个布尔秘密共享数组执行安全 AND 操作。
 * 
 * @param size      数组大小
 * @param A_shares  输入 A 的布尔份额 (0 或 1)
 * @param B_shares  输入 B 的布尔份额 (0 或 1)
 * @param C_shares  输出 C = A & B 的布尔份额
 */
void SecureAND(int32_t size, 
               const GroupElement* A_shares, 
               const GroupElement* B_shares, 
               GroupElement* C_shares)
{
    // === Dealer 离线阶段 ===
    if (party == DEALER) {
        auto keys = keyGenSecureAND(size);
        server->send_secureand_key(&keys.first,1); // 需要在 comms.h/.cpp 中添加这个函数
        client->send_secureand_key(&keys.second,1);
        
        // 释放内存
        delete[] keys.first.a_share; 
        delete[] keys.second.a_share; 
        return;
    }

    // === 计算方在线阶段 ===
    
    // 1. 从 Dealer 接收密钥包
    SecureANDKeyPack keys[1];
    dealer->recv_secureand_key(keys,1); // 需要在 comms.h/.cpp 添加
    peer->sync(); // 确保双方都收到了密钥

    // 2. 本地计算掩码 epsilon = A ^ a, delta = B ^ b
    GroupElement* epsilon_shares = new GroupElement[size];
    GroupElement* delta_shares = new GroupElement[size];

    //#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        // A_shares 是 GroupElement (uint64_t), 但我们只关心最低位
        std::cout<<party<<": "<<i<<std::endl;
        GroupElement temp;
        temp = A_shares[i];
        temp = keys[0].a_share[i];
        epsilon_shares[i] = temp;
        epsilon_shares[i] = (GroupElement)(A_shares[i] & 1) ^ keys[0].a_share[i];
        delta_shares[i]   = (GroupElement)(B_shares[i] & 1) ^ keys[0].b_share[i];
    }
    
    // 3. 交互一次，公开 epsilon 和 delta
    reconstruct_big_bool(size, epsilon_shares); // 现在 epsilon_shares 存的是公开的 epsilon
    reconstruct_big_bool(size, delta_shares);   // 现在 delta_shares 存的是公开的 delta
    
    GroupElement* epsilon_public = epsilon_shares;
    GroupElement* delta_public = delta_shares;
    
    // 4. 本地计算最终结果 C = c ^ (eps & B) ^ (A & del) ^ (eps & del)
    // CrypTen 公式: (b & eps) ^ (a & del) ^ (eps & del) ^ c
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        uint8_t eps = epsilon_public[i];
        uint8_t del = delta_public[i];

        // CrypTen 的公式需要 a 和 b 的份额，但这里用 A 和 B 也可以推导
        // 公式: C = c ^ (eps & B) ^ (del & A) ^ (eps & del)
        // 证明:
        // C = c ^ ( (A^a) & B ) ^ ( (B^b) & A ) ^ ( (A^a) & (B^b) )
        //   = c ^ (A&B ^ a&B) ^ (B&A ^ b&A) ^ (A&B ^ A&b ^ a&B ^ a&b)
        //   = (c^a&b) ^ (A&B ^ a&B) ^ (B&A ^ b&A) ^ (A&B ^ A&b ^ a&B)
        //   = (a&b^a&b) ^ (A&B) ... (所有项都抵消了)
        // 更简单的 Beaver 公式: C = (eps & del) ^ (eps & b) ^ (del & a) ^ c
        
        uint8_t term1 = eps & del;
        uint8_t term2 = eps & keys[0].b_share[i];
        uint8_t term3 = del & keys[0].a_share[i];
        
        C_shares[i] = (GroupElement)(term1 ^ term2 ^ term3 ^ keys[0].c_share[i]);
    }
    
    // 清理内存
    //delete[] keys[0];
    delete[] keys[0].a_share;
    delete[] keys[0].b_share;
    delete[] keys[0].c_share;
    delete[] epsilon_shares;
    delete[] delta_shares;
}

void extract_bit_shares(int32_t size, const GroupElement* arr, int bit_pos, GroupElement* out_bit_shares) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        // 本地右移并取最低位即可
        out_bit_shares[i] = (arr[i] >> bit_pos) & 1;
    }
}

// 串行全加器实现
void SecureAdd(int32_t size, 
               const GroupElement* A_shares, 
               const GroupElement* B_shares, 
               GroupElement* Sum_shares)
{
    if(party==DEALER){
        GroupElement dummy_input1[size];
        GroupElement dummy_input2[size];
        GroupElement dummy_output[size];
        for (int i = 0; i < bitlength; ++i) {
            SecureAND(size, dummy_input1, dummy_input2, dummy_output);
            SecureAND(size, dummy_input1, dummy_input2, dummy_output);
        }
        return;
    }
    const int bitlength = FSSConfig::bitlength; 

    // 初始化最终结果和当前进位份额为 0
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        Sum_shares[i] = 0;
    }
    GroupElement* carry_in_shares = new GroupElement[size](); // 初始化为0

    // 临时数组
    GroupElement* a_i = new GroupElement[size];
    GroupElement* b_i = new GroupElement[size];
    GroupElement* half_sum = new GroupElement[size];
    GroupElement* term1 = new GroupElement[size];
    GroupElement* term2 = new GroupElement[size];
    GroupElement* carry_out_shares = new GroupElement[size];

    // 从最低位到最高位，逐位计算
    for (int i = 0; i < bitlength; ++i) {
        // 1. 提取 A 和 B 的第 i 位
        extract_bit_shares(size, A_shares, i, a_i);
        extract_bit_shares(size, B_shares, i, b_i);

        // 2. 计算 Sum_i = a_i ^ b_i ^ carry_in
        #pragma omp parallel for
        for (int j = 0; j < size; ++j) {
            half_sum[j] = a_i[j] ^ b_i[j];
            GroupElement sum_i_share = half_sum[j] ^ carry_in_shares[j];
            
            // 将当前位的和的份额，加到最终结果的对应位置上
            // (1ULL << i) 是一个公开常数，乘法是本地操作
            Sum_shares[j] += sum_i_share * (1ULL << i);
        }

        // 3. 计算 Carry_out = (a_i & b_i) | (half_sum & carry_in)
        // C_out = t1 | t2 = t1 ^ t2 ^ (t1 & t2)
        
        // 计算 t1 = a_i & b_i
        SecureAND(size, a_i, b_i, term1);
        
        // 计算 t2 = half_sum & carry_in
        SecureAND(size, half_sum, carry_in_shares, term2);

        // 计算 C_out = t1 ^ t2
        #pragma omp parallel for
        for (int j = 0; j < size; ++j) {
            carry_out_shares[j] = term1[j] ^ term2[j];
        }

        // 将 carry_out_shares 作为下一轮的 carry_in_shares
        std::swap(carry_in_shares, carry_out_shares);
    }

    // 清理内存
    delete[] carry_in_shares;
    delete[] a_i;
    delete[] b_i;
    delete[] half_sum;
    delete[] term1;
    delete[] term2;
    delete[] carry_out_shares;
}

/**
 * @brief 高效并行加法器 (Kogge-Stone 架构)
 * 通信轮数: 1 (初始化) + 6 (树状压缩) = 7 轮
 * 适用于 FSS 框架下的 A2B 或大量算术加法
 */
/**
 * @brief 高效并行加法器 (Kogge-Stone 架构)
 * 核心：利用 SecureAND 的批量处理能力，在 log(64)=6 轮内完成进位合并
 */
void SecureAddParallel(int32_t size, 
                       const GroupElement* A, 
                       const GroupElement* B, 
                       GroupElement* Sum)
{
    const int k = FSSConfig::bitlength; // 64
    const int total_bits = size * k;

    // --- 1. Dealer 离线模拟阶段 ---
    if (party == DEALER) {
        GroupElement *dummy_input1 = new GroupElement[total_bits];
        GroupElement *dummy_input2 = new GroupElement[total_bits];
        GroupElement *dummy_output = new GroupElement[total_bits];
        // 同步逻辑：1 (初始化) + 6 (树状迭代)
        SecureAND(total_bits, dummy_input1, dummy_input2, dummy_output);
        dummy_input1 = new GroupElement[2*total_bits];
        dummy_input2 = new GroupElement[2*total_bits];
        dummy_output = new GroupElement[2*total_bits];
        for (int offset = 1; offset < k; offset <<= 1) {
            SecureAND(2 * total_bits, dummy_input1, dummy_input2, dummy_output);
        }
        return;
    }

    // --- 2. 在线阶段：铺平比特 (Flattening) ---
    GroupElement* bits_G = new GroupElement[total_bits];
    GroupElement* bits_P = new GroupElement[total_bits];
    GroupElement* flat_A = new GroupElement[total_bits];
    GroupElement* flat_B = new GroupElement[total_bits];

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        for (int b = 0; b < k; ++b) {
            flat_A[i * k + b] = (A[i] >> b) & 1;
            flat_B[i * k + b] = (B[i] >> b) & 1;
        }
    }

    // 初始化初始进位信号：G = A & B, P = A ^ B
    SecureAND(total_bits, flat_A, flat_B, bits_G);
    #pragma omp parallel for
    for (int i = 0; i < total_bits; ++i) {
        bits_P[i] = flat_A[i] ^ flat_B[i];
    }

    // --- 3. 树状迭代 (Kogge-Stone 核心循环) ---
    for (int offset = 1; offset < k; offset <<= 1) {
        GroupElement* batch_left  = new GroupElement[2 * total_bits];
        GroupElement* batch_right = new GroupElement[2 * total_bits];
        GroupElement* batch_res   = new GroupElement[2 * total_bits];

        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            for (int j = offset; j < k; ++j) {
                int curr = i * k + j;
                int prev = i * k + (j - offset);
                // 打包 G_new 和 P_new 的计算
                batch_left[curr]  = bits_P[curr];
                batch_right[curr] = bits_G[prev];
                batch_left[curr + total_bits]  = bits_P[curr];
                batch_right[curr + total_bits] = bits_P[prev];
            }
        }

        SecureAND(2 * total_bits, batch_left, batch_right, batch_res);

        #pragma omp parallel for
        for (int i = 0; i < total_bits; ++i) {
            int bit_idx = i % k;
            if (bit_idx >= offset) {
                bits_G[i] ^= batch_res[i];
                bits_P[i]  = batch_res[i + total_bits];
            }
        }
        delete[] batch_left; delete[] batch_right; delete[] batch_res;
    }

    // --- 4. 结果组装 ---
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        GroupElement res = (flat_A[i * k] ^ flat_B[i * k]);
        for (int b = 1; b < k; ++b) {
            GroupElement bit_sum = (flat_A[i * k + b] ^ flat_B[i * k + b]) ^ bits_G[i * k + (b - 1)];
            res |= (bit_sum << b);
        }
        Sum[i] = res;
    }

    delete[] bits_G; delete[] bits_P; delete[] flat_A; delete[] flat_B;
}

void A2B(int size, const GroupElement* arithmetic_shares, GroupElement* binary_shares) {
    if(party==DEALER){
        GroupElement dummy_input1[size];
        GroupElement dummy_input2[size];
        GroupElement dummy_output[size];
        SecureAddParallel(size, dummy_input1, dummy_input2, dummy_output);
        return;
    }
    GroupElement* all_shares_as_binary = new GroupElement[size * 2];

    for (int i = 0; i < 2; ++i) {
        if (party - 2 == i) {
            GroupElement share1[size];
            GroupElement share2[size];
            for(int i = 0;i<size;i++){
                auto shares = splitShareXor(arithmetic_shares[i],1);
                share1[i] = shares.first;
                share2[i] = shares.second;
            }
            memcpy(all_shares_as_binary,share1,size*sizeof(GroupElement));
            peer->send_batched_input(share2, size, bitlength);
            peer->recv_batched_input(share2, size, bitlength);
            memcpy(all_shares_as_binary,share2,size*sizeof(GroupElement));
            //memcpy(all_shares_as_binary+size*sizeof(GroupElement),share2,size*sizeof(GroupElement));
        } else {
            GroupElement share1[size];
            GroupElement share2[size];
            GroupElement temp[size];
            for(int i = 0;i<size;i++){
                auto shares = splitShareXor(arithmetic_shares[i],1);
                share1[i] = shares.first;
                share2[i] = shares.second;
            }
            memcpy(all_shares_as_binary+size,share1,size*sizeof(GroupElement));
            //memcpy(all_shares_as_binary+size*sizeof(GroupElement),share1,size*sizeof(GroupElement));
            peer->recv_batched_input(temp, size, bitlength);
            peer->send_batched_input(share2, size, bitlength);
            memcpy(all_shares_as_binary,temp,size*sizeof(GroupElement));

        }
    }
    peer->sync();
    memcpy(binary_shares, all_shares_as_binary, size * sizeof(GroupElement));


    SecureAddParallel(size, binary_shares, all_shares_as_binary +  size, binary_shares);
    
    delete[] all_shares_as_binary;
}

/**
 * @brief 对算术秘密共享数组执行安全 ReLU 操作。
 * 
 * @param size      数组大小
 * @param inArr     输入的算术份额数组 [x]
 * @param outArr    输出的算术份额数组 ReLU([x])
 */
void SecureReLU(int32_t size, 
                const GroupElement* inArr, 
                GroupElement* outArr,int scale)
{
    int world_size = 2;
    // === Dealer 逻辑 ===
    if (party == DEALER) {
        // Dealer 需要为所有底层的协议生成密钥
        // 1. A2B 内部的 SecureAdd -> SecureAND
        GroupElement dummy_input1[size];
        GroupElement dummy_input2[size];
        GroupElement dummy_output[size];
        uint8_t* dummy_int1 = new uint8_t[size];
        A2B(size,dummy_input1,dummy_input2);
        // 2. B2A
        B2A_Crypten(size, dummy_int1, dummy_input2);
        
        // 3. ElemWiseMul
        ElemWiseMul(size, dummy_input1, dummy_input1, dummy_input2, dummy_input2, dummy_output, dummy_output);
        
        return;
    }

    // === 计算方逻辑 ===

    // --- 步骤 1: 将算术份额 [x] 转换为二进制份额 <x> ---
    GroupElement* x_binary_shares = new GroupElement[size];
    // 这里需要一个 A2B 的实现，它内部会调用 SecureAdd 和 SecureAND
    // 我们假设 A2B_Protocol 封装了这个逻辑
    A2B(size, inArr, x_binary_shares);
    
    // --- 步骤 2: 提取符号位 <msb> 并计算比较结果 <res> = NOT <msb> ---
    uint8_t* comparison_bit_shares = new uint8_t[size]; // B2A_Crypten 需要 uint8_t
    
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        // a. 提取 MSB (本地右移)
        uint8_t msb_share = (x_binary_shares[i] >> (FSSConfig::bitlength - 1)) & 1;
        
        // b. 计算 [x > 0] = NOT [msb]
        //    NOT(a) = 1 ^ a. 只在一方执行异或1的操作。
        if (party == SERVER) {
            comparison_bit_shares[i] = 1 ^ msb_share;
        } else {
            comparison_bit_shares[i] = msb_share;
        }
    }
    delete[] x_binary_shares;

    // --- 步骤 3: 将比较结果的布尔份额 <res> 转换为算术份额 [res] ---
    GroupElement* comparison_arith_shares = new GroupElement[size];
    B2A_Crypten(size, comparison_bit_shares, comparison_arith_shares);
    delete[] comparison_bit_shares;
    
    // --- 步骤 4: 计算最终结果 [y] = [x] * [res] ---
    ElemWiseMul(size, inArr, nullptr, comparison_arith_shares, nullptr, outArr, nullptr);
    
    delete[] comparison_arith_shares;
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

inline GroupElement count_local_wrap(GroupElement a, GroupElement b) {
    // 将无符号的 GroupElement 转换为有符号的 int64_t 来进行判断
    int64_t signed_a = static_cast<int64_t>(a);
    int64_t signed_b = static_cast<int64_t>(b);
    
    // 加法仍然在 uint64_t 上进行，以模拟环的行为
    GroupElement next_unsigned = a + b;
    int64_t next_signed = static_cast<int64_t>(next_unsigned);

    // 检查上溢: 两个正数相加，结果为负数
    if (signed_a > 0 && signed_b > 0 && next_signed < 0) {
        return 1; // 上溢
    }
    
    // 检查下溢: 两个负数相加，结果为正数
    if (signed_a < 0 && signed_b < 0 && next_signed > 0) {
        return -1; // 下溢，返回 -1 (在环上是一个大正数)
    }

    return 0; // 没有溢出
}

// inline GroupElement count_local_wrap(GroupElement a, GroupElement b) {
//     // 将无符号的 GroupElement 转换为有符号的 int64_t 来进行判断
//     GroupElement c = a + b;
//     if(a>c||b>c){
//         return 1;
//     }else{
//         return 0;
//     }
// }

void ARS_CrypTen_Style(int32_t size, 
                       GroupElement* inArr, 
                       GroupElement* outArr, 
                       int32_t shift)
{
    // === Dealer 离线阶段 ===
    if (party == DEALER) {
        for (int i = 0; i < size; ++i) {
            auto keys = keyGenARS_CrypTen_Style(bitlength);
            server->send_ars_crypten_key(keys.first);
            client->send_ars_crypten_key(keys.second);
        }
        return;
    }
    GroupElement debug_beta_xr_share = 0;
    GroupElement debug_theta_r_share = 0;
    GroupElement debug_theta_z_share = 0;
    // === 计算方在线阶段 ===
    ARS_CrypTen_Style_KeyPack* keys = new ARS_CrypTen_Style_KeyPack[size];
    for(int i=0; i<size; ++i){
        keys[i] = dealer->recv_ars_crypten_key();
        if (i == 3) { // 保存第一个元素的 theta_r 份额用于调试
            debug_theta_r_share = keys[i].theta_r_share;
        }
    }

    GroupElement* z_shares = new GroupElement[size];
    GroupElement* beta_xr_shares = new GroupElement[size];
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        beta_xr_shares[i] = count_local_wrap(inArr[i], keys[i].r_share);
        z_shares[i] = inArr[i] + keys[i].r_share;
    }
    if (size > 0) { // 保存第一个元素的 beta_xr 份额用于调试
        debug_beta_xr_share = beta_xr_shares[3];
    }

    GroupElement* wrap_count_shares = new GroupElement[size];
    if (party == SERVER) {
        peer->send_batched_input(z_shares, size, bitlength);
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            wrap_count_shares[i] = beta_xr_shares[i] - keys[i].theta_r_share;
        }
        debug_theta_z_share = 0;
    } else { // CLIENT
        GroupElement* z_other_shares = new GroupElement[size];
        peer->recv_batched_input(z_other_shares, size, bitlength);
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            GroupElement theta_z = count_local_wrap(z_other_shares[i], z_shares[i]);
            if (i == 3) { // 保存第一个元素的 theta_z (真实值) 用于调试
                debug_theta_z_share = theta_z;
            }
            wrap_count_shares[i] = theta_z + beta_xr_shares[i] - keys[i].theta_r_share;
        }
        delete[] z_other_shares;
    }
    
    //if (size > 0) {
    //    printf("\n--- [Party %d] ARS Intermediate Value Debug ---\n", party);
    //    
    //    GroupElement debug_values[3];
    //    debug_values[0] = debug_beta_xr_share;
    //    debug_values[1] = debug_theta_r_share;
    //    debug_values[2] = debug_theta_z_share;
    //    printf("  beta_xr share for element 0: %llu\n", debug_values[0]);
    //    // 现在双方都持有各自的份额，可以一起调用reconstruct
    //    reconstruct(3, debug_values, bitlength);
    //
    //    // reconstruct之后，debug_values里存的是明文
    //    // 只有一方打印即可，避免重复输出
    //    if (party == SERVER) {
    //        printf("  Reconstructed beta_xr for element 0: %llu\n", debug_values[0]);
    //        printf("  Reconstructed theta_r for element 0: %llu\n", debug_values[1]);
    //        printf("  Reconstructed theta_z for element 0: %llu\n", debug_values[2]);
    //    }
    //}


    //GroupElement *temp = new GroupElement[size];
    //memcpy(temp,wrap_count_shares,size*sizeof(GroupElement));
    //reconstruct(size,temp,bitlength);
    //print_array("wrap count",party,size,temp,size);

    // 最终组合
    //GroupElement correction_term_multiplier = (1ULL << (bitlength - shift));
    //GroupElement correction_term_multiplier = 4ULL * ( (1ULL << (bitlength - 2)) >> shift );
    GroupElement correction_term_multiplier = (1ULL << (bitlength - shift));
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        GroupElement plain_truncate = static_cast<int64_t>(inArr[i]) >> shift;
        GroupElement correction = wrap_count_shares[i] * correction_term_multiplier;
        //correction = 0;
        outArr[i] = plain_truncate - correction;
    }

    delete[] keys;
    delete[] z_shares;
    delete[] beta_xr_shares;
    delete[] wrap_count_shares;
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

void print_double_array(const std::string& title, int party, int size, GroupElement* arr, int limit) {
    //if (party == DEALER) return; // Dealer 不打印

    std::cout << "\n--- [Party " << party << "] " << title << " ---" << std::endl;
    for (int i = 0; i < size && i < limit; ++i) {
        // 为了可读性，我们可以将 uint64_t 转换为 int64_t 来打印
        // 这样负数（在模运算下的大正数）会更容易看懂
        mod_array(arr,size,FSSConfig::bitlength);
        auto temp = fixed_to_double(arr[i],16);
        std::cout << "  [" << i << "]: " << temp << std::endl;
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
        
        peer->sync();
        
        uint64_t keysize_start = dealer->bytesReceived();
        key = dealer->recv_dpf_route_key(size, data_bw, rank_bw);
        GroupElement* y_plus_r_shares = new GroupElement[size];
        GroupElement* z_mul_s_shares = new GroupElement[size];

        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            y_plus_r_shares[i] = y_in[i] + key.r_shares[i];
        }
        reconstruct(size, y_plus_r_shares, FSSConfig::bitlength); 
        GroupElement* y_hat_public = y_plus_r_shares; 
        
        ElemWiseMul(size, 
                    z_in, z_in, 
                    key.s_shares, key.s_shares,
                    z_mul_s_shares, z_mul_s_shares);

        reconstruct(size, z_mul_s_shares, bitlength);
        GroupElement* z_tilde_public = z_mul_s_shares;
        
        int size = key.size;
        int rank_bin = key.rank_bin;
        int data_bin = key.data_bin;
        auto start_time = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int k = 0; k < size; ++k) {
            GroupElement target_rank_k = k;
            GroupElement result_share_k = 0;

            for (int i = 0; i < size; ++i) {
                GroupElement dpf_input = y_hat_public[i] - k;
                mod(dpf_input, rank_bin);

                GroupElement v_share_i = evalDPF_with_payload(party-2, key.routing_keys[i], dpf_input);
                GroupElement term = z_tilde_public[i] * v_share_i;

                result_share_k += term;
            }
            z_out[k] = result_share_k;
        }
                    // 3. 记录结束时间
        auto end_time = std::chrono::high_resolution_clock::now();

        // 4. 计算时间差并打印
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // 为了防止多个参与方都打印时间，可以只让一个 party (例如 party 0) 打印
        if (party == 2) {
            std::cout << "================================================" << std::endl;
            std::cout << "Total execution time: " << duration.count() << " milliseconds" << std::endl;
            std::cout << "Total execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
            std::cout << "================================================" << std::endl;
        }
        mod_array(z_out, size, data_bw);
    }
}

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
) {
    if (party == DEALER) {
        std::cout << "[Dealer] Generating and sending graph update keys..." << std::endl;
        
        // 1. 调用 keyGen 生成两方的密钥包
        auto key_pair = keyGenForGraphUpdate(
            target_node_v_star,
            n, c,
            A_old, A_new, A_bw, A_data_bw,
            F_old, F_new, F_bw, F_data_bw
        );
        
        server->send_graph_update_key(key_pair.first);
        client->send_graph_update_key(key_pair.second);

        std::cout << "[Dealer] Keys sent." << std::endl;

    } else {
        std::cout << "[Party " << party << "] Receiving graph update keys..." << std::endl;


        GraphUpdateKeyPack key = dealer->recv_graph_update_key();

        std::cout << "[Party " << party << "] Keys received. Synchronizing with peer..." << std::endl;

        // 2. 与另一个计算方同步，确保双方都收到了密钥再开始计算
        peer->sync();

        std::cout << "[Party " << party << "] Starting oblivious update computation..." << std::endl;

        // 3. 执行不经意更新的计算部分
        obliviousUpdate(party, n, c, A_share, F_share, key.keys_A,key.keys_F);

        std::cout << "[Party " << party << "] Oblivious update computation finished." << std::endl;
        peer->sync();
    }
}

void three_interval_check(
    uint8_t party,
    GroupElement x_share,
    GroupElement a, // Public boundary 1
    GroupElement b, // Public boundary 2
    uint8_t bin,
    OneHotShares& result_shares) 
{
    // --- 1. & 2. Dealer (离线) 准备并分发 DPF 密钥 [e_i] 和 i 的份额 [i] ---
    // (这部分代码与原来完全相同，保持不变)
    DPFKeyPack my_key(bin, 1);
    GroupElement my_i_share = 0;

    if (party == DEALER) {
        // ... (Dealer a逻辑, 省略) ...
    } else {
        my_key = dealer->recv_dpf_keypack(bin, 1);
        dealer->recv_ge_array(&my_i_share, 1);
    }
    if (party != DEALER) peer->sync();

    // ==================== 修改开始 ====================

    // --- Step 1 (Revised): 交互一次, 公开统一的移位量 d = x - i ---
    // 这个 d 代表了我们的秘密坐标系 x 相对于随机坐标系 i 的偏移量。
    GroupElement d_share = x_share - my_i_share;
    
    // 调用 reconstruct 来安全地公开 d。
    // reconstruct 内部处理通信，返回 d 的明文值。
    // 假设 reconstruct(share) 返回 share_0 + share_1
    GroupElement d ;//= reconstruct_single_value(d_share); 

    // --- Step 2 (Revised): 本地计算旋转后的边界点 a' 和 b' ---
    const GroupElement N = 1ULL << bin;
    const GroupElement N_half = N / 2;
    
    // 我们将所有关于 x 的比较，都转换成关于 i 的等价比较。
    // x < a  <=>  x - i < a - i  <=>  i > x - (a - x) ... 这样做太复杂
    // 正确的转换是: x < a  <=>  i < a - (x - i)  <=>  i < a - d
    
    // 计算旋转后的边界点。我们现在要在 i 的坐标系中观察 a 和 b。
    // (a - d) mod N
    const GroupElement a_prime = (a - d + N) % N; 
    // (b - d) mod N
    const GroupElement b_prime = (b - d + N) % N;

    // 2.1 准备所有需要查询的前缀端点
    // 比较 [i < a'] <=> i in [N/2 - a', N - a') mod N
    // 比较 [i < b'] <=> i in [N/2 - b', N - b') mod N
    const GroupElement endpoints_to_query[4] = {
        (N - a_prime + N) % N,
        (N_half - a_prime + N) % N,
        (N - b_prime + N) % N,
        (N_half - b_prime + N) % N
    };

    // ==================== 修改结束 ====================

    // --- Step 3: 一次性计算所有前缀的奇偶性份额 ---
    // (这部分代码与原来完全相同，保持不变)
    std::map<GroupElement, uint8_t>* parity_shares_map = compute_prefix_parities(
        party, my_key, endpoints_to_query, 4);

    // --- Step 4: 本地组合得到比较结果份额 c1 = [x < a] 和 c2 = [x < b] ---
    // 注意，我们现在用旋转后的端点 a_prime 和 b_prime 来从 map 中取值
    // [x < a] <=> [i < a']
    uint8_t c1_share = (*parity_shares_map)[(N - a_prime + N) % N] ^ (*parity_shares_map)[(N_half - a_prime + N) % N];
    // [x < b] <=> [i < b']
    uint8_t c2_share = (*parity_shares_map)[(N - b_prime + N) % N] ^ (*parity_shares_map)[(N_half - b_prime + N) % N];

    delete parity_shares_map;

    // --- Step 5: 本地组合生成 OneHotShares ---
    // (这部分代码与原来完全相同，逻辑是正确的)
    // s0 = [x < a]
    result_shares.s0 = c1_share;
    // s1 = [a <= x < b] <=> [x < b] XOR [x < a]
    result_shares.s1 = c1_share ^ c2_share;
    // s2 = [x >= b] <=> NOT [x < b]
    result_shares.s2 = 1 ^ c2_share;
}

void FastRelu(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), std::string prefix)
{
    if (party == DEALER)
    {
        // Dealer为每个元素生成密钥并发送
        for (int i = 0; i < size; i++)
        {
            auto keys = keyGenFastRelu(bitlength, bitlength);
            server->send_fast_relu_key(keys.first);
            client->send_fast_relu_key(keys.second);
            // 注意：需要释放keyGenFastRelu中为DCFKeyPack分配的内存
            // 这通常在KeyPack的析构函数中处理
            
        }
    }
    else
    {
        // 存储所有密钥
        std::vector<FastReluKeyPack> keys(size);
        for (int i = 0; i < size; i++) {
            keys[i] = dealer->recv_fast_relu_key(bitlength, bitlength);
        }

        peer->sync(); // 等待双方都接收完密钥

        // 遵循 Algorithm 4: EvalReLU
        
        // 1. 准备公开 x+r
        GroupElement* masked_x = new GroupElement[size];
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            masked_x[i] = inArr[i] + keys[i].r_sh;
        }

        // 2. 交互一次以重构 masked_x
        reconstruct(size, masked_x, bitlength);
        //print_array("Original Plaintext 'x+r'", party, size, masked_x,size);
        //print_double_array("Original Plaintext(double) 'x+r'", party, size, masked_x,size);
        // 3. 本地计算 FSS 并得到系数分享
        GroupElement* coeff_shares = new GroupElement[size * 2]; // 存储所有b0, b1的分享
        
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            // evalDCF需要一个数组来接收结果，因为groupSize=2
            GroupElement* b_shares_i = new GroupElement[2];
            // 对公开值 masked_x[i] (即 x+r) 进行求值
            evalDCF(party-2, b_shares_i, masked_x[i], keys[i].dcfKey);
            
            coeff_shares[i*2 + 0] = b_shares_i[0] + keys[i].b_sh[0]; // [b0]_p
            coeff_shares[i*2 + 1] = b_shares_i[1] + keys[i].b_sh[1]; // [b1]_p

            delete[] b_shares_i;
        }
        


        // GroupElement* coeff_shares_temp = new GroupElement[size * 2]; 
        // memcpy(coeff_shares_temp,coeff_shares,size*2);
        // reconstruct(size*2, coeff_shares_temp, bitlength);
        // print_array("Original Plaintext 'b'", party, size*2, coeff_shares_temp,size*2);
        
        // 4. 本地计算最终输出份额
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            GroupElement b0_sh = coeff_shares[i*2 + 0];
            GroupElement b1_sh = coeff_shares[i*2 + 1];

            // [y]_p = [b0]_p * (x+r) + [b1]_p
            // 这是一个公开数(masked_x[i])和秘密份额的乘法，是本地操作
            outArr[i] = b0_sh * masked_x[i] + b1_sh;
            mod(outArr[i], bitlength);
        }

        // GroupElement* temp = new GroupElement[size]; 
        // memcpy(temp,outArr,size * sizeof(GroupElement));
        // reconstruct(size, temp, bitlength);
        // print_double_array("Original Plaintext 'res'", party, size, temp,size);

        delete[] masked_x;
        delete[] coeff_shares;
    }
}


void OptFastRelu(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), std::string prefix)
{
    if (party == DEALER)
    {
        // Dealer为每个元素生成密钥并发送
        for (int i = 0; i < size; i++)
        {
            auto keys = keyGenFastRelu_DPFET(bitlength, bitlength);
            server->send_fast_relu_dpfet_key(keys.first);
            client->send_fast_relu_dpfet_key(keys.second);
            // 注意：需要释放keyGenFastRelu中为DCFKeyPack分配的内存
            // 这通常在KeyPack的析构函数中处理
            
        }
    }
    else
    {
        // 存储所有密钥
        std::vector<FastReluKeyPack> keys(size);
        // for (int i = 0; i < size; i++) {
        //     keys[i] = dealer->recv_fast_relu_dpfet_key(bitlength, bitlength);
        // }

        peer->sync(); // 等待双方都接收完密钥

        // 遵循 Algorithm 4: EvalReLU
        
        // 1. 准备公开 x+r
        GroupElement* masked_x = new GroupElement[size];
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            masked_x[i] = inArr[i] + keys[i].r_sh;
        }

        // 2. 交互一次以重构 masked_x
        reconstruct(size, masked_x, bitlength);
        //print_array("Original Plaintext 'x+r'", party, size, masked_x,size);
        //print_double_array("Original Plaintext(double) 'x+r'", party, size, masked_x,size);
        // 3. 本地计算 FSS 并得到系数分享
        GroupElement* coeff_shares = new GroupElement[size * 2]; // 存储所有b0, b1的分享
        
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            // evalDCF需要一个数组来接收结果，因为groupSize=2
            GroupElement* b_shares_i = new GroupElement[2];
            // 对公开值 masked_x[i] (即 x+r) 进行求值
            evalDCF(party-2, b_shares_i, masked_x[i], keys[i].dcfKey);
            
            coeff_shares[i*2 + 0] = b_shares_i[0] + keys[i].b_sh[0]; // [b0]_p
            coeff_shares[i*2 + 1] = b_shares_i[1] + keys[i].b_sh[1]; // [b1]_p

            delete[] b_shares_i;
        }
        


        // GroupElement* coeff_shares_temp = new GroupElement[size * 2]; 
        // memcpy(coeff_shares_temp,coeff_shares,size*2);
        // reconstruct(size*2, coeff_shares_temp, bitlength);
        // print_array("Original Plaintext 'b'", party, size*2, coeff_shares_temp,size*2);
        
        // 4. 本地计算最终输出份额
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            GroupElement b0_sh = coeff_shares[i*2 + 0];
            GroupElement b1_sh = coeff_shares[i*2 + 1];

            // [y]_p = [b0]_p * (x+r) + [b1]_p
            // 这是一个公开数(masked_x[i])和秘密份额的乘法，是本地操作
            outArr[i] = b0_sh * masked_x[i] + b1_sh;
            mod(outArr[i], bitlength);
        }

        // GroupElement* temp = new GroupElement[size]; 
        // memcpy(temp,outArr,size * sizeof(GroupElement));
        // reconstruct(size, temp, bitlength);
        // print_double_array("Original Plaintext 'res'", party, size, temp,size);

        delete[] masked_x;
        delete[] coeff_shares;
    }
}



void clip_with_relu(int32_t size,GroupElement *inArr,GroupElement *outArr, GroupElement lower, GroupElement upper){
    GroupElement* clip_relu_in = new GroupElement[size * 2];
    if(party == DEALER){
        FastRelu(size * 2, clip_relu_in, clip_relu_in, clip_relu_in, clip_relu_in);
    }else{        
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            if (party == SERVER) {
                clip_relu_in[i]        = inArr[i] - upper;
                clip_relu_in[i + size] = lower - inArr[i];
            } else {
                clip_relu_in[i]        = inArr[i];
                clip_relu_in[i + size] = -inArr[i];
            }
        }

        // GroupElement* temp = new GroupElement[size]; 
        // memcpy(temp,inArr,size * sizeof(GroupElement));
        // reconstruct(size, temp, bitlength);
        // print_double_array("Original Plaintext 'input softmax'", party, size, temp, size);

        // temp = new GroupElement[2*size]; 
        // memcpy(temp,clip_relu_in,2 * size * sizeof(GroupElement));
        // reconstruct(size*2, temp, bitlength);

        
        // print_array("Original Plaintext 'relu input'", party, size*2, temp, size*2);
        // print_double_array("Original Plaintext 'relu input(double)'", party, size*2, temp, size*2);

        GroupElement* clip_relu_out = new GroupElement[size * 2];
        FastRelu(size * 2, clip_relu_in, nullptr, clip_relu_out, nullptr);

        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            outArr[i] = inArr[i] + clip_relu_out[i + size] - clip_relu_out[i];
        }
        

        //GroupElement* temp = new GroupElement[size]; 
        //memcpy(temp,x,size * sizeof(GroupElement));
        //reconstruct(size, temp, bitlength);
        //print_double_array("Original Plaintext 'x'", party, size, temp, size);

        delete[] clip_relu_in;
        delete[] clip_relu_out;
    }
}


// =========================================================================
// == 辅助函数: DPF三区间判断 (这个可以放在一个新文件 dpf_interval.cpp/h 中)
// =========================================================================
/**
 * @brief 使用DPF安全地判断一个秘密值x相对于两个公开边界的位置。
 *
 * @param x_share 秘密输入x的加性份额。
 * @param lower 公开下界。
 * @param upper 公开上界。
 * @return OneHotShares_xor 包含三个区间判断结果的XOR秘密份额。
 */
OneHotShares_xor dpf_three_interval_check(GroupElement x_share, GroupElement lower, GroupElement upper)
{
    GroupElement shift = double_to_fixed(1000.0,16);
    const int bin = FSSConfig::bitlength;
    OneHotShares_xor result;
    x_share += shift/2;
    lower+=shift;
    upper+=shift;
    // --- Dealer 逻辑 (保持不变) ---
    if (party == DEALER) {
        GroupElement alpha = prngs[0].get<GroupElement>() % (1ULL << bin);
        alpha = 1024*64*1000;
        auto key_pair = keyGenDPFET(bin, alpha);
        auto alpha_split = splitShare(alpha, bin);
        server->send_dpfet_keypack(key_pair.first);
        client->send_dpfet_keypack(key_pair.second);
        server->send_ge_array(&alpha_split.first, 1);
        client->send_ge_array(&alpha_split.second, 1);
        return {};
    }

    // --- 计算方逻辑 ---
    
    // a. 接收密钥和份额
    auto my_key = dealer->recv_dpfet_keypack(bin);
    GroupElement my_alpha_share;
    dealer->recv_ge_array(&my_alpha_share, 1);

    // b. 重构公开移位量 d = x - alpha
    GroupElement d_share = x_share - my_alpha_share;
    reconstruct(1, &d_share, bin);
    GroupElement d = d_share;

    // c. 准备比较的输入值
    GroupElement C_lower = lower;
    GroupElement lower_prime = C_lower - d;
    
    GroupElement C_upper = upper;
    GroupElement upper_prime = C_upper - d;
    
    // 执行两次比较，得到XOR份额
    uint8_t res_ge_lower_share = evalDPFET_LT(party-2, my_key, lower_prime);
    uint8_t res_ge_upper_share = evalDPFET_LT(party-2, my_key, upper_prime);

    // =======================================================
    // ==         SIMPLE DEBUGGING PRINT BLOCK              ==
    // =======================================================
    // peer->sync(); 

    // // --- 准备重构 ---
    // GroupElement temp_x = x_share;
    // GroupElement temp_alpha = my_alpha_share;
    // uint8_t temp_res_ge_lower = res_ge_lower_share;
    // uint8_t temp_res_ge_upper = res_ge_upper_share;

    // // --- 执行重构 ---
    // reconstruct(1, &temp_x, bin);
    // reconstruct(1, &temp_alpha, bin);
    // reconstruct_bool(1, &temp_res_ge_lower);
    // reconstruct_bool(1, &temp_res_ge_upper);

    // // --- 只让 SERVER 打印 ---
    // if (party == SERVER) {
    //     const int scale = 16;
    //     std::cout << "\n--- DEBUG (x=" << fixed_to_double(temp_x, scale) << ") ---" << std::endl;
    //     std::cout << "x=" << (int64_t)temp_x << ", alpha=" << (int64_t)temp_alpha << ", d=" << (int64_t)d << std::endl;
    //     std::cout << "[LOWER] lower'=" << (int64_t)lower_prime 
    //               << ", eval([alpha >= lower]): " << (int)temp_res_ge_lower 
    //               << " (Expected: " << ((int64_t)temp_alpha >= (int64_t)lower) << ")" << std::endl;
    //     std::cout << "[UPPER] upper'=" << (int64_t)upper_prime 
    //               << ", eval([alpha >= upper]): " << (int)temp_res_ge_upper
    //               << " (Expected: " << ((int64_t)temp_alpha >= (int64_t)upper) << ")" << std::endl;
    //     std::cout << "---------------------------\n" << std::endl;
    // }
    // =======================================================

    // d. 本地计算最终的独热编码XOR份额
    result.s0_share = (party == SERVER) ? (1 ^ res_ge_lower_share) : res_ge_lower_share;
    uint8_t res_lt_upper_share = (party == SERVER) ? (1 ^ res_ge_upper_share) : res_ge_upper_share;
    result.s1_share = result.s0_share ^ res_lt_upper_share;
    result.s2_share = (party == SERVER) ? (1 ^ res_lt_upper_share) : res_lt_upper_share;
    
    peer->sync();

    return result;
}



void SlothTRfromWrap(int size, int bin, GroupElement *x, GroupElement *w, GroupElement *y, int scale, std::string parent)
{
    if (party == DEALER)
    {
        pair<SlothLRSKeyPack> *keys = new pair<SlothLRSKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            GroupElement rout = random_ge(1);
            keys[i] = keyGenSlothLRS(bin, scale, x[i], w[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_sloth_lrs_key(keys[i].first);
            client->send_sloth_lrs_key(keys[i].second);
        }

        delete[] keys;
    }
    else
    {
        SlothLRSKeyPack *keys = new SlothLRSKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        uint64_t keyread_time = time_this_block([&]()
                                                {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_sloth_lrs_key(bin, scale);
            } });

        peer->sync();

        uint64_t compute_time = time_this_block([&]()
                                                {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                y[i] = evalSlothLRS(party - 2, x[i], w[i], keys[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, y, bin - scale); });

        FSS::stat_t stat = {
            parent,
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

void SlothTR(int size, int bin, GroupElement *x, GroupElement *y, int scale, std::string prefix)
{
    GroupElement *w = new GroupElement[size];
    GroupElement *x0 = w;

    auto t = time_this_block([&]()
                             {
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        x0[i] = x[i];
        mod(x0[i], scale);
    } });

    SlothWrap(size, scale, x0, w, prefix + "TruncateReduce");
    SlothTRfromWrap(size, bin, x, w, y, scale, prefix + "TruncateReduce");

    delete[] w;

    if (party != DEALER)
        FSS::push_stats({prefix + "TruncateReduce::Misc", 0, t, 0, 0, 0});
}

void SlothSignExtendFromWrap(int size, int bin, int bout, GroupElement *x, GroupElement *w, GroupElement *y, std::string parent)
{
    if (party == DEALER)
    {
        pair<SlothSignExtendKeyPack> *keys = new pair<SlothSignExtendKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            GroupElement rout = random_ge(bout);
            keys[i] = keyGenSlothSignExtend(bin, bout, x[i], w[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_sloth_sign_extend_key(keys[i].first);
            client->send_sloth_sign_extend_key(keys[i].second);
        }

        delete[] keys;
    }
    else
    {
        SlothSignExtendKeyPack *keys = new SlothSignExtendKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        uint64_t keyread_time = time_this_block([&]()
                                                {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_sloth_sign_extend_key(bin, bout);
            } });

        peer->sync();

        uint64_t compute_time = time_this_block([&]()
                                                {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                y[i] = evalSlothSignExtend(party - 2, x[i], w[i], keys[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, y, bout); });

        FSS::stat_t stat = {
            parent,
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

void SlothFaithfulARS(int size, int bin, GroupElement *x, GroupElement *y, int scale, std::string prefix)
{
    GroupElement *w = new GroupElement[size];

    SlothTR(size, bin, x, y, scale, prefix + "FaithfulARS::");

    if (party != DEALER)
    {
        auto t = time_this_block([&]()
                                 {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            y[i] = y[i] + (1LL << (bin-scale-1));
            mod(y[i], bin-scale);
        } });

        FSS::push_stats({prefix + "FaithfulARS::Misc", 0, t, 0, 0, 0});
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            mod(y[i], bin - scale);
        }
    }

    SlothWrap(size, bin - scale, y, w, prefix + "FaithfulARS");
    SlothSignExtendFromWrap(size, bin - scale, bin, y, w, y, prefix + "FaithfulARS");

    delete[] w;
}

// =========================================================================
// == 最终的裁剪函数 (这个可以放在你的 protocol/clip.cpp 或类似文件中)
// =========================================================================
/**
 * @brief 使用DPF三区间判断安全地将输入裁剪到 [lower, upper] 区间。
 */
void clip_with_dpf(
    int32_t size,
    GroupElement *inArr,
    GroupElement *outArr,
    GroupElement lower,
    GroupElement upper,
    int scale)
{
    const int bin = FSSConfig::bitlength;
    const GroupElement shift = double_to_fixed(1000.0, 16);

    // --- Dealer Logic (Offline Phase) ---
    if (party == DEALER) {
        for (int i = 0; i < size; ++i) {
            // !! PRESERVING USER'S HARDCODED ALPHA LOGIC !!
            GroupElement alpha = 1024*64*1000;
            auto key_pair = keyGenDPFET(bin, alpha);
            auto alpha_split = splitShare(alpha, bin);

            server->send_dpfet_keypack(key_pair.first);
            client->send_dpfet_keypack(key_pair.second);
            server->send_ge_array(&alpha_split.first, 1);
            client->send_ge_array(&alpha_split.second, 1);
        }
        
        GroupElement* dummy1 = new GroupElement[size];
        GroupElement* dummy2 = new GroupElement[size];
        GroupElement* dummy3 = new GroupElement[size*3];
        uint8_t* dummy_int1 = new uint8_t[size * 3];
        B2A_Crypten(size*3,dummy_int1,dummy3);
        // 2. 为 `ElemWiseMul` 准备 Beaver 三元组
        ElemWiseMul(size, dummy1, dummy1, dummy2, dummy2, dummy2, dummy2);
        
        return;
    }

    // --- Computing Party Logic (Online Phase) ---

    // -- Step 1: Receive all keys and randomness for the entire batch --
    DPFETKeyPack* all_keys = new DPFETKeyPack[size];
    GroupElement* all_alpha_shares = new GroupElement[size];
    for (int i = 0; i < size; ++i) {
        all_keys[i] = dealer->recv_dpfet_keypack(bin);
        dealer->recv_ge_array(&all_alpha_shares[i], 1);
    }

    // -- Step 2: Locally compute all `d_share` values for the batch --
    GroupElement* all_d_shares = new GroupElement[size];
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        // !! PRESERVING USER'S SHIFT/2 LOGIC !!
        GroupElement shifted_x_share = inArr[i] + shift/2;
        all_d_shares[i] = shifted_x_share - all_alpha_shares[i];
    }

    // -- Step 3: Perform ONE-SHOT BATCHED RECONSTRUCTION --
    reconstruct(size, all_d_shares, bin);
    GroupElement* all_d_public = all_d_shares;

    // -- Step 4: Perform all local computations for the batch --
    uint8_t* res_lower_shares = new uint8_t[size];
    uint8_t* res_upper_shares = new uint8_t[size];
    
    // Apply shift to public boundaries
    GroupElement shifted_lower = lower + shift;
    GroupElement shifted_upper = upper + shift;

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        GroupElement lower_prime = shifted_lower - all_d_public[i];
        GroupElement upper_prime = shifted_upper - all_d_public[i];

        // Perform the two DPF evaluations locally and store their shares
        res_lower_shares[i] = evalDPFET_LT(party - 2, all_keys[i], lower_prime);
        res_upper_shares[i] = evalDPFET_LT(party - 2, all_keys[i], upper_prime);
    }

    // -- Step 5: Continue with the original logic (Local combination + B2A) --
    uint8_t* s0_xor_shares = new uint8_t[size];
    uint8_t* s1_xor_shares = new uint8_t[size];
    uint8_t* s2_xor_shares = new uint8_t[size];
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        s0_xor_shares[i] = (party == SERVER) ? (1 ^ res_lower_shares[i]) : res_lower_shares[i];
        uint8_t res_lt_upper_share = (party == SERVER) ? (1 ^ res_upper_shares[i]) : res_upper_shares[i];
        s1_xor_shares[i] = s0_xor_shares[i] ^ res_lt_upper_share;
        s2_xor_shares[i] = (party == SERVER) ? (1 ^ res_lt_upper_share) : res_lt_upper_share;
    }

    // =========================================================================
    // ==               ENHANCED DEBUGGING BLOCK START                        ==
    // =========================================================================
    //peer->sync(); // 同步，确保双方在同一点开始重构，避免日志混乱

    // 1. 创建临时数组用于存储待重构的份额副本
    //GroupElement* temp_inArr_plain = new GroupElement[size];
    //uint8_t* temp_res_lower_plain = new uint8_t[size];
    //uint8_t* temp_res_upper_plain = new uint8_t[size];
    //uint8_t* temp_s0_plain = new uint8_t[size];
    //uint8_t* temp_s1_plain = new uint8_t[size];
    //uint8_t* temp_s2_plain = new uint8_t[size];

    // 2. 复制份额数据到临时数组
    //memcpy(temp_inArr_plain, inArr, size * sizeof(GroupElement));
    //memcpy(temp_res_lower_plain, res_lower_shares, size * sizeof(uint8_t));
    //memcpy(temp_res_upper_plain, res_upper_shares, size * sizeof(uint8_t));
    //memcpy(temp_s0_plain, s0_xor_shares, size * sizeof(uint8_t));
    //memcpy(temp_s1_plain, s1_xor_shares, size * sizeof(uint8_t));
    //memcpy(temp_s2_plain, s2_xor_shares, size * sizeof(uint8_t));
    
    // 3. 重构临时数组，得到明文
    //reconstruct(size, temp_inArr_plain, bin);
    //reconstruct_bool(size, temp_res_lower_plain);
    //reconstruct_bool(size, temp_res_upper_plain);
    //reconstruct_bool(size, temp_s0_plain);
    //reconstruct_bool(size, temp_s1_plain);
    //reconstruct_bool(size, temp_s2_plain);
    
    // 4. 只让一个计算方 (例如 SERVER) 打印，避免重复输出
    //if (party == SERVER) {
    //    std::cout << "\n--- [DEBUG] clip_with_dpf Plaintext Values (size=" << size << ") ---" << std::endl;
    //    for (int i = 0; i < size; ++i) {
            // 提取明文值
   //         GroupElement x_plain = temp_inArr_plain[i];
            
            // 计算 shift 前后的值
    //        GroupElement x_shifted_plain = x_plain + shift;
    //        GroupElement lower_shifted_plain = lower + shift;
    //        GroupElement upper_shifted_plain = upper + shift;


    //            std::cout << "Element[" << i << "]:\n"
     //                 << "  --- Inputs (Before Shift) ---\n"
      //                << "  x (uint64)    : " << x_plain << " (" << fixed_to_double(x_plain, scale) << ")\n"
       //               << "  lower (uint64): " << lower   << " (" << fixed_to_double(lower, scale)   << ")\n"
        //              << "  upper (uint64): " << upper   << " (" << fixed_to_double(upper, scale)   << ")\n"
         //             << "  --- Inputs (After Shift) ---\n"
          //            << "  x' (uint64)   : " << x_shifted_plain << "\n"
            //          << "  lower'(uint64): " << lower_shifted_plain << "\n"
             //         << "  upper'(uint64): " << upper_shifted_plain << "\n"
               //       << "  --- MPC Results vs. Ground Truth ---\n"
                 //     << "  evalDPFET (is [x'>=lower']): Got " << (int)temp_res_lower_plain[i]  << "\n"
                  //    << "  evalDPFET (is [x'>=upper']): Got " << (int)temp_res_upper_plain[i]  << "\n"
                    //  << "  s0 ([x'<lower'])          : Got " << (int)temp_s0_plain[i] << "\n"
                     // << "  s1 ([lower'<=x'<upper'])   : Got " << (int)temp_s1_plain[i]  << "\n"
                      //<< "  s2 ([x'>=upper'])          : Got " << (int)temp_s2_plain[i]  << "\n"
                      //<< "--------------------------------------------------" << std::endl;
        //}
    //}
    
    // 5. 清理临时数组
    //delete[] temp_inArr_plain;
    //delete[] temp_res_lower_plain;
    //delete[] temp_res_upper_plain;
    //delete[] temp_s0_plain;
    //delete[] temp_s1_plain;
    //delete[] temp_s2_plain;

    //peer->sync(); // 再次同步，确保调试通信结束后再继续主协议
    // =========================================================================
    // ==                ENHANCED DEBUGGING BLOCK END                         ==
    // =========================================================================

    uint8_t* all_bool_shares = new uint8_t[size * 3];
    memcpy(all_bool_shares, s0_xor_shares, size * sizeof(uint8_t));
    memcpy(all_bool_shares + size, s1_xor_shares, size * sizeof(uint8_t));
    memcpy(all_bool_shares + size * 2, s2_xor_shares, size * sizeof(uint8_t));
    
    GroupElement* all_add_shares = new GroupElement[size * 3];
    //B2A_Crypten(size * 3, all_bool_shares, all_add_shares);
auto stats = time_comm_this_block([&]() {
            B2A_Crypten(size * 3, all_bool_shares, all_add_shares);
        });

        // stats.first  = 时间 (微秒)
        // stats.second = 通信量 (字节)
        
        // 将整体作为一个原子操作推送到统计中
        // 结构体参数通常是: Name, keyread, compute, reconstruct, comm_bytes, key_size
        // 这里我们把整个函数的时间主要归为 compute 或 reconstruct 都可以，或者分开
        // 为了简单，我们全部算作 compute_time，通信量如实记录
        FSS::push_stats({
            "B2A_Crypten_Total", 
            0,              // key_read
            stats.first,    // compute_time (包含通信等待时间)
            0,              // reconstruct_time
            stats.second,   // comm_bytes
            0               // key_size
        });
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        // This loop targets the s1_add_shares part of the array
        all_add_shares[i + size] = all_add_shares[i + size] << scale;
    }

    GroupElement* s0_add_shares = all_add_shares;
    GroupElement* s1_add_shares = all_add_shares + size;
    GroupElement* s2_add_shares = all_add_shares + size * 2;
    
    // -- Step 6: Secure Arithmetic Computation --
    GroupElement* term1_add_shares = new GroupElement[size];
    ElemWiseMul(size, s1_add_shares, nullptr, inArr, nullptr, term1_add_shares, nullptr);

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        GroupElement term0_add_share = s0_add_shares[i] * lower;
        GroupElement term2_add_share = s2_add_shares[i] * upper;
        outArr[i] = term0_add_share + term1_add_shares[i] + term2_add_share;
    }

    // -- Cleanup --
    delete[] all_keys;
    delete[] all_alpha_shares;
    delete[] all_d_shares;
    delete[] res_lower_shares;
    delete[] res_upper_shares;
    delete[] s0_xor_shares;
    delete[] s1_xor_shares;
    delete[] s2_xor_shares;
    delete[] all_bool_shares;
    delete[] all_add_shares;
    delete[] term1_add_shares;
}


void SoftmaxODE(int32_t size, 
                GroupElement *inArr, GroupElement *inArr_mask, // MASK_PAIR 展开
                GroupElement *outArr, GroupElement *outArr_mask,
                int iter_num, bool clip)
{
    // === 0. Dealer 直接退出 ===
    const int scale = 16; 
    GroupElement upper = (12LL << scale);
    GroupElement lower = (-4LL << scale);

    if (party == DEALER) {
        // Dealer 的代码是线上计算的“蓝图”，它为每一个安全协议调用生成密钥。
        // 它不关心变量的真实值，只关心操作的类型和尺寸。
        // 因此我们使用占位符/虚拟变量。
        GroupElement *dummy1 = new GroupElement[size];
        GroupElement *dummy2 = new GroupElement[size];
        GroupElement *dummy3 = new GroupElement[size];
        GroupElement *dummy_double_size = new GroupElement[size * 2];

        // 对应在线代码的步骤 2: (可选) 安全裁剪
        if (clip) {
            //clip_with_relu(size,inArr,outArr,lower,upper);
            clip_with_dpf(size,inArr,outArr,lower,upper,scale);
        }

        // 对应在线代码的步骤 3: 初始化 x = x / iter_num
        int log2_iter_num = (int)log2(iter_num);
        ARS_CrypTen_Style(size,dummy1,dummy2,log2_iter_num);
        //SlothARS(size, dummy1, dummy1, log2_iter_num, "SoftmaxODE::");
        //ARS_CrypTen_Style(size, dummy1, dummy1, log2_iter_num);

        // 对应在线代码的步骤 5: ODE 迭代
        for (int k = 0; k < iter_num; ++k) {
            // 为步骤 1 的 ElemWiseMul(g, x) 生成密钥
            ElemWiseMul(size, dummy1, dummy1, dummy2, dummy2, dummy3, dummy3);

            // 为 ElemWiseMul(diff, g) 生成密钥
            ElemWiseMul(size, dummy1, dummy1, dummy2, dummy2, dummy3, dummy3);

        }

        // Dealer 不知道真实的输出，所以将输出掩码设置为0
        // (或者一个随机值，具体取决于框架设计)
        for(int i=0; i<size; ++i) outArr_mask[i] = 0;

        // 清理内存
        delete[] dummy1;
        delete[] dummy2;
        delete[] dummy3;
        delete[] dummy_double_size;
        
        return;
    }

    // === 1. 初始化和参数设置 (仅计算方执行) ===

    GroupElement* x = new GroupElement[size];
    memcpy(x, inArr, size * sizeof(GroupElement));

    // === 2. (可选) 高效安全裁剪 ===
    if (clip) {
        //clip_with_relu(size,inArr,x,lower,upper);
        clip_with_dpf(size,inArr,x,lower,upper,scale);
    }
    
    GroupElement* temp_x = new GroupElement[size]; 
    memcpy(temp_x,x, size* sizeof(GroupElement));
    reconstruct(size, temp_x, bitlength);
    
    //print_double_array("x ",party,size,temp_x,size);
    // === 3. 初始化 x = x / iter_num ===
    int log2_iter_num = (int)log2(iter_num);
    peer->sync();
    //ARS_CrypTen_Style(size, x, x, log2_iter_num);
    GroupElement *t = new GroupElement[size];
    ARS_CrypTen_Style(size, x, x, log2_iter_num); 

    //GroupElement* temp_sx = new GroupElement[size]; 
    //memcpy(temp_sx,x, size* sizeof(GroupElement));
    //reconstruct(size, temp_sx, bitlength);
    
    
    //print_double_array("x / iter_num",party,size,temp_sx,size);
    // === 4. 初始化 g ===
    GroupElement* g = new GroupElement[size](); // 初始化为0
    if (party == SERVER) {
        GroupElement initial_g_val = (1LL << scale) / size;
        for (int i = 0; i < size; ++i) g[i] = initial_g_val;
    }

    // === 5. ODE 迭代 ===
    GroupElement* gx_prod = new GroupElement[size];
    GroupElement* dot_prod_broadcast = new GroupElement[size];
    GroupElement* term3 = new GroupElement[size];
    
    //GroupElement* temp = new GroupElement[size]; 
    //memcpy(temp,g,size * sizeof(GroupElement));
    //reconstruct(size, temp, bitlength);
    //print_double_array("Original Plaintext 'g0'", party, size, temp, size);

    for (int k = 0; k < iter_num; ++k) {
        ElemWiseMul(size, g, nullptr, x, nullptr, gx_prod, nullptr);
        // 步骤 3: 计算点积。现在 dot_prod_share 的小数位数是 16
        GroupElement dot_prod_share = 0;
        for (int i = 0; i < size; ++i) {
            dot_prod_share += gx_prod[i];
        }

        // 步骤 4: 安全计算 (g·x)*g
        // 4.1: 将标量秘密分享 [g·x] 广播成一个向量
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            dot_prod_broadcast[i] = dot_prod_share;
        }
        //GroupElement* temp = new GroupElement[1]; 
        //GroupElement* temp_start = new GroupElement[size]; 
        //temp_x = new GroupElement[size]; 
        //GroupElement* temp_gx = new GroupElement[size]; 

        //memcpy(temp_start,g, size* sizeof(GroupElement));
        //memcpy(temp_x,x, size* sizeof(GroupElement));
        //memcpy(temp_gx,gx_prod, size* sizeof(GroupElement));

        //reconstruct(size, temp_start, bitlength);
        //reconstruct(size, temp_x, bitlength);
        //reconstruct(size, temp_gx, bitlength);



        //print_double_array("g_start",party,size,temp_start,size);
        //print_double_array("x_start",party,size,temp_x,size);
        //print_double_array("gx_prod",party,size,temp_gx,size);


        //memcpy(temp,dot_prod_share,  sizeof(GroupElement));
        //temp[0] = dot_prod_share;
        //reconstruct(1, temp, bitlength);
        //print_double_array("Original Plaintext 'dot_prod_share", party, 1, temp, 1);

        // 4.2: 【这里是关键】使用 ElemWiseMul 进行安全乘法。
        //      输入 g (scale=16) 和 dot_prod_broadcast (scale=16)。
        ElemWiseMul(size, g, nullptr, dot_prod_broadcast, nullptr, term3, nullptr);

        // 步骤 6: 更新g。现在所有项 (g, gx_prod, term3) 的小数位数都是 16，计算正确。
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            g[i] = g[i] + gx_prod[i] - term3[i];
            mod(g[i], bitlength);
        }
        //temp = new GroupElement[size]; 
        //memcpy(temp,g,size * sizeof(GroupElement));
        //reconstruct(size, temp, bitlength);
        //print_double_array("Original Plaintext 'g"+std::to_string(k)+"'", party, size, temp, size);
    }
    
    // === 6. 返回结果 ===
    memcpy(outArr, g, size * sizeof(GroupElement));

    // === 7. 清理内存 ===
    delete[] x;
    delete[] g;
    delete[] gx_prod;
}

// in .../FSS/api/api.cpp

// Include necessary headers, e.g., "api.h", "protocol/nonlinear.h"

// =========================================================================
// == HELPER FUNCTION: Secure Exponential Approximation
// =========================================================================
// ================= DEBUG HELPER START (新增调试函数) =================

// ================= DEBUG HELPER END =================
// ================= 使用 Relu2Round 协议计算 dReLU =================
// 这个函数专门封装 evalRelu2_drelu，用于替换 SlothDrelu
void FastRelu2RoundDrelu(int32_t size, int bin, GroupElement *inArr, GroupElement *outArr, std::string prefix)
{
    // Relu2Round 需要的有效位宽，通常等于 bitlength (64)
    // 如果你知道输入范围较小，也可以传更小的值，但默认传 bitlength 最稳妥
    int effectiveBw = bitlength; 

    if (party == DEALER) {
        pair<Relu2RoundKeyPack> *keys = new pair<Relu2RoundKeyPack>[size];
        
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            // 我们只需要计算 dReLU (符号位)
            // dReLU 的掩码: routDrelu (这是我们要的)
            // ReLU 的掩码: rout (这里不需要，给个随机数即可)
            GroupElement routDrelu = random_ge(1); 
            GroupElement routDummy = random_ge(bitlength); 

            // 注意：inArr[i] 在 Dealer 端是掩码
            keys[i] = keyGenRelu2Round(effectiveBw, bitlength, inArr[i], routDrelu, routDummy);
            
            // 保存 dReLU 的掩码到 outArr (这是加法分享的掩码)
            outArr[i] = routDrelu; 
        }

        for (int i = 0; i < size; ++i) {
            server->send_relu_2round_key(keys[i].first);
            client->send_relu_2round_key(keys[i].second);
            // 释放内存 (注意：keyGenRelu2Round 内部可能分配了 dcfKey 的内存)
            // 如果你的库有专门的 free 函数请使用，或者依赖析构
            freeRelu2RoundKeyPackPair(keys[i]); 
        }
        delete[] keys;
    } 
    else 
    {
        Relu2RoundKeyPack *keys = new Relu2RoundKeyPack[size];
        
        // 接收密钥
        for (int i = 0; i < size; ++i) {
            // recv 函数需要 effectiveBw 参数
            keys[i] = dealer->recv_relu_2round_key(effectiveBw, bitlength);
        }
        
        peer->sync();

        // 执行计算
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            // party - 2: Server传0, Client传1
            // 结果直接存入 outArr
            outArr[i] = evalRelu2_drelu(party - 2, inArr[i], keys[i]);
        }

        // 释放密钥
        for (int i = 0; i < size; ++i) {
            freeRelu2RoundKeyPack(keys[i]);
        }
        delete[] keys;
    }
}

void SecureExpApprox(int32_t size, 
                     MASK_PAIR(GroupElement *inArr),
                     MASK_PAIR(GroupElement *outArr),
                     int scale, 
                     int taylor_n)
{
    // --- 1. Dealer 逻辑 ---
    if (party == DEALER) {
        // 这里的初始化很重要，防止脏数据影响 KeyGen
        GroupElement* dummy1 = new GroupElement[size](); 
        GroupElement* dummy2 = new GroupElement[size]();
        
        // 1. Taylor Keys
        ARS_CrypTen_Style(size, dummy1, dummy2, taylor_n);
        for (int i = 0; i < taylor_n; ++i) {
            SecureSquare(size, dummy2, dummy2, dummy1, dummy1);
            ARS_CrypTen_Style(size, dummy1, dummy1, scale);
            //SlothARS(size, dummy1, dummy1, scale, "ExpApprox::Taylor");
        }

        // // 2. Clip Keys (SlothDrelu)
        // SlothDrelu(size, bitlength, dummy1, dummy2, "ExpApprox::Clip");
        // //FastRelu2RoundDrelu(size, bitlength, dummy1, dummy2, "ExpApprox::Clip");
        
        // // 3. Select Keys
        // Select(size, dummy2, dummy1, dummy1, "ExpApprox::Select");

        delete[] dummy1; 
        delete[] dummy2;
        return;
    }

    // --- 2. Computing Party Logic ---

    // [DEBUG] 输入
    //debug_reconstruct_and_print("SecureExpApprox: Input (Centered X)", size, inArr, scale);

    // === Part 1: Taylor Calculation ===
    GroupElement* taylor_results = new GroupElement[size];
    {
        GroupElement* term_shares = new GroupElement[size];
        ARS_CrypTen_Style(size, inArr, term_shares, taylor_n);
        
        GroupElement one_fixed = (1LL << scale);
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            if (party == SERVER) term_shares[i] += one_fixed;
        }

        GroupElement* current_res = term_shares;
        GroupElement* next_res = new GroupElement[size];
        for (int i = 0; i < taylor_n; ++i) {
            SecureSquare(size, current_res, nullptr, next_res, nullptr);
            //debug_reconstruct_and_print("SecureExpApprox: Taylor temp square result", size, next_res, scale*2); // 调试可注释
            ARS_CrypTen_Style(size, next_res, next_res, scale);
            //SlothARS(size, next_res, next_res, scale, "SecureExpApprox::Taylor");
            //debug_reconstruct_and_print("SecureExpApprox: Taylor temp ars result", size, next_res, scale); // 调试可注释
            std::swap(current_res, next_res);
        }
        memcpy(taylor_results, current_res, size * sizeof(GroupElement));
        
        if (current_res != term_shares) delete[] current_res;
        if (next_res != term_shares) delete[] next_res;
        else delete[] current_res;
    }

    // [DEBUG] Taylor 结果 (这里有些负数/下溢是正常的，关键看后面选不选它)
    //debug_reconstruct_and_print("SecureExpApprox: Taylor Raw Result", size, taylor_results, scale);
    memcpy(outArr,taylor_results, size * sizeof(GroupElement));
    // === Part 2: Clip Condition (SlothDrelu) ===
    // const double T_exp_double = -13.0; // 阈值
    // GroupElement T_exp_fixed = double_to_fixed(T_exp_double, scale);

    // GroupElement* diff_shares = new GroupElement[size];
    // GroupElement* is_ge_shares = new GroupElement[size]; // [x >= -13]

    // #pragma omp parallel for
    // for (int i = 0; i < size; ++i) {
    //     if (party == SERVER) diff_shares[i] = inArr[i] - T_exp_fixed;
    //     else diff_shares[i] = inArr[i];
    // }
    // debug_reconstruct_and_print("SecureExpApprox:  inArr[i] - T_exp_fixed", size, diff_shares, scale);
    // // 计算 [diff >= 0] 即 [x >= -13]
    // SlothDrelu(size, bitlength, diff_shares, is_ge_shares, "ExpApprox::Clip");
    // //FastRelu2RoundDrelu(size, bitlength, diff_shares, is_ge_shares, "ExpApprox::Clip");
    // // [DEBUG] 打印 "是否 >= -13" (1=保留, 0=置零)
    // for(int i=0; i<size; ++i){
    //     if(party==SERVER){
    //         std::cout << is_ge_shares[i] << " ";
    //     }
    // }
    // std::cout << std::endl;
    // //debug_reconstruct_and_print("SecureExpApprox: Is GreaterOrEqual -13? (1=Keep, 0=Zero)", size, is_ge_shares, 0);

    // // === Part 3: Selection ===
    // // 【核心修复】直接使用 is_ge_shares 作为选择位
    // // 如果 is_ge 为 1，结果 = 1 * taylor = taylor
    // // 如果 is_ge 为 0，结果 = 0 * taylor = 0
    // Select(size, is_ge_shares, taylor_results, outArr, "ExpApprox::Select");

    // debug_reconstruct_and_print("SecureExpApprox: Final Output", size, outArr, scale);

    // delete[] taylor_results;
    // delete[] diff_shares;
    // delete[] is_ge_shares;
}

// ================= SoftmaxBumbleBee (带调试功能的完整版) =================
void SoftmaxBumbleBee(int32_t size, 
                      MASK_PAIR(GroupElement *inArr),
                      MASK_PAIR(GroupElement *outArr),
                      int scale, 
                      int taylor_n)
{
    // --- 1. Dealer 逻辑 (完整保留，无省略) ---
    if (party == DEALER) {
        GroupElement* dummy_vec = new GroupElement[size];
        GroupElement* dummy_single = new GroupElement[1];
        
        // 1. SecureMaxVector
        SlothMaxpool(1, size, bitlength, dummy_vec, dummy_single, "Softmax::");

        // 2. SecureExpApprox
        SecureExpApprox(size, dummy_vec, dummy_vec, dummy_vec, dummy_vec, scale, taylor_n);
        
        // 3. Final ElemWiseMul
        
        //ElemWiseMul(size, dummy_vec, dummy_vec, dummy_vec, dummy_vec, dummy_vec, dummy_vec);
        ARS_CrypTen_Style(size, dummy_vec, dummy_vec, scale);
        delete[] dummy_vec;
        delete[] dummy_single;
        return;
    }

    // --- 2. Computing Party Logic (带调试埋点) ---

    // [DEBUG] Step 0: 原始输入
    //debug_reconstruct_and_print("Softmax: Step 0 - Input", size, inArr, scale);

    // === 1. Securely find the maximum value ===
    GroupElement* max_val_share = new GroupElement[1];
    // We treat the 1D input vector as a 1xSIZE matrix for SlothMaxPool
    SlothMaxpool(1, size, bitlength, inArr, max_val_share, "Softmax::");
    
    // [DEBUG] Step 1: 打印最大值
    ////debug_reconstruct_and_print("Softmax: Step 1 - Max Value", 1, max_val_share, scale);

    // === 2. Locally compute centered input x' = x - max(x) ===
    GroupElement* x_prime_shares = new GroupElement[size];
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        x_prime_shares[i] = inArr[i] - max_val_share[0];
    }
    
    // [DEBUG] Step 2: 减去最大值后的输入 (应全 <= 0)
    //debug_reconstruct_and_print("Softmax: Step 2 - Shifted Input (x - max)", size, x_prime_shares, scale);

    // === 3. Securely compute exp(x') for each element ===
    GroupElement* exp_shares = new GroupElement[size];
    // 注意：这里调用的是我们上面修改过的带 Debug 的 SecureExpApprox
    SecureExpApprox(size, x_prime_shares, nullptr, exp_shares, nullptr, scale, taylor_n);
    
    // [DEBUG] Step 3: Exp 计算结果
    //debug_reconstruct_and_print("Softmax: Step 3 - Exp(x')", size, exp_shares, scale);

    // === 4. Locally sum up all the exp shares ===
    GroupElement sum_exp_share = 0;
    for (int i = 0; i < size; ++i) {
        sum_exp_share += exp_shares[i];
    }
    
    // === 步骤 5: 公开 SUM 并计算倒数 (BumbleBee 的方式) ===
    GroupElement sum_exp_plain = sum_exp_share;
    // 调用 reconstruct 来公开 sum 的值
    reconstruct(1, &sum_exp_plain, bitlength); // 现在 sum_exp_plain 是明文
    
    // [DEBUG] Step 4: 打印 Exp 的和 (Scale 必须正确)
    if (party == SERVER) {
        double sum_val = fixed_to_double(sum_exp_plain, scale);
        std::cout << "\n[DEBUG] Softmax: Step 4 - Sum Exp = " << sum_val 
                  << " (Raw: " << (int64_t)sum_exp_plain << ")" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    // 在明文中计算倒数
    double sum_double = fixed_to_double(sum_exp_plain, scale);
    if (abs(sum_double) < 1e-9) sum_double = 1e-9; // 防止除以零
    double inv_sum_double = 1.0 / sum_double;
    GroupElement inv_sum_fixed = double_to_fixed(inv_sum_double, scale);

    // === 步骤 6: 公开数与秘密份额的乘法 ===
    // 这是一个纯本地操作
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        outArr[i] = exp_shares[i] * inv_sum_fixed;
    }
    std::cout << "-----------"<<fixed_to_double(inv_sum_fixed, scale) <<"---------------------------" << std::endl;
    //debug_reconstruct_and_print("Softmax: Step 4.5 - * inv_sum", size, outArr, scale*2);
    // 截断 (因为进行了乘法，定点数倍数增加，需要右移)
    ARS_CrypTen_Style(size, outArr, outArr, scale);
    //SlothFaithfulARS(size,bitlength, outArr, outArr, scale, "Softmax::FinalARS");
    // [DEBUG] Step 5: 最终结果
    //debug_reconstruct_and_print("Softmax: Step 5 - Final Output", size, outArr, scale);

    // === Cleanup ===
    delete[] max_val_share;
    delete[] x_prime_shares;
    delete[] exp_shares;
}