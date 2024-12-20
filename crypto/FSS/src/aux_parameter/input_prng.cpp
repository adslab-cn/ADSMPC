#include <aux_parameter/input_prng.h>
#include <aux_parameter/comms.h>
#include <aux_parameter/config.h>
#include <chrono>
#include <thread>
#include <aux_parameter/stats.h>

using namespace FSSConfig;

osuCrypto::AES inputPrng[2];
int counter[2] = {0, 0};

void input_prng_init()
{
    if (party == DEALER) {
        osuCrypto::AES aesSeed(prngs[0].get<osuCrypto::block>());
        auto seed0 = aesSeed.ecbEncBlock(osuCrypto::ZeroBlock);
        server->send_block(seed0);
        auto seed1 = aesSeed.ecbEncBlock(osuCrypto::OneBlock);
        client->send_block(seed1);
        inputPrng[0] = osuCrypto::AES(seed0);
        inputPrng[1] = osuCrypto::AES(seed1);
    }
    else {
        auto seed = dealer->recv_block();
        inputPrng[party - SERVER] = osuCrypto::AES(seed);
    }
}

osuCrypto::block get_input_mask_pair(int idx, int owner)
{
    osuCrypto::block val = inputPrng[owner - SERVER].ecbEncBlock(osuCrypto::toBlock(0, counter[owner - SERVER] + idx));
    return val;
}

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

void input_layer_dealer_thread(int thread_idx, int size, int owner, GroupElement *x_mask)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i++)
    {
        auto pair = get_input_mask_pair(i, owner);
        x_mask[2*i] = _mm_extract_epi64(pair, 0);
        x_mask[2*i+1] = _mm_extract_epi64(pair, 1);
    }
}

void input_layer_owner_thread(int thread_idx, int size, int owner, GroupElement *x, GroupElement *x_mask)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i++)
    {
        auto pair = get_input_mask_pair(i, owner);
        x_mask[2*i] = _mm_extract_epi64(pair, 0);
        x_mask[2*i+1] = _mm_extract_epi64(pair, 1);
        x[2*i] = x[2*i] + x_mask[2*i];
        x[2*i+1] = x[2*i+1] + x_mask[2*i+1];
    }
}

void input_layer(GroupElement *x, GroupElement *x_mask, int size, int owner)
{
    if (size == 0) return;
    if (party == DEALER) {
        TIME_THIS_BLOCK_FOR_INPUT_IF(
            std::thread thread_pool[num_threads];
            for(int i = 0; i < num_threads; ++i)
            {
                thread_pool[i] = std::thread(input_layer_dealer_thread, i, size/2, owner, x_mask);
            }
            for(int i = 0; i < num_threads; ++i)
            {
                thread_pool[i].join();
            }
            if (size % 2 == 1) {
                auto pair = get_input_mask_pair(size/2, owner);
                x_mask[size-1] = _mm_extract_epi64(pair, 0);
            }
        , true, accumulatedInputTimeOffline)
    }
    else if (party == owner) {
        // for(int i = 0; i < size; ++i) {
        //     std::cin >> x[i];
        // }
        // generate and add masks
        TIME_THIS_BLOCK_FOR_INPUT_IF(
            std::thread thread_pool[num_threads];
            for(int i = 0; i < num_threads; ++i)
            {
                thread_pool[i] = std::thread(input_layer_owner_thread, i, size/2, owner, x, x_mask);
            }
            for(int i = 0; i < num_threads; ++i)
            {
                thread_pool[i].join();
            }
            if (size % 2 == 1) {
                auto pair = get_input_mask_pair(size/2, owner);
                x_mask[size-1] = _mm_extract_epi64(pair, 0);
                x[size-1] = x[size-1] + x_mask[size-1];
            }
        , true, accumulatedInputTimeOffline)
        
        TIME_THIS_BLOCK_FOR_INPUT_IF(
            peer->send_batched_input(x, size, bitlength);
        , true, (owner == SERVER ? accumulatedInputTimeOffline : accumulatedInputTimeOnline))
    }
    else {
        uint64_t *tmp = new uint64_t[size];
        TIME_THIS_BLOCK_FOR_INPUT_IF(
        peer->recv_batched_input(tmp, size, bitlength);
        , true, (owner == SERVER ? accumulatedInputTimeOffline : accumulatedInputTimeOnline))
        // todo: parallelize this maybe?
        for(int i = 0; i < size; ++i) {
            x[i] = tmp[i];
        }
        delete[] tmp;
    }
    counter[owner - SERVER] += size;
}