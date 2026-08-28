#include <FSS/cryptgnn.h>

#include <FSS/api.h>
#include <FSS/assert.h>
#include <FSS/comms.h>
#include <FSS/config.h>
#include <FSS/prng.h>
#include <FSS/stats.h>
#include <cryptoTools/Crypto/AES.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

using namespace FSSConfig;
using osuCrypto::AES;
using osuCrypto::block;
using osuCrypto::toBlock;

namespace {

inline int imod(int64_t x, int n) {
    x %= n;
    if (x < 0) x += n;
    return static_cast<int>(x);
}

inline GroupElement aes_word(const AES &aes, uint64_t domain, uint64_t index) {
    const block value = aes.ecbEncBlock(toBlock(domain, index));
    return static_cast<uint64_t>(_mm_extract_epi64(value, 0));
}

inline GroupElement noise_at(const AES &aes, int batch, int row, int col,
                             int vertices, int dim) {
    const uint64_t index = (static_cast<uint64_t>(batch) * vertices + row) * dim + col;
    return aes_word(aes, 0x43525950544d504cULL, index);
}

void send_plan(Peer *channel, const CryptMPLPlan &p,
               const std::vector<int> &ss, const std::vector<int> &ds) {
    channel->send_ge(GroupElement(p.vertices), 64);
    channel->send_ge(GroupElement(p.edges), 64);
    channel->send_ge(GroupElement(p.batches), 64);
    for (int b = 0; b <= p.batches; ++b)
        channel->send_ge(GroupElement(p.batch_begin[b]), 64);
    for (int b = 0; b < p.batches; ++b) {
        channel->send_ge(GroupElement(ss[b]), 64);
        channel->send_ge(GroupElement(ds[b]), 64);
    }
    for (int e = 0; e < p.edges; ++e) {
        channel->send_ge(GroupElement(p.relative_src[e]), 64);
        channel->send_ge(GroupElement(p.relative_dst[e]), 64);
    }
}

void exchange(std::vector<GroupElement> &send_buf,
              std::vector<GroupElement> &recv_buf) {
    always_assert(send_buf.size() == recv_buf.size());
    const int count = static_cast<int>(send_buf.size());
#pragma omp parallel sections
    {
#pragma omp section
        { peer->send_batched_input(send_buf.data(), count, bitlength); }
#pragma omp section
        { peer->recv_batched_input(recv_buf.data(), count, bitlength); }
    }
    ++numRounds;
}

} // namespace

CryptMPLPlan CryptMPLPrepare2P1(int vertices,
                                const std::vector<int> &src,
                                const std::vector<int> &dst,
                                int requested_batches) {
    always_assert(vertices > 0);
    always_assert(src.size() == dst.size());
    CryptMPLPlan p;
    if (party == DEALER) {
        p.vertices = vertices;
        p.edges = static_cast<int>(src.size());
        p.batches = std::max(1, std::min(requested_batches, std::max(1, p.edges)));
        p.batch_begin.resize(p.batches + 1);
        p.first_src.resize(p.batches);
        p.first_dst.resize(p.batches);
        p.relative_src.resize(p.edges);
        p.relative_dst.resize(p.edges);
        for (int b = 0; b <= p.batches; ++b)
            p.batch_begin[b] = static_cast<int>((static_cast<int64_t>(p.edges) * b) / p.batches);
        for (int b = 0; b < p.batches; ++b) {
            const int begin = p.batch_begin[b];
            p.first_src[b] = src[begin];
            p.first_dst[b] = dst[begin];
            for (int e = begin; e < p.batch_begin[b + 1]; ++e) {
                always_assert(src[e] >= 0 && src[e] < vertices);
                always_assert(dst[e] >= 0 && dst[e] < vertices);
                p.relative_src[e] = imod(src[e] - p.first_src[b], vertices);
                p.relative_dst[e] = imod(dst[e] - p.first_dst[b], vertices);
            }
        }
        std::vector<int> ss0(p.batches), ss1(p.batches), ds0(p.batches), ds1(p.batches);
        for (int b = 0; b < p.batches; ++b) {
            ss0[b] = static_cast<int>(random_ge(64) % vertices);
            ds0[b] = static_cast<int>(random_ge(64) % vertices);
            ss1[b] = imod(p.first_src[b] - ss0[b], vertices);
            ds1[b] = imod(p.first_dst[b] - ds0[b], vertices);
        }
        send_plan(server, p, ss0, ds0);
        send_plan(client, p, ss1, ds1);
        return p;
    }

    p.vertices = static_cast<int>(dealer->recv_ge(64));
    p.edges = static_cast<int>(dealer->recv_ge(64));
    p.batches = static_cast<int>(dealer->recv_ge(64));
    p.batch_begin.resize(p.batches + 1);
    p.first_src_share.resize(p.batches);
    p.first_dst_share.resize(p.batches);
    p.relative_src.resize(p.edges);
    p.relative_dst.resize(p.edges);
    for (int b = 0; b <= p.batches; ++b)
        p.batch_begin[b] = static_cast<int>(dealer->recv_ge(64));
    for (int b = 0; b < p.batches; ++b) {
        p.first_src_share[b] = static_cast<int>(dealer->recv_ge(64));
        p.first_dst_share[b] = static_cast<int>(dealer->recv_ge(64));
    }
    for (int e = 0; e < p.edges; ++e) {
        p.relative_src[e] = static_cast<int>(dealer->recv_ge(64));
        p.relative_dst[e] = static_cast<int>(dealer->recv_ge(64));
    }
    return p;
}

void CryptMPL2P1(const CryptMPLPlan &p, int dim,
                 GroupElement *input, GroupElement *input_mask,
                 GroupElement *output, GroupElement *output_mask,
                 const std::string &prefix) {
    always_assert(p.vertices > 0 && p.batches > 0 && dim > 0);
    const int n = p.vertices;
    const int elements = n * dim;
    const size_t batch_elements = static_cast<size_t>(p.batches) * elements;

    if (party == DEALER) {
        std::vector<GroupElement> in0(elements), in1(elements);
        for (int i = 0; i < elements; ++i) {
            in0[i] = random_ge(bitlength);
            in1[i] = input_mask[i] - in0[i];
            output_mask[i] = random_ge(bitlength);
        }
        server->send_ge_array(in0.data(), elements);
        client->send_ge_array(in1.data(), elements);

        const block read0 = prngs[0].get<block>();
        const block read1 = prngs[0].get<block>();
        const block write0 = prngs[0].get<block>();
        const block write1 = prngs[0].get<block>();
        server->send_block(read0); server->send_block(write0);
        client->send_block(read1); client->send_block(write1);

        const AES read_aes0(read0), read_aes1(read1);
        const AES write_aes0(write0), write_aes1(write1);
        std::vector<GroupElement> correction(elements, GroupElement(0));
        for (int b = 0; b < p.batches; ++b) {
            for (int e = p.batch_begin[b]; e < p.batch_begin[b + 1]; ++e) {
                const int src = imod(p.first_src[b] + p.relative_src[e], n);
                const int dst = imod(p.first_dst[b] + p.relative_dst[e], n);
                for (int k = 0; k < dim; ++k)
                    correction[dst * dim + k] +=
                        noise_at(read_aes0, b, src, k, n, dim) +
                        noise_at(read_aes1, b, src, k, n, dim);
            }
            for (int row = 0; row < n; ++row) {
                const int noise_row = imod(row - p.first_dst[b], n);
                for (int k = 0; k < dim; ++k)
                    correction[row * dim + k] +=
                        noise_at(write_aes0, b, noise_row, k, n, dim) +
                        noise_at(write_aes1, b, noise_row, k, n, dim);
            }
        }
        std::vector<GroupElement> q0(elements), q1(elements);
        for (int i = 0; i < elements; ++i) {
            q0[i] = random_ge(bitlength);
            q1[i] = output_mask[i] - correction[i] - q0[i];
        }
        server->send_ge_array(q0.data(), elements);
        client->send_ge_array(q1.data(), elements);
        return;
    }

    const auto key_begin = std::chrono::high_resolution_clock::now();
    const uint64_t key_bytes_begin = dealer->bytesReceived();
    std::vector<GroupElement> input_mask_share(elements), q(elements);
    dealer->recv_ge_array(input_mask_share.data(), elements);
    const block read_seed = dealer->recv_block();
    const block write_seed = dealer->recv_block();
    dealer->recv_ge_array(q.data(), elements);
    const uint64_t key_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - key_begin).count();

    const auto online_begin = std::chrono::high_resolution_clock::now();
    const uint64_t comm_begin = peer->bytesSent() + peer->bytesReceived();
    const AES read_aes(read_seed), write_aes(write_seed);
    std::vector<GroupElement> additive_input(elements);
    for (int i = 0; i < elements; ++i)
        additive_input[i] = (party == SERVER) ? input[i] - input_mask_share[i]
                                              : GroupElement(0) - input_mask_share[i];

    std::vector<GroupElement> send_buf(batch_elements), recv_buf(batch_elements);
    for (int b = 0; b < p.batches; ++b) {
        GroupElement *segment = send_buf.data() + static_cast<size_t>(b) * elements;
        const int shift = imod(p.first_src_share[b], n);
        for (int row = 0; row < n; ++row) {
            const int rotated_row = imod(row - shift, n); // left rotation
            for (int k = 0; k < dim; ++k)
                segment[rotated_row * dim + k] = additive_input[row * dim + k] +
                    noise_at(read_aes, b, row, k, n, dim);
        }
    }
    exchange(send_buf, recv_buf); // all batched secure reads: one round

    std::vector<GroupElement> messages(static_cast<size_t>(p.edges) * dim);
    for (int b = 0; b < p.batches; ++b) {
        const GroupElement *segment = recv_buf.data() + static_cast<size_t>(b) * elements;
        const int shift = imod(p.first_src_share[b], n);
        for (int e = p.batch_begin[b]; e < p.batch_begin[b + 1]; ++e) {
            const int row = imod(p.relative_src[e] + shift, n);
            std::copy_n(segment + row * dim, dim, messages.data() + static_cast<size_t>(e) * dim);
        }
    }

    std::fill(recv_buf.begin(), recv_buf.end(), GroupElement(0));
    for (int b = 0; b < p.batches; ++b) {
        GroupElement *plain_write = recv_buf.data() + static_cast<size_t>(b) * elements;
        for (int row = 0; row < n; ++row)
            for (int k = 0; k < dim; ++k)
                plain_write[row * dim + k] = noise_at(write_aes, b, row, k, n, dim);
        for (int e = p.batch_begin[b]; e < p.batch_begin[b + 1]; ++e) {
            const int row = p.relative_dst[e];
            for (int k = 0; k < dim; ++k)
                plain_write[row * dim + k] += messages[static_cast<size_t>(e) * dim + k];
        }
        GroupElement *rotated = send_buf.data() + static_cast<size_t>(b) * elements;
        const int shift = imod(p.first_dst_share[b], n);
        for (int row = 0; row < n; ++row)
            std::copy_n(plain_write + row * dim, dim,
                        rotated + imod(row + shift, n) * dim); // right rotation
    }
    exchange(send_buf, recv_buf); // all batched secure writes: one round

    std::fill(output, output + elements, GroupElement(0));
    for (int b = 0; b < p.batches; ++b) {
        const GroupElement *segment = recv_buf.data() + static_cast<size_t>(b) * elements;
        const int shift = imod(p.first_dst_share[b], n);
        for (int row = 0; row < n; ++row) {
            const int source_row = imod(row - shift, n);
            for (int k = 0; k < dim; ++k)
                output[row * dim + k] += segment[source_row * dim + k];
        }
    }
    for (int i = 0; i < elements; ++i) output[i] += q[i];
    reconstruct(elements, output, bitlength); // additive share -> masked value

    const uint64_t online_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - online_begin).count();
    const uint64_t comm = peer->bytesSent() + peer->bytesReceived() - comm_begin;
    FSS::push_stats({prefix + "BatchedReadWrite", key_us, online_us, 0, comm,
                     dealer->bytesReceived() - key_bytes_begin});
}
