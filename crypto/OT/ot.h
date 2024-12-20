

#ifndef OT_H__
#define OT_H__
#include "utils/emp-tool.h"
namespace sci {
template <typename T> class OT {
public:
  void send(const block128 *data0, const block128 *data1, int length) {
    static_cast<T *>(this)->send_impl(data0, data1, length);
  }
  void recv(block128 *data, const bool *b, int length) {
    static_cast<T *>(this)->recv_impl(data, b, length);
  }
  void send(const block256 *data0, const block256 *data1, int length) {
    static_cast<T *>(this)->send_impl(data0, data1, length);
  }
  void recv(block256 *data, const bool *b, int length) {
    static_cast<T *>(this)->recv_impl(data, b, length);
  }
  void send(block128 **data, int length, int N) {
    static_cast<T *>(this)->send_impl(data, length, N);
  }
  void recv(block128 *data, const uint8_t *b, int length, int N) {
    static_cast<T *>(this)->recv_impl(data, b, length, N);
  }
  void send(uint8_t **data, int length, int N, int l) {
    static_cast<T *>(this)->send_impl(data, length, N, l);
  }
  void recv(uint8_t *data, const uint8_t *b, int length, int N, int l) {
    static_cast<T *>(this)->recv_impl(data, b, length, N, l);
  }
  void recv(uint8_t *data, uint8_t *b, int length, int N, int l) {
    static_cast<T *>(this)->recv_impl(data, b, length, N, l);
  }
  void send(uint8_t **data, int length, int l) {
    static_cast<T *>(this)->send_impl(data, length, l);
  }
  void recv(uint8_t *data, const uint8_t *b, int length, int l) {
    static_cast<T *>(this)->recv_impl(data, b, length, l);
  }
  void recv(uint8_t *data, uint8_t *b, int length, int l) {
    static_cast<T *>(this)->recv_impl(data, b, length, l);
  }
  void send(uint64_t **data, int length, int l) {
    static_cast<T *>(this)->send_impl(data, length, l);
  }
  void recv(uint64_t *data, const uint8_t *b, int length, int l) {
    static_cast<T *>(this)->recv_impl(data, b, length, l);
  }
  void recv(uint64_t *data, uint8_t *b, int length, int l) {
    static_cast<T *>(this)->recv_impl(data, b, length, l);
  }

  void send_cot(uint64_t *data0, uint64_t *corr, int length, int l) {
    static_cast<T *>(this)->send_cot(data0, corr, length, l);
  }
  void recv_cot(uint64_t *data, bool *b, int length, int l) {
    static_cast<T *>(this)->recv_cot(data, b, length, l);
  }

  template <typename intType>
  void send_cot_matmul(intType *rdata, const intType *corr,
                       const uint32_t *chunkSizes, const uint32_t *numChunks,
                       const int numOTs, int senderMatmulDims) {
    static_cast<T *>(this)->send_cot_matmul(rdata, corr, chunkSizes, numChunks,
                                            numOTs, senderMatmulDims);
  }

  template <typename intType>
  void recv_cot_matmul(intType *data, const uint8_t *choices,
                       const uint32_t *chunkSizes, const uint32_t *numChunks,
                       const int numOTs, int senderMatmulDims) {
    static_cast<T *>(this)->recv_cot_matmul(
        data, choices, chunkSizes, numChunks, numOTs, senderMatmulDims);
  }

  void send(uint8_t **data, int length, int N, int l, bool type) {
    static_cast<T *>(this)->send_impl(data, length, N, l, type);
  }
  void recv(uint8_t *data, const uint8_t *b, int length, int N, int l,
            bool type) {
    static_cast<T *>(this)->recv_impl(data, b, length, N, l, type);
  }
  void recv(uint8_t *data, uint8_t *b, int length, int N, int l, bool type) {
    static_cast<T *>(this)->recv_impl(data, b, length, N, l, type);
  }
  void send(uint8_t **data, int length, int l, bool type) {
    static_cast<T *>(this)->send_impl(data, length, l, type);
  }
  void recv(uint8_t *data, const uint8_t *b, int length, int l, bool type) {
    static_cast<T *>(this)->recv_impl(data, b, length, l, type);
  }
  void recv(uint8_t *data, uint8_t *b, int length, int l, bool type) {
    static_cast<T *>(this)->recv_impl(data, b, length, l, type);
  }
};
} // namespace sci
#endif // OT_H__
