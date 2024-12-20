#pragma once
#include <array>
// Array initializers
template <typename T> T *make_array(std::size_t s1) { return new T[s1]; }
template <typename T> T *make_array(std::size_t s1, std::size_t s2) {
    return new T[s1 * s2];
}
template <typename T> T *make_array(std::size_t s1, std::size_t s2, std::size_t s3) {
    return new T[s1 * s2 * s3];
}
template <typename T>
T *make_array(std::size_t s1, std::size_t s2, std::size_t s3, std::size_t s4) {
    return new T[s1 * s2 * s3 * s4];
}
template <typename T>
T *make_array(std::size_t s1, std::size_t s2, std::size_t s3, std::size_t s4, std::size_t s5) {
    return new T[s1 * s2 * s3 * s4 * s5];
}

// Indexing Helpers, we use 1D pointers for any dimension array, hence these macros are necessary
// Copied from SCI
#define Arr2DIdx(arr, s0, s1, i, j) (*((arr) + (i) * (s1) + (j)))
// #define Arr3D(arr, s0, s1, s2, i, j, k)                                 \
  (*((arr) + (i) * (s1) * (s2) + (j) * (s2) + (k)))
#define Arr4DIdx(arr, s0, s1, s2, s3, i, j, k, l)                          \
  (*((arr) + (i) * (s1) * (s2) * (s3) + (j) * (s2) * (s3) + (k) * (s3) + (l)))
#define Arr5DIdx(arr, s0, s1, s2, s3, s4, i, j, k, l, m)                   \
  (*((arr) + (i) * (s1) * (s2) * (s3) * (s4) + (j) * (s2) * (s3) * (s4) +      \
     (k) * (s3) * (s4) + (l) * (s4) + (m)))

