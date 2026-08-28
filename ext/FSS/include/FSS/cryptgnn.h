#pragma once

#include <FSS/group_element.h>
#include <cstdint>
#include <string>
#include <vector>

// Two-computing-party + dealer adaptation of CryptGNN's batched CryptMPL.
// Only the first source/destination index in every batch is secret-shared;
// the remaining indices are represented by the public relative offsets used
// by Algorithm 6 of CryptGNN.
struct CryptMPLPlan {
    int vertices = 0;
    int edges = 0;
    int batches = 0;
    std::vector<int> batch_begin;
    std::vector<int> relative_src;
    std::vector<int> relative_dst;
    std::vector<int> first_src;       // populated only by the dealer
    std::vector<int> first_dst;       // populated only by the dealer
    std::vector<int> first_src_share; // populated by an online party
    std::vector<int> first_dst_share; // populated by an online party
};

CryptMPLPlan CryptMPLPrepare2P1(int vertices,
                                const std::vector<int> &src,
                                const std::vector<int> &dst,
                                int requested_batches = 20);

// Input/output follow the framework's masked-value convention: the dealer
// owns *_mask and both online parties own the common masked value. Internally
// the protocol converts to ordinary additive shares, executes batched secure
// read/write, and converts the result back to a masked value.
void CryptMPL2P1(const CryptMPLPlan &plan, int feature_dim,
                 GroupElement *input, GroupElement *input_mask,
                 GroupElement *output, GroupElement *output_mask,
                 const std::string &prefix = "CryptMPL::");
