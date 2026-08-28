#include <FSS/graphiti.h>
#include <FSS/config.h>
#include <FSS/assert.h>
#include <FSS/comms.h>
#include <algorithm>
#include <chrono>
#include <iostream>

void GraphitiGraph::validate() const {
    always_assert(num_vertices >= 0);
    always_assert(src.size() == dst.size());
    for (size_t e = 0; e < src.size(); ++e) {
        always_assert(src[e] >= 0 && src[e] < num_vertices);
        always_assert(dst[e] >= 0 && dst[e] < num_vertices);
    }
}

void GraphitiGCNAggregate(const GraphitiGraph &graph, int dim,
                          const GroupElement *vertex_payload,
                          GroupElement *vertex_aggregate,
                          const std::string &label) {
    graph.validate();
    always_assert(dim > 0);
    const auto begin = std::chrono::high_resolution_clock::now();
    std::fill(vertex_aggregate,
              vertex_aggregate + static_cast<size_t>(graph.num_vertices) * dim,
              GroupElement(0));
    // Propagate, identity ApplyE, and additive Gather.
    for (size_t e = 0; e < graph.src.size(); ++e) {
        const GroupElement *message = vertex_payload + graph.src[e] * dim;
        GroupElement *destination = vertex_aggregate + graph.dst[e] * dim;
        for (int j = 0; j < dim; ++j) {
            destination[j] += message[j];
            mod(destination[j], FSSConfig::bitlength);
        }
    }
    if (!label.empty()) {
        const auto end = std::chrono::high_resolution_clock::now();
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        std::cerr << "   [Graphiti] " << label << ": V=" << graph.num_vertices
                  << ", E=" << graph.num_edges() << ", time=" << us / 1000.0
                  << " ms" << std::endl;
    }
}

GraphitiGraph GraphitiPrepare2P1(const GraphitiGraph &dealer_graph) {
    using namespace FSSConfig;
    if (party == DEALER) {
        dealer_graph.validate();
        const int edges = dealer_graph.num_edges();
        server->send_ge(GroupElement(dealer_graph.num_vertices), 64);
        server->send_ge(GroupElement(edges), 64);
        client->send_ge(GroupElement(dealer_graph.num_vertices), 64);
        client->send_ge(GroupElement(edges), 64);
        std::vector<GroupElement> packed(static_cast<size_t>(edges) * 2);
        for (int e = 0; e < edges; ++e) {
            packed[2 * e] = GroupElement(dealer_graph.src[e]);
            packed[2 * e + 1] = GroupElement(dealer_graph.dst[e]);
        }
        if (!packed.empty()) {
            server->send_ge_array(packed.data(), packed.size());
            client->send_ge_array(packed.data(), packed.size());
        }
        return dealer_graph;
    }
    const int vertices = static_cast<int>(dealer->recv_ge(64));
    const int edges = static_cast<int>(dealer->recv_ge(64));
    std::vector<GroupElement> packed(static_cast<size_t>(edges) * 2);
    if (!packed.empty()) dealer->recv_ge_array(packed.data(), packed.size());
    std::vector<int> src(edges), dst(edges);
    for (int e = 0; e < edges; ++e) {
        src[e] = static_cast<int>(packed[2 * e]);
        dst[e] = static_cast<int>(packed[2 * e + 1]);
    }
    GraphitiGraph result(vertices, std::move(src), std::move(dst));
    result.validate();
    return result;
}
