#pragma once
#include <FSS/group_element.h>
#include <string>
#include <utility>
#include <vector>

struct GraphitiGraph {
    int num_vertices = 0;
    std::vector<int> src;
    std::vector<int> dst;
    GraphitiGraph() = default;
    GraphitiGraph(int n, std::vector<int> s, std::vector<int> d)
        : num_vertices(n), src(std::move(s)), dst(std::move(d)) {}
    int num_edges() const { return static_cast<int>(src.size()); }
    void validate() const;
};

void GraphitiGCNAggregate(const GraphitiGraph &graph, int dim,
                          const GroupElement *vertex_payload,
                          GroupElement *vertex_aggregate,
                          const std::string &label = "");

// Dealer-preprocessed routing plan. Call once for the stable base graph and
// retain the returned plan; call again only for a changed delta graph.
GraphitiGraph GraphitiPrepare2P1(const GraphitiGraph &dealer_graph);
