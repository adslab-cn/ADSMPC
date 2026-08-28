#include <FSS/graphiti.h>
#include <FSS/config.h>
#include <cassert>
#include <iostream>

int main() {
    FSSConfig::bitlength = 64;
    GraphitiGraph graph(3, {0, 0, 1, 2}, {0, 1, 2, 1});
    GroupElement input[] = {1, 10, 2, 20, 3, 30};
    GroupElement output[6] = {};
    GraphitiGCNAggregate(graph, 2, input, output, "unit-test");
    const GroupElement expected[] = {1, 10, 4, 40, 2, 20};
    for (int i = 0; i < 6; ++i) assert(output[i] == expected[i]);
    std::cout << "Graphiti aggregation: SUCCESS\n";
    return 0;
}
