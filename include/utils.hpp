//
// Created by Ali Hamza Azam on 25/04/2025.
//

#ifndef PARALLEL_SSSP_UTILS_H
#define PARALLEL_SSSP_UTILS_H

#include "graph.h"
#include <vector>
#include <string>

// Function declarations
Graph load_graph(const std::string& filename);
std::vector<EdgeChange> load_edge_changes(const std::string& filename);

#endif //PARALLEL_SSSP_UTILS_H
