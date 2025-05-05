// utils.hpp - Utility functions for graph loading, updates parsing, and SSSP
// --------------------------------------------------------------------------
// Provides routines to load graphs, parse edge-change sequences, and run Dijkstra.

#ifndef PARALLEL_SSSP_UTILS_H
#define PARALLEL_SSSP_UTILS_H

#include "graph.hpp"
#include <vector>
#include <string>

// load_graph: load a Graph from Matrix Market (.mtx) or simple edge list (.edges)
Graph load_graph(const std::string& filename);

// load_edge_changes: parse edge updates file into a list of EdgeChange
std::vector<EdgeChange> load_edge_changes(const std::string& filename);

// dijkstra: compute single-source shortest paths using a priority queue
SSSPResult dijkstra(const Graph& g, int source);

#endif // PARALLEL_SSSP_UTILS_H
