//
// Created by Ali Hamza Azam on 25/04/2025.
//

#ifndef PARALLEL_SSSP_GRAPH_H
#define PARALLEL_SSSP_GRAPH_H

#include <vector>
#include <limits>
#include <queue>
#include <atomic> // For potential atomic operations if needed later
#include <stdexcept> // For exceptions
#include <metis.h> // Include METIS header

// Add Weight type alias
using Weight = double;

const Weight INFINITY_WEIGHT = std::numeric_limits<Weight>::infinity();

struct Edge {
    int to;
    Weight weight; // Use Weight type alias
};

struct Graph {
    int num_vertices;
    std::vector<std::vector<Edge>> adj; // Adjacency list

    Graph(int n) : num_vertices(n), adj(n) {}

    // Declare add_edge here
    void add_edge(int u, int v, Weight weight);

    // Placeholder for removing an edge - requires finding the edge first
    void remove_edge(int u, int v);

    // Add const overload for neighbors
    const std::vector<Edge>& neighbors(int u) const {
         if (u < 0 || u >= num_vertices) {
             throw std::out_of_range("Vertex index out of range in neighbors()");
         }
        return adj[u];
    }

    // Method to convert graph to METIS CSR format - Declaration moved here
    void to_metis_csr(std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy, std::vector<idx_t>& adjwgt) const;

    // Method to get the number of unique edges (handles undirected representation)
    size_t get_edge_count() const {
        size_t count = 0;
        for (int u = 0; u < num_vertices; ++u) {
            for (const auto& edge : adj[u]) {
                // Count edge only once in undirected graph
                if (u < edge.to) {
                    count++;
                }
            }
        }
        return count;
    }

    // Check if an edge exists between two vertices
    bool has_edge(int u, int v) const;
};

// Structure to hold SSSP results
struct SSSPResult {
    std::vector<Weight> dist; // Use Weight type alias
    std::vector<int> parent;

    // Constructor initializing vectors
    SSSPResult(int n = 0) : dist(n, INFINITY_WEIGHT), parent(n, -1) {}
};

// Enum for types of edge changes
enum class ChangeType {
    INSERT,
    DELETE,
    INCREASE, // Weight increase (treated like DELETE in some contexts)
    DECREASE  // Weight decrease (treated like INSERT in some contexts)
};

// Structure for edge changes
struct EdgeChange {
    int u, v;
    Weight weight; // Use Weight type alias
    ChangeType type; // Type of change
};

#endif //PARALLEL_SSSP_GRAPH_H
