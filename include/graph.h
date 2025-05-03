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

    void add_edge(int u, int v, Weight weight) {
        // Assuming undirected graph for simplicity, add edges in both directions
        if (u < 0 || u >= num_vertices || v < 0 || v >= num_vertices) {
             throw std::out_of_range("Vertex index out of range in add_edge");
        }
        adj[u].push_back({v, weight});
        adj[v].push_back({u, weight});
    }

    // Placeholder for removing an edge - requires finding the edge first
    void remove_edge(int u, int v);

    // Add const overload for neighbors
    const std::vector<Edge>& neighbors(int u) const {
         if (u < 0 || u >= num_vertices) {
             throw std::out_of_range("Vertex index out of range in neighbors()");
         }
        return adj[u];
    }

     // Non-const version - return const reference as it doesn't modify state
     // This was causing issues when called from const Graph objects.
     // Returning const& should be safe and resolve the const-correctness errors.
     const std::vector<Edge>& neighbors(int u) { // Changed return type
          if (u < 0 || u >= num_vertices) {
             throw std::out_of_range("Vertex index out of range in neighbors()");
         }
         return adj[u]; // Return const reference
     }

    // Method to convert graph to METIS CSR format
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

    SSSPResult(int n) : dist(n, INFINITY_WEIGHT), parent(n, -1) {}
};

// Structure for edge changes
struct EdgeChange {
    int u, v;
    Weight weight; // Use Weight type alias
    bool is_insertion;
};

#endif //PARALLEL_SSSP_GRAPH_H
