// graph.hpp - Graph and SSSP data structures, plus edge-update definitions
// -------------------------------------------------------------
// Defines Graph (adjacency list), SSSPResult, ChangeType, and EdgeChange.

#ifndef PARALLEL_SSSP_GRAPH_H
#define PARALLEL_SSSP_GRAPH_H

#include <vector>
#include <limits>
#include <stdexcept>
#include <metis.h>

using Weight = double;                                // Numeric type for edge weights and distances
static constexpr Weight INFINITY_WEIGHT =             // Value representing 'infinite' distance
    std::numeric_limits<Weight>::infinity();

// Edge: destination vertex index and associated weight
struct Edge {
    int to;                                           // Neighbor vertex index
    Weight weight;                                    // Weight of the edge
};

// Graph: weighted undirected graph stored as adjacency lists
struct Graph {
    int num_vertices;                                 // Number of vertices in the graph
    std::vector<std::vector<Edge>> adj;               // Adjacency lists: for each u, vector of Edge{to,weight}

    Graph(int n = 0): num_vertices(n), adj(n) {}

    // add_edge: insert an undirected edge u<->v with weight w
    void add_edge(int u, int v, Weight w);

    // remove_edge: remove edge u<->v if present (no-op if missing)
    void remove_edge(int u, int v);

    // neighbors: return list of outgoing edges from u
    const std::vector<Edge>& neighbors(int u) const {
         if (u < 0 || u >= num_vertices) {
             throw std::out_of_range("Vertex index out of range in neighbors()");
         }
        return adj[u];
    }

    // to_metis_csr: populate CSR arrays for METIS_PartGraphKway
    //   xadj: offsets into adjncy/adjwgt, size num_vertices+1
    //   adjncy: concatenated target indices
    //   adjwgt: corresponding edge weights (idx_t)
    void to_metis_csr(
        std::vector<idx_t>& xadj,
        std::vector<idx_t>& adjncy,
        std::vector<idx_t>& adjwgt) const;

    // Method to get the number of unique edges (handles undirected representation)
    [[nodiscard]] size_t get_edge_count() const {
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
    [[nodiscard]] bool has_edge(int u, int v) const;
};

// SSSPResult: stores distances and parent pointers for SSSP tree
struct SSSPResult {
    std::vector<Weight> dist;                         // dist[u] = shortest-path distance from source
    std::vector<int> parent;                          // parent[u] = predecessor of u in the SSSP tree, or -1
    SSSPResult(int n = 0): dist(n, INFINITY_WEIGHT), parent(n, -1) {}
};

// ChangeType: type of structural update to apply to graph
enum class ChangeType { INSERT, DELETE, INCREASE, DECREASE };

// EdgeChange: represent a single update operation on an edge
struct EdgeChange {
    int u, v;                                         // Endpoints of the edge
    Weight weight;                                    // New weight for INSERT/DECREASE, ignored for DELETE/INCREASE
    ChangeType type;                                  // Type of update
};

#endif // PARALLEL_SSSP_GRAPH_H
