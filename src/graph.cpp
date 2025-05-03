#include "../include/graph.hpp"
#include <stdexcept> // Required for std::runtime_error in to_metis_csr

// Definition of add_edge moved here
void Graph::add_edge(int u, int v, Weight weight) {
    // Assuming undirected graph for simplicity, add edges in both directions
    if (u < 0 || u >= num_vertices || v < 0 || v >= num_vertices) {
         throw std::out_of_range("Vertex index out of range in add_edge");
    }
    adj[u].push_back({v, weight});
    adj[v].push_back({u, weight});
}

bool Graph::has_edge(int from, int to) const {
    if (from < 0 || from >= num_vertices) return false;
    for (const auto& edge : adj[from]) {
        if (edge.to == to) {
            return true;
        }
    }
    return false;
}

void Graph::remove_edge(int u, int v) {
    if (u < 0 || u >= num_vertices || v < 0 || v >= num_vertices) {
        throw std::out_of_range("Vertex index out of range in remove_edge");
    }
    auto remove = [&](int from, int to) {
        auto& neighbors = adj[from];
        for (auto it = neighbors.begin(); it != neighbors.end(); ++it) {
            if (it->to == to) {
                neighbors.erase(it);
                return true;
            }
        }
        return false;
    };
    remove(u, v);
    remove(v, u);
}

// Definition of Graph::to_metis_csr moved here from utils.cpp
void Graph::to_metis_csr(std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy, std::vector<idx_t>& adjwgt) const {
    xadj.clear();
    adjncy.clear();
    adjwgt.clear();

    xadj.resize(num_vertices + 1);
    // Calculate exact total degree first for precise allocation
    size_t total_degree = 0;
    for (int i = 0; i < num_vertices; ++i) {
        total_degree += adj[i].size();
    }
    adjncy.resize(total_degree);
    adjwgt.resize(total_degree);

    idx_t current_adj_idx = 0;
    xadj[0] = 0;

    for (int i = 0; i < num_vertices; ++i) {
        for (const auto& edge : adj[i]) {
            if (current_adj_idx >= total_degree) {
                 // This should ideally not happen if total_degree was calculated correctly
                 throw std::runtime_error("CSR array index out of bounds during creation.");
            }
            adjncy[current_adj_idx] = static_cast<idx_t>(edge.to);
            // METIS expects integer weights. Cast or scale your Weight (double).
            // Simple cast for now. Consider rounding or scaling based on weight distribution.
            idx_t weight_val = static_cast<idx_t>(edge.weight);
            // METIS requires positive weights. If you have 0 or negative, adjust.
            adjwgt[current_adj_idx] = (weight_val <= 0) ? 1 : weight_val;
            current_adj_idx++;
        }
        xadj[i + 1] = current_adj_idx;
    }

    // Verify final index matches total degree
    if (current_adj_idx != total_degree) {
         throw std::runtime_error("CSR final index does not match total degree.");
    }
}


