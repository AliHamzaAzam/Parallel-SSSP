#include "../include/graph.h"

// ...existing code...

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

// ...existing code...

