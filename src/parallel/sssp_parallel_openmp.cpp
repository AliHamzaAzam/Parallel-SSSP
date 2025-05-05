// sssp_parallel_openmp.cpp - OpenMP parallel dynamic SSSP update routines
// ---------------------------------------------------------------------
// Implements functions to identify and update affected vertices in the
// shortest-path tree under edge insertions and deletions, using OpenMP

#include <atomic>
#include <iostream>
#include <vector>

#include "../../include/graph.hpp"
#include "../../include/utils.hpp"

// Helper: check if edge (u,v) is in the current SSSP tree
inline bool is_edge_in_tree(int u, int v,
    const std::vector<int>& Parent,
    const std::vector<Weight>& Dist,
    Weight weight)
{
    const double eps = 1e-9;
    if (Parent[v] == u && std::abs(Dist[u] + weight - Dist[v]) < eps) return true;
    if (Parent[u] == v && std::abs(Dist[v] + weight - Dist[u]) < eps) return true;
    return false;
}

// Extract direct children of a vertex in the SSSP tree
std::vector<int> get_children(int v, int n, const std::vector<int>& Parent)
{
    std::vector<int> children;
    for (int i = 0; i < n; ++i) {
        if (Parent[i] == v) children.push_back(i);
    }
    return children;
}

// ProcessChanges_OpenMP: apply all edge changes (insert/delete) in parallel
// - G: input graph
// - changes: list of EdgeChange (type, endpoints, weight)
// - T: current SSSP results (dist, parent) updated in-place
// - Affected_Del/Affected: flags to mark vertices whose distance or tree membership changed
void ProcessChanges_OpenMP(
    const Graph& G,
    const std::vector<EdgeChange>& changes,
    SSSPResult& T,
    std::vector<std::atomic<bool>>& Affected_Del,
    std::vector<std::atomic<bool>>& Affected)
{
    const auto n = G.num_vertices;
    #pragma omp parallel for schedule(dynamic)
    for (const auto & c : changes) {
        int u = c.u, v = c.v;
        if (u < 0 || v < 0 || u >= n || v >= n) continue;

        if (c.type == ChangeType::INSERT || c.type == ChangeType::DECREASE) {
            // Handle insert or decrease: relax endpoint with larger dist
            Weight w = c.weight;
            bool u_inf = (T.dist[u] == INFINITY_WEIGHT);
            bool v_inf = (T.dist[v] == INFINITY_WEIGHT);
            int x = (v_inf || (!u_inf && T.dist[u] <= T.dist[v])) ? u : v;
            int y = (x == u ? v : u);
            Weight newd = (T.dist[x] == INFINITY_WEIGHT ? INFINITY_WEIGHT : T.dist[x] + w);
            if (newd < T.dist[y]) {
                #pragma omp critical
                {
                    if (newd < T.dist[y]) {
                        T.dist[y] = newd;
                        T.parent[y] = x;
                        Affected[y].store(true, std::memory_order_relaxed);
                    }
                }
            }
        } else {
            // Handle delete or increase: invalidate if tree edge removed
            Weight dw = INFINITY_WEIGHT;
            for (auto& e : G.neighbors(u)) if (e.to == v) { dw = e.weight; break; }
            if (dw == INFINITY_WEIGHT) continue;
            if (is_edge_in_tree(u, v, T.parent, T.dist, dw)) {
                int y = (T.parent[v] == u ? v : u);
                #pragma omp critical
                {
                    if (is_edge_in_tree(u, v, T.parent, T.dist, dw) && T.dist[y] != INFINITY_WEIGHT) {
                        T.dist[y] = INFINITY_WEIGHT;
                        T.parent[y] = -1;
                        Affected_Del[y].store(true, std::memory_order_relaxed);
                        Affected[y].store(true, std::memory_order_relaxed);
                    }
                }
            }
        }
    }
}

// UpdateAffectedVertices_OpenMP: Bellman-Ford relaxations on affected set
// - G: input graph
// - T: SSSP results
// - Affected: atomic flags marking vertices to update (reset when processed)
void UpdateAffectedVertices_OpenMP(
    const Graph& G,
    SSSPResult& T,
    std::vector<std::atomic<bool>>& Affected)
{
    int n = G.num_vertices;
    bool changed = true;
    while (changed) {
        changed = false;
        #pragma omp parallel for reduction(||:changed) schedule(dynamic)
        for (int v = 0; v < n; ++v) {
            if (Affected[v].load(std::memory_order_relaxed)) {
                Weight best = INFINITY_WEIGHT;
                int p = -1;
                for (auto& e : G.neighbors(v)) {
                    int u = e.to;
                    if (!Affected[u].load() && T.dist[u] + e.weight < best) {
                        best = T.dist[u] + e.weight;
                        p = u;
                    }
                }
                if (best < T.dist[v]) {
                    T.dist[v] = best;
                    T.parent[v] = p;
                    changed = true;
                }
                Affected[v].store(false);
            }
        }
    }
}

// BatchUpdate_OpenMP: full update sequence for a batch of changes
// - runs ProcessChanges then UpdateAffectedVertices
void BatchUpdate_OpenMP(
    const Graph& G,
    SSSPResult& T,
    const std::vector<EdgeChange>& changes)
{
    int n = G.num_vertices;
    if (n == 0) return;
    std::vector<std::atomic<bool>> Affected_Del(n), Affected(n);
    for (int i = 0; i < n; ++i) {
        Affected_Del[i].store(false);
        Affected[i].store(false);
    }
    ProcessChanges_OpenMP(G, changes, T, Affected_Del, Affected);
    UpdateAffectedVertices_OpenMP(G, T, Affected);
}

// TestDynamicSSSPWorkflow_OpenMP: example driver to run and print results
void TestDynamicSSSPWorkflow_OpenMP(
    Graph& G,
    SSSPResult& T,
    const std::vector<EdgeChange>& changes)
{
    BatchUpdate_OpenMP(G, T, changes);
    std::cout << "--- After OpenMP Dynamic SSSP ---\n";
    for (int i = 0; i < G.num_vertices; ++i) {
        std::cout << "Vertex " << i << ": Dist=" << T.dist[i]
                  << ", Parent=" << T.parent[i] << '\n';
    }
}
