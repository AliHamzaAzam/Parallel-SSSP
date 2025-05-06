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
    const double eps = 1e-9; // A small epsilon for floating-point comparisons
    // Check if u is parent of v AND the distance matches
    if (Parent[v] == u && std::abs(Dist[u] + weight - Dist[v]) < eps) return true;
    // Check if v is parent of u AND the distance matches (for undirected graphs)
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
            // Handle delete or increase: invalidate tree edge if present (use parent pointers)
            int y = -1;
            if (T.parent[v] == u) {
                y = v;
            } else if (T.parent[u] == v) {
                y = u;
            } else {
                continue; // not part of tree
            }
            #pragma omp critical
            {
                if (T.dist[y] != INFINITY_WEIGHT) {
                    T.dist[y] = INFINITY_WEIGHT;
                    T.parent[y] = -1;
                    Affected_Del[y].store(true, std::memory_order_relaxed);
                    Affected[y].store(true, std::memory_order_relaxed);
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

// Asynchronous relaxations using OpenMP tasks (Algorithm 4)
void AsyncUpdateAffectedVertices_OpenMP(
    const Graph& G,
    SSSPResult& T,
    std::vector<std::atomic<bool>>& Affected) // Affected[v] is true if v needs processing
{
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            for (int v_idx = 0; v_idx < G.num_vertices; ++v_idx) {
                if (Affected[v_idx].load(std::memory_order_relaxed)) {
                    #pragma omp task firstprivate(v_idx) shared(G, T, Affected)
                    {
                        std::vector<int> task_local_stack;
                        task_local_stack.push_back(v_idx);

                        while (!task_local_stack.empty()) {
                            int curr = task_local_stack.back();
                            task_local_stack.pop_back();

                            Affected[curr].store(false, std::memory_order_relaxed);

                            // Step 1: Try to relax 'curr' itself using its neighbors (pull update for curr)
                            Weight best_dist_for_curr_candidate = T.dist[curr]; 
                            int best_parent_for_curr_candidate = T.parent[curr];
                            bool curr_path_can_be_improved = false;

                            for (const auto& edge_to_curr : G.neighbors(curr)) {
                                int potential_parent = edge_to_curr.to;
                                Weight weight_pp_to_curr = edge_to_curr.weight;

                                if (T.dist[potential_parent] != INFINITY_WEIGHT) {
                                    if (T.dist[potential_parent] + weight_pp_to_curr < best_dist_for_curr_candidate) {
                                        best_dist_for_curr_candidate = T.dist[potential_parent] + weight_pp_to_curr;
                                        best_parent_for_curr_candidate = potential_parent;
                                        curr_path_can_be_improved = true;
                                    }
                                }
                            }

                            if (curr_path_can_be_improved) {
                                #pragma omp critical (sssp_node_update_lock)
                                {
                                    if (best_dist_for_curr_candidate < T.dist[curr]) {
                                        T.dist[curr] = best_dist_for_curr_candidate;
                                        T.parent[curr] = best_parent_for_curr_candidate;
                                        // curr_was_actually_updated_in_pull = true; // Not strictly needed for current logic flow
                                    }
                                }
                            }

                            // Step 2: If 'curr' now has a finite distance, try to relax its neighbors (push update from curr)
                            if (T.dist[curr] != INFINITY_WEIGHT) {
                                for (const auto& edge_from_curr : G.neighbors(curr)) {
                                    int u_neighbor = edge_from_curr.to;
                                    Weight new_dist_for_u_neighbor = T.dist[curr] + edge_from_curr.weight;
                                    bool neighbor_dist_was_updated = false;

                                    #pragma omp critical (sssp_node_update_lock)
                                    {
                                        if (new_dist_for_u_neighbor < T.dist[u_neighbor]) {
                                            T.dist[u_neighbor] = new_dist_for_u_neighbor;
                                            T.parent[u_neighbor] = curr;
                                            neighbor_dist_was_updated = true;
                                        }
                                    }

                                    if (neighbor_dist_was_updated) {
                                        if (!Affected[u_neighbor].exchange(true, std::memory_order_relaxed)) {
                                            task_local_stack.push_back(u_neighbor);
                                        }
                                    }
                                }
                            }
                        } // end while (!task_local_stack.empty())
                    } // end task for v_idx
                }
            }
            #pragma omp taskwait
        } // end single
    } // end parallel
}

// Asynchronous batch update: apply structural changes then async dynamic update
void AsyncBatchUpdate_OpenMP(
    const Graph& G,
    SSSPResult& T,
    const std::vector<EdgeChange>& changes)
{
    int n = G.num_vertices;
    if (n == 0) return;
    std::vector<std::atomic<bool>> Affected(n), Affected_Del(n);
    for (int i = 0; i < n; ++i) {
        Affected[i].store(false, std::memory_order_relaxed);
        Affected_Del[i].store(false, std::memory_order_relaxed);
    }

    // Phase 1: Apply changes to T based on the edge changes and mark directly affected nodes.
    ProcessChanges_OpenMP(G, changes, T, Affected_Del, Affected);

    // Phase 2: Iteratively propagate invalidations (INFINITY_WEIGHT) down the SSSP tree.
    // This ensures that if a parent's distance becomes INF, its children also become INF
    // and are marked Affected, forcing them to find new paths.
    // This loop should be executed single-threaded or with proper synchronization if parallelized.
    // For simplicity here, it's a sequential loop before the parallel async update.
    bool changed_invalidation_pass;
    do {
        changed_invalidation_pass = false;
        for (int k = 0; k < n; ++k) {
            if (T.dist[k] != INFINITY_WEIGHT) { // If k itself is not already INF
                int parent_of_k = T.parent[k];
                if (parent_of_k != -1) { // If k has a parent
                    if (T.dist[parent_of_k] == INFINITY_WEIGHT) { // And parent's distance is INF
                        T.dist[k] = INFINITY_WEIGHT;
                        T.parent[k] = -1; // Orphan k
                        Affected[k].store(true, std::memory_order_relaxed);
                        changed_invalidation_pass = true;
                    }
                }
            }
        }
    } while (changed_invalidation_pass);

    // Phase 3: Asynchronous relaxation for all Affected nodes to find new shortest paths.
    AsyncUpdateAffectedVertices_OpenMP(G, T, Affected);
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
    std::cout << "--- After OpenMP Dynamic SSSP (Asynchronous Algorithm 4) ---\n";
    AsyncBatchUpdate_OpenMP(G, T, changes);
    for (int i = 0; i < G.num_vertices; ++i) {
        std::cout << "Vertex " << i << ": Dist=" << T.dist[i]
                  << ", Parent=" << T.parent[i] << '\n';
    }
}
