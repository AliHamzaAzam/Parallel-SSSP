//
// Created by Ali Hamza Azam on 25/04/2025.
//

#include "../../include/graph.hpp"
#include "../../include/utils.hpp"
#include <vector>
#include <limits>
#include <omp.h>
#include <atomic>
#include <iostream>
#include <numeric> // For std::iota if needed

// --- Helper Functions ---

// Check if edge exists in the SSSP tree T
// Needs edge weight for accurate check
inline bool is_edge_in_tree(int u, int v, const std::vector<int>& Parent, const std::vector<Weight>& Dist, Weight edge_weight) {
    // Check parent relationship and if the distance matches
    // Handle potential floating point inaccuracies if necessary
    const double epsilon = 1e-9; // Adjust epsilon as needed
    if (Parent[v] == u && std::abs((Dist[u] + edge_weight) - Dist[v]) < epsilon) return true;
    if (Parent[u] == v && std::abs((Dist[v] + edge_weight) - Dist[u]) < epsilon) return true;
    return false;
}

// Get children of v in the SSSP tree T
std::vector<int> get_children(int v, int num_vertices, const std::vector<int>& Parent) {
    std::vector<int> children;
    // This is inefficient for large graphs; a better approach stores child lists or uses adjacency info.
    // Consider pre-calculating child lists if performance is critical.
    for (int i = 0; i < num_vertices; ++i) {
        if (Parent[i] == v) {
            children.push_back(i);
        }
    }
    return children;
}


// Algorithm 2: Parallel Identification of Affected Vertices (OpenMP)
// Takes combined changes vector
void ProcessChanges_OpenMP(
    const Graph& G,
    const std::vector<EdgeChange>& changes, // Combined changes
    SSSPResult& T, // Pass SSSPResult by reference
    std::vector<std::atomic<bool>>& Affected_Del,
    std::vector<std::atomic<bool>>& Affected
) {
    #pragma omp parallel for schedule(dynamic) // Use dynamic scheduling for potentially uneven workload
    for (size_t i = 0; i < changes.size(); ++i) {
        const auto& change = changes[i];
        int u = change.u;
        int v = change.v;

        // Basic bounds check for safety
        if (u < 0 || u >= G.num_vertices || v < 0 || v >= G.num_vertices) {
             // Optionally log this error
             // std::cerr << "Warning: Change " << i << " involves out-of-bounds vertex." << std::endl;
             continue; // Skip this change
        }


        if (change.type == ChangeType::INSERT || change.type == ChangeType::DECREASE) {
            Weight w = change.weight;
            // Determine closer (x) and farther (y) vertices based on current Dist
            int x, y;
             // Handle potential INFINITY_WEIGHT during comparison
            bool u_inf = (T.dist[u] == INFINITY_WEIGHT);
            bool v_inf = (T.dist[v] == INFINITY_WEIGHT);

            if (u_inf && v_inf) continue; // Both unreachable, insertion doesn't help yet
            else if (u_inf) { x = v; y = u; } // u is farther (inf)
            else if (v_inf) { x = u; y = v; } // v is farther (inf)
            else if (T.dist[u] > T.dist[v]) { x = v; y = u; } // v is closer
            else { x = u; y = v; } // u is closer or equal

            Weight potential_dist_y = (T.dist[x] == INFINITY_WEIGHT) ? INFINITY_WEIGHT : T.dist[x] + w;

            // Use atomic compare-and-swap or critical section for thread safety
            // Using critical section for simplicity here
            if (potential_dist_y < T.dist[y]) {
                #pragma omp critical (DistUpdateInsert)
                {
                    // Re-check condition inside critical section
                    if (potential_dist_y < T.dist[y]) {
                        T.dist[y] = potential_dist_y;
                        T.parent[y] = x;
                        Affected[y].store(true, std::memory_order_relaxed); // Mark as affected
                    }
                }
            }
             // Check the other direction too (x might be updated via y)
             Weight potential_dist_x = (T.dist[y] == INFINITY_WEIGHT) ? INFINITY_WEIGHT : T.dist[y] + w;
             if (potential_dist_x < T.dist[x]) {
                 #pragma omp critical (DistUpdateInsert) // Use the same critical section name
                 {
                     // Re-check condition inside critical section
                     if (potential_dist_x < T.dist[x]) {
                         T.dist[x] = potential_dist_x;
                         T.parent[x] = y;
                         Affected[x].store(true, std::memory_order_relaxed);
                     }
                 }
             }

        } else { // Deletion
            // Find the weight of the edge being deleted (needed for tree check)
            // This requires looking up the edge weight.
            Weight deleted_edge_weight = INFINITY_WEIGHT; // Initialize to invalid weight
             try {
                 // Use the const version of neighbors
                 for(const auto& edge : G.neighbors(u)) {
                     if (edge.to == v) {
                         deleted_edge_weight = edge.weight;
                         break;
                     }
                 }
             } catch (const std::out_of_range& oor) {
                 // Handle cases where u might be invalid (though checked above)
                 continue; // Skip if neighbor lookup fails
             }

             // If weight wasn't found, the edge likely didn't exist (or was already removed)
             if (deleted_edge_weight == INFINITY_WEIGHT) {
                 continue;
             }


            // Check if the edge was part of the SSSP tree T using the found weight
            if (is_edge_in_tree(u, v, T.parent, T.dist, deleted_edge_weight)) {
                 // Determine which vertex depended on the edge (the one with the larger distance)
                 int y = -1;
                 if (T.parent[v] == u && std::abs((T.dist[u] + deleted_edge_weight) - T.dist[v]) < 1e-9) {
                     y = v;
                 } else if (T.parent[u] == v && std::abs((T.dist[v] + deleted_edge_weight) - T.dist[u]) < 1e-9) {
                     y = u;
                 }


                 if (y != -1) { // If a dependent vertex 'y' was found
                     // Use atomic operations or critical section
                     #pragma omp critical (DistUpdateDelete)
                     {
                         // Re-check if still dependent inside critical section
                         if (is_edge_in_tree(u, v, T.parent, T.dist, deleted_edge_weight)) {
                             // Ensure we are invalidating the correct vertex 'y' that depended on the edge
                             if ((y == v && T.parent[v] == u) || (y == u && T.parent[u] == v)) {
                                 if (T.dist[y] != INFINITY_WEIGHT) { // Avoid redundant work
                                     T.dist[y] = INFINITY_WEIGHT;
                                     T.parent[y] = -1; // Reset parent
                                     Affected_Del[y].store(true, std::memory_order_relaxed); // Mark as affected by deletion
                                     Affected[y].store(true, std::memory_order_relaxed);     // Mark as generally affected
                                 }
                             }
                         }
                     }
                 }
            }
        }
    }
}


// Improved Algorithm 2: Mark and invalidate affected vertices after deletions
void IdentifyAndInvalidateAffected_OpenMP(
    const Graph& G,
    SSSPResult& T,
    const std::vector<EdgeChange>& changes,
    std::vector<std::atomic<bool>>& Affected
) {
    int n = G.num_vertices;
    // Step 1: Mark directly affected vertices (those whose parent edge is deleted)
    #pragma omp parallel for
    for (size_t i = 0; i < changes.size(); ++i) {
        const auto& change = changes[i];
        if (change.type == ChangeType::INSERT || change.type == ChangeType::DECREASE) continue; // Only process deletions
        int u = change.u, v = change.v;
        // If the parent of v is u, and the edge (u,v) was in the SSSP tree, mark v
        if (T.parent[v] == u && !G.has_edge(u, v)) {
            Affected[v] = true;
            T.dist[v] = INFINITY_WEIGHT;
            T.parent[v] = -1;
        }
        // If the parent of u is v, and the edge (v,u) was in the SSSP tree, mark u
        if (T.parent[u] == v && !G.has_edge(v, u)) {
            Affected[u] = true;
            T.dist[u] = INFINITY_WEIGHT;
            T.parent[u] = -1;
        }
    }
    // Step 2: Propagate affected marking to descendants in the SSSP tree
    bool changed = true;
    while (changed) {
        changed = false;
        #pragma omp parallel for
        for (int v = 0; v < n; ++v) {
            if (Affected[v]) {
                for (int w = 0; w < n; ++w) {
                    if (T.parent[w] == v && !Affected[w]) {
                        Affected[w] = true;
                        T.dist[w] = INFINITY_WEIGHT;
                        T.parent[w] = -1;
                        changed = true;
                    }
                }
            }
        }
    }
}


// Algorithm 2: Parallel Identification of Affected Vertices (OpenMP)
// Marks all vertices whose shortest path is affected by edge deletions
void IdentifyAffectedVertices_OpenMP(
    const Graph& G,
    const std::vector<EdgeChange>& changes, // Only deletions
    const SSSPResult& T, // Current SSSP tree
    std::vector<std::atomic<bool>>& Affected // Output: affected vertices
) {
    int n = G.num_vertices;
    // Step 1: Mark directly affected vertices (those whose parent edge is deleted)
    #pragma omp parallel for
    for (size_t i = 0; i < changes.size(); ++i) {
        const auto& change = changes[i];
        if (change.type == ChangeType::INSERT || change.type == ChangeType::DECREASE) continue; // Only process deletions
        int u = change.u, v = change.v;
        // If the parent of v is u, and the edge (u,v) was in the SSSP tree, mark v
        if (T.parent[v] == u) {
            Affected[v] = true;
        }
        // If the parent of u is v, and the edge (v,u) was in the SSSP tree, mark u
        if (T.parent[u] == v) {
            Affected[u] = true;
        }
    }
    // Step 2: Propagate affected marking to descendants in the SSSP tree
    // (If a parent is affected, all its children are affected)
    bool changed = true;
    while (changed) {
        changed = false;
        #pragma omp parallel for
        for (int v = 0; v < n; ++v) {
            if (Affected[v]) {
                for (int w = 0; w < n; ++w) {
                    if (T.parent[w] == v && !Affected[w]) {
                        Affected[w] = true;
                        changed = true;
                    }
                }
            }
        }
    }
}


// Algorithm 3: Parallel Update of Affected Vertices (OpenMP)
// Updates distances and parents for all affected vertices after deletions
void UpdateAffectedVertices_OpenMP(
    const Graph& G,
    SSSPResult& T, // Pass SSSPResult by reference
    std::vector<std::atomic<bool>>& Affected // Input/Output: affected vertices
) {
    int n = G.num_vertices;
    bool changed = true;
    while (changed) {
        changed = false;
        #pragma omp parallel for
        for (int v = 0; v < n; ++v) {
            if (Affected[v]) {
                Weight min_dist = INFINITY_WEIGHT;
                int min_parent = -1;
                // Try to find a better path to v from its neighbors
                for (const auto& edge : G.neighbors(v)) {
                    int u = edge.to;
                    if (!Affected[u] && T.dist[u] + edge.weight < min_dist) {
                        min_dist = T.dist[u] + edge.weight;
                        min_parent = u;
                    }
                }
                if (min_dist < T.dist[v]) {
                    T.dist[v] = min_dist;
                    T.parent[v] = min_parent;
                    changed = true;
                    Affected[v] = false; // No longer affected
                }
            }
        }
    }
}


// Algorithm 4: Asynchronous Update of Affected Vertices (OpenMP)
// This function performs asynchronous updates for affected vertices after deletions/insertions
void AsyncUpdate_OpenMP(
    const Graph& G,
    SSSPResult& T, // Pass SSSPResult by reference
    std::vector<std::atomic<bool>>& Affected // Input/Output: affected vertices
) {
    int n = G.num_vertices;
    bool changed = true;
    while (changed) {
        changed = false;
        #pragma omp parallel for schedule(dynamic)
        for (int v = 0; v < n; ++v) {
            if (Affected[v]) {
                Weight min_dist = INFINITY_WEIGHT;
                int min_parent = -1;
                for (const auto& edge : G.neighbors(v)) {
                    int u = edge.to;
                    if (!Affected[u] && T.dist[u] + edge.weight < min_dist) {
                        min_dist = T.dist[u] + edge.weight;
                        min_parent = u;
                    }
                }
                if (min_dist < T.dist[v]) {
                    T.dist[v] = min_dist;
                    T.parent[v] = min_parent;
                    changed = true;
                    Affected[v] = false;
                }
            }
        }
    }
}


// Wrapper function for batch updates using OpenMP Algorithms 2 & 3
void BatchUpdate_OpenMP(
    const Graph& G,
    SSSPResult& T, // Pass SSSPResult by reference
    const std::vector<EdgeChange>& changes // Combined changes
) {
    int n = G.num_vertices;
    if (n == 0) return; // Handle empty graph case

    // Use std::vector<std::atomic<bool>> for thread safety
    std::vector<std::atomic<bool>> Affected_Del(n);
    std::vector<std::atomic<bool>> Affected(n);

    // Initialize flags in parallel
    #pragma omp parallel for schedule(static)
    for(int i=0; i<n; ++i) {
        // Use relaxed memory order for initialization
        Affected_Del[i].store(false, std::memory_order_relaxed);
        Affected[i].store(false, std::memory_order_relaxed);
    }

    // Step 1: Identify initially affected vertices (Algorithm 2)
    auto start_proc = std::chrono::high_resolution_clock::now();
    ProcessChanges_OpenMP(G, changes, T, Affected_Del, Affected);
    auto end_proc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> proc_time = end_proc - start_proc;
    // std::cout << "  [OMP] ProcessChanges finished in " << proc_time.count() << " ms." << std::endl;


    // Step 2: Update distances iteratively (Algorithm 3)
    auto start_update = std::chrono::high_resolution_clock::now();
    UpdateAffectedVertices_OpenMP(G, T, Affected);
    auto end_update = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> update_time = end_update - start_update;
    // std::cout << "  [OMP] UpdateAffectedVertices finished in " << update_time.count() << " ms." << std::endl;


    // Note: Algorithm 4 (AsyncUpdate) is separate and not called here by default.
}

// Test function for the full dynamic SSSP workflow using OpenMP algorithms
void TestDynamicSSSPWorkflow_OpenMP(
    Graph& G,
    SSSPResult& T,
    const std::vector<EdgeChange>& changes
) {
    int n = G.num_vertices;
    std::vector<std::atomic<bool>> Affected(n);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) Affected[i] = false;

    // 1. Identify and invalidate affected vertices (improved Algorithm 2)
    IdentifyAndInvalidateAffected_OpenMP(G, T, changes, Affected);

    // 2. Update affected vertices (Algorithm 3)
    UpdateAffectedVertices_OpenMP(G, T, Affected);

    // 3. Optionally, run asynchronous update (Algorithm 4)
    AsyncUpdate_OpenMP(G, T, Affected);

    // 4. Print updated SSSP result
    std::cout << "--- After Dynamic SSSP Update (OpenMP) ---" << std::endl;
    for (int v = 0; v < n; ++v) {
        std::cout << "Vertex " << v << ": Dist = " << T.dist[v] << ", Parent = " << T.parent[v] << std::endl;
    }
}
