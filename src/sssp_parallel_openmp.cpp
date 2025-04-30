//
// Created by Ali Hamza Azam on 25/04/2025.
//

#include "../include/graph.h" // Use definitions from graph.h
#include "../include/utils.hpp" // Include utils for EdgeChange if needed, though graph.h has it now
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


        if (change.is_insertion) {
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


// Algorithm 3: Parallel Update of Affected Vertices (OpenMP)
void UpdateAffectedVertices_OpenMP(
    const Graph& G,
    SSSPResult& T, // Pass SSSPResult by reference
    std::vector<std::atomic<bool>>& Affected_Del,
    std::vector<std::atomic<bool>>& Affected
) {
    int n = G.num_vertices;
    std::atomic<bool> changed_del(true); // Use atomic for loop control

    // Phase 1: Propagate Deletions (Setting distances to infinity)
    while (changed_del.load(std::memory_order_acquire)) {
        changed_del.store(false, std::memory_order_relaxed); // Assume no changes in this iteration
        std::vector<int> newly_affected_del_indices;
        // Consider reserving based on previous iteration size or a fraction of n
        // newly_affected_del_indices.reserve(n / 10);

        #pragma omp parallel
        {
            std::vector<int> local_newly_affected_del;

            #pragma omp for nowait schedule(dynamic) // Process vertices marked for deletion propagation
            for (int v = 0; v < n; ++v) {
                // Use load with acquire memory order for visibility
                if (Affected_Del[v].load(std::memory_order_acquire)) {
                    // Use store with release memory order
                    Affected_Del[v].store(false, std::memory_order_release); // Reset flag for next deletion iteration

                    // Find children in the SSSP tree and mark them
                    // Use the helper function
                    for (int child : get_children(v, n, T.parent)) {
                         // Basic bounds check
                         if (child < 0 || child >= n) continue;

                         if (T.dist[child] != INFINITY_WEIGHT) { // Avoid redundant work
                            // This block needs atomicity if multiple parents could invalidate the same child
                            #pragma omp critical (ChildInvalidation)
                            {
                                // Re-check inside critical section
                                // Ensure the parent is still 'v' before invalidating
                                if (T.parent[child] == v && T.dist[child] != INFINITY_WEIGHT) {
                                     T.dist[child] = INFINITY_WEIGHT;
                                     T.parent[child] = -1; // Reset parent
                                     // Use relaxed memory order for flags updated within critical section
                                     Affected[child].store(true, std::memory_order_relaxed); // Mark as generally affected for Phase 2
                                     local_newly_affected_del.push_back(child); // Mark for next deletion round
                                }
                            }
                        }
                    }
                }
            } // end omp for

            // Collect results from local lists using a critical section
            #pragma omp critical (CollectDeleted)
            {
                if (!local_newly_affected_del.empty()) {
                    newly_affected_del_indices.insert(newly_affected_del_indices.end(),
                                                     local_newly_affected_del.begin(),
                                                     local_newly_affected_del.end());
                }
            }
        } // end parallel region

        // Mark the children collected in this pass as affected for the *next* deletion propagation pass
        if (!newly_affected_del_indices.empty()) {
            // Use store with release memory order
            changed_del.store(true, std::memory_order_release); // Signal that changes occurred
            #pragma omp parallel for schedule(static) // Static might be okay here
            for (size_t i = 0; i < newly_affected_del_indices.size(); ++i) {
                 int idx = newly_affected_del_indices[i];
                 // Bounds check already done when adding to list, but double check is safe
                 if (idx >= 0 && idx < n) {
                    // Use store with release memory order
                    Affected_Del[idx].store(true, std::memory_order_release);
                 }
            }
        }
    } // end while changed_del


    // Phase 2: Propagate Distance Updates
    std::atomic<bool> changed_dist(true); // Use atomic for loop control
    while (changed_dist.load(std::memory_order_acquire)) {
        changed_dist.store(false, std::memory_order_relaxed); // Assume no changes in this iteration
        // Use a temporary vector for the next affected set, non-atomic initially
        std::vector<bool> next_affected_flags(n, false);


        #pragma omp parallel
        {
            bool local_changed = false; // Track changes within this thread

            #pragma omp for nowait schedule(dynamic) // Iterate over potentially affected vertices
            for (int v = 0; v < n; ++v) {
                // Use load with acquire memory order
                if (Affected[v].load(std::memory_order_acquire)) {
                    // Relax outgoing edges
                    if (T.dist[v] != INFINITY_WEIGHT) { // Only relax if source 'v' is reachable
                        try {
                            for (const auto& edge : G.neighbors(v)) { // Use const G.neighbors()
                                int n_neighbor = edge.to;
                                Weight weight = edge.weight;
                                // Bounds check neighbor
                                if (n_neighbor < 0 || n_neighbor >= n) continue;

                                Weight new_dist = T.dist[v] + weight;

                                // Use atomic compare-and-swap or critical section
                                // Using critical section for simplicity
                                if (new_dist < T.dist[n_neighbor]) {
                                    #pragma omp critical (DistUpdateRelax)
                                    {
                                        // Re-check condition inside critical section
                                        if (new_dist < T.dist[n_neighbor]) {
                                            T.dist[n_neighbor] = new_dist;
                                            T.parent[n_neighbor] = v;
                                            // Mark in the non-atomic temporary vector first
                                            if (!next_affected_flags[n_neighbor]) {
                                                 next_affected_flags[n_neighbor] = true;
                                            }
                                            local_changed = true;
                                        }
                                    }
                                }
                            }
                        } catch (const std::out_of_range& oor) { /* Ignore error if v is invalid */ }
                    }

                    // Relax incoming edges (Pseudocode Step: else if Dist[v] > Dist[n] + W(n, v))
                    try {
                        for (const auto& edge : G.neighbors(v)) { // Iterate neighbors 'n' of 'v'
                            int n_neighbor = edge.to;
                            Weight weight = edge.weight; // Weight(n_neighbor, v) - assuming undirected
                            // Bounds check neighbor
                            if (n_neighbor < 0 || n_neighbor >= n) continue;

                            if (T.dist[n_neighbor] != INFINITY_WEIGHT) { // Check if neighbor 'n' is reachable
                                Weight new_dist_v = T.dist[n_neighbor] + weight;
                                if (new_dist_v < T.dist[v]) {
                                    #pragma omp critical (DistUpdateRelax) // Use same critical section
                                    {
                                        // Re-check condition inside critical section
                                        if (new_dist_v < T.dist[v]) {
                                            T.dist[v] = new_dist_v;
                                            T.parent[v] = n_neighbor;
                                            // Mark 'v' itself in the non-atomic temporary vector
                                             if (!next_affected_flags[v]) {
                                                 next_affected_flags[v] = true;
                                             }
                                            local_changed = true;
                                        }
                                    }
                                }
                            }
                        }
                     } catch (const std::out_of_range& oor) { /* Ignore error if v is invalid */ }
                }
            } // end for loop

            // If any thread made a change, signal the outer loop atomically
            if (local_changed) {
                 changed_dist.store(true, std::memory_order_relaxed); // Relaxed is fine here, check is acquire
            }

        } // end parallel region

        // Update Affected flags for the next iteration based on next_affected_flags
        #pragma omp parallel for schedule(static)
        for(int i=0; i<n; ++i) {
             // Use store with release memory order
             Affected[i].store(next_affected_flags[i], std::memory_order_release);
        }

    } // end while changed_dist
}


// Algorithm 4: Asynchronous Update with OpenMP (Placeholder)
void AsyncUpdate_OpenMP(
    const std::vector<EdgeChange>& changes,
    const Graph& G,
    SSSPResult& T // Pass SSSPResult by reference
    // int asynchronyLevel // Parameter from pseudocode
) {
    std::cerr << "AsyncUpdate_OpenMP (Algorithm 4) is not fully implemented." << std::endl;
    // Implementation depends heavily on the chosen asynchronous model (e.g., work queues, tasking).
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
    UpdateAffectedVertices_OpenMP(G, T, Affected_Del, Affected);
    auto end_update = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> update_time = end_update - start_update;
    // std::cout << "  [OMP] UpdateAffectedVertices finished in " << update_time.count() << " ms." << std::endl;


    // Note: Algorithm 4 (AsyncUpdate) is separate and not called here by default.
}
