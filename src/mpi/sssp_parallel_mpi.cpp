// sssp_parallel_mpi.cpp - MPI distributed dynamic SSSP update routines
// ------------------------------------------------------------
// Implements the distributed update algorithm for SSSP under edge changes:
// 1. Distributed_UpdateAffected_MPI: iterative allreduce + local relaxations
// 2. Distributed_DynamicSSSP_MPI: orchestrates invalidation and update phases

#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>

#include "../../include/graph.hpp"
#include "../../include/utils.hpp"

// Type alias for boundary edges (must match main_mpi.cpp)
// Assumed structure: map<local_vertex_idx, vector<Edge{remote_global_vertex_idx, weight}>>
// i.e., maps a local vertex to the list of incoming edges from remote vertices.
using BoundaryEdges = std::map<int, std::vector<Edge>>;

// Note: This file defines the core distributed update routines; graph data is passed via parameters

// Forward declaration for asynchronous update variant (Algorithm 4)
void Distributed_UpdateAffected_MPI_Async(
    const Graph& local_graph,
    const BoundaryEdges& local_boundary_edges,
    const std::vector<int>& local_to_global,
    std::vector<double>& dist,
    std::vector<int>& parent,
    int my_rank,
    const std::vector<int>& part,
    int source_global_id, // Added source_global_id
    bool debug_output
);

// Forward declaration for sparse communication variant
void Distributed_UpdateAffected_MPI_Sparse(
    const Graph& local_graph,
    const BoundaryEdges& local_boundary_edges,
    const std::vector<int>& local_to_global,
    std::vector<double>& dist,
    std::vector<int>& parent,
    int my_rank,
    const std::vector<int>& part,
    int source_global_id,
    bool debug_output
);

// Distributed_IdentifyAffected_MPI: determine which local vertices are affected by edge changes
// Inputs:
//   local_graph        : subgraph owned by this rank
//   local_to_global    : map from local index to global vertex ID
//   changes            : full list of edge updates (DELETE/INCREASE)
//   parent             : current SSSP parent array (global IDs)
//   part               : global partition vector (global ID -> owner rank)
// Output:
//   affected (size n_local): marks true for local vertices whose distances/tree need update
void Distributed_IdentifyAffected_MPI(
    const Graph& local_graph,
    const std::vector<int>& local_to_global,
    const std::vector<EdgeChange>& changes,
    std::vector<bool>& affected,
    const std::vector<int>& parent,
    const std::vector<int>& part
) {
    int n_local = local_graph.num_vertices;
    int n_global = part.size();
    // Step 1: Mark directly affected (deleted/increased) child vertices in the SSSP tree
    std::vector<bool> affected_global(n_global, false);
    for (const auto& change : changes) {
        if (change.type == ChangeType::DELETE || change.type == ChangeType::INCREASE) {
            int u = change.u;
            int v = change.v;
            int y = -1;
            // Identify which endpoint was the child in the SSSP tree
            if (v >= 0 && v < n_global && parent[v] == u) {
                y = v;
            } else if (u >= 0 && u < n_global && parent[u] == v) {
                y = u;
            }
            if (y != -1) {
                affected_global[y] = true;
            }
        }
    }
    // Step 2: Propagate affected status to descendants in the SSSP tree (global)
    bool changed = true;
    while (changed) {
        changed = false;
        for (int v = 0; v < n_global; ++v) {
            if (!affected_global[v] && parent[v] != -1 && affected_global[parent[v]]) {
                affected_global[v] = true;
                changed = true;
            }
        }
    }
    // Step 3: Set local affected array
    for (int u_local = 0; u_local < n_local; ++u_local) {
        int u_global = local_to_global[u_local];
        affected[u_local] = affected_global[u_global];
    }
}

// Distributed_UpdateAffected_MPI: iteratively relax local and boundary edges across ranks
// Inputs:
//   local_graph           : subgraph owned by this rank
//   local_boundary_edges  : incoming edges from remote vertices
//   local_to_global       : map from local index to global ID
//   dist, parent          : local SSSP arrays (updated in-place)
//   my_rank, part         : MPI rank and global partition mapping
//   source_global_id      : global ID of the SSSP source vertex
//   debug_output          : enable verbose iteration logs
// Executes allreduce of global distances and local relaxations until convergence
void Distributed_UpdateAffected_MPI(
    const Graph& local_graph,
    const BoundaryEdges& local_boundary_edges,
    const std::vector<int>& local_to_global,
    std::vector<double>& dist,
    std::vector<int>& parent,
    int my_rank,
    const std::vector<int>& part,
    int source_global_id, // Added source_global_id
    bool debug_output
) {
    // OPTIMIZATION: Check if sparse communication should be used
    bool use_sparse = false;
    if (char* sparse_env = getenv("SSSP_SPARSE")) {
        use_sparse = (strcmp(sparse_env, "1") == 0 || 
                     strcmp(sparse_env, "true") == 0 || 
                     strcmp(sparse_env, "yes") == 0);
    } else {
        // Choose sparse automatically if graph is large or has many edge changes
        int n_local = local_graph.num_vertices;
        int n_global = part.size();
        use_sparse = (n_global > 10000 || n_local > 5000);
    }
    
    // Use sparse communication for large graphs or when explicitly requested
    if (use_sparse) {
        if (my_rank == 0) {
            std::cout << "[Rank 0] Using sparse communication optimization" << std::endl;
        }
        Distributed_UpdateAffected_MPI_Sparse(
            local_graph, local_boundary_edges, local_to_global,
            dist, parent, my_rank, part, source_global_id, debug_output
        );
        return;
    }
    
    // Original dense implementation continues below
    int n_local = local_graph.num_vertices;
    int n_global = part.size(); // Get global size from partition array

    // Safety check to prevent allocation of very large buffers
    if (n_global > 1000000) {
        if (my_rank == 0) {
            std::cerr << "WARNING: Very large global vertex count: " << n_global
                      << ". This may cause memory issues." << std::endl;
        }
    }

    if (my_rank == 0) std::cout << "[Rank 0] UpdateAffected: Starting. n_local=" << n_local << ", n_global=" << n_global << std::endl;

    bool local_changed_in_iter = true;
    int iteration = 0;
    int consecutive_no_changes = 0; // OPTIMIZATION: Early termination counter
    const int early_term_threshold = 3; // OPTIMIZATION: Stop after this many iterations with small changes

    // Allocate buffers with safety checks
    std::vector<double> send_dist_buffer;
    std::vector<double> current_global_dist;
    std::vector<double> prev_global_dist; // OPTIMIZATION: Track previous state to detect significant changes

    try {
        // Buffer to hold distances from this rank to be sent in Allreduce
        send_dist_buffer.resize(n_global, INFINITY_WEIGHT);
        // Buffer to receive the minimum distances across all ranks for all global vertices
        current_global_dist.resize(n_global, INFINITY_WEIGHT);
        prev_global_dist.resize(n_global, INFINITY_WEIGHT); // OPTIMIZATION: For change tracking
    } catch (const std::bad_alloc& e) {
        std::cerr << "[Rank " << my_rank << "] ERROR: Memory allocation failed: " << e.what()
                  << ". Try reducing problem size or using more processes." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    // OPTIMIZATION: Track changed vertices for sparse updates
    std::vector<bool> changed_vertices(n_local, true);  // Initially all are marked as changed
    std::vector<int> changed_global_indices; // List of global indices that changed
    changed_global_indices.reserve(n_local); // Reserve to avoid reallocations

    // Check for asynchronous update variant via environment variable
    if (char* async_env = getenv("SSSP_ASYNC"); async_env != nullptr &&
        (strcmp(async_env, "1") == 0 || strcmp(async_env, "true") == 0 || strcmp(async_env, "yes") == 0)) {
        if (my_rank == 0) std::cout << "[Rank 0] Using asynchronous update variant (Algorithm 4)" << std::endl;
        Distributed_UpdateAffected_MPI_Async(local_graph, local_boundary_edges, local_to_global,
                                            dist, parent, my_rank, part, source_global_id, debug_output); // Pass source_global_id
        return;
    }

    // OPTIMIZATION: Prepare a priority queue for better edge relaxation ordering
    struct VertexDist {
        int vertex;
        double distance;
        bool operator>(const VertexDist& other) const { return distance > other.distance; }
    };

    while (true) {
        iteration++;
        local_changed_in_iter = false;
        
        // OPTIMIZATION: Save previous global distances for significant change detection
        prev_global_dist = current_global_dist;

        // --- DEBUG PRINT: State at start of iteration (Rank 3 only) ---
        if (debug_output && my_rank == 3) {
            std::cout << "[Rank 3 DEBUG] Iter " << iteration << " Start | Dist: [ ";
            for(double d : dist) std::cout << (d == INFINITY_WEIGHT ? "INF " : std::to_string(d) + " ");
            std::cout << "] | Parent: [ ";
            for(int p : parent) std::cout << p << " ";
            std::cout << "]" << std::endl;
        }
        // --- END DEBUG ---

        // --- Communication Step: Gather current minimum distances for all global vertices --- //
        // OPTIMIZATION: Sparse communication - only include vertices that changed
        std::fill(send_dist_buffer.begin(), send_dist_buffer.end(), INFINITY_WEIGHT);
        changed_global_indices.clear();
        
        #pragma omp parallel for schedule(static)
        for (int u_local = 0; u_local < n_local; ++u_local) {
            int u_global = local_to_global[u_local];
            // Ensure u_global is valid before accessing send_dist_buffer
            if (u_global >= 0 && u_global < n_global) {
                send_dist_buffer[u_global] = dist[u_local];
                if (changed_vertices[u_local]) {
                    #pragma omp critical
                    changed_global_indices.push_back(u_global);
                }
            } else {
                 // This case should ideally not happen if mappings are correct
                 if (my_rank == 0) std::cerr << "Warning: Invalid global index " << u_global << " for local index " << u_local << " in rank " << my_rank << std::endl;
            }
        }

        // Reset change tracking for next iteration
        std::fill(changed_vertices.begin(), changed_vertices.end(), false);

        // 2. Perform Allreduce to get the minimum distance for each global vertex across all ranks
        // Added error handling for MPI operations
        int mpi_result = MPI_Allreduce(send_dist_buffer.data(), current_global_dist.data(), n_global, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        if (mpi_result != MPI_SUCCESS) {
            std::cerr << "[Rank " << my_rank << "] MPI_Allreduce failed. Aborting." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, mpi_result);
            return;
        }

        // OPTIMIZATION: Check if significant changes occurred in the global state
        double max_change = 0.0;
        for (int i = 0; i < n_global; ++i) {
            if (prev_global_dist[i] != INFINITY_WEIGHT && current_global_dist[i] != INFINITY_WEIGHT) {
                max_change = std::max(max_change, std::abs(prev_global_dist[i] - current_global_dist[i]));
            }
            else if (prev_global_dist[i] != current_global_dist[i]) {
                max_change = INFINITY_WEIGHT; // Consider INF to finite or finite to INF a significant change
                break;
            }
        }
        
        // If change is small, count toward early termination threshold
        if (max_change < 1e-6) {
            consecutive_no_changes++;
        } else {
            consecutive_no_changes = 0;
        }

        // Self-Invalidation Pass based on parent's new distance
        bool changed_by_self_invalidation = false;
        #pragma omp parallel for schedule(dynamic) reduction(||:changed_by_self_invalidation)
        for (int u_local = 0; u_local < n_local; ++u_local) {
            int p_global = parent[u_local]; // Current parent (global ID)
            if (p_global != -1 && p_global >= 0 && p_global < n_global) { // Valid parent
                if (current_global_dist[p_global] == INFINITY_WEIGHT) { // Parent became INF
                    if (dist[u_local] != INFINITY_WEIGHT) {
                        dist[u_local] = INFINITY_WEIGHT;
                        parent[u_local] = -1;
                        changed_vertices[u_local] = true;
                        changed_by_self_invalidation = true;
                        if (debug_output) {
                            int u_g = local_to_global[u_local];
                            std::cout << "[Rank " << my_rank << " Sync Self-Invalidate] Iter " << iteration
                                      << ": u_local=" << u_local << " (global " << u_g
                                      << ") parent " << p_global << " became INF. Dist now INF." << std::endl;
                        }
                    }
                }
            }
        }
        if (changed_by_self_invalidation) {
            local_changed_in_iter = true;
        }

        // --- Local Relaxation Step --- //
        bool changed_this_pass = false;

        // OPTIMIZATION: Use priority queue for better edge relaxation ordering
        std::vector<VertexDist> priority_list;
        for (int u_local = 0; u_local < n_local; ++u_local) {
            if (dist[u_local] != INFINITY_WEIGHT) {
                priority_list.push_back({u_local, dist[u_local]});
            }
        }
        std::sort(priority_list.begin(), priority_list.end(), 
            [](const VertexDist& a, const VertexDist& b) { return a.distance < b.distance; });

        // Process vertices in priority order (better vertices first)
        #pragma omp parallel for schedule(dynamic, 16) reduction(||:changed_this_pass)
        for (size_t i = 0; i < priority_list.size(); ++i) {
            int u_local = priority_list[i].vertex;
            double dist_u = dist[u_local];
            if (dist_u == INFINITY_WEIGHT) continue;  // Skip if it became INF after preparation

            // Internal edges relaxation - process outgoing edges from u_local
            for (const auto& edge : local_graph.neighbors(u_local)) {
                const int v_local = edge.to;
                if (const double weight = edge.weight; dist_u + weight < dist[v_local]) {
                    #pragma omp atomic write
                    dist[v_local] = dist_u + weight;
                    parent[v_local] = local_to_global[u_local];
                    changed_vertices[v_local] = true;
                    changed_this_pass = true;
                }
            }

            // Boundary edges relaxation - check if we can be improved by remote vertices
            if (auto boundary_it = local_boundary_edges.find(u_local); boundary_it != local_boundary_edges.end()) {
                for (const auto& [to, weight] : boundary_it->second) {
                    const int remote_u = to;
                    const double w = weight;
                    if (const double remote_dist = current_global_dist[remote_u]; remote_dist + w < dist[u_local]) {
                        #pragma omp atomic write
                        dist[u_local] = remote_dist + w;
                        parent[u_local] = remote_u;
                        changed_vertices[u_local] = true;
                        changed_this_pass = true;
                    }
                }
            }
        }
        local_changed_in_iter = changed_this_pass;

        // --- DEBUG PRINT: Changed Flag (Rank 3 only) ---
        if (debug_output && my_rank == 3) {
            std::cout << "[Rank 3 DEBUG] Iter " << iteration << " End | changed_this_pass=" << (changed_this_pass ? "true" : "false") << std::endl;
        }
        // --- END DEBUG ---

        // --- Global Convergence Check --- //
        int local_flag = local_changed_in_iter ? 1 : 0;
        int global_flag = 0;
        mpi_result = MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (mpi_result != MPI_SUCCESS) {
            std::cerr << "[Rank " << my_rank << "] MPI_Allreduce for convergence check failed. Aborting." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, mpi_result);
            return;
        }

        if (my_rank == 0) std::cout << "[Rank 0] UpdateAffected Iteration " << iteration << ", Global Change Flag: " << global_flag << std::endl;

        // OPTIMIZATION: Early termination checks
        if (global_flag == 0 || consecutive_no_changes >= early_term_threshold) {
            if (consecutive_no_changes >= early_term_threshold && my_rank == 0) {
                std::cout << "[Rank 0] Early termination after " << iteration 
                          << " iterations. No significant changes for " 
                          << consecutive_no_changes << " iterations." << std::endl;
            }
            break;
        }

        // Add a safeguard against infinite loops (e.g., negative cycles, though SSSP assumes non-negative)
        // Or just very slow convergence. n_global iterations should be sufficient for Bellman-Ford like convergence.
        if (iteration > n_global + 1) { // Allow one extra iteration for propagation
             if (my_rank == 0) std::cerr << "Warning: UpdateAffected reached max iterations (" << iteration << "). Potential issue or slow convergence?" << std::endl;
             break;
        }

        // Secondary safeguard to prevent excessive iterations
        if (iteration > 1000) {
            std::cerr << "[Rank " << my_rank << "] WARNING: Excessive iterations reached (" << iteration
                      << "). Forcing termination." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
    } // end while loop

    std::cout << "[Rank " << my_rank << "] UpdateAffected: Finished after " << iteration << " iterations." << std::endl;
}

// Asynchronous variant (Algorithm 4): non-blocking allreduce overlapped with local relaxations
void Distributed_UpdateAffected_MPI_Async(
    const Graph& local_graph,
    const BoundaryEdges& local_boundary_edges,
    const std::vector<int>& local_to_global,
    std::vector<double>& dist,
    std::vector<int>& parent,
    int my_rank,
    const std::vector<int>& part,
    int source_global_id, // Added source_global_id
    bool debug_output
) {
    int n_local = local_graph.num_vertices;
    int n_global = part.size();
    std::vector<double> send_buf(n_global, INFINITY_WEIGHT);
    std::vector<double> global_buf(n_global, INFINITY_WEIGHT);
    std::vector<double> prev_global_buf(n_global, INFINITY_WEIGHT); // OPTIMIZATION: To track changes
    MPI_Request req;
    bool local_changed = true;
    int iteration = 0;
    
    // OPTIMIZATION: Track consecutive iterations without significant changes
    int stalled_iterations = 0;
    const int early_term_threshold = 3;
    const double delta = 1.0; // OPTIMIZATION: Delta parameter for delta-stepping

    // OPTIMIZATION: Track changed vertices for better update efficiency
    std::vector<bool> changed_vertices(n_local, true); // All vertices initially marked changed
    
    // OPTIMIZATION: Create neighborhood awareness - which ranks own neighboring vertices
    std::vector<int> neighbor_ranks;
    for (const auto& [local_vertex, edges] : local_boundary_edges) {
        for (const auto& edge : edges) {
            int remote_global_id = edge.to;
            if (remote_global_id >= 0 && remote_global_id < n_global) {
                int remote_rank = part[remote_global_id];
                if (std::find(neighbor_ranks.begin(), neighbor_ranks.end(), remote_rank) == neighbor_ranks.end()) {
                    neighbor_ranks.push_back(remote_rank);
                }
            }
        }
    }
    std::sort(neighbor_ranks.begin(), neighbor_ranks.end());
    neighbor_ranks.erase(std::unique(neighbor_ranks.begin(), neighbor_ranks.end()), neighbor_ranks.end());
    
    if (debug_output) {
        std::cout << "[Rank " << my_rank << "] Connected to " << neighbor_ranks.size() 
                  << " neighboring ranks" << std::endl;
    }
    
    // OPTIMIZATION: Use delta-stepping buckets for better vertex processing order
    const int MAX_BUCKETS = 128; // Limit number of buckets to avoid excessive overhead
    
    while (true) {
        iteration++;
        // prepare send buffer from local dist
        std::fill(send_buf.begin(), send_buf.end(), INFINITY_WEIGHT);
        
        // OPTIMIZATION: Store previous state before updating
        prev_global_buf = global_buf;
        
        // OPTIMIZATION: Batch update vertices by partitioning them into buckets
        std::vector<std::vector<int>> buckets(MAX_BUCKETS);
        
        // Fill send buffer from local distances
        #pragma omp parallel for schedule(static, 64)
        for (int u_local = 0; u_local < n_local; ++u_local) {
            int u_global = local_to_global[u_local];
            if (u_global >= 0 && u_global < n_global) {
                send_buf[u_global] = dist[u_local];
                
                // OPTIMIZATION: Assign vertex to a bucket based on its distance (delta-stepping)
                if (dist[u_local] != INFINITY_WEIGHT) {
                    int bucket_idx = std::min(static_cast<int>(dist[u_local] / delta), MAX_BUCKETS - 1);
                    #pragma omp critical
                    buckets[bucket_idx].push_back(u_local);
                }
            }
        }
        
        // start non-blocking allreduce
        MPI_Iallreduce(send_buf.data(), global_buf.data(), n_global, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD, &req);

        // OPTIMIZATION: Local relaxations using delta-stepping buckets
        local_changed = false;
        
        // For each bucket in increasing order of distance
        for (int bucket_idx = 0; bucket_idx < MAX_BUCKETS; ++bucket_idx) {
            if (buckets[bucket_idx].empty()) continue;
            
            // Process all vertices in this bucket
            #pragma omp parallel for schedule(dynamic, 16) reduction(||:local_changed)
            for (size_t i = 0; i < buckets[bucket_idx].size(); ++i) {
                int u_local = buckets[bucket_idx][i];
                double du = dist[u_local];
                if (du == INFINITY_WEIGHT) continue;
                
                // internal edges
                for (const auto& e : local_graph.neighbors(u_local)) {
                    if (du + e.weight < dist[e.to]) {
                        #pragma omp atomic write
                        dist[e.to] = du + e.weight;
                        parent[e.to] = local_to_global[u_local];
                        changed_vertices[e.to] = true;
                        local_changed = true;
                    }
                }
                
                // boundary edges - using potentially stale global_buf, but that's OK in async algorithm
                if (auto it = local_boundary_edges.find(u_local); it != local_boundary_edges.end()) {
                    for (const auto& be : it->second) {
                        int g_to = be.to;
                        double w = be.weight;
                        if (global_buf[g_to] + w < dist[u_local]) {
                            #pragma omp atomic write
                            dist[u_local] = global_buf[g_to] + w;
                            parent[u_local] = g_to;
                            changed_vertices[u_local] = true;
                            local_changed = true;
                        }
                    }
                }
            }
        }

        // wait for global reduction result
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        
        // OPTIMIZATION: Check for significant changes in global state
        double max_global_change = 0.0;
        for (int i = 0; i < n_global; ++i) {
            if (prev_global_buf[i] != INFINITY_WEIGHT && global_buf[i] != INFINITY_WEIGHT) {
                max_global_change = std::max(max_global_change, 
                                             std::abs(prev_global_buf[i] - global_buf[i]));
            } 
            else if (prev_global_buf[i] != global_buf[i]) {
                max_global_change = INFINITY_WEIGHT; // Consider INF->finite or finite->INF as significant
                break;
            }
        }
        
        if (max_global_change < 1e-6) {
            stalled_iterations++;
        } else {
            stalled_iterations = 0;
        }
        
        // update local dist with fresh global_buf
        bool changed_by_global_min_update = false;
        #pragma omp parallel for schedule(dynamic, 32) reduction(||:changed_by_global_min_update)
        for (int u_local = 0; u_local < n_local; ++u_local) {
            int u_global = local_to_global[u_local];
            double gd = global_buf[u_global];
            if (gd < dist[u_local]) {
                dist[u_local] = gd;
                parent[u_local] = u_global;
                changed_vertices[u_local] = true;
                changed_by_global_min_update = true;
            }
        }
        if (changed_by_global_min_update) local_changed = true;

        // Self-Invalidation Pass based on parent's new distance (using fresh global_buf)
        bool changed_by_async_self_invalidation = false;
        #pragma omp parallel for schedule(dynamic, 32) reduction(||:changed_by_async_self_invalidation)
        for (int u_local = 0; u_local < n_local; ++u_local) {
            int p_global = parent[u_local]; // Current parent (global ID)
            if (p_global != -1 && p_global >= 0 && p_global < n_global) { // Valid parent
                if (global_buf[p_global] == INFINITY_WEIGHT) { // Parent became INF
                    if (dist[u_local] != INFINITY_WEIGHT) {
                        dist[u_local] = INFINITY_WEIGHT;
                        parent[u_local] = -1;
                        changed_vertices[u_local] = true;
                        changed_by_async_self_invalidation = true;
                        if (debug_output) {
                            int u_g = local_to_global[u_local];
                            std::cout << "[Rank " << my_rank << " Async Self-Invalidate] Iter " << iteration
                                      << ": u_local=" << u_local << " (global " << u_g
                                      << ") parent " << p_global << " became INF. Dist now INF." << std::endl;
                        }
                    }
                }
            }
        }
        if (changed_by_async_self_invalidation) {
            local_changed = true;
        }
        
        // Update local dist with fresh global_buf values if they are better (e.g. source or path from another rank)
        // This step is crucial for incorporating the globally agreed minimums.
        bool changed_by_global_min_update2 = false;
        #pragma omp parallel for schedule(dynamic, 32) reduction(||:changed_by_global_min_update2)
        for (int u_local = 0; u_local < n_local; ++u_local) {
            int u_global = local_to_global[u_local];
            if (u_global >= 0 && u_global < n_global) { // Bounds check for u_global
                 double gd = global_buf[u_global];
                 if (gd < dist[u_local]) {
                    dist[u_local] = gd;
                    // If this node is the source and its distance became 0, parent is -1.
                    // Otherwise, if we take a global minimum, the parent is not immediately known from this step alone.
                    // Setting parent to -1 forces relaxation to find the true parent unless it's the source.
                    parent[u_local] = (u_global == source_global_id && gd == 0.0) ? -1 : -2; // -2 to indicate unknown, to be fixed by relaxation
                    changed_vertices[u_local] = true;
                    changed_by_global_min_update2 = true;
                 } else if (u_global == source_global_id && dist[u_local] > 0.0 && global_buf[u_global] == 0.0) {
                    // Ensure source node is correctly set if global_buf confirms its distance is 0
                    dist[u_local] = 0.0;
                    parent[u_local] = -1;
                    changed_vertices[u_local] = true;
                    changed_by_global_min_update2 = true;
                 }
            }
        }
        if (changed_by_global_min_update2) local_changed = true;

        // OPTIMIZATION: An additional full relaxation pass for vertices with unknown parents (-2)
        // to correctly establish parents after the global_min_update
        bool fixed_parents = false;
        #pragma omp parallel for schedule(dynamic) reduction(||:fixed_parents)
        for (int u_local = 0; u_local < n_local; ++u_local) {
            if (parent[u_local] == -2) {
                // Find the correct parent by checking all possible edges
                double best_dist = dist[u_local];
                int best_parent = -1;
                
                // Check local neighbors
                for (int v_local = 0; v_local < n_local; ++v_local) {
                    // Skip same vertex
                    if (v_local == u_local) continue;
                    
                    // Check if v_local has an edge to u_local
                    for (const auto& e : local_graph.neighbors(v_local)) {
                        if (e.to == u_local) {
                            // This is an incoming edge to u_local
                            if (dist[v_local] + e.weight <= best_dist) {
                                best_dist = dist[v_local] + e.weight;
                                best_parent = local_to_global[v_local];
                            }
                            break; // Found the edge, no need to check more
                        }
                    }
                }
                
                // Check boundary edges
                for (const auto& [from_local, edges] : local_boundary_edges) {
                    if (from_local == u_local) {
                        for (const auto& e : edges) {
                            int v_global = e.to;
                            if (global_buf[v_global] + e.weight <= best_dist) {
                                best_dist = global_buf[v_global] + e.weight;
                                best_parent = v_global;
                            }
                        }
                    }
                }
                
                if (best_parent != -1) {
                    parent[u_local] = best_parent;
                    fixed_parents = true;
                }
            }
        }
        if (fixed_parents) local_changed = true;

        // convergence check across ranks
        int flag = local_changed ? 1 : 0;
        int global_flag = 0;
        MPI_Allreduce(&flag, &global_flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        
        // OPTIMIZATION: Early termination conditions
        if (global_flag == 0 || stalled_iterations >= early_term_threshold) {
            if (stalled_iterations >= early_term_threshold && my_rank == 0) {
                std::cout << "[Rank 0] Async early termination after " << iteration 
                          << " iterations. No significant changes for " 
                          << stalled_iterations << " iterations." << std::endl;
            }
            break;
        }
        
        // Safety check to avoid infinite loops
        if (iteration > n_global + 1) {
            if (my_rank == 0) {
                std::cout << "[Rank 0] Maximum iterations reached in async update." << std::endl;
            }
            break;
        }
    }
    
    if (my_rank == 0) std::cout << "[Rank 0] Async UpdateAffected finished in " << iteration << " iterations." << std::endl;
}

// Distributed_DynamicSSSP_MPI: top-level driver for dynamic SSSP update per rank
// Inputs:
//   local_graph           : subgraph owned by this rank
//   local_boundary_edges  : incoming edges from remote ranks
//   local_to_global       : map local index -> global ID
//   global_to_local       : map global ID -> local index or -1
//   initial_affected_del  : flags for vertices to invalidate (deletion impact)
//   initial_affected      : flags for vertices needing update (insertion impact)
//   source                : global source vertex ID
//   part                  : global partition vector
//   my_rank, debug_output : MPI rank and verbose flag
// Operates by invalidating relevant vertices then calling UpdateAffected
void Distributed_DynamicSSSP_MPI(
    const Graph& local_graph,
    const BoundaryEdges& local_boundary_edges,
    const std::vector<int>& local_to_global,
    const std::vector<int>& global_to_local,
    std::vector<double>& dist,
    std::vector<int>& parent,
    int my_rank,
    const std::vector<int>& part,
    int source, // This is the source_global_id
    const std::vector<bool>& initial_affected_del,
    const std::vector<bool>& initial_affected,
    bool debug_output
) {
    if (debug_output && my_rank == 0) {
        std::cout << "[Rank 0 DEBUG] Entering Distributed_DynamicSSSP_MPI. Source: " << source << std::endl;
    }
    int n_local = local_graph.num_vertices;

    // OPTIMIZATION: Create a set to track affected vertices for faster lookup
    std::vector<bool> affected(n_local, false);
    
    // OPTIMIZATION: Gather precomputed information about local parent-child relationships to speedup invalidation
    std::vector<std::vector<int>> local_children(n_local);
    #pragma omp parallel for schedule(static)
    for (int u_local = 0; u_local < n_local; ++u_local) {
        int u_global = local_to_global[u_local];
        for (int v_local = 0; v_local < n_local; ++v_local) {
            if (parent[v_local] == u_global) {
                #pragma omp critical
                local_children[u_local].push_back(v_local);
            }
        }
    }

    // OPTIMIZATION: Use queue-based method for invalidation to avoid redundant work
    std::vector<int> invalidation_queue;
    
    // Step 1: Initial invalidation based on initial_affected_del flags.
    // These flags indicate that the SSSP parent edge for this vertex was deleted or its weight increased significantly.
    #pragma omp parallel for schedule(static)
    for (int v_local = 0; v_local < n_local; ++v_local) {
        if (initial_affected_del[v_local]) {
            if (debug_output) {
                std::cout << "[Rank " << my_rank << " DEBUG] Initial invalidation for local vertex " << v_local
                          << " (global " << local_to_global[v_local] << "). Old dist: " << dist[v_local] 
                          << ", Old parent: " << parent[v_local] << std::endl;
            }
            dist[v_local] = INFINITY_WEIGHT;
            parent[v_local] = -1;
            affected[v_local] = true;
            
            #pragma omp critical
            invalidation_queue.push_back(v_local);
        }
    }

    // Step 2: Propagate invalidations down the *local* SSSP tree using a queue-based approach.
    // This is much more efficient than the original iterative approach.
    if (!invalidation_queue.empty()) {
        int invalidated_count = 0;
        
        while (!invalidation_queue.empty()) {
            // Process multiple vertices in parallel
            std::vector<int> current_queue;
            #pragma omp critical
            {
                current_queue.swap(invalidation_queue);
            }
            
            std::vector<int> next_queue;
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < current_queue.size(); ++i) {
                int u_local = current_queue[i];
                
                // Process all children using the precomputed relationships
                for (int v_local_child : local_children[u_local]) {
                    if (dist[v_local_child] != INFINITY_WEIGHT) {
                        // This child needs to be invalidated
                        #pragma omp atomic write
                        dist[v_local_child] = INFINITY_WEIGHT;
                        parent[v_local_child] = -1;
                        affected[v_local_child] = true;
                        invalidated_count++;
                        
                        if (debug_output) {
                            #pragma omp critical
                            {
                                std::cout << "[Rank " << my_rank << " DEBUG] Queue-based invalidation of vertex " << v_local_child
                                          << " (global " << local_to_global[v_local_child] << ") from parent " 
                                          << local_to_global[u_local] << std::endl;
                            }
                        }
                        
                        // Add to next queue level
                        #pragma omp critical
                        next_queue.push_back(v_local_child);
                    }
                }
            }
            
            // Merge next_queue into invalidation_queue
            #pragma omp critical
            {
                invalidation_queue.swap(next_queue);
            }
        }
        
        if (debug_output) {
            std::cout << "[Rank " << my_rank << " DEBUG] Queue-based invalidation propagation completed. "
                      << "Total invalidated: " << invalidated_count << " vertices." << std::endl;
        }
    }

    // Additional check for source vertex - ensure it has correct distance
    int source_local = global_to_local[source];
    if (source_local >= 0 && source_local < n_local) {
        // The source is on this rank
        if (dist[source_local] != 0.0) {
            dist[source_local] = 0.0;
            parent[source_local] = -1;
            affected[source_local] = true;
            
            if (debug_output) {
                std::cout << "[Rank " << my_rank << " DEBUG] Resetting source vertex " << source_local
                          << " (global " << source << ") to distance 0." << std::endl;
            }
        }
    }

    // OPTIMIZATION: Avoid empty invocation of UpdateAffected
    // Check if any vertex is affected
    bool has_affected = false;
    for (int v_local = 0; v_local < n_local; ++v_local) {
        if (affected[v_local] || initial_affected[v_local]) {
            has_affected = true;
            break;
        }
    }

    if (!has_affected) {
        if (debug_output) {
            std::cout << "[Rank " << my_rank << " DEBUG] No vertices affected. Skipping update phase." << std::endl;
        }
        return;
    }

    // Step 3: Call the iterative relaxation process.
    if (debug_output) {
        std::cout << "[Rank " << my_rank << " DEBUG] Starting Distributed_UpdateAffected_MPI with "
                  << "potentially updated dist/parent arrays..." << std::endl;
    }

    // OPTIMIZATION: Check environment variable to choose best update strategy
    // based on problem characteristics or allow runtime selection
    bool use_async = false;
    if (char* async_env = getenv("SSSP_ASYNC")) {
        use_async = (strcmp(async_env, "1") == 0 || 
                    strcmp(async_env, "true") == 0 || 
                    strcmp(async_env, "yes") == 0);
    } else {
        // If not specified, decide based on problem characteristics:
        // Async is usually better for large, sparse graphs
        use_async = (n_local > 1000 || local_boundary_edges.size() > (n_local / 4));
    }
    
    if (use_async) {
        if (debug_output && my_rank == 0) {
            std::cout << "[Rank 0 DEBUG] Using asynchronous update variant." << std::endl;
        }
        Distributed_UpdateAffected_MPI_Async(
            local_graph,
            local_boundary_edges,
            local_to_global,
            dist,    // Pass the modified dist array
            parent,  // Pass the modified parent array
            my_rank,
            part,
            source,  // Pass source_global_id
            debug_output
        );
    } else {
        if (debug_output && my_rank == 0) {
            std::cout << "[Rank 0 DEBUG] Using synchronous update variant." << std::endl;
        }
        Distributed_UpdateAffected_MPI(
            local_graph,
            local_boundary_edges,
            local_to_global,
            dist,    // Pass the modified dist array
            parent,  // Pass the modified parent array
            my_rank,
            part,
            source,  // Pass source_global_id
            debug_output
        );
    }

    if (debug_output && my_rank == 0) {
        std::cout << "[Rank 0 DEBUG] Exiting Distributed_DynamicSSSP_MPI." << std::endl;
    }
}

// OPTIMIZATION: Sparse communication variant that only exchanges changed vertices
// This significantly reduces communication overhead for large graphs with localized changes
void Distributed_UpdateAffected_MPI_Sparse(
    const Graph& local_graph,
    const BoundaryEdges& local_boundary_edges,
    const std::vector<int>& local_to_global,
    std::vector<double>& dist,
    std::vector<int>& parent,
    int my_rank,
    const std::vector<int>& part,
    int source_global_id,
    bool debug_output
) {
    int n_local = local_graph.num_vertices;
    int n_global = part.size();
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    
    if (my_rank == 0) std::cout << "[Rank 0] Sparse UpdateAffected: Starting with n_local=" << n_local << ", n_global=" << n_global << std::endl;
    
    bool local_changed = true;
    int iteration = 0;
    int consecutive_no_changes = 0;
    const int early_term_threshold = 3;
    
    // Current global state - maintained locally on each rank
    std::vector<double> global_dist(n_global, INFINITY_WEIGHT);
    
    // Track which vertices changed in each iteration
    std::vector<bool> changed_vertices(n_local, true); // Initially all are considered changed
    
    // Create auxiliary data for sparse communication
    std::vector<int> send_counts(num_ranks, 0);
    std::vector<int> send_displs(num_ranks, 0);
    std::vector<int> recv_counts(num_ranks, 0);
    std::vector<int> recv_displs(num_ranks, 0);
    
    // Temporary buffers for sparse communication
    std::vector<int> send_indices;     // Global indices to send
    std::vector<double> send_values;   // Values to send
    std::vector<int> recv_indices;     // Global indices received
    std::vector<double> recv_values;   // Values received
    
    // Priority queue for processing vertices in distance order
    struct VertexDist {
        int vertex;
        double distance;
        bool operator>(const VertexDist& other) const { return distance > other.distance; }
    };
    
    // Main iteration loop
    while (true) {
        iteration++;
        local_changed = false;
        
        // --- Prepare data to send (only changed vertices) ---
        send_indices.clear();
        send_values.clear();
        
        for (int u_local = 0; u_local < n_local; ++u_local) {
            if (changed_vertices[u_local]) {
                int u_global = local_to_global[u_local];
                send_indices.push_back(u_global);
                send_values.push_back(dist[u_local]);
            }
        }
        
        // Reset change tracking for next iteration
        std::fill(changed_vertices.begin(), changed_vertices.end(), false);
        
        // --- Communicate the size of data each rank will send ---
        int send_size = send_indices.size();
        MPI_Allgather(&send_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        // Calculate displacements for Allgatherv
        recv_displs[0] = 0;
        for (int i = 1; i < num_ranks; i++) {
            recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
        }
        
        // Calculate total receive size
        int total_recv_size = 0;
        for (int i = 0; i < num_ranks; i++) {
            total_recv_size += recv_counts[i];
        }
        
        // Allocate receive buffers
        recv_indices.resize(total_recv_size);
        recv_values.resize(total_recv_size);
        
        // Exchange the indices of changed vertices
        MPI_Allgatherv(send_indices.data(), send_size, MPI_INT,
                      recv_indices.data(), recv_counts.data(), recv_displs.data(),
                      MPI_INT, MPI_COMM_WORLD);
        
        // Exchange the values of changed vertices
        MPI_Allgatherv(send_values.data(), send_size, MPI_DOUBLE,
                      recv_values.data(), recv_counts.data(), recv_displs.data(),
                      MPI_DOUBLE, MPI_COMM_WORLD);
        
        // --- Update global state with received data ---
        for (int i = 0; i < total_recv_size; i++) {
            int g_idx = recv_indices[i];
            double g_val = recv_values[i];
            
            if (g_idx >= 0 && g_idx < n_global && g_val < global_dist[g_idx]) {
                global_dist[g_idx] = g_val;
            }
        }
        
        // Check if source vertex has correct distance
        if (source_global_id >= 0 && source_global_id < n_global) {
            global_dist[source_global_id] = 0.0;
        }
        
        // Verify number of significant changes for early termination
        if (total_recv_size < n_global / 100) { // Less than 1% of vertices changed
            consecutive_no_changes++;
        } else {
            consecutive_no_changes = 0;
        }
        
        // --- Self-Invalidation Pass ---
        bool changed_by_self_invalidation = false;
        #pragma omp parallel for schedule(dynamic) reduction(||:changed_by_self_invalidation)
        for (int u_local = 0; u_local < n_local; ++u_local) {
            int p_global = parent[u_local]; // Current parent (global ID)
            if (p_global != -1 && p_global >= 0 && p_global < n_global) { // Valid parent
                if (global_dist[p_global] == INFINITY_WEIGHT) { // Parent became INF
                    if (dist[u_local] != INFINITY_WEIGHT) {
                        dist[u_local] = INFINITY_WEIGHT;
                        parent[u_local] = -1;
                        changed_vertices[u_local] = true;
                        changed_by_self_invalidation = true;
                    }
                }
            }
        }
        
        if (changed_by_self_invalidation) {
            local_changed = true;
        }
        
        // --- Local Relaxation Step using priority queue ---
        std::vector<VertexDist> priority_list;
        for (int u_local = 0; u_local < n_local; ++u_local) {
            if (dist[u_local] != INFINITY_WEIGHT) {
                priority_list.push_back({u_local, dist[u_local]});
            }
        }
        
        std::sort(priority_list.begin(), priority_list.end(), 
                 [](const VertexDist& a, const VertexDist& b) { return a.distance < b.distance; });
        
        // Process vertices in priority order
        bool changed_this_pass = false;
        #pragma omp parallel for schedule(dynamic, 16) reduction(||:changed_this_pass)
        for (size_t i = 0; i < priority_list.size(); ++i) {
            int u_local = priority_list[i].vertex;
            double dist_u = dist[u_local];
            if (dist_u == INFINITY_WEIGHT) continue;
            
            // Internal edges relaxation
            for (const auto& edge : local_graph.neighbors(u_local)) {
                const int v_local = edge.to;
                if (const double weight = edge.weight; dist_u + weight < dist[v_local]) {
                    #pragma omp atomic write
                    dist[v_local] = dist_u + weight;
                    parent[v_local] = local_to_global[u_local];
                    changed_vertices[v_local] = true;
                    changed_this_pass = true;
                }
            }
            
            // Boundary edges relaxation
            if (auto boundary_it = local_boundary_edges.find(u_local); boundary_it != local_boundary_edges.end()) {
                for (const auto& [to, weight] : boundary_it->second) {
                    const int remote_u = to;
                    const double w = weight;
                    if (const double remote_dist = global_dist[remote_u]; remote_dist + w < dist[u_local]) {
                        #pragma omp atomic write
                        dist[u_local] = remote_dist + w;
                        parent[u_local] = remote_u;
                        changed_vertices[u_local] = true;
                        changed_this_pass = true;
                    }
                }
            }
        }
        
        local_changed = local_changed || changed_this_pass;
        
        // --- Global Convergence Check ---
        int local_flag = local_changed ? 1 : 0;
        int global_flag = 0;
        MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        
        if (my_rank == 0) {
            std::cout << "[Rank 0] Sparse UpdateAffected Iteration " << iteration 
                     << ", Global Change Flag: " << global_flag 
                     << ", Changed vertices: " << total_recv_size << " (" 
                     << (100.0 * total_recv_size / n_global) << "%)" << std::endl;
        }
        
        // Check for termination
        if (global_flag == 0 || (consecutive_no_changes >= early_term_threshold && iteration > 3)) {
            if (consecutive_no_changes >= early_term_threshold && my_rank == 0) {
                std::cout << "[Rank 0] Sparse update early termination after " << iteration 
                         << " iterations. Few changes for " << consecutive_no_changes 
                         << " consecutive iterations." << std::endl;
            }
            break;
        }
        
        // Safeguard against infinite loops
        if (iteration > std::min(1000, n_global + 1)) {
            if (my_rank == 0) {
                std::cout << "[Rank 0] Maximum iterations reached in sparse update." << std::endl;
            }
            break;
        }
    }
    
    if (my_rank == 0) {
        std::cout << "[Rank 0] Sparse UpdateAffected finished after " << iteration << " iterations." << std::endl;
    }
}