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
    // Step 1: Mark directly affected vertices locally
    std::vector<bool> affected_global(n_global, false);
    #pragma omp parallel for schedule(static)
    for (const auto& change : changes) {
        if (change.type == ChangeType::DELETE || change.type == ChangeType::INCREASE) { // Deletion or weight increase
            int u = change.u, v = change.v;
            if (u >= 0 && u < n_global) affected_global[u] = true;
            if (v >= 0 && v < n_global) affected_global[v] = true;
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

// Forward declaration for asynchronous update variant (Algorithm 4)
void Distributed_UpdateAffected_MPI_Async(
    const Graph& local_graph,
    const BoundaryEdges& local_boundary_edges,
    const std::vector<int>& local_to_global,
    std::vector<double>& dist,
    std::vector<int>& parent,
    int my_rank,
    const std::vector<int>& part,
    bool debug_output
);

// Distributed_UpdateAffected_MPI: iteratively relax local and boundary edges across ranks
// Inputs:
//   local_graph           : subgraph owned by this rank
//   local_boundary_edges  : incoming edges from remote vertices
//   local_to_global       : map from local index to global ID
//   dist, parent          : local SSSP arrays (updated in-place)
//   my_rank, part         : MPI rank and global partition mapping
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
    bool debug_output
) {
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

    // Allocate buffers with safety checks
    std::vector<double> send_dist_buffer;
    std::vector<double> current_global_dist;

    try {
        // Buffer to hold distances from this rank to be sent in Allreduce
        send_dist_buffer.resize(n_global, INFINITY_WEIGHT);
        // Buffer to receive the minimum distances across all ranks for all global vertices
        current_global_dist.resize(n_global, INFINITY_WEIGHT);
    } catch (const std::bad_alloc& e) {
        std::cerr << "[Rank " << my_rank << "] ERROR: Memory allocation failed: " << e.what()
                  << ". Try reducing problem size or using more processes." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    // Check for asynchronous update variant via environment variable
    if (char* async_env = getenv("SSSP_ASYNC"); async_env != nullptr &&
        (strcmp(async_env, "1") == 0 || strcmp(async_env, "true") == 0 || strcmp(async_env, "yes") == 0)) {
        if (my_rank == 0) std::cout << "[Rank 0] Using asynchronous update variant (Algorithm 4)" << std::endl;
        Distributed_UpdateAffected_MPI_Async(local_graph, local_boundary_edges, local_to_global,
                                            dist, parent, my_rank, part, debug_output);
        return;
    }

    while (true) {
        iteration++;
        local_changed_in_iter = false;

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
        // 1. Prepare local distances for sending (map local -> global)
        std::fill(send_dist_buffer.begin(), send_dist_buffer.end(), INFINITY_WEIGHT);
        if (n_local > 0) {
            #pragma omp parallel for schedule(static)
            for (int u_local = 0; u_local < n_local; ++u_local) {
                int u_global = local_to_global[u_local];
                // Ensure u_global is valid before accessing send_dist_buffer
                if (u_global >= 0 && u_global < n_global) {
                    send_dist_buffer[u_global] = dist[u_local];
                } else {
                     // This case should ideally not happen if mappings are correct
                     if (my_rank == 0) std::cerr << "Warning: Invalid global index " << u_global << " for local index " << u_local << " in rank " << my_rank << std::endl;
                }
            }
        }

        // 2. Perform Allreduce to get the minimum distance for each global vertex across all ranks
        // Added error handling for MPI operations
        int mpi_result = MPI_Allreduce(send_dist_buffer.data(), current_global_dist.data(), n_global, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        if (mpi_result != MPI_SUCCESS) {
            std::cerr << "[Rank " << my_rank << "] MPI_Allreduce failed. Aborting." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, mpi_result);
            return;
        }

        // --- Local Relaxation Step --- //
        bool changed_this_pass = false;

        // Parallelize per-vertex relaxation using OpenMP
        #pragma omp parallel for schedule(dynamic) reduction(||:changed_this_pass)
        for (int u_local = 0; u_local < n_local; ++u_local) {
            double dist_u = dist[u_local];
            if (dist_u == INFINITY_WEIGHT) continue;

            // Internal edges relaxation
            #pragma omp parallel for schedule(dynamic) reduction(+:changed_this_pass)
            for (const auto& edge : local_graph.neighbors(u_local)) {
                const int v_local = edge.to;
                if (const double weight = edge.weight; dist_u + weight < dist[v_local]) {
                    dist[v_local] = dist_u + weight;
                    parent[v_local] = local_to_global[u_local];
                    changed_this_pass = true;
                }
            }

            // Boundary edges relaxation
            if (auto boundary_it = local_boundary_edges.find(u_local); boundary_it != local_boundary_edges.end()) {
                for (const auto& [to, weight] : boundary_it->second) {
                    const int remote_u = to;
                    const double w = weight;
                    if (const double remote_dist = current_global_dist[remote_u]; remote_dist + w < dist[u_local]) {
                        dist[u_local] = remote_dist + w;
                        parent[u_local] = remote_u;
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

        // Check for convergence
        if (global_flag == 0) {
            break; // No rank made any changes in this iteration
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
    bool debug_output
) {
    int n_local = local_graph.num_vertices;
    int n_global = part.size();
    std::vector<double> send_buf(n_global, INFINITY_WEIGHT);
    std::vector<double> global_buf(n_global, INFINITY_WEIGHT);
    MPI_Request req;
    bool local_changed = true;
    int iteration = 0;

    while (true) {
        iteration++;
        // prepare send buffer from local dist
        std::fill(send_buf.begin(), send_buf.end(), INFINITY_WEIGHT);
        for (int u_local = 0; u_local < n_local; ++u_local) {
            int u_global = local_to_global[u_local];
            if (u_global >= 0 && u_global < n_global) send_buf[u_global] = dist[u_local];
        }
        // start non-blocking allreduce
        MPI_Iallreduce(send_buf.data(), global_buf.data(), n_global, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD, &req);

        // local relaxations using current global_buf (stale values)
        local_changed = false;
        for (int u_local = 0; u_local < n_local; ++u_local) {
            double du = dist[u_local];
            if (du == INFINITY_WEIGHT) continue;
            // internal edges
            for (const auto& e : local_graph.neighbors(u_local)) {
                if (du + e.weight < dist[e.to]) {
                    dist[e.to] = du + e.weight;
                    parent[e.to] = local_to_global[u_local];
                    local_changed = true;
                }
            }
            // boundary edges
            if (auto it = local_boundary_edges.find(u_local); it != local_boundary_edges.end()) {
                for (const auto& be : it->second) {
                    int g_to = be.to;
                    double w = be.weight;
                    if (global_buf[g_to] + w < dist[u_local]) {
                        dist[u_local] = global_buf[g_to] + w;
                        parent[u_local] = g_to;
                        local_changed = true;
                    }
                }
            }
        }

        // wait for global reduction result
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        // update local dist with fresh global_buf
        for (int u_local = 0; u_local < n_local; ++u_local) {
            int u_global = local_to_global[u_local];
            double gd = global_buf[u_global];
            if (gd < dist[u_local]) {
                dist[u_local] = gd;
                parent[u_local] = u_global;
                local_changed = true;
            }
        }

        // convergence check across ranks
        int flag = local_changed ? 1 : 0;
        int global_flag = 0;
        MPI_Allreduce(&flag, &global_flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (global_flag == 0) break;
        if (iteration > n_global + 1) break;
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
    int source,
    const std::vector<bool>& initial_affected_del,
    const std::vector<bool>& initial_affected,
    bool debug_output
) {
    const int n_local = local_graph.num_vertices;

    // Use initial_affected instead of recalculating from scratch
    std::vector<bool> affected = initial_affected;

    // Invalidate affected vertices and descendants
    std::cout << "[Rank " << my_rank << "] Invalidating locally affected vertices..." << std::endl;
    int invalidated_count = 0;
    for (int u_local = 0; u_local < n_local; ++u_local) {
        // Use initial_affected_del to decide which vertices to fully invalidate
        if (u_local < initial_affected_del.size() && initial_affected_del[u_local]) {
            dist[u_local] = INFINITY_WEIGHT;
            parent[u_local] = -1;
            if (u_local < affected.size()) affected[u_local] = true; // Ensure marked for update
            invalidated_count++;
        }
        // Ensure vertices only in initial_affected are also marked
        else if (u_local < initial_affected.size() && initial_affected[u_local]) {
            if (u_local < affected.size()) affected[u_local] = true;
        }
    }
    std::cout << "[Rank " << my_rank << "] Invalidated " << invalidated_count << " vertices based on initial_affected_del." << std::endl;

    // Re-initialize the source node if it is local
    if (source >= 0 && source < static_cast<int>(global_to_local.size()) && global_to_local[source] != -1) {
        if (const int src_local = global_to_local[source]; src_local >= 0 && src_local < n_local) { // Bounds check
            dist[src_local] = 0.0;
            parent[src_local] = -1; // Source has no parent
            if (src_local < affected.size()) affected[src_local] = true; // Source might need to propagate updates
            std::cout << "[Rank " << my_rank << "] Re-initialized local source vertex " << src_local << " (global " << source << ")." << std::endl;
        }
    }

    // Update affected vertices (full reconnection)
    Distributed_UpdateAffected_MPI(
        local_graph,                // Input: local graph structure
        local_boundary_edges,       // Input: boundary edges map
        local_to_global,            // Input: mapping local index -> global ID
        dist,                    // Input/Output: local distances (in/out)
        parent,                  // Input/Output: local parents (in/out)
        my_rank,                    // Current MPI rank
        part,                       // Global partition vector
        debug_output                // Pass debug_output parameter to control verbose output
    );

    std::cout << "[Rank " << my_rank << "] Exiting Distributed_DynamicSSSP_MPI." << std::endl;
}