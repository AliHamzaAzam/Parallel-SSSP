#include "../../include/graph.hpp"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm> // For std::min
#include "../../include/utils.hpp"
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <queue> // Added for Incremental_SSSP_MPI
#include <map>   // Added for BoundaryEdges type alias
#include <omp.h> // For potential OpenMP usage

// Type alias for boundary edges (must match main_mpi.cpp)
// Assumed structure: map<local_vertex_idx, vector<Edge{remote_global_vertex_idx, weight}>>
// i.e., maps a local vertex to the list of incoming edges from remote vertices.
using BoundaryEdges = std::map<int, std::vector<Edge>>;

// Forward declare the global graph variable (assuming it's accessible here, which is NOT ideal design)
// A better design would pass necessary global graph info explicitly.

// Structure to find minimum distance and its owner rank using MPI_MINLOC
struct MinLocResult {
    Weight dist;
    int rank;
};

// Placeholder implementation for the MPI SSSP function
// Replace this with your actual MPI SSSP logic.
void SSSP_MPI(const Graph& graph, int source, SSSPResult& result, int argc, char* argv[]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = graph.num_vertices;
    std::vector<Weight>& dist = result.dist; // Use the result's dist vector directly
    std::vector<int>& parent = result.parent; // Use the result's parent vector
    std::vector<bool> visited(n, false); // Keep track of visited vertices globally

    // Initialize distances: All infinity, source is 0
    // This happens on all processes, but only the owner's value matters initially
    std::fill(dist.begin(), dist.end(), INFINITY_WEIGHT);
    std::fill(parent.begin(), parent.end(), -1);
    if (source >= 0 && source < n) {
         // Everyone sets the source distance, simplifies logic slightly
         dist[source] = 0;
    } else if (rank == 0) {
        std::cerr << "Error: Invalid source vertex " << source << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1); // Abort all processes
        return; // Should not be reached
    }

    // --- Distributed Dijkstra Main Loop ---
    for (int i = 0; i < n; ++i) {
        // 1. Find local minimum among unvisited vertices owned by this process
        Weight local_min_dist = INFINITY_WEIGHT;
        int local_u = -1;
        for (int u = 0; u < n; ++u) {
            // Check ownership (cyclic partitioning) and if not visited
            if (u % size == rank && !visited[u]) {
                if (dist[u] < local_min_dist) {
                    local_min_dist = dist[u];
                    local_u = u;
                }
            }
        }

        // 2. Find global minimum distance and the rank owning it
        MinLocResult local_min = {local_min_dist, rank};
        MinLocResult global_min;

        // Use MPI_MINLOC to find the minimum distance and the rank of the process holding it
        // Need to define a custom MPI_Op or use MPI_DOUBLE_INT if Weight is double
        // Assuming Weight is double for MPI_DOUBLE_INT
        static_assert(std::is_same<Weight, double>::value, "Weight must be double for MPI_DOUBLE_INT");
        MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

        // 3. Check for termination (no reachable vertices left)
        if (global_min.dist == INFINITY_WEIGHT) {
            break; // All remaining vertices are unreachable
        }

        // 4. Get the global minimum vertex index 'u_glob'
        // The process owning the global minimum needs to broadcast it
        int u_glob = -1; // Initialize u_glob
        if (rank == global_min.rank) {
            u_glob = local_u; // This rank found the global minimum vertex
        }
        // Broadcast the global minimum vertex index from the owner rank
        MPI_Bcast(&u_glob, 1, MPI_INT, global_min.rank, MPI_COMM_WORLD);

        // 5. Mark the global minimum vertex as visited (all processes)
        if (u_glob != -1) { // Ensure a valid vertex was found
             visited[u_glob] = true;
        } else {
             // Should not happen if global_min.dist wasn't INFINITY
             if (rank == 0) std::cerr << "Error: u_glob is -1 despite non-infinite distance." << std::endl;
             MPI_Abort(MPI_COMM_WORLD, 1);
        }


        // 6. Relax edges outgoing from u_glob (all processes do this)
        // Need access to neighbors of u_glob. Graph is replicated.
        if (u_glob != -1) { // Check again for safety
            // Use the const version of neighbors
            for (const auto& edge : graph.neighbors(u_glob)) {
                int v = edge.to;
                Weight weight = edge.weight;
                // Check if v is already visited
                if (!visited[v]) {
                    Weight new_dist = global_min.dist + weight;
                    // Update distance if a shorter path is found
                    // No need to check ownership here, as dist is replicated (for now)
                    if (new_dist < dist[v]) {
                        dist[v] = new_dist;
                        parent[v] = u_glob;
                        // Note: In a more optimized version with distributed dist,
                        // the owner of 'v' would update its local value.
                        // Here, all processes update their copy of dist[v].
                    }
                }
            }
        }
    } // End of Dijkstra main loop

    // Result is already in result.dist and result.parent on all processes
    // because we worked on replicated data structures.
    // No final gather needed with this approach.

    // Optional: Barrier for synchronization before exiting
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "[SSSP_MPI] Computation finished." << std::endl;
        // You might want to print results or timing here
        // Timing 
        

    }
}

void Incremental_SSSP_MPI(Graph& graph, int source, SSSPResult& result, const std::vector<EdgeChange>& updates, int argc, char* argv[]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = graph.num_vertices;
    std::vector<Weight>& dist = result.dist;
    std::vector<int>& parent = result.parent;

    // Step 1: Apply updates to the graph (all processes do this)
    for (const auto& update : updates) {
        if (update.type == ChangeType::INSERT || update.type == ChangeType::DECREASE) { // Insertion or weight decrease
            // Handle insertion
            graph.add_edge(update.u, update.v, update.weight);
        } else {
            // Handle deletion
            graph.remove_edge(update.u, update.v);
        }
    }

    // Step 2: Identify affected vertices (all processes do this)
    std::queue<int> affected_vertices;
    std::vector<bool> in_queue(n, false);

    for (const auto& update : updates) {
        if (!in_queue[update.u]) {
            affected_vertices.push(update.u);
            in_queue[update.u] = true;
        }
        if (!in_queue[update.v]) {
            affected_vertices.push(update.v);
            in_queue[update.v] = true;
        }
    }

    // Step 3: Incrementally update shortest paths
    while (!affected_vertices.empty()) {
        int u = affected_vertices.front();
        affected_vertices.pop();
        in_queue[u] = false;

        for (const auto& edge : graph.neighbors(u)) {
            int v = edge.to;
            Weight weight = edge.weight;

            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                parent[v] = u;

                if (!in_queue[v]) {
                    affected_vertices.push(v);
                    in_queue[v] = true;
                }
            }
        }
    }

    // Optional: Barrier for synchronization before exiting
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "[Incremental_SSSP_MPI] Incremental computation finished." << std::endl;
    }
}

// Distributed Bellman-Ford SSSP for partitioned graphs (MPI + OpenMP)
void Distributed_BellmanFord_MPI(
    Graph& local_graph,
    const std::vector<int>& local_to_global, // local vertex id -> global id
    const std::vector<int>& global_to_local, // global vertex id -> local id (or -1)
    const std::vector<idx_t>& part, // partition vector
    int my_rank,
    int num_ranks,
    int global_source,
    std::vector<double>& global_dist // Output: global distances (gathered on rank 0)
) {
    int n_local = local_graph.num_vertices;
    std::vector<double> dist(n_local, INFINITY_WEIGHT);
    // Set local source if owned
    if (global_to_local[global_source] != -1) {
        dist[global_to_local[global_source]] = 0.0;
    }
    bool local_changed = true;
    bool global_changed = true;
    int n_global = part.size();
    // Identify boundary vertices (those with neighbors in other ranks)
    std::set<int> boundary_globals;
    for (int u_local = 0; u_local < n_local; ++u_local) {
        int u_global = local_to_global[u_local];
        for (const auto& edge : local_graph.neighbors(u_local)) {
            int v_global = edge.to < local_to_global.size() ? local_to_global[edge.to] : -1;
            if (v_global != -1 && part[v_global] != my_rank) {
                boundary_globals.insert(u_global);
            }
        }
    }
    // Main Bellman-Ford loop
    while (true) {
        local_changed = false;
        // Local relaxation (OpenMP parallel)
        #pragma omp parallel for schedule(dynamic)
        for (int u_local = 0; u_local < n_local; ++u_local) {
            for (const auto& edge : local_graph.neighbors(u_local)) {
                int v_local = edge.to;
                double weight = edge.weight;
                if (dist[u_local] + weight < dist[v_local]) {
                    #pragma omp critical
                    {
                        if (dist[u_local] + weight < dist[v_local]) {
                            dist[v_local] = dist[u_local] + weight;
                            local_changed = true;
                        }
                    }
                }
            }
        }
        // Exchange boundary distances with all ranks
        std::vector<double> sendbuf(n_global, INFINITY_WEIGHT);
        for (int u_local = 0; u_local < n_local; ++u_local) {
            int u_global = local_to_global[u_local];
            sendbuf[u_global] = dist[u_local];
        }
        std::vector<double> recvbuf(n_global, INFINITY_WEIGHT);
        MPI_Allreduce(sendbuf.data(), recvbuf.data(), n_global, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        // Update local distances with received values
        for (int u_local = 0; u_local < n_local; ++u_local) {
            int u_global = local_to_global[u_local];
            if (recvbuf[u_global] < dist[u_local]) {
                dist[u_local] = recvbuf[u_global];
                local_changed = true;
            }
        }
        // Check for global convergence
        int local_flag = local_changed ? 1 : 0;
        int global_flag = 0;
        MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (global_flag == 0) break;
    }
    // Gather global distances on rank 0
    std::vector<double> local_dist_out(n_global, INFINITY_WEIGHT);
    for (int u_local = 0; u_local < n_local; ++u_local) {
        int u_global = local_to_global[u_local];
        local_dist_out[u_global] = dist[u_local];
    }
    if (my_rank == 0) {
        global_dist.resize(n_global, INFINITY_WEIGHT);
    }
    MPI_Reduce(local_dist_out.data(), my_rank == 0 ? global_dist.data() : nullptr, n_global, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
}

// Distributed Algorithm 2: Identify affected vertices (deletions/insertions) with descendant propagation
void Distributed_IdentifyAffected_MPI(
    const Graph& local_graph,
    const std::vector<int>& local_to_global,
    const std::vector<int>& global_to_local,
    const std::vector<EdgeChange>& changes,
    std::vector<bool>& affected,
    const std::vector<int>& parent, // Use current parent array for propagation
    int my_rank,
    int num_ranks,
    const std::vector<idx_t>& part
) {
    int n_local = local_graph.num_vertices;
    int n_global = part.size();
    // Step 1: Mark directly affected vertices locally
    std::vector<bool> affected_global(n_global, false);
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

// Struct for distance and parent for MPI reduction
struct DistParent {
    double dist;
    int parent;
};

// Custom MPI reduction for DistParent (min distance, update parent)
void reduce_min_distparent(void* in, void* inout, int* len, MPI_Datatype* datatype) {
    DistParent* in_vals = (DistParent*)in;
    DistParent* inout_vals = (DistParent*)inout;
    for (int i = 0; i < *len; ++i) {
        if (in_vals[i].dist < inout_vals[i].dist) {
            inout_vals[i] = in_vals[i];
        }
    }
}

// Distributed Algorithm 3: Update affected vertices (Bellman-Ford style, using local_boundary_edges)
void Distributed_UpdateAffected_MPI(
    Graph& local_graph,
    const BoundaryEdges& local_boundary_edges, // Map: local_v -> vector<Edge{remote_u_global, weight}>
    const std::vector<int>& local_to_global,
    const std::vector<int>& global_to_local, // Needed to map global parent back to local if necessary (though not used in current relaxation)
    std::vector<double>& dist, // Local distances (in/out)
    std::vector<int>& parent, // Local parents (in/out) - stores local index or global index
    std::vector<bool>& affected, // Note: This isn't actually used in the current relaxation logic, but kept for signature compatibility
    int my_rank,
    int num_ranks,
    const std::vector<int>& part // Global partition array (int, not idx_t)
) {
    int n_local = local_graph.num_vertices;
    int n_global = part.size(); // Get global size from partition array
    if (my_rank == 0) std::cout << "[Rank 0] UpdateAffected: Starting. n_local=" << n_local << ", n_global=" << n_global << std::endl;

    bool local_changed_in_iter = true;
    int iteration = 0;

    // Buffer to hold distances from this rank to be sent in Allreduce
    std::vector<double> send_dist_buffer(n_global, INFINITY_WEIGHT);
    // Buffer to receive the minimum distances across all ranks for all global vertices
    std::vector<double> current_global_dist(n_global, INFINITY_WEIGHT);

    while (true) {
        iteration++;
        local_changed_in_iter = false;

        // --- DEBUG PRINT: State at start of iteration (Rank 3 only) ---
        if (my_rank == 3) {
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
        std::fill(current_global_dist.begin(), current_global_dist.end(), INFINITY_WEIGHT); // Reset receive buffer
        MPI_Allreduce(send_dist_buffer.data(), current_global_dist.data(), n_global, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        // --- Local Relaxation Step --- //
        bool changed_this_pass = false;
        if (n_local > 0) {
            // Iterate through each local vertex v_local
            for (int v_local = 0; v_local < n_local; ++v_local) {
                int v_global = local_to_global[v_local]; // Global ID of the current local vertex
                double current_dist_v = dist[v_local]; // Current best distance to v_local
                int current_parent_v = parent[v_local]; // Current parent of v_local (can be local index or global index)
                bool updated = false;

                // A) Relax using incoming INTERNAL edges (u_local -> v_local)
                // Iterate through potential local predecessors u_local
                // Note: This requires iterating through the adjacency list *backwards* or having a reverse graph.
                // A more efficient way is to iterate through all local u and relax its outgoing edges.
                // Let's switch to the standard forward relaxation approach:
                // Iterate through all local u_local and relax edges u_local -> v_local where v_local is also local.

                // B) Relax using incoming BOUNDARY edges (u_global_remote -> v_local)
                // Use the precomputed local_boundary_edges map.
                // Key: v_local (local destination), Value: vector of Edge{u_global_remote, weight}
                auto boundary_it = local_boundary_edges.find(v_local);
                if (boundary_it != local_boundary_edges.end()) {
                    const auto& incoming_edges = boundary_it->second;
                    for (const auto& edge : incoming_edges) {
                        int u_global_remote = edge.to; // The 'to' field stores the global source vertex ID
                        double weight = edge.weight;
                        double dist_u_global = INFINITY_WEIGHT;

                        // Get the current best distance to the remote source u_global_remote from the gathered data
                        if (u_global_remote >= 0 && u_global_remote < n_global) {
                             dist_u_global = current_global_dist[u_global_remote];
                        } else {
                             if (my_rank == 0) std::cerr << "Warning: Invalid remote global source index " << u_global_remote << " in boundary edge for local vertex " << v_local << " (global " << v_global << ") in rank " << my_rank << std::endl;
                             continue; // Skip this edge if the source index is invalid
                        }

                        // Relaxation check
                        if (dist_u_global != INFINITY_WEIGHT && dist_u_global + weight < current_dist_v) {
                            current_dist_v = dist_u_global + weight;
                            // Parent is remote, store its global ID.
                            current_parent_v = u_global_remote;
                            updated = true;
                            // Debug log (optional)
                            // std::cout << "[Rank " << my_rank << "] Relax Boundary: " << u_global_remote << " -> " << v_local << " (" << v_global << ") | NewDist=" << current_dist_v << " | OldDist=" << dist[v_local] << std::endl;
                        }
                    }
                }

                // Update the local distance and parent if relaxation occurred *from boundary edges*
                // We need to handle internal edges separately.
                // Let's restructure the relaxation: Iterate through local U, relax its outgoing edges.

            } // end loop over v_local (OLD STRUCTURE - TO BE REPLACED)


            // --- REVISED Local Relaxation Step --- //
            // Iterate through all local source vertices u_local
            for (int u_local = 0; u_local < n_local; ++u_local) {
                 double dist_u = dist[u_local];
                 if (dist_u == INFINITY_WEIGHT) continue; // Cannot relax from infinity

                 // A) Relax outgoing INTERNAL edges (u_local -> v_local)
                 for (const auto& edge : local_graph.neighbors(u_local)) {
                     int v_local = edge.to; // Destination is local
                     double weight = edge.weight;
                     if (dist_u + weight < dist[v_local]) {
                         // --- DEBUG PRINT: Internal Relaxation (Rank 3 only) ---
                         if (my_rank == 3) {
                             std::cout << "[Rank 3 DEBUG] Iter " << iteration << " Relax Internal: "
                                       << local_to_global[u_local] << " -> " << local_to_global[v_local]
                                       << " | NewDist=" << (dist_u + weight) << " | OldDist=" << dist[v_local] << std::endl;
                         }
                         // --- END DEBUG ---
                         dist[v_local] = dist_u + weight;
                         // Store the GLOBAL ID of the parent
                         if (u_local >= 0 && u_local < local_to_global.size()) { // Bounds check
                            parent[v_local] = local_to_global[u_local];
                         } else {
                            parent[v_local] = -1; // Should not happen
                         }
                         changed_this_pass = true;
                         // Debug log (optional)
                         // std::cout << "[Rank " << my_rank << "] Relax Internal: " << u_local << " (" << local_to_global[u_local] << ") -> " << v_local << " (" << local_to_global[v_local] << ") | NewDist=" << dist[v_local] << std::endl;
                     }
                 }

                 // B) Relax outgoing BOUNDARY edges (u_local -> v_global_remote)
                 // This step is implicitly handled by other ranks relaxing their *incoming* boundary edges
                 // using the distances gathered via MPI_Allreduce. We don't need to explicitly relax outgoing
                 // boundary edges here because the effect is captured when the destination rank (which owns v_global_remote)
                 // performs its relaxation step using the `current_global_dist` which includes our `dist_u`.

                 // C) Relax incoming BOUNDARY edges (u_global_remote -> u_local)
                 // This needs to be done for the current u_local as the destination.
                 int u_global = local_to_global[u_local]; // Global ID of the current local vertex u_local
                 auto boundary_it = local_boundary_edges.find(u_local); // Find incoming edges to u_local
                 if (boundary_it != local_boundary_edges.end()) {
                     const auto& incoming_edges = boundary_it->second;
                     for (const auto& edge : incoming_edges) {
                         int source_global_remote = edge.to; // Global ID of the remote source
                         double weight = edge.weight;
                         double dist_source_global = INFINITY_WEIGHT;

                         // Get distance to remote source from gathered data
                         if (source_global_remote >= 0 && source_global_remote < n_global) {
                             dist_source_global = current_global_dist[source_global_remote];
                         } else {
                              if (my_rank == 0) std::cerr << "Warning: Invalid remote global source index " << source_global_remote << " in boundary edge for local vertex " << u_local << " (global " << u_global << ") in rank " << my_rank << std::endl;
                              continue;
                         }

                         // Relaxation check
                         if (dist_source_global != INFINITY_WEIGHT && dist_source_global + weight < dist[u_local]) {
                             // --- DEBUG PRINT: Boundary Relaxation (Rank 3 only) ---
                             if (my_rank == 3) {
                                 std::cout << "[Rank 3 DEBUG] Iter " << iteration << " Relax Boundary: "
                                           << source_global_remote << " -> " << u_global
                                           << " | NewDist=" << (dist_source_global + weight) << " | OldDist=" << dist[u_local] << std::endl;
                             }
                             // --- END DEBUG ---
                             dist[u_local] = dist_source_global + weight;
                             parent[u_local] = source_global_remote; // Parent is remote, store global ID
                             changed_this_pass = true;
                         }
                     }
                 }
            } // end loop over u_local (REVISED STRUCTURE)

        } // end if (n_local > 0)

        local_changed_in_iter = changed_this_pass;

        // --- DEBUG PRINT: Changed Flag (Rank 3 only) ---
        if (my_rank == 3) {
            std::cout << "[Rank 3 DEBUG] Iter " << iteration << " End | changed_this_pass=" << (changed_this_pass ? "true" : "false") << std::endl;
        }
        // --- END DEBUG ---


        // --- Global Convergence Check --- //
        int local_flag = local_changed_in_iter ? 1 : 0;
        int global_flag = 0;
        MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

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
    } // end while loop

    std::cout << "[Rank " << my_rank << "] UpdateAffected: Finished after " << iteration << " iterations." << std::endl;
}

// Distributed dynamic SSSP update: applies changes and updates SSSP
void Distributed_DynamicSSSP_MPI(
    Graph& local_graph,
    const BoundaryEdges& local_boundary_edges, // Add boundary edges map
    const std::vector<int>& local_to_global,
    const std::vector<int>& global_to_local,
    std::vector<double>& dist,
    std::vector<int>& parent,
    const std::vector<EdgeChange>& changes,
    int my_rank,
    int num_ranks,
    const std::vector<int>& part, // Changed idx_t to int
    int source, // <-- new argument
    const std::vector<bool>& initial_affected_del, // Added missing parameter
    const std::vector<bool>& initial_affected      // Added missing parameter
) {
    int n_local = local_graph.num_vertices;
    int n_global = part.size(); // Keep this definition

    // 1. Apply changes to local subgraph (insertions/deletions)
    // This section seems incorrect based on the logic in main_mpi.cpp
    // where changes are applied globally before partitioning.
    // If changes are applied *before* scattering, this local application is redundant
    // and potentially harmful if local_graph doesn't perfectly match the global state.
    // Commenting out for now, assuming main_mpi.cpp handles changes correctly before this call.
    /*
    for (const auto& change : changes) {
        int u = change.u, v = change.v;
        int u_local = (u >= 0 && u < (int)global_to_local.size()) ? global_to_local[u] : -1;
        int v_local = (v >= 0 && v < (int)global_to_local.size()) ? global_to_local[v] : -1;
        if (change.type == ChangeType::INSERT || change.type == ChangeType::DECREASE) { // Insertion or weight decrease
            if (u_local != -1 && v_local != -1) {
                // Check if edge already exists before adding? Graph::add_edge might handle this.
                local_graph.add_edge(u_local, v_local, change.weight);
            }
        } else { // Deletion or weight increase (treat increase as delete+insert?)
                 // The logic here might need refinement depending on how increases are handled.
            if (u_local != -1 && v_local != -1) {
                local_graph.remove_edge(u_local, v_local);
            }
        }
    }
    */

    // 2. Identify affected vertices (with descendant propagation)
    // Use initial_affected instead of recalculating from scratch
    std::vector<bool> affected = initial_affected; // Initialize with provided status

    // Remove redundant definition of n_global
    // int n_global = part.size();

    // Gather global parent array for propagation (if needed) - This seems unnecessary if initial_affected is correct
    /*
    std::vector<int> global_parent(n_global, -1);
    std::vector<int> local_parent_out(n_global, -1);
    for (int u_local = 0; u_local < (int)local_to_global.size(); ++u_local) {
        int u_global = local_to_global[u_local];
        if (u_global >= 0 && u_global < n_global) { // Bounds check
             if (u_local >= 0 && u_local < parent.size()) { // Bounds check
                local_parent_out[u_global] = parent[u_local];
             }
        }
    }
    // Using MPI_MAX might be incorrect for parent IDs (-1 vs valid IDs). MPI_REPLACE might be better if needed.
    MPI_Allreduce(local_parent_out.data(), global_parent.data(), n_global, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    */
    // Distributed_IdentifyAffected_MPI(...) call removed as initial_affected is used directly.


    // 3. Invalidate affected vertices and descendants
    std::cout << "[Rank " << my_rank << "] Invalidating locally affected vertices..." << std::endl; // Added log
    int invalidated_count = 0; // Added counter
    for (int u_local = 0; u_local < n_local; ++u_local) {
        // Use initial_affected_del to decide which vertices to fully invalidate
        if (u_local < initial_affected_del.size() && initial_affected_del[u_local]) { // Bounds check
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
     std::cout << "[Rank " << my_rank << "] Invalidated " << invalidated_count << " vertices based on initial_affected_del." << std::endl; // Added log


    // Re-initialize the source node if it is local
    if (source >= 0 && source < (int)global_to_local.size() && global_to_local[source] != -1) {
        int src_local = global_to_local[source];
         if (src_local >= 0 && src_local < n_local) { // Bounds check
            dist[src_local] = 0.0;
            parent[src_local] = -1; // Source has no parent
             if (src_local < affected.size()) affected[src_local] = true; // Source might need to propagate updates
             std::cout << "[Rank " << my_rank << "] Re-initialized local source vertex " << src_local << " (global " << source << ")." << std::endl; // Added log
         }
    }

    // 4. Update affected vertices (full reconnection)
    // Pass local_boundary_edges
    Distributed_UpdateAffected_MPI(local_graph, local_boundary_edges, local_to_global, global_to_local, dist, parent, affected, my_rank, num_ranks, part);

     std::cout << "[Rank " << my_rank << "] Exiting Distributed_DynamicSSSP_MPI." << std::endl; // Added log
}
