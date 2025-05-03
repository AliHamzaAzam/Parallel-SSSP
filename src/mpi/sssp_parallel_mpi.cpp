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

// Distributed Algorithm 3: Update affected vertices (Bellman-Ford style, with parent tracking and full reconnection)
void Distributed_UpdateAffected_MPI(
    Graph& local_graph,
    const std::vector<int>& local_to_global,
    const std::vector<int>& global_to_local,
    std::vector<double>& dist,
    std::vector<int>& parent,
    std::vector<bool>& affected,
    int my_rank,
    int num_ranks,
    const std::vector<idx_t>& part
) {
    int n_local = local_graph.num_vertices;
    int n_global = part.size();
    bool local_changed = true;

    // Create MPI_Datatype for DistParent
    MPI_Datatype MPI_DistParent;
    int blocklengths[2] = {1, 1};
    MPI_Aint offsets[2];
    offsets[0] = offsetof(DistParent, dist);
    offsets[1] = offsetof(DistParent, parent);
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_INT};
    MPI_Type_create_struct(2, blocklengths, offsets, types, &MPI_DistParent);
    MPI_Type_commit(&MPI_DistParent);

    // Create custom reduction op
    MPI_Op min_distparent_op;
    MPI_Op_create(&reduce_min_distparent, 1, &min_distparent_op);

    std::vector<DistParent> local_dp(n_global, {INFINITY_WEIGHT, -1});
    std::vector<DistParent> global_dp(n_global, {INFINITY_WEIGHT, -1});

    // Remove debug prints and max_iters for production
    while (true) {
        local_changed = false;
        // For all affected vertices, try to reconnect using all incoming edges (global relaxation)
        #pragma omp parallel for schedule(dynamic)
        for (int v_local = 0; v_local < n_local; ++v_local) {
            if (!affected[v_local]) continue;
            int v_global = local_to_global[v_local];
            double min_dist = INFINITY_WEIGHT;
            int min_parent = -1;
            // For distributed: check all possible incoming edges from all partitions
            for (int u_global = 0; u_global < n_global; ++u_global) {
                int u_local = (u_global < (int)global_to_local.size()) ? global_to_local[u_global] : -1;
                if (u_local == -1) continue;
                for (const auto& edge : local_graph.neighbors(u_local)) {
                    if (edge.to == v_local) {
                        if (dist[u_local] + edge.weight < min_dist) {
                            min_dist = dist[u_local] + edge.weight;
                            min_parent = local_to_global[u_local];
                        }
                    }
                }
            }
            if (min_dist < dist[v_local]) {
                dist[v_local] = min_dist;
                parent[v_local] = min_parent;
                local_changed = true;
            }
        }
        // Prepare local DistParent for reduction
        for (int u_local = 0; u_local < n_local; ++u_local) {
            int u_global = local_to_global[u_local];
            local_dp[u_global].dist = dist[u_local];
            local_dp[u_global].parent = parent[u_local];
        }
        // Allreduce to get global min distance and parent
        MPI_Allreduce(local_dp.data(), global_dp.data(), n_global, MPI_DistParent, min_distparent_op, MPI_COMM_WORLD);
        // Update local dist/parent with global values
        for (int u_local = 0; u_local < n_local; ++u_local) {
            int u_global = local_to_global[u_local];
            if (global_dp[u_global].dist < dist[u_local] || parent[u_local] != global_dp[u_global].parent) {
                dist[u_local] = global_dp[u_global].dist;
                parent[u_local] = global_dp[u_global].parent;
                local_changed = true;
            }
        }
        // Check for global convergence
        int local_flag = local_changed ? 1 : 0;
        int global_flag = 0;
        MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (global_flag == 0) break;
    }
    MPI_Type_free(&MPI_DistParent);
    MPI_Op_free(&min_distparent_op);
}

// Distributed Algorithm 4: Asynchronous update (optional, can be similar to above)
void Distributed_AsyncUpdate_MPI(
    Graph& local_graph,
    const std::vector<int>& local_to_global,
    const std::vector<int>& global_to_local,
    std::vector<double>& dist,
    std::vector<int>& parent,
    std::vector<bool>& affected,
    int my_rank,
    int num_ranks,
    const std::vector<idx_t>& part
) {
    // For simplicity, call Distributed_UpdateAffected_MPI (can be improved for async)
    Distributed_UpdateAffected_MPI(local_graph, local_to_global, global_to_local, dist, parent, affected, my_rank, num_ranks, part);
}

// Distributed dynamic SSSP update: applies changes and updates SSSP
void Distributed_DynamicSSSP_MPI(
    Graph& local_graph,
    const std::vector<int>& local_to_global,
    const std::vector<int>& global_to_local,
    std::vector<double>& dist,
    std::vector<int>& parent,
    const std::vector<EdgeChange>& changes,
    int my_rank,
    int num_ranks,
    const std::vector<idx_t>& part,
    int source, // <-- new argument
    const std::vector<bool>& initial_affected_del, // Added missing parameter
    const std::vector<bool>& initial_affected      // Added missing parameter
) {
    int n_local = local_graph.num_vertices;
    // 1. Apply changes to local subgraph (insertions/deletions)
    for (const auto& change : changes) {
        int u = change.u, v = change.v;
        int u_local = (u < (int)global_to_local.size()) ? global_to_local[u] : -1;
        int v_local = (v < (int)global_to_local.size()) ? global_to_local[v] : -1;
        if (change.type == ChangeType::INSERT || change.type == ChangeType::DECREASE) { // Insertion or weight decrease
            if (u_local != -1 && v_local != -1) {
                local_graph.add_edge(u_local, v_local, change.weight);
            }
        } else {
            if (u_local != -1 && v_local != -1) {
                local_graph.remove_edge(u_local, v_local);
            }
        }
    }
    // 2. Identify affected vertices (with descendant propagation)
    // Use initial_affected instead of recalculating from scratch
    std::vector<bool> affected = initial_affected; // Initialize with provided status
    // The propagation logic might need adjustment based on how initial_affected is calculated
    // For now, assume initial_affected already includes direct effects.
    // We might still need global propagation if initial_affected only covers rank 0's view.

    // Gather global parent array for propagation (if needed)
    int n_global = part.size();
    std::vector<int> global_parent(n_global, -1);
    std::vector<int> local_parent_out(n_global, -1);
    for (int u_local = 0; u_local < (int)local_to_global.size(); ++u_local) {
        int u_global = local_to_global[u_local];
        local_parent_out[u_global] = parent[u_local];
    }
    MPI_Allreduce(local_parent_out.data(), global_parent.data(), n_global, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    // Distributed_IdentifyAffected_MPI(local_graph, local_to_global, global_to_local, changes, affected, global_parent, my_rank, num_ranks, part);
    // ^^^ Consider if this is still needed or if initial_affected is sufficient ^^^ 

    // 3. Invalidate affected vertices and descendants
    for (int u_local = 0; u_local < n_local; ++u_local) {
        if (affected[u_local]) {
            dist[u_local] = INFINITY_WEIGHT;
            parent[u_local] = -1;
        }
    }
    // Re-initialize the source node if it is local
    if (source >= 0 && source < (int)global_to_local.size() && global_to_local[source] != -1) {
        int src_local = global_to_local[source];
        dist[src_local] = 0.0;
        parent[src_local] = -1;
    }
    // 4. Update affected vertices (full reconnection)
    Distributed_UpdateAffected_MPI(local_graph, local_to_global, global_to_local, dist, parent, affected, my_rank, num_ranks, part);
    // 5. Optionally, run async update
    Distributed_AsyncUpdate_MPI(local_graph, local_to_global, global_to_local, dist, parent, affected, my_rank, num_ranks, part);
}
