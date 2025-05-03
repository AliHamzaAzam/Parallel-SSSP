#include <iostream>
#include <vector>
#include <string>
#include <chrono> // Include for timing
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <mpi.h>
#include <metis.h>

#include "../../include/graph.hpp"
#include "../../include/utils.hpp"

// Forward declaration for the MPI SSSP function
void SSSP_MPI(const Graph& graph, int source, SSSPResult& result, int argc, char* argv[]);
// Forward declaration for sequential Dijkstra (needed for baseline/initial compute on rank 0)
SSSPResult dijkstra(const Graph& g, int source); // Ensure this is declared
// Forward declaration for distributed Bellman-Ford SSSP (might be removed later)
void Distributed_BellmanFord_MPI(
    Graph& local_graph,
    const std::vector<int>& local_to_global,
    const std::vector<int>& global_to_local,
    const std::vector<idx_t>& part,
    int my_rank,
    int num_ranks,
    int global_source,
    std::vector<double>& global_dist
);
// Forward declaration for distributed dynamic SSSP update (signature updated)
void Distributed_DynamicSSSP_MPI(
    Graph& local_graph,
    const std::vector<int>& local_to_global,
    const std::vector<int>& global_to_local,
    std::vector<double>& dist,         // Local distances (output)
    std::vector<int>& parent,         // Local parents (output)
    const std::vector<EdgeChange>& changes, // Full changes list (may not be needed if initial affected is enough)
    int my_rank,
    int num_ranks,
    const std::vector<idx_t>& part,     // Global partition vector
    int source,
    const std::vector<bool>& initial_affected_del, // Initial affected_del status for local vertices
    const std::vector<bool>& initial_affected      // Initial affected status for local vertices
);

// Helper: Extract local subgraph for this rank from the global graph and partition vector
// NOTE: This function might need modification to include boundary edge information
Graph extract_local_subgraph(const Graph& global_graph, const std::vector<idx_t>& part, int my_rank) {
    int n = global_graph.num_vertices;
    std::vector<int> local_vertices;
    for (int v = 0; v < n; ++v) {
        if (part[v] == my_rank) local_vertices.push_back(v);
    }
    // Map global vertex id to local id
    std::vector<int> global_to_local(n, -1);
    for (size_t i = 0; i < local_vertices.size(); ++i) {
        global_to_local[local_vertices[i]] = i;
    }
    Graph local_graph(local_vertices.size());
    // Add edges where both endpoints are local
    for (size_t i = 0; i < local_vertices.size(); ++i) {
        int u_global = local_vertices[i];
        for (const auto& edge : global_graph.neighbors(u_global)) {
            int v_global = edge.to;
            if (part[v_global] == my_rank) {
                int u_local = i;
                int v_local = global_to_local[v_global];
                if (v_local != -1) {
                    local_graph.add_edge(u_local, v_local, edge.weight);
                }
            }
        }
    }
    return local_graph;
}

int mpi_main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string filename;
    int start_node = -1;
    std::string changes_filename = "";
    idx_t num_partitions = size;
    Graph graph(0); // Global graph loaded by all ranks for now
    std::vector<EdgeChange> changes;
    int num_changes = 0;
    SSSPResult initial_sssp_result(0); // Initialize with size 0 using the constructor
    std::vector<bool> affected_del; // For Rank 0 initial calculation
    std::vector<bool> affected;     // For Rank 0 initial calculation
    std::vector<idx_t> part;         // Partition vector (size known after graph load)

    // --- Argument Parsing and Broadcasting ---
    if (rank == 0) {
        // ... (Argument parsing logic as before - sets filename, start_node, changes_filename, num_partitions) ...
        if (argc < 2) { std::cerr << "Usage (from main): <graph_file.ext> <start_node> [changes_file] [num_partitions]" << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
        filename = argv[0];
        try { start_node = std::stoi(argv[1]); } catch (const std::exception& e) { std::cerr << "Error: Invalid start node provided: " << argv[1] << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
        if (argc > 2) changes_filename = argv[2];
        if (argc > 3) { try { num_partitions = std::max((idx_t)1, (idx_t)std::stoi(argv[3])); } catch (const std::exception& e) { std::cerr << "Warning: Invalid number of partitions provided: '" << argv[3] << "'. Using default (" << size << "). Error: " << e.what() << std::endl; num_partitions = size; } }
        else { if (rank == 0) std::cout << "Number of partitions not specified. Defaulting to number of MPI ranks (" << size << ")." << std::endl; num_partitions = size; } // Default if not provided

        // Broadcast necessary parameters
        int fn_len = filename.length();
        MPI_Bcast(&fn_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(const_cast<char*>(filename.c_str()), fn_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        int cfn_len = changes_filename.length();
        MPI_Bcast(&cfn_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (cfn_len > 0) MPI_Bcast(const_cast<char*>(changes_filename.c_str()), cfn_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(&start_node, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&num_partitions, 1, MPI_INT64_T, 0, MPI_COMM_WORLD); // Assuming idx_t is 64-bit
    } else {
        // Receive parameters
        int fn_len; MPI_Bcast(&fn_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<char> fn_buffer(fn_len + 1); MPI_Bcast(fn_buffer.data(), fn_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        filename = std::string(fn_buffer.data());
        int cfn_len; MPI_Bcast(&cfn_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (cfn_len > 0) { std::vector<char> cfn_buffer(cfn_len + 1); MPI_Bcast(cfn_buffer.data(), cfn_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD); changes_filename = std::string(cfn_buffer.data()); }
        else { changes_filename = ""; }
        MPI_Bcast(&start_node, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&num_partitions, 1, MPI_INT64_T, 0, MPI_COMM_WORLD); // Assuming idx_t is 64-bit
    }

    // --- Graph Loading (All ranks load for now) ---
    try {
        if (rank == 0) std::cout << "Rank " << rank << " loading graph from " << filename << "..." << std::endl;
        graph = load_graph(filename); // All ranks load the full graph
        if (rank == 0) std::cout << "Graph loaded: " << graph.num_vertices << " vertices." << std::endl;
        part.resize(graph.num_vertices); // Resize partition vector on all ranks
        if (rank == 0) { // Also resize Rank 0 specific vectors
             initial_sssp_result.dist.resize(graph.num_vertices);
             initial_sssp_result.parent.resize(graph.num_vertices);
             affected_del.resize(graph.num_vertices);
             affected.resize(graph.num_vertices);
        }
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " error loading graph: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
     // Basic validation after load
    if (start_node < 0 || start_node >= graph.num_vertices) {
        if (rank == 0) std::cerr << "Error: Start node " << start_node << " is out of range [0, " << graph.num_vertices - 1 << "]." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

     // --- Load Changes (Rank 0 loads, broadcasts count and data) ---
    if (rank == 0) {
        if (!changes_filename.empty()) {
            std::cout << "\nLoading changes from " << changes_filename << "..." << std::endl;
            try {
                changes = load_edge_changes(changes_filename);
                num_changes = changes.size();
                std::cout << "Changes loaded: " << num_changes << " total." << std::endl;
            } catch (const std::exception& e) { std::cerr << "Warning: Failed to load changes file '" << changes_filename << "'. Error: " << e.what() << std::endl; num_changes = 0; changes.clear(); }
        } else { std::cout << "\nNo changes file provided." << std::endl; num_changes = 0; }
    }
    MPI_Bcast(&num_changes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (num_changes > 0) {
        if (rank != 0) changes.resize(num_changes);
        MPI_Bcast(changes.data(), num_changes * sizeof(EdgeChange), MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
        changes.clear(); // Ensure empty on all ranks
    }

    // --- Rank 0: Initial SSSP, Partitioning, Initial Affected Calc ---
    std::chrono::duration<double, std::milli> initial_sssp_time_rank0; // Timing variable
    if (rank == 0) {
        std::cout << "Rank 0: Performing initial SSSP..." << std::endl;
        auto start_initial_sssp = std::chrono::high_resolution_clock::now();
        initial_sssp_result = dijkstra(graph, start_node); // Run Dijkstra on the full graph
        auto end_initial_sssp = std::chrono::high_resolution_clock::now();
        initial_sssp_time_rank0 = end_initial_sssp - start_initial_sssp;
        std::cout << "Rank 0: Initial SSSP completed in " << initial_sssp_time_rank0.count() << " ms." << std::endl;

        // --- METIS Partitioning (as before) ---
        idx_t objval = 0;
        if (graph.num_vertices > 0 && num_partitions > 1) {
             std::cout << "Rank 0 partitioning graph into " << num_partitions << " parts using METIS..." << std::endl;
             std::vector<idx_t> xadj, adjncy, adjwgt;
             graph.to_metis_csr(xadj, adjncy, adjwgt);
             idx_t nVertices = graph.num_vertices;
             idx_t ncon = 1;
             idx_t* adjwgt_ptr = adjwgt.empty() ? NULL : adjwgt.data();
             idx_t nParts = num_partitions;
             if (!adjncy.empty()) {
                 int metis_ret = METIS_PartGraphKway(&nVertices, &ncon, xadj.data(), adjncy.data(), NULL, NULL, adjwgt_ptr, &nParts, NULL, NULL, NULL, &objval, part.data());
                 if (metis_ret != METIS_OK) { std::cerr << "METIS partitioning failed (Rank 0). Error code: " << metis_ret << ". Aborting." << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
                 else { std::cout << "METIS partitioning successful (Rank 0). Edge cut: " << objval << std::endl; }
             } else { std::cerr << "Warning (Rank 0): Graph has no edges. Assigning vertices sequentially." << std::endl; for(idx_t i = 0; i < nVertices; ++i) part[i] = i % nParts; }
        } else {
             if (rank == 0) std::cout << "Skipping partitioning (num_partitions <= 1 or no vertices). Assigning all to partition 0." << std::endl;
             std::fill(part.begin(), part.end(), 0);
        }

        // --- Initial Affected Calculation (Algorithm 2 on Rank 0) ---
        // IMPORTANT: This modifies a temporary copy of distances/parents for calculation.
        // The actual distributed algorithm needs the *original* state + affected flags.
        std::cout << "Rank 0: Calculating initial affected vertices..." << std::endl;
        std::fill(affected_del.begin(), affected_del.end(), false);
        std::fill(affected.begin(), affected.end(), false);
        SSSPResult temp_sssp_result = initial_sssp_result; // Copy original state for calculation

        // Process Deletions/Increases
        for (const auto& change : changes) {
            // Treat INCREASE like DELETE for initial affected_del calculation
            if (change.type == ChangeType::DECREASE || change.type == ChangeType::INSERT) continue;

            // Check if edge e(u, v) was in the original SSSP tree T
            // Note: This check assumes an undirected graph representation in the parent array
            // or that the edge direction matches parent relationship.
            bool in_tree = (temp_sssp_result.parent[change.v] == change.u);
            // If the graph is stored undirected, you might need:
            // bool in_tree = (temp_sssp_result.parent[change.v] == change.u) || (temp_sssp_result.parent[change.u] == change.v);

            if (in_tree) {
                 int u = change.u;
                 int v = change.v;
                 // y = argmax_{x in {u,v}} {Dist[x]}
                 // We mark the child node 'v' as affected if the edge (u,v) from parent u is deleted/increased.
                 int y = v; // The vertex whose distance might become infinite due to parent edge removal
                 if (temp_sssp_result.dist[y] != INFINITY_WEIGHT) { // Only mark if reachable
                     temp_sssp_result.dist[y] = INFINITY_WEIGHT; // Change Dist[y] to infinity in the *temporary* result
                     affected_del[y] = true;
                     affected[y] = true;
                     // TODO: Potentially propagate INF distance change in temp_sssp_result for accurate affected calculation?
                     // This might require a temporary BFS/DFS from y in the tree. For now, just mark y.
                 }
            }
        }

        // Process Insertions/Decreases
        for (const auto& change : changes) {
             // Only handle decrease/insert here
             if (change.type != ChangeType::DECREASE && change.type != ChangeType::INSERT) continue;

             int u = change.u;
             int v = change.v;
             double weight = change.weight;

             // Relax edge based on current *temporary* distances
             if (temp_sssp_result.dist[u] != INFINITY_WEIGHT && temp_sssp_result.dist[v] > temp_sssp_result.dist[u] + weight) {
                 temp_sssp_result.dist[v] = temp_sssp_result.dist[u] + weight; // Update temporary result
                 temp_sssp_result.parent[v] = u;
                 affected[v] = true;
             }
             // Check relaxation in the other direction too
             if (temp_sssp_result.dist[v] != INFINITY_WEIGHT && temp_sssp_result.dist[u] > temp_sssp_result.dist[v] + weight) {
                 temp_sssp_result.dist[u] = temp_sssp_result.dist[v] + weight; // Update temporary result
                 temp_sssp_result.parent[u] = v;
                 affected[u] = true;
             }
        }
         std::cout << "Rank 0: Initial affected calculation done." << std::endl;
         // Now, 'affected_del' and 'affected' contain the initial status based on the original SSSP tree.
         // 'initial_sssp_result' still holds the *original* Dijkstra result.
    }

    // --- Broadcast Partition Vector ---
    MPI_Bcast(part.data(), graph.num_vertices, MPI_INT64_T, 0, MPI_COMM_WORLD); // Assuming idx_t is 64-bit

    // --- Data Scattering (Rank 0 -> Others) ---
    // TODO: Implement scattering of subgraph structures and initial state arrays
    // This involves:
    // 1. Rank 0 iterating 1 to size-1:
    //    - Determine vertices for rank `p`.
    //    - Build local graph structure for `p` (including boundary edges).
    //    - Extract local portions of initial_sssp_result.dist, .parent, affected_del, affected.
    //    - Send this data using MPI_Send (potentially multiple sends or packed buffer).
    // 2. Ranks 1 to size-1:
    //    - Receive data using MPI_Recv.
    //    - Reconstruct local graph and state arrays.
    // 3. Rank 0 also needs to set up its own local graph and state.

    Graph local_graph(0); // Placeholder for local graph
    std::vector<int> local_to_global;
    std::vector<int> global_to_local; // Map for local rank: global_idx -> local_idx
    std::vector<double> local_dist;
    std::vector<int> local_parent;
    std::vector<bool> local_affected_del;
    std::vector<bool> local_affected;

    // --- Placeholder: Setup local data for Rank 0 ---
    if (rank == 0) {
        std::cout << "Rank 0: Setting up local data..." << std::endl;
        // TODO: Extract Rank 0's local data from global data calculated above
        // local_graph = extract_local_subgraph_with_boundaries(graph, part, rank); // Needs modification
        // Populate local_dist, local_parent etc. from initial_sssp_result, affected_del, affected
        // Populate local_to_global, global_to_local mappings for rank 0
        std::cout << "Rank 0: Local data setup placeholder complete." << std::endl;
    } else {
         std::cout << "Rank " << rank << ": Waiting to receive local data..." << std::endl;
         // TODO: Receive local data from Rank 0
         std::cout << "Rank " << rank << ": Received local data placeholder complete." << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD); // Wait for all ranks to finish setup/receive


    // --- Distributed Update ---
    if (rank == 0) std::cout << "Starting distributed update phase..." << std::endl;
    auto start_update = MPI_Wtime();

    // Call the function that implements Algorithm 3/4
    Distributed_DynamicSSSP_MPI(
        local_graph,        // Local graph structure (needs boundary info)
        local_to_global,    // Mapping: local index -> global ID
        global_to_local,    // Mapping: global ID -> local index
        local_dist,         // Local distance array (input/output)
        local_parent,       // Local parent array (input/output)
        changes,            // Full changes list (might not be needed)
        rank,
        size,
        part,               // Global partition vector
        start_node,
        local_affected_del, // Initial affected_del status for local vertices
        local_affected      // Initial affected status for local vertices
    );

    auto end_update = MPI_Wtime();
    double update_time_ms = (end_update - start_update) * 1000.0;
    if (rank == 0) std::cout << "Distributed update phase finished in " << update_time_ms << " ms." << std::endl;


    // --- Gather Results ---
    int n_global = graph.num_vertices; // Use graph.num_vertices for global size
    std::vector<double> final_global_dist;
    std::vector<int> final_global_parent;
    if (rank == 0) {
        final_global_dist.resize(n_global, INFINITY_WEIGHT); // Initialize on rank 0
        final_global_parent.resize(n_global, -1);          // Initialize on rank 0
    }

    // Gather distances (using Reduce with MPI_IN_PLACE on rank 0)
    std::vector<double> local_dist_out(n_global, INFINITY_WEIGHT);
    // Ensure local_to_global is populated before this loop
    if (!local_to_global.empty()) { // Add check to prevent crash if setup is incomplete
        for (int u_local = 0; u_local < (int)local_to_global.size(); ++u_local) { // Use local_to_global size
            int u_global = local_to_global[u_local];
            if (u_global >= 0 && u_global < n_global && u_local < local_dist.size()) { // Bounds check for local_dist too
                 local_dist_out[u_global] = local_dist[u_local]; // Use final local_dist
            }
        }
    }
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : local_dist_out.data(),
               rank == 0 ? final_global_dist.data() : nullptr,
               n_global, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    // Gather parents (using Gather)
    std::vector<int> local_parent_out(n_global, -1);
    // Ensure local_to_global is populated before this loop
    if (!local_to_global.empty()) { // Add check to prevent crash if setup is incomplete
         for (int u_local = 0; u_local < (int)local_to_global.size(); ++u_local) { // Use local_to_global size
            int u_global = local_to_global[u_local];
             if (u_global >= 0 && u_global < n_global && u_local < local_parent.size()) { // Bounds check for local_parent too
                // Map local parent index back to global index if not -1
                int p_local = local_parent[u_local]; // Use final local_parent
                if (p_local != -1 && p_local >= 0 && p_local < local_to_global.size()) { // Bounds check for parent index mapping
                     local_parent_out[u_global] = local_to_global[p_local];
                } else {
                     local_parent_out[u_global] = -1;
                }
            }
        }
    }
    std::vector<int> gathered_parents;
    if (rank == 0) gathered_parents.resize(n_global * size);
    MPI_Gather(local_parent_out.data(), n_global, MPI_INT,
               gathered_parents.data(), n_global, MPI_INT, 0, MPI_COMM_WORLD);

    // Combine gathered parents on Rank 0
    if (rank == 0) {
        for(int i=0; i<n_global; ++i) {
            final_global_parent[i] = -1; // Initialize
            for(int r=0; r<size; ++r) {
                // Check bounds for gathered_parents access
                if (r * n_global + i < gathered_parents.size()) {
                    int val = gathered_parents[r * n_global + i];
                    // Simple strategy: take the first valid parent found.
                    // Might need refinement if multiple ranks report a parent due to boundary edges.
                    if (val != -1) {
                        final_global_parent[i] = val;
                        break;
                    }
                }
            }
        }
    }


    // --- Output Results (Rank 0) ---
    if (rank == 0) {
        std::cout << "\n--- Final SSSP Result (MPI) ---" << std::endl;
        for (int i = 0; i < n_global; ++i) {
            std::cout << "Vertex " << i << ": Dist = ";
            if (i < final_global_dist.size() && final_global_dist[i] == INFINITY_WEIGHT) std::cout << "INF"; // Bounds check
            else if (i < final_global_dist.size()) std::cout << final_global_dist[i]; // Bounds check
            else std::cout << "ERR"; // Indicate error if out of bounds
            std::cout << ", Parent = ";
            if (i < final_global_parent.size()) std::cout << final_global_parent[i]; // Bounds check
            else std::cout << "ERR"; // Indicate error if out of bounds
            std::cout << std::endl;
        }

        std::cout << "\n--- Timings (MPI) ---" << std::endl;
        std::cout << "Initial SSSP (Rank 0): " << initial_sssp_time_rank0.count() << " ms" << std::endl;
        std::cout << "Distributed Update Time: " << update_time_ms << " ms" << std::endl;
        std::cout << "\nExecution finished." << std::endl;
    }

    MPI_Finalize();
    return 0;
}

// NOTE: Remember to update the actual implementation of Distributed_DynamicSSSP_MPI
// in sssp_parallel_mpi.cpp to match the new signature and use the initial affected flags.
// Also, the TODO sections for data scattering and local setup need to be implemented.
// The extract_local_subgraph function likely needs modification.
