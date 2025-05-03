#include <iostream>
#include <vector>
#include <string>
#include <chrono>
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
SSSPResult dijkstra(const Graph& g, int source);
// Forward declaration for distributed Bellman-Ford SSSP
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
// Forward declaration for distributed dynamic SSSP update
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
    int source
);

// Helper: Extract local subgraph for this rank from the global graph and partition vector
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

    // --- Argument Parsing (Rank 0 handles and broadcasts) ---
    // Expected arguments after shift from main:
    // argv[0]: graph_file
    // argv[1]: start_node
    // argv[2]: changes_file (optional)
    // argv[3]: num_partitions (optional, defaults to number of ranks)
    std::string filename;
    int start_node = -1;
    std::string changes_filename = ""; // Initialize changes filename
    idx_t num_partitions = size; // Default partitions to number of MPI ranks

    if (rank == 0) {
        if (argc < 2) { // Need at least graph_file and start_node
            std::cerr << "Usage (from main): <graph_file.ext> <start_node> [changes_file] [num_partitions]" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        filename = argv[0];
        try {
            start_node = std::stoi(argv[1]);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid start node provided: " << argv[1] << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (argc > 2) {
            changes_filename = argv[2]; // Optional changes file
        }
        if (argc > 3) { // Optional num_partitions
             try {
                // Ensure at least 1 partition, default was already 'size'
                num_partitions = std::max((idx_t)1, (idx_t)std::stoi(argv[3]));
             } catch (const std::exception& e) {
                 std::cerr << "Warning: Invalid number of partitions provided: '" << argv[3] << "'. Using default (" << size << "). Error: " << e.what() << std::endl;
                 num_partitions = size; // Use default (number of ranks)
             }
        } else {
             std::cout << "Number of partitions not specified. Defaulting to number of MPI ranks (" << size << ")." << std::endl;
             num_partitions = size; // Explicitly set default if not provided
        }

        // Broadcast necessary parameters
        int fn_len = filename.length();
        MPI_Bcast(&fn_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(const_cast<char*>(filename.c_str()), fn_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

        int cfn_len = changes_filename.length(); // Length of changes filename
        MPI_Bcast(&cfn_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (cfn_len > 0) { // Only broadcast if changes_filename is not empty
             MPI_Bcast(const_cast<char*>(changes_filename.c_str()), cfn_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        }

        MPI_Bcast(&start_node, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&num_partitions, 1, MPI_INT64_T, 0, MPI_COMM_WORLD); // Assuming idx_t is 64-bit

    } else {
        // Receive parameters on non-root ranks
        int fn_len;
        MPI_Bcast(&fn_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<char> fn_buffer(fn_len + 1);
        MPI_Bcast(fn_buffer.data(), fn_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        filename = std::string(fn_buffer.data());

        int cfn_len;
        MPI_Bcast(&cfn_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (cfn_len > 0) {
            std::vector<char> cfn_buffer(cfn_len + 1);
            MPI_Bcast(cfn_buffer.data(), cfn_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
            changes_filename = std::string(cfn_buffer.data());
        } else {
            changes_filename = ""; // Ensure it's empty if length was 0
        }

        MPI_Bcast(&start_node, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&num_partitions, 1, MPI_INT64_T, 0, MPI_COMM_WORLD); // Assuming idx_t is 64-bit
    }

    // --- Graph Loading (All ranks load the full graph) ---
    Graph graph(0);
    try {
        if (rank == 0) std::cout << "Rank " << rank << " loading graph from " << filename << "..." << std::endl;
        graph = load_graph(filename);
        if (rank == 0) std::cout << "Graph loaded: " << graph.num_vertices << " vertices." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " error loading graph: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Basic validation after load
    if (start_node < 0 || start_node >= graph.num_vertices) {
        if (rank == 0) {
             std::cerr << "Error: Start node " << start_node << " is out of range [0, " << graph.num_vertices - 1 << "]." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --- METIS Partitioning (Rank 0 performs, broadcasts assignments) ---
    std::vector<idx_t> part(graph.num_vertices);
    if (rank == 0) {
        idx_t objval = 0;
        if (graph.num_vertices > 0 && num_partitions > 1) {
            std::cout << "Rank 0 partitioning graph into " << num_partitions << " parts using METIS..." << std::endl;
            std::vector<idx_t> xadj, adjncy, adjwgt;
            graph.to_metis_csr(xadj, adjncy, adjwgt);

            idx_t nVertices = graph.num_vertices;
            idx_t ncon = 1; // Vertex balance
            idx_t* adjwgt_ptr = adjwgt.empty() ? NULL : adjwgt.data();
            if (adjwgt_ptr && adjwgt.size() != adjncy.size()) adjwgt_ptr = NULL; // Basic validation
            idx_t nParts = num_partitions;

            if (!adjncy.empty()) {
                int metis_ret = METIS_PartGraphKway(&nVertices, &ncon, xadj.data(), adjncy.data(),
                                                    NULL, NULL, adjwgt_ptr, &nParts, NULL, NULL, NULL,
                                                    &objval, part.data());
                if (metis_ret != METIS_OK) {
                    std::cerr << "METIS partitioning failed (Rank 0). Error code: " << metis_ret << ". Aborting." << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                } else {
                    std::cout << "METIS partitioning successful (Rank 0). Edge cut: " << objval << std::endl;
                    // Print partition assignments for debugging
                    std::cout << "METIS partition assignments:" << std::endl;
                    for (size_t i = 0; i < part.size(); ++i) {
                        std::cout << "Vertex " << i << " -> Partition " << part[i] << std::endl;
                    }
                    // Print number of vertices per partition
                    std::vector<int> partition_counts(nParts, 0);
                    for (size_t i = 0; i < part.size(); ++i) {
                        partition_counts[part[i]]++;
                    }
                    std::cout << "Vertices per partition:" << std::endl;
                    for (idx_t p = 0; p < nParts; ++p) {
                        std::cout << "Partition " << p << ": " << partition_counts[p] << " vertices" << std::endl;
                    }
                    // Print all cross-partition edges
                    std::cout << "Cross-partition edges:" << std::endl;
                    for (int u = 0; u < graph.num_vertices; ++u) {
                        for (const auto& edge : graph.neighbors(u)) {
                            int v = edge.to;
                            if (part[u] != part[v]) {
                                std::cout << "Edge (" << u << "," << v << ") crosses partitions " << part[u] << " and " << part[v] << std::endl;
                            }
                        }
                    }
                }
            } else {
                 std::cerr << "Warning (Rank 0): Graph has no edges. Assigning vertices sequentially." << std::endl;
                 for(idx_t i = 0; i < nVertices; ++i) part[i] = i % nParts;
            }
        } else {
            if (rank == 0) std::cout << "Skipping partitioning (num_partitions <= 1 or no vertices). Assigning all to partition 0." << std::endl;
            std::fill(part.begin(), part.end(), 0);
        }
    }
    // Broadcast partition assignments to all ranks
    // Use MPI_INT64_T assuming idx_t is 64-bit. Use MPI_INT if idx_t is 32-bit.
    MPI_Bcast(part.data(), graph.num_vertices, MPI_INT64_T, 0, MPI_COMM_WORLD); 

    // --- Build local subgraph for each rank ---
    Graph local_graph = extract_local_subgraph(graph, part, rank);
    if (rank == 0) {
        std::cout << "Partitioning and local subgraph extraction complete. Local vertices on rank 0: " << local_graph.num_vertices << std::endl;
    }

    // --- Build local_to_global and global_to_local mappings ---
    std::vector<int> local_to_global;
    std::vector<int> global_to_local(graph.num_vertices, -1);
    for (int v = 0, local_idx = 0; v < graph.num_vertices; ++v) {
        if (part[v] == rank) {
            local_to_global.push_back(v);
            global_to_local[v] = local_idx++;
        }
    }

    // --- Load and broadcast edge changes (if filename provided) ---
    std::vector<EdgeChange> changes;
    int num_changes = 0; // Initialize num_changes
    if (rank == 0) {
        if (!changes_filename.empty()) { // Check if changes filename was provided
            std::cout << "\nLoading changes from " << changes_filename << "..." << std::endl;
            try {
                changes = load_edge_changes(changes_filename);
                std::cout << "Changes loaded: " << changes.size() << " total." << std::endl;
                num_changes = changes.size(); // Set num_changes based on loaded vector
            } catch (const std::exception& e) {
                 std::cerr << "Warning: Failed to load changes file '" << changes_filename << "'. Error: " << e.what() << std::endl;
                 num_changes = 0; // Ensure num_changes is 0 if loading fails
                 changes.clear();
            }
        } else {
             std::cout << "\nNo changes file provided." << std::endl;
             num_changes = 0; // Ensure num_changes is 0 if no file provided
        }
    }
    // Broadcast number of changes
    MPI_Bcast(&num_changes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize and broadcast changes data only if num_changes > 0
    if (num_changes > 0) {
        if (rank != 0) changes.resize(num_changes);
        MPI_Bcast(changes.data(), num_changes * sizeof(EdgeChange), MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
        // Ensure changes vector is empty on all ranks if num_changes is 0
        changes.clear();
    }

    // --- Distributed Bellman-Ford SSSP ---
    std::vector<double> global_dist; // Only filled on rank 0
    Distributed_BellmanFord_MPI(
        local_graph,
        local_to_global,
        global_to_local,
        part,
        rank,
        size,
        start_node,
        global_dist
    );

    if (rank == 0) {
        std::cout << "\n--- Distributed Bellman-Ford SSSP Result (Distances from source " << start_node << ") ---" << std::endl;
        for (size_t i = 0; i < global_dist.size(); ++i) {
            std::cout << "Vertex " << i << ": Dist = ";
            if (global_dist[i] == INFINITY_WEIGHT) std::cout << "INF";
            else std::cout << global_dist[i];
            std::cout << std::endl;
        }
    }

    // --- SSSP Calculation ---
    SSSPResult sssp_result(graph.num_vertices);
    if (rank == 0) {
         std::cout << "\nRank 0 starting MPI SSSP calculation..." << std::endl;
    }
    auto start_time_mpi = MPI_Wtime(); // Use MPI timer

    // Call the main MPI SSSP function (which is currently a placeholder)
    SSSP_MPI(graph, start_node, sssp_result, argc, argv);

    auto end_time_mpi = MPI_Wtime();
    double mpi_sssp_time_ms = (end_time_mpi - start_time_mpi) * 1000.0;

    // --- Distributed Dynamic SSSP Update or Initial Bellman-Ford ---
    std::vector<double> dist(local_graph.num_vertices, INFINITY_WEIGHT);
    std::vector<int> parent(local_graph.num_vertices, -1);
    // Set local source distance if owned
    int local_source_idx = global_to_local[start_node];
    if (local_source_idx != -1) {
        dist[local_source_idx] = 0.0;
    }

    auto start_time_update = MPI_Wtime(); // Timer for update/initial compute

    if (num_changes > 0) {
        if (rank == 0) std::cout << "Processing " << num_changes << " changes using Distributed Dynamic SSSP..." << std::endl;
        // Call the dynamic update function
        Distributed_DynamicSSSP_MPI(
            local_graph,
            local_to_global,
            global_to_local,
            dist,
            parent,
            changes,
            rank,
            size,
            part,
            start_node
        );
    } else {
        if (rank == 0) std::cout << "No changes detected. Running initial Distributed Bellman-Ford..." << std::endl;
        // Run initial Bellman-Ford if no changes
        // Note: Bellman-Ford implementation needs to populate 'dist' correctly.
        // The current Distributed_BellmanFord_MPI signature returns results in 'global_dist' on rank 0 only.
        // We need a version that populates the local 'dist' vector on each rank.
        // For now, let's assume Distributed_DynamicSSSP_MPI handles the initial case if changes is empty,
        // or we need to adjust the Bellman-Ford call.
        // --- TEMPORARY: Assuming Dynamic handles initial if changes empty ---
         Distributed_DynamicSSSP_MPI(
            local_graph,
            local_to_global,
            global_to_local,
            dist, // Pass local dist vector
            parent, // Pass local parent vector
            changes, // Pass empty changes vector
            rank,
            size,
            part,
            start_node
        );
        // --- END TEMPORARY ---
        // TODO: Replace above with a proper call to a distributed Bellman-Ford
        //       that populates the local 'dist' and 'parent' vectors, or ensure
        //       Distributed_DynamicSSSP_MPI handles the empty changes case correctly
        //       as an initial computation.
    }
    auto end_time_update = MPI_Wtime();
    double update_time_ms = (end_time_update - start_time_update) * 1000.0;

    // --- Gather and print results (distances and parents) on rank 0 ---
    int n_global = part.size();
    if (rank == 0) {
        global_dist.assign(n_global, INFINITY_WEIGHT); // Resize and initialize on rank 0
    }
    std::vector<int> global_parent(n_global, -1);
    // Gather distances
    std::vector<double> local_dist_out(n_global, INFINITY_WEIGHT);
    for (int u_local = 0; u_local < (int)local_to_global.size(); ++u_local) {
        int u_global = local_to_global[u_local];
        local_dist_out[u_global] = dist[u_local];
    }
    // MPI_Reduce needs a valid buffer on rank 0, even if it's overwritten.
    // Other ranks pass their local_dist_out. Rank 0 passes its local_dist_out as well.
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : local_dist_out.data(), // Source buffer (MPI_IN_PLACE for root)
               rank == 0 ? global_dist.data() : nullptr,         // Receive buffer (only root receives)
               n_global, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    // Gather parents (Need careful reduction for parents, MAX might not be correct)
    // For simplicity, let's gather all parent arrays to rank 0
    std::vector<int> local_parent_out(n_global, -1);
     for (int u_local = 0; u_local < (int)local_to_global.size(); ++u_local) {
        int u_global = local_to_global[u_local];
        // Map local parent index back to global index if not -1
        // Ensure parent[u_local] is within the bounds of local_to_global
        if (parent[u_local] != -1 && parent[u_local] >= 0 && parent[u_local] < local_to_global.size()) { 
             local_parent_out[u_global] = local_to_global[parent[u_local]];
        } else {
             local_parent_out[u_global] = -1; // Keep as -1 if no parent or invalid index
        }
    }
    // Use MPI_Gather on rank 0 to collect all local_parent_out arrays
    std::vector<int> gathered_parents;
    if (rank == 0) gathered_parents.resize(n_global * size);
    MPI_Gather(local_parent_out.data(), n_global, MPI_INT,
               gathered_parents.data(), n_global, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Combine gathered parent arrays: take the first non -1 value found for each vertex
        for(int i=0; i<n_global; ++i) {
            global_parent[i] = -1; // Initialize
            for(int r=0; r<size; ++r) {
                int val = gathered_parents[r * n_global + i];
                if (val != -1) {
                    global_parent[i] = val;
                    break; // Found a parent, move to next global vertex
                }
            }
        }

        std::cout << "\n--- After Update/Recompute (MPI) ---" << std::endl;
        std::cout << "Current SSSP Result:" << std::endl;
        for (int i = 0; i < n_global; ++i) {
            std::cout << "Vertex " << i << ": Dist = ";
            if (global_dist[i] == INFINITY_WEIGHT) std::cout << "INF";
            else std::cout << global_dist[i];
            std::cout << ", Parent = " << global_parent[i] << std::endl;
        }
    }

    // --- Output Timings (Rank 0) ---
    if (rank == 0) {
        std::cout << "\n--- Timings (MPI) ---" << std::endl;
        // std::cout << "MPI SSSP Calculation: " << mpi_sssp_time_ms << " ms" << std::endl; // Old timer location
        std::cout << "MPI Update/Compute Time: " << update_time_ms << " ms" << std::endl; // New timer location
        std::cout << "\nExecution finished." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
