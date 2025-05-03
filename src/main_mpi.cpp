#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <mpi.h>
#include <metis.h>

#include "../include/graph.h"
#include "../include/utils.hpp"
#include "sssp_sequential.cpp" // For process_batch_sequential

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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- Argument Parsing (Rank 0 handles and broadcasts) ---
    std::string filename;
    int start_node = -1;
    idx_t num_partitions = 1; // Default partitions
    // Add other parameters as needed

    if (rank == 0) {
        if (argc < 3) {
            std::cerr << "Usage: mpirun -np <num_procs> " << argv[0] << " <graph_file.ext> <start_node> [num_partitions]" << std::endl;
            std::cerr << "Note: .ext can be .mtx or .edges" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        filename = argv[1];
        try {
            start_node = std::stoi(argv[2]);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid start node provided: " << argv[2] << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (argc > 3) {
             try {
                num_partitions = std::max(1, std::stoi(argv[3]));
             } catch (const std::exception& e) {
                 std::cerr << "Warning: Invalid number of partitions provided: " << argv[3] << ". Using default (1)." << std::endl;
                 num_partitions = 1;
             }
        }
        // Broadcast necessary parameters (filename length, filename, start_node, num_partitions)
        int fn_len = filename.length();
        MPI_Bcast(&fn_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(const_cast<char*>(filename.c_str()), fn_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD); // +1 for null terminator
        MPI_Bcast(&start_node, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // Use MPI_INT64_T assuming idx_t is 64-bit. Use MPI_INT if idx_t is 32-bit.
        MPI_Bcast(&num_partitions, 1, MPI_INT64_T, 0, MPI_COMM_WORLD); 

    } else {
        // Receive parameters on non-root ranks
        int fn_len;
        MPI_Bcast(&fn_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<char> fn_buffer(fn_len + 1);
        MPI_Bcast(fn_buffer.data(), fn_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        filename = std::string(fn_buffer.data());
        MPI_Bcast(&start_node, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // Use MPI_INT64_T assuming idx_t is 64-bit. Use MPI_INT if idx_t is 32-bit.
        MPI_Bcast(&num_partitions, 1, MPI_INT64_T, 0, MPI_COMM_WORLD); 
    }

    // --- Graph Loading (All ranks load the full graph for now) ---
    // Optimization: Could have rank 0 load and broadcast, or use parallel I/O.
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

    // --- Load and broadcast edge changes (if any) ---
    std::vector<EdgeChange> changes;
    if (rank == 0) {
        // Try to load changes file if provided
        if (argc > 4) {
            std::string changes_file = argv[4];
            std::cout << "\nLoading changes from " << changes_file << "..." << std::endl;
            changes = load_edge_changes(changes_file);
            std::cout << "Changes loaded: " << changes.size() << " total." << std::endl;
        }
    }
    // Broadcast number of changes
    int num_changes = changes.size();
    MPI_Bcast(&num_changes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) changes.resize(num_changes);
    // Broadcast changes data
    if (num_changes > 0) {
        MPI_Bcast(changes.data(), num_changes * sizeof(EdgeChange), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    // --- Distributed Dynamic SSSP Update (if changes exist) ---
    std::vector<double> dist(local_graph.num_vertices, INFINITY_WEIGHT);
    std::vector<int> parent(local_graph.num_vertices, -1);
    // Set local source if owned
    if (global_to_local[start_node] != -1) {
        dist[global_to_local[start_node]] = 0.0;
    }
    if (num_changes > 0) {
        if (size == 1) {
            // Single rank: use the same logic as the serial/OpenMP version
            SSSPResult sssp_result(graph.num_vertices);
            sssp_result.dist[start_node] = 0.0;
            for (int i = 0; i < graph.num_vertices; ++i) sssp_result.parent[i] = -1;
            process_batch_sequential(graph, sssp_result, changes);
            // Copy results to dist/parent for output
            for (int i = 0; i < graph.num_vertices; ++i) {
                dist[i] = sssp_result.dist[i];
                parent[i] = sssp_result.parent[i];
            }
        } else {
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
                start_node // Pass source argument
            );
        }
    } else {
        // If no changes, just run distributed Bellman-Ford
        Distributed_BellmanFord_MPI(
            local_graph,
            local_to_global,
            global_to_local,
            part,
            rank,
            size,
            start_node,
            dist
        );
    }

    // --- Gather and print results (distances and parents) on rank 0 ---
    int n_global = part.size();
    std::vector<int> global_parent(n_global, -1);
    // Gather distances
    std::vector<double> local_dist_out(n_global, INFINITY_WEIGHT);
    for (int u_local = 0; u_local < (int)local_to_global.size(); ++u_local) {
        int u_global = local_to_global[u_local];
        local_dist_out[u_global] = dist[u_local];
    }
    MPI_Reduce(local_dist_out.data(), global_dist.data(), n_global, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    // Gather parents
    std::vector<int> local_parent_out(n_global, -1);
    for (int u_local = 0; u_local < (int)local_to_global.size(); ++u_local) {
        int u_global = local_to_global[u_local];
        local_parent_out[u_global] = parent[u_local];
    }
    MPI_Reduce(local_parent_out.data(), global_parent.data(), n_global, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n--- After Update/Recompute (MPI) ---" << std::endl;
        std::cout << "Current SSSP Result:" << std::endl;
        for (int i = 0; i < n_global; ++i) {
            std::cout << "Vertex " << i << ": Dist = ";
            if (global_dist[i] == INFINITY_WEIGHT) std::cout << "INF";
            else std::cout << global_dist[i];
            std::cout << ", Parent = " << global_parent[i] << std::endl;
        }
    }

    // --- Output Results (Rank 0) ---
    if (rank == 0) {
        std::cout << "\n--- Timings (MPI) ---" << std::endl;
        std::cout << "MPI SSSP Calculation: " << mpi_sssp_time_ms << " ms" << std::endl;
        std::cout << "\nExecution finished." << std::endl;
    }

    MPI_Finalize();
    return 0;
}

// --- Need actual implementation or linking for Dijkstra if used ---
// Placeholder if not linked elsewhere
/*
SSSPResult dijkstra(const Graph& g, int source) {
    if (source < 0 || source >= g.num_vertices) {
        throw std::runtime_error("Source node out of range in Dijkstra");
    }
    SSSPResult result(g.num_vertices);
    result.dist[source] = 0;
    std::priority_queue<std::pair<Weight, int>, std::vector<std::pair<Weight, int>>, std::greater<std::pair<Weight, int>>> pq;
    pq.push({0, source});

    while (!pq.empty()) {
        Weight d = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (d > result.dist[u]) continue;

        for (const auto& edge : g.neighbors(u)) {
            int v = edge.to;
            Weight weight = edge.weight;
            if (result.dist[u] + weight < result.dist[v]) {
                result.dist[v] = result.dist[u] + weight;
                result.parent[v] = u;
                pq.push({result.dist[v], v});
            }
        }
    }
    return result;
}
*/
