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

// Forward declaration for the MPI SSSP function
void SSSP_MPI(const Graph& graph, int source, SSSPResult& result, int argc, char* argv[]);
// Forward declaration for sequential Dijkstra (needed for baseline/initial compute on rank 0)
SSSPResult dijkstra(const Graph& g, int source);

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
                    // Fallback might be complex in MPI, aborting for now.
                    MPI_Abort(MPI_COMM_WORLD, 1);
                } else {
                    std::cout << "METIS partitioning successful (Rank 0). Edge cut: " << objval << std::endl;
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

    // --- Output Results (Rank 0) ---
    if (rank == 0) {
        std::cout << "\n--- Timings (MPI) ---" << std::endl;
        std::cout << "MPI SSSP Calculation: " << mpi_sssp_time_ms << " ms" << std::endl;

        // Optional: Verify results or save Dist/Parent arrays
        // Example: Print distance to a specific node
        // int target_node = std::min(10, graph.num_vertices - 1);
        // if (target_node >= 0 && target_node < graph.num_vertices) {
        //     std::cout << "\nFinal distance (Rank 0) to node " << target_node << ": ";
        //     if (sssp_result.dist[target_node] == INFINITY_WEIGHT) {
        //         std::cout << "INF" << std::endl;
        //     } else {
        //         std::cout << sssp_result.dist[target_node] << std::endl;
        //     }
        // }
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
