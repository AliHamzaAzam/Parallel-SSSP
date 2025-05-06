// main_parallel.cpp - OpenMP-based parallel dynamic SSSP driver
// -------------------------------------------------------------------
// Entry point for OpenMP mode: loads graph, runs initial SSSP, applies updates,
// and performs parallel dynamic update using OpenMP routines.

#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include <chrono>

#include "../../include/graph.hpp"
#include "../../include/utils.hpp"

// Forward declarations
SSSPResult dijkstra(const Graph& g, int source);
void TestDynamicSSSPWorkflow_OpenMP(Graph& G, SSSPResult& T, const std::vector<EdgeChange>& changes);

int parallel_main(int argc, char* argv[]) {
    // Args: graph_file, start_node, num_threads, num_partitions, [changes_file]
    if (argc < 4) { // Need at least graph, start, threads, partitions
        std::cerr << "Usage (from main): <graph_file> <start_node> <num_threads> <num_partitions> [changes_file]" << std::endl;
        return 1;
    }
    std::string filename = argv[0]; // Correct index after shift
    int start_node = -1;
    int num_threads = 1;
    int num_partitions = 1; // Use int for consistency, METIS uses idx_t if needed
    std::string changes_file = "";

    try {
        start_node = std::stoi(argv[1]);     // Correct index
        num_threads = std::stoi(argv[2]);    // Parse num_threads
        num_partitions = std::stoi(argv[3]); // Parse num_partitions
        if (argc > 4) {
            changes_file = argv[4];             // Correct index for changes_file
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments in parallel_main: " << e.what() << std::endl;
        return 1;
    }

    // Set the number of threads for OpenMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
        std::cout << "OpenMP using " << num_threads << " threads." << std::endl;
    } else {
        std::cerr << "Warning: Invalid number of threads specified (" << num_threads << "). Using default." << std::endl;
        num_threads = omp_get_max_threads(); // Use default if invalid
        omp_set_num_threads(num_threads);
         std::cout << "OpenMP using default " << num_threads << " threads." << std::endl;
    }

    // Note: num_partitions is parsed but not used yet in this function.
    // Add partitioning logic here if needed for the OpenMP version.
    std::cout << "Number of partitions requested: " << num_partitions << " (currently unused in parallel_main)." << std::endl;

    try {
        std::cout << "Loading graph from " << filename << "..." << std::endl;
        Graph graph = load_graph(filename);
        std::cout << "Graph loaded: " << graph.num_vertices << " vertices." << std::endl;
        // Validate source
        if (start_node < 0 || start_node >= graph.num_vertices) {
            std::cerr << "Error: Start node " << start_node << " out of range." << std::endl;
            return 1;
        }

        // --- Initial SSSP using Dijkstra ---
        std::cout << "\nPerforming initial SSSP from source " << start_node << "..." << std::endl;
        auto t0 = std::chrono::high_resolution_clock::now();
        SSSPResult sssp_result = dijkstra(graph, start_node);
        auto t1 = std::chrono::high_resolution_clock::now();
        double initial_time = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Initial SSSP completed in " << initial_time << " ms." << std::endl;

        double update_time = 0.0;
        if (!changes_file.empty()) {
            std::cout << "\nLoading changes from " << changes_file << "..." << std::endl;
            std::vector<EdgeChange> changes = load_edge_changes(changes_file);
            std::cout << "Changes loaded: " << changes.size() << " total." << std::endl;

            // Apply changes to graph structure *before* calling the update function
            // (Assuming TestDynamicSSSPWorkflow_OpenMP expects the graph to be pre-modified)
             std::cout << "Applying structural changes to graph..." << std::endl;
             auto apply_start = std::chrono::high_resolution_clock::now();
             int skipped_changes = 0;
             for (const auto& change : changes) {
                 try {
                     if (change.u < 0 || change.u >= graph.num_vertices || change.v < 0 || change.v >= graph.num_vertices) {
                         std::cerr << "Warning: Skipping change due to out-of-bounds vertex index: "
                                   << (change.type == ChangeType::INSERT || change.type == ChangeType::DECREASE ? "i/d " : "d/i ")
                                   << change.u << " " << change.v << std::endl;
                         skipped_changes++;
                         continue;
                     }
                     if (change.type == ChangeType::INSERT || change.type == ChangeType::DECREASE) {
                         graph.add_edge(change.u, change.v, change.weight);
                     } else {
                         graph.remove_edge(change.u, change.v);
                     }
                 } catch (const std::exception& e) {
                      std::cerr << "Warning: Error applying change ("
                                << (change.type == ChangeType::INSERT || change.type == ChangeType::DECREASE ? "insert/decrease" : "delete/increase")
                                << " " << change.u << " " << change.v
                                << "). Skipping change. Error: " << e.what() << std::endl;
                      skipped_changes++;
                 }
             }
             auto apply_end = std::chrono::high_resolution_clock::now();
             std::chrono::duration<double, std::milli> apply_time = apply_end - apply_start;
             std::cout << "Structural changes applied in " << apply_time.count() << " ms.";
             if (skipped_changes > 0) {
                 std::cout << " (" << skipped_changes << " changes skipped due to errors or invalid indices)";
             }
             std::cout << std::endl;

            // --- Parallel dynamic update ---
            std::cout << "Starting parallel dynamic update..." << std::endl;
            auto u0 = std::chrono::high_resolution_clock::now();
            TestDynamicSSSPWorkflow_OpenMP(graph, sssp_result, changes);
            auto u1 = std::chrono::high_resolution_clock::now();
            update_time = std::chrono::duration<double, std::milli>(u1 - u0).count();
            std::cout << "Dynamic update completed in " << update_time << " ms." << std::endl;
        }

        // --- Timings Summary ---
        std::cout << "\n--- Timings (OpenMP) ---" << std::endl;
        std::cout << "Initial SSSP: " << initial_time << " ms" << std::endl;
        if (update_time > 0.0) {
            std::cout << "Dynamic Update Time: " << update_time << " ms" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Execution finished." << std::endl;
    return 0;
}
