#include <iostream>
#include <vector>
#include <string>
#include <omp.h> // Include OpenMP header
#include <chrono> // Include for timing
#include "../../include/graph.hpp"
#include "../../include/utils.hpp"

// Forward declarations for OpenMP SSSP logic
SSSPResult dijkstra(const Graph& g, int source);
void TestDynamicSSSPWorkflow_OpenMP(Graph& G, SSSPResult& T, const std::vector<EdgeChange>& changes);

int parallel_main(int argc, char* argv[]) {
    // Expected arguments after shift from main:
    // argv[0]: graph_file
    // argv[1]: start_node
    // argv[2]: num_threads
    // argv[3]: num_partitions
    // argv[4]: changes_file (optional)
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
            changes_file = argv[4];          // Correct index for changes_file
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
        if (start_node < 0 || start_node >= graph.num_vertices) {
            std::cerr << "Error: Start node " << start_node << " is out of range [0, " << graph.num_vertices - 1 << "]." << std::endl;
            return 1;
        }
        std::cout << "\nCalculating initial SSSP from source " << start_node << " using Dijkstra..." << std::endl;
        SSSPResult sssp_result = dijkstra(graph, start_node);
        std::cout << "Initial SSSP calculated." << std::endl;

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

            std::cout << "Processing batch using OpenMP (dynamic workflow)..." << std::endl;
            TestDynamicSSSPWorkflow_OpenMP(graph, sssp_result, changes);
        } else {
            std::cout << "No changes file provided. Only initial SSSP computed." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "\nError: An exception occurred: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "\nExecution finished." << std::endl;
    return 0;
}
