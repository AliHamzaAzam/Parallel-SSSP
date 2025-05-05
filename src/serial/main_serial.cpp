// main_serial.cpp - Serial dynamic SSSP driver
// ---------------------------------------------------
// Loads graph, computes initial SSSP, applies batch updates sequentially,
// and reports final distances and timings.

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>

#include "../../include/graph.hpp"
#include "../../include/utils.hpp"

// Forward: static SSSP via Dijkstra
SSSPResult dijkstra(const Graph& g, int source);
// Forward: sequential batch update (modifies graph and SSSPResult)
void process_batch_sequential(Graph& g, SSSPResult& sssp_result, const std::vector<EdgeChange>& batch);

// print_sssp_result: display dist and parent arrays
void print_sssp_result(const SSSPResult& sssp_result) {
    std::cout << "Current SSSP Result:" << std::endl;
    for (size_t i = 0; i < sssp_result.dist.size(); ++i) {
        std::cout << "Vertex " << i
                  << ": Dist = " << sssp_result.dist[i]
                  << ", Parent = " << sssp_result.parent[i]
                  << std::endl;
    }
}

// main: entry for serial mode
// Args: <graph_file> <start_node> [changes_file]
int main(int argc, char* argv[]) {
    // --- Argument Parsing ---
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> <start_node> [changes_file]" << std::endl;
        return 1;
    }
    std::string filename = argv[1];
    int start_node = std::stoi(argv[2]);
    std::string changes_file = (argc > 3 ? argv[3] : "");

    try {
        // --- Graph Loading ---
        std::cout << "Loading graph from " << filename << "..." << std::endl;
        Graph graph = load_graph(filename);
        std::cout << "Graph loaded: " << graph.num_vertices << " vertices." << std::endl;
        if (start_node < 0 || start_node >= graph.num_vertices) {
            std::cerr << "Error: start_node out of range." << std::endl;
            return 1;
        }

        // --- Initial SSSP ---
        std::cout << "\nRunning initial SSSP (Dijkstra) from " << start_node << "..." << std::endl;
        auto t0 = std::chrono::high_resolution_clock::now();
        SSSPResult sssp_result = dijkstra(graph, start_node);
        auto t1 = std::chrono::high_resolution_clock::now();
        double initial_time = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Initial SSSP completed in " << initial_time << " ms." << std::endl;
        print_sssp_result(sssp_result);

        double update_time = 0.0;
        if (!changes_file.empty()) {
            // --- Load Updates ---
            std::cout << "\nLoading changes from " << changes_file << "..." << std::endl;
            auto start_load = std::chrono::high_resolution_clock::now();
            std::vector<EdgeChange> changes = load_edge_changes(changes_file);
            auto end_load = std::chrono::high_resolution_clock::now();
            std::cout << "Changes loaded: " << changes.size() << " (" 
                      << std::chrono::duration<double, std::milli>(end_load - start_load).count()
                      << " ms)." << std::endl;

            // --- Sequential Batch Update ---
            std::cout << "Running sequential dynamic update..." << std::endl;
            auto t2 = std::chrono::high_resolution_clock::now();
            process_batch_sequential(graph, sssp_result, changes);
            auto t3 = std::chrono::high_resolution_clock::now();
            update_time = std::chrono::duration<double, std::milli>(t3 - t2).count();

            // --- Results After Update ---
            std::cout << "\n--- After Update (sequential) ---" << std::endl;
            print_sssp_result(sssp_result);
        }

        // --- Timings Summary ---
        std::cout << "\n--- Timings (Serial) ---" << std::endl;
        std::cout << "Initial Dijkstra: " << initial_time << " ms" << std::endl;
        if (update_time > 0.0) {
            std::cout << "Sequential Update: " << update_time << " ms" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nExecution finished." << std::endl;
    return 0;
}
