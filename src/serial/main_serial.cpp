#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>
#include "../../include/graph.hpp"
#include "../../include/utils.hpp"

SSSPResult dijkstra(const Graph& g, int source);
void process_batch_sequential(Graph& g, SSSPResult& sssp_result, const std::vector<EdgeChange>& batch);

void print_sssp_result(const SSSPResult& sssp_result) {
    std::cout << "Current SSSP Result:" << std::endl;
    for (size_t i = 0; i < sssp_result.dist.size(); ++i) {
        std::cout << "Vertex " << i << ": Dist = " << sssp_result.dist[i]
                  << ", Parent = " << sssp_result.parent[i] << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> <start_node> [changes_file]" << std::endl;
        return 1;
    }
    std::string filename = argv[1];
    int start_node = std::stoi(argv[2]);
    std::string changes_file = (argc > 3) ? argv[3] : "";

    try {
        std::cout << "Loading graph from " << filename << "..." << std::endl;
        Graph graph = load_graph(filename);
        std::cout << "Graph loaded: " << graph.num_vertices << " vertices." << std::endl;
        if (start_node < 0 || start_node >= graph.num_vertices) {
            std::cerr << "Error: Start node " << start_node << " is out of range [0, " << graph.num_vertices - 1 << "]." << std::endl;
            return 1;
        }
        std::cout << "\nCalculating initial SSSP from source " << start_node << " using Dijkstra..." << std::endl;
        auto start_time_initial = std::chrono::high_resolution_clock::now();
        SSSPResult sssp_result = dijkstra(graph, start_node);
        auto end_time_initial = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> initial_sssp_time = end_time_initial - start_time_initial;
        std::cout << "Initial SSSP calculated in " << initial_sssp_time.count() << " ms." << std::endl;
        print_sssp_result(sssp_result);

        if (!changes_file.empty()) {
            std::cout << "\nLoading changes from " << changes_file << "..." << std::endl;
            std::vector<EdgeChange> changes = load_edge_changes(changes_file);
            std::cout << "Changes loaded: " << changes.size() << " total." << std::endl;
            std::cout << "Processing batch sequentially..." << std::endl;
            auto start_time_update = std::chrono::high_resolution_clock::now();
            process_batch_sequential(graph, sssp_result, changes);
            auto end_time_update = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> update_time = end_time_update - start_time_update;
            std::cout << "\n--- After Update/Recompute (sequential) ---" << std::endl;
            print_sssp_result(sssp_result);
            std::cout << "\n--- Timings ---" << std::endl;
            std::cout << "Initial Dijkstra: " << initial_sssp_time.count() << " ms" << std::endl;
            std::cout << "Update/Recompute (sequential): " << update_time.count() << " ms" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "\nError: An exception occurred: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "\nExecution finished." << std::endl;
    return 0;
}
