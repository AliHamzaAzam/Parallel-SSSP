#include <iostream>
#include <vector>
#include <string>
#include "../../include/graph.hpp"
#include "../../include/utils.hpp"

// Forward declarations for OpenMP SSSP logic
SSSPResult dijkstra(const Graph& g, int source);
void TestDynamicSSSPWorkflow_OpenMP(Graph& G, SSSPResult& T, const std::vector<EdgeChange>& changes);

int parallel_main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: <graph_file> <start_node> [changes_file]" << std::endl;
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
        SSSPResult sssp_result = dijkstra(graph, start_node);
        std::cout << "Initial SSSP calculated." << std::endl;

        if (!changes_file.empty()) {
            std::cout << "\nLoading changes from " << changes_file << "..." << std::endl;
            std::vector<EdgeChange> changes = load_edge_changes(changes_file);
            std::cout << "Changes loaded: " << changes.size() << " total." << std::endl;
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
