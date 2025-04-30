#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <tuple> // For std::ignore

#include "../include/graph.h"
#include "../include/utils.hpp"

// Helper function to check if an edge exists (considers undirected nature)
bool edge_exists(int u, int v, const std::set<std::pair<int, int>>& existing_edges) {
    return existing_edges.count({std::min(u, v), std::max(u, v)});
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_graph_file> <output_updates_file> <num_updates>" << std::endl;
        return 1;
    }

    std::string input_graph_file = argv[1];
    std::string output_updates_file = argv[2];
    int num_updates = 0;
    try {
        num_updates = std::stoi(argv[3]);
        if (num_updates <= 0) {
            throw std::invalid_argument("Number of updates must be positive.");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid number of updates provided: " << argv[3] << ". " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Loading graph from: " << input_graph_file << std::endl;
    // Initialize with 0 vertices
    Graph graph(0);
    try {
        graph = load_graph(input_graph_file);
    } catch (const std::exception& e) {
        std::cerr << "Error loading graph: " << e.what() << std::endl;
        return 1;
    }

    if (graph.num_vertices == 0) {
        std::cerr << "Error: Graph is empty, cannot generate updates." << std::endl;
        return 1;
    }

    std::cout << "Generating " << num_updates << " updates for graph with " << graph.num_vertices << " vertices." << std::endl;

    std::vector<EdgeChange> generated_updates;
    generated_updates.reserve(num_updates);

    // Use a set to keep track of existing edges for efficient lookup during insertion
    // Store edges in a canonical form (min_node, max_node) to handle undirected nature easily
    std::set<std::pair<int, int>> existing_edges;
    std::vector<std::pair<int, int>> edge_list; // For selecting random edges to delete
    edge_list.reserve(graph.get_edge_count()); // Approximate size

    for (int u = 0; u < graph.num_vertices; ++u) {
        for (const auto& edge : graph.adj[u]) {
            // Only add edge once for undirected representation in the set/list
            if (u < edge.to) {
                existing_edges.insert({u, edge.to});
                edge_list.push_back({u, edge.to});
            }
        }
    }

    if (edge_list.empty() && graph.num_vertices > 0) {
        std::cout << "Warning: Graph has vertices but no edges. Only insertions will be generated." << std::endl;
    }


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> vertex_dist(0, graph.num_vertices - 1);
    std::uniform_real_distribution<double> weight_dist(1.0, 100.0); // Example weight range
    std::uniform_real_distribution<> action_dist(0.0, 1.0); // For choosing insert/delete

    int generated_count = 0;
    int max_attempts = num_updates * 100; // Limit attempts to avoid infinite loops
    int attempts = 0;

    while (generated_count < num_updates && attempts < max_attempts) {
        attempts++;
        bool try_insert = (edge_list.empty() || action_dist(gen) < 0.5); // 50% chance, or always insert if no edges to delete

        if (try_insert) {
            // --- Try Insertion ---
            int u = vertex_dist(gen);
            int v = vertex_dist(gen);
            if (u == v) continue; // Avoid self-loops

            if (!edge_exists(u, v, existing_edges)) {
                double weight = weight_dist(gen);
                generated_updates.push_back({u, v, weight, true});
                existing_edges.insert({std::min(u, v), std::max(u, v)});
                // Add to edge_list as well, so it *could* be deleted later in the sequence
                edge_list.push_back({std::min(u, v), std::max(u, v)}); 
                generated_count++;
            }
            // Else: edge already exists, try again in the next iteration

        } else {
            // --- Try Deletion ---
            if (!edge_list.empty()) {
                 std::uniform_int_distribution<> edge_idx_dist(0, edge_list.size() - 1);
                 int edge_idx = edge_idx_dist(gen);
                 
                 std::pair<int, int> edge_to_delete = edge_list[edge_idx];
                 int u = edge_to_delete.first;
                 int v = edge_to_delete.second;

                 // Check if it *still* exists (might have been deleted in a previous step)
                 if (edge_exists(u, v, existing_edges)) {
                     generated_updates.push_back({u, v, 0.0, false}); // Weight doesn't matter for deletion
                     existing_edges.erase({u, v}); // Remove canonical form
                     
                     // Remove from edge_list efficiently: swap with last and pop
                     std::swap(edge_list[edge_idx], edge_list.back());
                     edge_list.pop_back();

                     generated_count++;
                 }
                 // Else: edge was already deleted (or somehow removed), try again
            }
             // If edge_list became empty during generation, the next iteration will force insertion
        }
    }

     if (generated_count < num_updates) {
        std::cerr << "Warning: Could only generate " << generated_count << " unique updates after " << attempts << " attempts. The graph might be too dense or too sparse." << std::endl;
    }


    std::cout << "Writing " << generated_updates.size() << " updates to: " << output_updates_file << std::endl;
    std::ofstream outfile(output_updates_file);
    if (!outfile.is_open()) {
        std::cerr << "Error opening output file: " << output_updates_file << std::endl;
        return 1;
    }

    outfile << "# Generated updates for graph: " << input_graph_file << std::endl;
    outfile << "# Total updates: " << generated_updates.size() << std::endl;
    for (const auto& change : generated_updates) {
        if (change.is_insertion) {
            outfile << "i " << change.u << " " << change.v << " " << change.weight << std::endl;
        } else {
            outfile << "d " << change.u << " " << change.v << std::endl;
        }
    }

    outfile.close();
    std::cout << "Update file generated successfully." << std::endl;

    return 0;
}
