// generate_updates.cpp - Generate random edge-change sequences for dynamic SSSP
// ----------------------------------------------------------------------------
// Loads a graph, then produces a mix of INSERT and DELETE operations
// and writes them to an updates file.

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <tuple>

#include "../include/graph.hpp"
#include "../include/utils.hpp"

// edge_exists: returns true if undirected edge (u,v) is in existing_edges
inline bool edge_exists(int u, int v, const std::set<std::pair<int, int>>& existing_edges) {
    return existing_edges.count({std::min(u, v), std::max(u, v)});
}

// main: generate 'num_updates' edge changes for 'input_graph_file' and write to 'output_updates_file'
// Args:
//   argc==4: input_graph_file, output_updates_file, num_updates
// Exits with code 0 on success, 1 on errors.
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

    // Build set and list of current edges in canonical form
    std::set<std::pair<int, int>> existing_edges;
    std::vector<std::pair<int, int>> edge_list; // For selecting random edges to delete
    // Accumulate current edge weights to compute global mean
    double sum_weights = 0.0;
    int edge_count0 = 0;
    edge_list.reserve(graph.get_edge_count()); // Approximate size

    for (int u = 0; u < graph.num_vertices; ++u) {
        for (const auto& edge : graph.adj[u]) {
            if (u < edge.to) {
                existing_edges.insert({u, edge.to});
                edge_list.push_back({u, edge.to});
                sum_weights += edge.weight;
                edge_count0++;
            }
        }
    }

    if (edge_list.empty() && graph.num_vertices > 0) {
        std::cout << "Warning: Graph has vertices but no edges. Only insertions will be generated." << std::endl;
    }
    // Compute mean and set weight distribution within ±10% of mean
    double global_mean = edge_count0 > 0 ? sum_weights / edge_count0 : 1.0;
    double dev_ratio = 0.1; // 10% deviation allowed
    double min_weight = std::max(0.0, global_mean * (1.0 - dev_ratio));
    double max_weight = global_mean * (1.0 + dev_ratio);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> vertex_dist(0, graph.num_vertices - 1);
    std::uniform_real_distribution<double> weight_dist(min_weight, max_weight); // Centered on global mean ±10%
    std::uniform_real_distribution<> action_dist(0.0, 1.0); // For choosing insert/delete

    int generated_count = 0;
    int max_attempts = num_updates * 100; // Limit attempts to avoid infinite loops
    int attempts = 0;

    // Randomly generate INSERT or DELETE operations until desired count
    while (generated_count < num_updates && attempts < max_attempts) {
        attempts++;
        bool try_insert = (edge_list.empty() || action_dist(gen) < 0.5); // 50% chance, or always insert if no edges to delete

        if (try_insert) {
            // Select a pair of distinct vertices without an existing edge
            int u, v;
            int attempts_inner = 0;
            do {
                if (++attempts_inner > max_attempts) break;
                u = vertex_dist(gen);
                v = vertex_dist(gen);
            } while (u == v || edge_exists(u, v, existing_edges));
            if (u != v && !edge_exists(u, v, existing_edges)) {
                double weight = weight_dist(gen);
                generated_updates.push_back({u, v, weight, ChangeType::INSERT});
                existing_edges.insert({std::min(u, v), std::max(u, v)});
                edge_list.push_back({std::min(u, v), std::max(u, v)});
                generated_count++;
            }

        } else {
            if (!edge_list.empty()) {
                std::uniform_int_distribution<> edge_idx_dist(0, edge_list.size() - 1);
                int edge_idx = edge_idx_dist(gen);

                std::pair<int, int> edge_to_delete = edge_list[edge_idx];
                int u = edge_to_delete.first;
                int v = edge_to_delete.second;

                if (edge_exists(u, v, existing_edges)) {
                    generated_updates.push_back({u, v, 0.0, ChangeType::DELETE}); // Weight doesn't matter for deletion
                    existing_edges.erase({u, v}); // Remove canonical form

                    std::swap(edge_list[edge_idx], edge_list.back());
                    edge_list.pop_back();

                    generated_count++;
                }
            }
        }
    }

    if (generated_count < num_updates) {
        std::cerr << "Warning: Could only generate " << generated_count << " unique updates after " << attempts << " attempts. The graph might be too dense or too sparse." << std::endl;
    }

    // Write updates to output file in lines 'i u v weight' or 'd u v'
    std::cout << "Writing " << generated_updates.size() << " updates to: " << output_updates_file << std::endl;
    std::ofstream outfile(output_updates_file);
    if (!outfile.is_open()) {
        std::cerr << "Error opening output file: " << output_updates_file << std::endl;
        return 1;
    }

    outfile << "# Generated updates for graph: " << input_graph_file << std::endl;
    outfile << "# Total updates: " << generated_updates.size() << std::endl;
    for (const auto& change : generated_updates) {
        if (change.type == ChangeType::INSERT) {
            outfile << "i " << change.u << " " << change.v << " " << change.weight << std::endl;
        } else {
            outfile << "d " << change.u << " " << change.v << std::endl;
        }
    }

    outfile.close();
    std::cout << "Update file generated successfully." << std::endl;

    return 0;
}
