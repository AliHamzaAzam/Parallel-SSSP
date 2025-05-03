//
// Created by Ali Hamza Azam on 25/04/2025.
//

#include "../include/utils.hpp"
#include "../include/graph.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <tuple> // Required for std::tuple
#include <algorithm> // Required for std::max
#include <cstdlib> // Required for exit, EXIT_FAILURE
#include <stdexcept> // Required for std::runtime_error

// Function to load a graph from a file (supports simple edge list and .mtx)
Graph load_graph(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    std::string line;

    // --- Check file format based on extension --- 
    bool is_mtx = false;
    if (filename.length() >= 5 && filename.substr(filename.length() - 4) == ".mtx") {
        is_mtx = true;
    }
    // Add check for .edges extension, treat it as simple edge list
    bool is_edges = false;
    if (filename.length() >= 6 && filename.substr(filename.length() - 6) == ".edges") {
        is_edges = true;
        is_mtx = false; // Ensure it's not treated as mtx if extension is .edges
    }

    // If not explicitly .mtx or .edges, assume simple edge list for now
    // A more robust check might involve peeking file content
    if (!is_mtx && !is_edges) {
        std::cout << "Warning: Unknown file extension for '" << filename << "'. Assuming simple edge list format (u v weight)." << std::endl;
    }

    Graph g(0); // Initialize with 0 vertices, will resize later
    int num_vertices = 0;
    long long num_edges_expected = 0; // Use long long for potentially large files
    long long edge_count = 0;

    if (is_mtx) {
        std::cout << "Reading Matrix Market file: " << filename << std::endl;
        // --- Read MTX Header --- 
        getline(infile, line); // Read the %%MatrixMarket or %MatrixMarket line
        // Check for symmetry, format (coordinate), field (real, integer, pattern)
        // For now, assume coordinate real/integer/pattern and general (non-symmetric)

        // --- Skip Comments --- 
        while (getline(infile, line) && line[0] == '%') {
            // Skip comment lines
        }

        // --- Read Dimensions --- 
        std::stringstream ss_dims(line);
        int rows, cols;
        if (!(ss_dims >> rows >> cols >> num_edges_expected)) {
            throw std::runtime_error("Error reading MTX dimensions line: " + line);
        }
        if (rows <= 0 || cols <= 0) {
             throw std::runtime_error("Invalid dimensions in MTX file: " + line);
        }
        // Assuming graph is represented by the matrix, rows determine vertices
        num_vertices = rows; 
        g = Graph(num_vertices); // Resize graph
        std::cout << "MTX dimensions: " << rows << "x" << cols << ", expecting " << num_edges_expected << " edges." << std::endl;

        // --- Read Edges (MTX) --- 
        int u_idx, v_idx;
        double weight_val = 1.0; // Default weight for pattern graphs
        int line_num = 0; // Line counter after dimensions line
        // We already read the dimension line, start reading edges
        while (getline(infile, line)) {
            line_num++;
            // Trim leading/trailing whitespace (basic)
            line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
            line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

            if (line.empty() || line[0] == '%') continue; // Skip empty lines/comments

            std::stringstream ss_edge(line);
            if (ss_edge >> u_idx >> v_idx) {
                 // MTX is 1-based, adjust to 0-based
                 int u = u_idx - 1;
                 int v = v_idx - 1;

                 // Try reading weight, if it fails, use default (for pattern)
                 // Reset weight_val to default before trying to read
                 weight_val = 1.0;
                 if (!(ss_edge >> weight_val)) {
                     // Weight reading failed or wasn't present, keep default 1.0
                     // Clear potential error state from failed weight read
                     ss_edge.clear(); 
                 }

                 // Optional: Check if anything remains unparsed on the line
                 std::string remaining;
                 if (ss_edge >> remaining) {
                     std::cerr << "Warning: Extra data found on edge line " << line_num << ": '" << remaining << "' in file: " << filename << ". Line: '" << line << "'" << std::endl;
                 }


                 if (u < 0 || u >= num_vertices || v < 0 || v >= num_vertices) {
                     std::cerr << "Warning: Skipping edge line " << line_num << " with out-of-bounds vertex index (" 
                               << u_idx << "," << v_idx << ") in MTX file. Max vertex index should be " << num_vertices -1 << "." << std::endl;
                     continue;
                 }
                 g.add_edge(u, v, weight_val);
                 edge_count++;
                 // Note: For symmetric MTX, add_edge adds both directions, which is correct.
                 // edge_count tracks lines read, matching num_edges_expected from header.
            } else {
                 // This is where the error occurs
                 std::cerr << "Warning: Skipping malformed edge line " << line_num << " in MTX file: '" << line << "'" << std::endl;
                 // Add more diagnostics: what is the stream state?
                 std::cerr << "    Stream state after trying to read u,v: good=" << ss_edge.good() << ", eof=" << ss_edge.eof() << ", fail=" << ss_edge.fail() << ", bad=" << ss_edge.bad() << std::endl;
            }
        }
        if (edge_count != num_edges_expected) {
             std::cerr << "Warning: Expected " << num_edges_expected << " edges, but read " << edge_count << " edges from MTX file." << std::endl;
        }

    } else { // Handle .edges or other simple edge list formats
        if (is_edges) {
             std::cout << "Reading .edges file: " << filename << std::endl;
        } else {
             std::cout << "Reading simple edge list file: " << filename << std::endl;
        }
        // --- Read Simple Edge List Format (u v weight) --- 
        std::vector<std::tuple<int, int, double>> edges;
        int max_vertex_id = -1;
        int line_num = 0;
        while (getline(infile, line)) {
            line_num++;
            // Trim leading/trailing whitespace (basic)
            line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
            line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

            if (line.empty() || line[0] == '#' || line[0] == '%') continue; // Skip comments/empty
            std::stringstream ss(line);
            int u, v;
            double weight;
            // Expecting 0-based indices directly from .edges file
            if (ss >> u >> v >> weight) {
                // Basic validation for non-negative indices
                if (u < 0 || v < 0) {
                    std::cerr << "Warning: Skipping edge line " << line_num << " with negative vertex index (" 
                              << u << "," << v << ") in file: " << filename << ". Line: '" << line << "'" << std::endl;
                    continue;
                }
                edges.emplace_back(u, v, weight);
                max_vertex_id = std::max({max_vertex_id, u, v});
                edge_count++;
            } else {
                 std::cerr << "Warning: Skipping malformed edge line " << line_num << " in file: " << filename << ". Line: '" << line << "'" << std::endl;
            }
        }

        if (max_vertex_id == -1) {
            // Handle case where file might be empty or contain only comments
            if (edge_count == 0) {
                 std::cout << "Warning: No valid edges found in file: " << filename << ". Creating an empty graph." << std::endl;
                 g = Graph(0);
                 return g; // Return empty graph
            } else {
                 // This case should ideally not happen if max_vertex_id tracks correctly
                 throw std::runtime_error("Error: No valid vertex IDs found despite reading edges.");
            }
        }

        num_vertices = max_vertex_id + 1; // Vertices are 0 to max_vertex_id
        g = Graph(num_vertices); // Resize graph

        for (const auto& edge : edges) {
            g.add_edge(std::get<0>(edge), std::get<1>(edge), std::get<2>(edge));
        }
    }

    std::cout << "Loaded graph with " << num_vertices << " vertices and " << edge_count << " edges (from file lines)." << std::endl;

    return g;
}

// Function to load edge changes from a file (format: type u v [weight])
// type: 'i' for insertion, 'd' for deletion
std::vector<EdgeChange> load_edge_changes(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<EdgeChange> changes;
    std::string line;
    char type;
    int u, v;
    double weight;

    while (getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        if (ss >> type >> u >> v) {
            if (type == 'i') {
                if (ss >> weight) {
                    changes.push_back({u, v, weight, ChangeType::INSERT});
                } else {
                    std::cerr << "Warning: Skipping malformed insertion line (missing weight): " << line << std::endl;
                }
            } else if (type == 'd') {
                changes.push_back({u, v, 0.0, ChangeType::DELETE}); // Weight ignored for deletion
            } else {
                 std::cerr << "Warning: Skipping malformed line (invalid type): " << line << std::endl;
            }
        } else {
            std::cerr << "Warning: Skipping malformed line: " << line << std::endl;
        }
    }
     std::cout << "Loaded " << changes.size() << " edge changes." << std::endl;
    return changes;
}

// Dijkstra SSSP implementation for shared use
SSSPResult dijkstra(const Graph& g, int source) {
    int n = g.num_vertices;
    SSSPResult result(n);
    result.dist[source] = 0;
    using P = std::pair<double, int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
    pq.push({0, source});
    while (!pq.empty()) {
        double d = pq.top().first;
        int u = pq.top().second;
        pq.pop();
        if (d > result.dist[u]) continue;
        for (const auto& edge : g.neighbors(u)) {
            int v = edge.to;
            double weight = edge.weight;
            if (result.dist[u] + weight < result.dist[v]) {
                result.dist[v] = result.dist[u] + weight;
                result.parent[v] = u;
                pq.push({result.dist[v], v});
            }
        }
    }
    return result;
}
