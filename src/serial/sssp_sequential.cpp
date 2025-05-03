//
// Created by Ali Hamza Azam on 25/04/2025.
//

#include "../../include/graph.hpp"
#include "../../include/utils.hpp"
#include <vector>
#include <queue>
#include <limits>
#include <iostream>
#include <stdexcept> // For std::runtime_error

// Constants
#define INF std::numeric_limits<double>::max()

// Baseline Dijkstra Algorithm - REMOVED definition, declared in utils.hpp and defined in utils.cpp
// SSSPResult dijkstra(const Graph& g, int source) { ... }


// Overload for internal use? Check if this is still needed.
void dijkstra(const Graph& graph, int start_node, SSSPResult& result) {
    int num_vertices = graph.num_vertices;
    result.dist.assign(num_vertices, INF);
    result.parent.assign(num_vertices, -1);
    result.dist[start_node] = 0;

    // Use Weight (double) for priority queue distance
    using PDI = std::pair<Weight, int>;
    std::priority_queue<PDI, std::vector<PDI>, std::greater<PDI>> pq;
    pq.push({0.0, start_node}); // Use 0.0 for double

    while (!pq.empty()) {
        Weight d = pq.top().first; // Use Weight
        int u = pq.top().second;
        pq.pop();

        // Use INFINITY_WEIGHT for comparison
        if (d > result.dist[u] && result.dist[u] != INFINITY_WEIGHT) {
            continue;
        }

        for (const auto& edge : graph.neighbors(u)) { // Use neighbors() for const correctness
            int v = edge.to;
            Weight weight = edge.weight; // Use Weight

            if (result.dist[u] != INFINITY_WEIGHT && result.dist[u] + weight < result.dist[v]) {
                result.dist[v] = result.dist[u] + weight;
                result.parent[v] = u;
                pq.push({result.dist[v], v});
            }
        }
    }
}

// Helper function for SingleChange (part of Algorithm 1)
// Propagates updates from a single vertex z
// Returns true if any neighbor's distance was updated, false otherwise
// NOTE: This helper is not directly used in the refined SingleChange below.
// bool UpdatedSSSP(int z, const Graph& G, SSSPResult& T) { ... }


// Algorithm 1: Sequential SSSP Update for a Single Change
// IMPORTANT REFACTOR: This version assumes the graph 'G' ALREADY reflects the change.
// It needs the 'before' state information (like old weight or parent relationship)
// passed implicitly or explicitly to correctly handle deletions.
// The current implementation tries to deduce based on the *current* graph state,
// which is complex and potentially incorrect after the change is applied.
// A better approach for sequential updates is often to NOT pre-modify the graph
// and let SingleChange handle both the graph update and SSSP propagation.
// ---
// REVERTING to the logic where SingleChange handles the graph modification itself.
// This means main.cpp should NOT pre-modify the graph when calling the sequential version.
void SingleChange(const EdgeChange& change, Graph& G, SSSPResult& T) {
    int u = change.u;
    int v = change.v;
    Weight weight = change.weight; // Weight for insertion

    // --- State before the change ---
    Weight old_dist_u = T.dist[u];
    Weight old_dist_v = T.dist[v];
    int old_parent_u = T.parent[u];
    int old_parent_v = T.parent[v];
    Weight existing_weight = INFINITY_WEIGHT; // Weight of edge (u,v) if it exists before change

    // Find existing edge weight (needed for deletion logic)
    bool edge_existed = false;
    try {
        for (const auto& edge : G.neighbors(u)) {
            if (edge.to == v) {
                existing_weight = edge.weight;
                edge_existed = true;
                break;
            }
        }
    } catch (const std::out_of_range& oor) { /* Ignore if u is invalid */ }


    // --- Apply the change to the graph structure ---
    try {
        if (change.is_insertion) {
            // If edge exists, update weight? Or assume new edge? Let's assume add_edge handles duplicates if needed.
            G.add_edge(u, v, weight);
        } else { // Deletion
            if (edge_existed) {
                G.remove_edge(u, v);
            } else {
                // std::cerr << "Warning: Attempting to delete non-existent edge (" << u << "," << v << ")" << std::endl;
                // No structural change, but might still trigger SSSP update if paths change indirectly (rare)
                // For simplicity, we can return if the edge didn't exist.
                return;
            }
        }
    } catch (const std::exception& e) {
         std::cerr << "Error modifying graph in SingleChange: " << e.what() << std::endl;
         return; // Stop processing this change if graph modification fails
    }


    // --- SSSP Update Logic ---
    std::queue<int> pq; // Queue for vertices needing re-evaluation (BFS-like propagation)
    // Use std::vector<bool> to track nodes in the queue to avoid duplicates
    std::vector<bool> in_queue(G.num_vertices, false);

    auto add_to_queue = [&](int node) {
        if (node >= 0 && node < G.num_vertices && !in_queue[node]) {
            pq.push(node);
            in_queue[node] = true;
        }
    };


    if (change.is_insertion) {
        // Check if the new/updated edge offers a shorter path
        if (old_dist_u != INFINITY_WEIGHT && old_dist_u + weight < T.dist[v]) {
            T.dist[v] = old_dist_u + weight;
            T.parent[v] = u;
            add_to_queue(v);
        }
        if (old_dist_v != INFINITY_WEIGHT && old_dist_v + weight < T.dist[u]) {
            T.dist[u] = old_dist_v + weight;
            T.parent[u] = v;
            add_to_queue(u);
        }
    } else { // Deletion
        // If the deleted edge was part of the SSSP tree for either u or v
        const double epsilon = 1e-9;
        bool invalidated = false;
        if (old_parent_v == u && std::abs((old_dist_u + existing_weight) - old_dist_v) < epsilon) {
            // v depended on u via the deleted edge
            T.dist[v] = INFINITY_WEIGHT;
            T.parent[v] = -1;
            add_to_queue(v);
            invalidated = true;
            // Need to invalidate children of v as well
            std::queue<int> children_q;
            children_q.push(v);
            while(!children_q.empty()){
                int curr = children_q.front();
                children_q.pop();
                // Inefficiently find children
                 for(int i=0; i<G.num_vertices; ++i){
                     if(T.parent[i] == curr){
                         if(T.dist[i] != INFINITY_WEIGHT){
                             T.dist[i] = INFINITY_WEIGHT;
                             T.parent[i] = -1;
                             add_to_queue(i); // Add invalidated child to main queue
                             children_q.push(i); // Add to explore its children
                         }
                     }
                 }
            }

        }
        // Symmetrically check if u depended on v
        if (old_parent_u == v && std::abs((old_dist_v + existing_weight) - old_dist_u) < epsilon) {
             // u depended on v via the deleted edge
            if (T.dist[u] != INFINITY_WEIGHT) { // Avoid double invalidation if symmetric check runs
                T.dist[u] = INFINITY_WEIGHT;
                T.parent[u] = -1;
                add_to_queue(u);
                invalidated = true;
                 // Invalidate children of u
                std::queue<int> children_q;
                children_q.push(u);
                while(!children_q.empty()){
                    int curr = children_q.front();
                    children_q.pop();
                     for(int i=0; i<G.num_vertices; ++i){
                         if(T.parent[i] == curr){
                             if(T.dist[i] != INFINITY_WEIGHT){
                                 T.dist[i] = INFINITY_WEIGHT;
                                 T.parent[i] = -1;
                                 add_to_queue(i);
                                 children_q.push(i);
                             }
                         }
                     }
                }
            }
        }

        // If not part of the tree, deletion might still enable alternative paths for u or v
        // Add u and v to the queue to re-evaluate their paths just in case
        if (!invalidated) {
             add_to_queue(u);
             add_to_queue(v);
        }
    }

    // --- Propagate updates (Bellman-Ford/SPFA-like relaxation) ---
    while (!pq.empty()) {
        int z = pq.front();
        pq.pop();
        in_queue[z] = false; // Mark as removed from queue

        // Try to find a better path *to* z from its neighbors (especially if z was invalidated)
        // This step is crucial after deletions to potentially reconnect nodes
        if (T.dist[z] == INFINITY_WEIGHT) { // If z is currently unreachable
             Weight min_dist_z = INFINITY_WEIGHT;
             int best_parent_z = -1;
             try {
                 for (const auto& edge : G.neighbors(z)) {
                     int neighbor = edge.to;
                     if (neighbor >= 0 && neighbor < G.num_vertices && T.dist[neighbor] != INFINITY_WEIGHT) {
                         if (T.dist[neighbor] + edge.weight < min_dist_z) {
                             min_dist_z = T.dist[neighbor] + edge.weight;
                             best_parent_z = neighbor;
                         }
                     }
                 }
             } catch (const std::out_of_range& oor) { /* Ignore if z is invalid */ }

             if (min_dist_z < T.dist[z]) { // Found a new path to z
                 T.dist[z] = min_dist_z;
                 T.parent[z] = best_parent_z;
                 // Since z's distance improved, it needs to be processed again to update its neighbors
                 add_to_queue(z);
                 continue; // Skip the neighbor update below for this iteration, do it next time
             }
        }


        // Propagate updates *from* z to its neighbors
        if (T.dist[z] != INFINITY_WEIGHT) { // Only propagate if z is reachable
             try {
                 for (const auto& edge : G.neighbors(z)) {
                     int n = edge.to;
                     Weight weight_zn = edge.weight;
                      if (n >= 0 && n < G.num_vertices) {
                         if (T.dist[z] + weight_zn < T.dist[n]) {
                             T.dist[n] = T.dist[z] + weight_zn;
                             T.parent[n] = z;
                             add_to_queue(n); // Add neighbor to queue for further propagation
                         }
                     }
                 }
             } catch (const std::out_of_range& oor) { /* Ignore if z is invalid */ }
        }
    }
}


// Batch processing function for sequential updates
// IMPORTANT: Assumes main.cpp does NOT pre-modify the graph for sequential mode.
// This function calls SingleChange which modifies the graph internally.
void process_batch_sequential(Graph& g, SSSPResult& sssp_result, const std::vector<EdgeChange>& batch) {
    std::cout << "Processing batch of " << batch.size() << " changes sequentially (modifying graph internally)..." << std::endl;
    auto start_batch = std::chrono::high_resolution_clock::now();
    int count = 0;
    for (const auto& change : batch) {
        // std::cout << "  Processing change " << ++count << "/" << batch.size() << "..." << std::endl;
        // Basic validation of change indices before calling SingleChange
        if (change.u < 0 || change.u >= g.num_vertices || change.v < 0 || change.v >= g.num_vertices) {
             std::cerr << "Warning: Skipping change in batch involving out-of-bounds vertex: "
                       << (change.is_insertion ? "insert " : "delete ") << change.u << " " << change.v << std::endl;
             continue;
        }
        SingleChange(change, g, sssp_result); // Modifies 'g' and 'sssp_result'
    }
    auto end_batch = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> batch_time = end_batch - start_batch;
     std::cout << "Sequential batch processing finished in " << batch_time.count() << " ms." << std::endl;
}

