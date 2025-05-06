// sssp_sequential.cpp - Sequential single-change and batch dynamic SSSP
// -----------------------------------------------------------------
// Implements in-place update of SSSPResult for single edge changes or batches
// using SPFA-like relaxation and tree-invalidation logic.

#include <queue>
#include <iostream>

#include "../../include/graph.hpp"

// SingleChange: apply one edge insertion or deletion to graph G and update T
// - change: EdgeChange (type, endpoints, weight)
// - G: graph to be modified (insert/delete edge)
// - T: current SSSP result (dist, parent) to be updated incrementally
void SingleChange(const EdgeChange& change, Graph& G, SSSPResult& T) {
    const int u = change.u;
    const int v = change.v;
    const Weight weight = change.weight; // Weight for insertion

    // --- State before the change ---
    const Weight old_dist_u = T.dist[u];
    const Weight old_dist_v = T.dist[v];
    const int old_parent_u = T.parent[u];
    const int old_parent_v = T.parent[v];
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
    } catch ([[maybe_unused]] const std::out_of_range& oor) { /* Ignore if u is invalid */ }


    // --- Apply the change to the graph structure ---
    try {
        if (change.type == ChangeType::INSERT) {
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


    if (change.type == ChangeType::INSERT) {
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
        // Invalidate the removed edge if it was a tree edge
        bool invalidated = false;
        std::queue<int> children_q;
        // Check v->u relation
        if (T.parent[v] == u) {
            T.dist[v] = INFINITY_WEIGHT;
            T.parent[v] = -1;
            add_to_queue(v);
            invalidated = true;
            children_q.push(v);
        } else if (T.parent[u] == v) {
            T.dist[u] = INFINITY_WEIGHT;
            T.parent[u] = -1;
            add_to_queue(u);
            invalidated = true;
            children_q.push(u);
        }
        // Invalidate all descendants of the removed tree edge
        while (!children_q.empty()) {
            int curr = children_q.front();
            children_q.pop();
            for (int i = 0; i < G.num_vertices; ++i) {
                if (T.parent[i] == curr) {
                    T.dist[i] = INFINITY_WEIGHT;
                    T.parent[i] = -1;
                    add_to_queue(i);
                    children_q.push(i);
                }
            }
        }
        // If edge not in tree, enqueue endpoints for potential new paths
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
             } catch ([[maybe_unused]] const std::out_of_range& oor) { /* Ignore if z is invalid */ }

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
             } catch ([[maybe_unused]] const std::out_of_range& oor) { /* Ignore if z is invalid */ }
        }
    }
}

// process_batch_sequential: apply a batch of changes sequentially
// - g: graph (modified in-place)
// - sssp_result: SSSPResult to update for all changes
// - batch: list of EdgeChange
void process_batch_sequential(Graph& g, SSSPResult& sssp_result, const std::vector<EdgeChange>& batch) {
    std::cout << "Processing batch of " << batch.size() << " changes sequentially (modifying graph internally)..." << std::endl;
    auto start_batch = std::chrono::high_resolution_clock::now();
    for (const auto& change : batch) {
        // std::cout << "  Processing change " << ++count << "/" << batch.size() << "..." << std::endl;
        // Basic validation of change indices before calling SingleChange
        if (change.u < 0 || change.u >= g.num_vertices || change.v < 0 || change.v >= g.num_vertices) {
             std::cerr << "Warning: Skipping change in batch involving out-of-bounds vertex: "
                       << (change.type == ChangeType::INSERT ? "insert " : "delete ") << change.u << " " << change.v << std::endl;
             continue;
        }
        SingleChange(change, g, sssp_result); // Modifies 'g' and 'sssp_result'
    }
    auto end_batch = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> batch_time = end_batch - start_batch;
     std::cout << "Sequential batch processing finished in " << batch_time.count() << " ms." << std::endl;
}
