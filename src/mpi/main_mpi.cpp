// main_mpi.cpp - Distributed Dynamic SSSP using MPI and OpenMP
// ------------------------------------------------------------
// Entry point and orchestration for loading the graph, partitioning,
// scattering subgraphs, and performing dynamic SSSP updates in parallel.
// Coordinates MPI communication and collects results back on rank 0.

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <mpi.h>
#include <metis.h>
#include <map>
#include <cstdlib>
#include <cstring>

#include "../../include/graph.hpp"
#include "../../include/utils.hpp"

// Add a type alias for boundary edges before forward declarations
using BoundaryEdges = std::map<int,std::vector<Edge>>;
// Forward declaration for distributed dynamic SSSP update (signature updated)
void Distributed_DynamicSSSP_MPI(
    const Graph& local_graph,                               // Input: local graph structure
    const BoundaryEdges& local_boundary_edges,              // Input: boundary edges
    const std::vector<int>& local_to_global,                // Input: local to global mapping
    const std::vector<int>& global_to_local,                // Input: global to local mapping
    std::vector<double>& dist,                              // Input/Output: local distances (in/out)
    std::vector<int>& parent,                               // Input/Output: local parents (in/out)
    int my_rank,                                            // Input: current MPI rank
    const std::vector<int>& part,                           // Input: partition assignment for each global vertex
    int source,                                             // Input: source vertex for SSSP
    const std::vector<bool>& initial_affected_del,          // Initial affected_del status for local vertices
    const std::vector<bool>& initial_affected,              // Initial affected status for local vertices
    bool debug_output = false                               // Input: debug output flag (default: false)
);

// setup_local_data: extract the subgraph owned by this rank
// - global_graph: full graph on rank 0
// - part: partition assignment for each global vertex
// - my_rank: current MPI rank
// Outputs:
//   local_graph         : induced subgraph of owned vertices
//   local_to_global     : map from local indices to global IDs
//   global_to_local     : inverse map (global ID -> local index or -1)
//   local_boundary_edges: incoming edges from remote vertices
//   local_dist, local_parent, local_affected_del, local_affected:
//                        initial state for dynamic update
void setup_local_data(
    const Graph& global_graph,                              // Input: full graph on rank 0
    const std::vector<idx_t>& part,                         // Input: partition assignment for each global vertex
    int my_rank,                                            // Input: current MPI rank
    Graph& local_graph,                                     // Output: local graph structure
    std::vector<int>& local_to_global,                      // Output: mapping local index -> global index
    std::vector<int>& global_to_local,                      // Output: mapping global index -> local index (-1 if not local)
    BoundaryEdges& local_boundary_edges,                    // Output: boundary edges map (INCOMING edges: local_v -> {global_u, weight})
    const SSSPResult& initial_sssp_result,                  // Input: Full initial result (only needed on rank 0)
    const std::vector<bool>& initial_affected_del_global,   // Input: Full initial affected_del (only needed on rank 0)
        const std::vector<bool>& initial_affected_global,   // Input: Full initial affected (only needed on rank 0)
    std::vector<double>& local_dist,                        // Output: local distances
    std::vector<int>& local_parent,                         // Output: local parents (stores local index if parent is local, global index if remote)
    std::vector<bool>& local_affected_del,                  // Output: local affected_del
    std::vector<bool>& local_affected                       // Output: local affected
) {
    const int n_global = global_graph.num_vertices;
    std::cout << "[Rank " << my_rank << "] Starting local data setup for " << n_global << " global vertices." << std::endl;

    // 1. Identify local vertices and create initial mappings
    local_to_global.clear();
    global_to_local.assign(n_global, -1);
    for (int v_global = 0; v_global < n_global; ++v_global) {
        if (part[v_global] == my_rank) {
            // Assign next available local index
            const int v_local = local_to_global.size();
            local_to_global.push_back(v_global);
            global_to_local[v_global] = v_local;
        }
    }
    const int n_local = local_to_global.size();
    std::cout << "[Rank " << my_rank << "] Identified " << n_local << " local vertices." << std::endl;

    // 2. Initialize local graph and state vectors
    local_graph = Graph(n_local); // Initialize local graph with correct size
    local_dist.resize(n_local);
    local_parent.resize(n_local);
    local_affected_del.resize(n_local);
    local_affected.resize(n_local);
    local_boundary_edges.clear(); // Clear the boundary edges map

    // 3. Populate local state vectors from global initial state (only rank 0 has full initial state initially)
    //    (For ranks > 0, this will be overwritten by received data later)
    if (my_rank == 0) {
         std::cout << "[Rank 0] Populating local state from initial SSSP result." << std::endl;
         for (int u_local = 0; u_local < n_local; ++u_local) {
             const int u_global = local_to_global[u_local];
             // Add bounds check for safety
             if (u_global >= 0 && u_global < initial_sssp_result.dist.size()) {
                 local_dist[u_local] = initial_sssp_result.dist[u_global];
                 // Store global parent ID initially. Will be converted below.
                 local_parent[u_local] = initial_sssp_result.parent[u_global];
             } else {
                 local_dist[u_local] = INFINITY_WEIGHT; // Default if mapping is wrong
                 local_parent[u_local] = -1;
             }
             if (u_global >= 0 && u_global < initial_affected_del_global.size()) {
                 local_affected_del[u_local] = initial_affected_del_global[u_global];
             } else {
                 local_affected_del[u_local] = false;
             }
              if (u_global >= 0 && u_global < initial_affected_global.size()) {
                 local_affected[u_local] = initial_affected_global[u_global];
             } else {
                 local_affected[u_local] = false;
             }
         }
         std::cout << "[Rank 0] Local state populated." << std::endl;
    }


    // 4. Build local graph (internal edges) AND populate incoming boundary edges map
    std::cout << "[Rank " << my_rank << "] Building local graph structure and boundary edges..." << std::endl;
    int internal_edges = 0;
    int boundary_edges_count = 0; // Count of incoming boundary edges

    // Iterate through all global vertices u_global to find edges
    for (int u_global = 0; u_global < n_global; ++u_global) {
        try {
            for (const auto& edge : global_graph.neighbors(u_global)) {
                int v_global = edge.to;
                Weight weight = edge.weight;

                // Check if the destination v_global is local to this rank
                if (v_global >= 0 && v_global < n_global && part[v_global] == my_rank) {
                    int v_local = global_to_local[v_global];
                    if (v_local == -1) continue; // Should not happen

                    // Now check if the source u_global is local or remote
                    if (part[u_global] == my_rank) {
                        // Source u_global is also local -> Internal edge
                        int u_local = global_to_local[u_global];
                        if (u_local != -1) {
                            local_graph.add_edge(u_local, v_local, weight);
                            internal_edges++;
                        }
                    } else {
                        // Source u_global is remote -> Incoming boundary edge
                        // Store edge: key=local destination (v_local), value={global source (u_global), weight}
                        local_boundary_edges[v_local].push_back({u_global, weight});
                        boundary_edges_count++;
                    }
                }
                // else: Destination v_global is remote, ignore this edge for local setup
            }
        } catch ([[maybe_unused]] const std::out_of_range& oor) {
             std::cerr << "[Rank " << my_rank << "] Warning: Out-of-range access for neighbors of global vertex " << u_global << ". Skipping." << std::endl;
        }
    }

     std::cout << "[Rank " << my_rank << "] Local graph built. Internal edges added: " << internal_edges << ". Incoming boundary edges stored: " << boundary_edges_count << std::endl;
     std::cout << "[Rank " << my_rank << "] Local data setup finished." << std::endl;
}

// mpi_main: parse arguments, broadcast parameters, load graph,
// perform initial Dijkstra on rank 0, partition graph, compute initial affected set,
// distribute subgraphs to worker ranks, invoke Distributed_DynamicSSSP_MPI,
// and gather and print final results on rank 0.
int mpi_main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Add debug flag to control verbose output
    bool debug_output = false;
    // Check for environment variable to enable debug output
    char* debug_env = getenv("SSSP_DEBUG");
    if (debug_env != nullptr && (strcmp(debug_env, "1") == 0 ||
                                strcmp(debug_env, "true") == 0 ||
                                strcmp(debug_env, "yes") == 0)) {
        debug_output = true;
        if (rank == 0) std::cout << "Debug output enabled via SSSP_DEBUG environment variable" << std::endl;
    }

    std::string filename;
    int start_node = -1;
    std::string changes_filename;
    idx_t num_partitions = size;
    Graph graph(0); // Local graph variable
    std::vector<EdgeChange> changes;
    int num_changes = 0;
    SSSPResult initial_sssp_result(0);        // Initialize with size 0 using the constructor
    std::vector<bool> affected_del;             // For Rank 0 initial calculation
    std::vector<bool> affected;                 // For Rank 0 initial calculation
    std::vector<int> part;                      // Partition vector (size known after graph load, use int for MPI_INT)

    // --- Argument Parsing and Broadcasting ---
    if (rank == 0) {
        // ... (Argument parsing logic as before - sets filename, start_node, changes_filename, num_partitions) ...
        if (argc < 2) { std::cerr << "Usage (from main): <graph_file.ext> <start_node> [changes_file] [num_partitions]" << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
        filename = argv[0];
        try { start_node = std::stoi(argv[1]); } catch ([[maybe_unused]] const std::exception& e) { std::cerr << "Error: Invalid start node provided: " << argv[1] << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
        if (argc > 2) changes_filename = argv[2];
        if (argc > 3) { try { num_partitions = std::max((idx_t)1, (idx_t)std::stoi(argv[3])); } catch (const std::exception& e) { std::cerr << "Warning: Invalid number of partitions provided: '" << argv[3] << "'. Using default (" << size << "). Error: " << e.what() << std::endl; num_partitions = size; } }
        else { if (rank == 0) std::cout << "Number of partitions not specified. Defaulting to number of MPI ranks (" << size << ")." << std::endl; num_partitions = size; } // Default if not provided

        // Broadcast necessary parameters
        int fn_len = filename.length();
        MPI_Bcast(&fn_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(const_cast<char*>(filename.c_str()), fn_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        int cfn_len = changes_filename.length();
        MPI_Bcast(&cfn_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (cfn_len > 0) MPI_Bcast(const_cast<char*>(changes_filename.c_str()), cfn_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(&start_node, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&num_partitions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        // Receive parameters
        int fn_len; MPI_Bcast(&fn_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<char> fn_buffer(fn_len + 1); MPI_Bcast(fn_buffer.data(), fn_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        filename = std::string(fn_buffer.data());
        int cfn_len; MPI_Bcast(&cfn_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (cfn_len > 0) { std::vector<char> cfn_buffer(cfn_len + 1); MPI_Bcast(cfn_buffer.data(), cfn_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD); changes_filename = std::string(cfn_buffer.data()); }
        else { changes_filename = ""; }
        MPI_Bcast(&start_node, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&num_partitions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // --- Graph Loading (All ranks load for now) ---
    try {
        if (rank == 0) std::cout << "Rank " << rank << " loading graph from " << filename << "..." << std::endl;
        graph = load_graph(filename); // All ranks load the full graph
        if (rank == 0) std::cout << "Graph loaded: " << graph.num_vertices << " vertices." << std::endl;
        part.resize(graph.num_vertices); // Resize partition vector on all ranks
        if (rank == 0) { // Also resize Rank 0 specific vectors
             initial_sssp_result.dist.resize(graph.num_vertices);
             initial_sssp_result.parent.resize(graph.num_vertices);
             affected_del.resize(graph.num_vertices);
             affected.resize(graph.num_vertices);
        }
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " error loading graph: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
     // Basic validation after load
    if (start_node < 0 || start_node >= graph.num_vertices) {
        if (rank == 0) std::cerr << "Error: Start node " << start_node << " is out of range [0, " << graph.num_vertices - 1 << "]." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

     // --- Load Changes (Rank 0 loads, broadcasts count and data) ---
    if (rank == 0) {
        if (!changes_filename.empty()) {
            std::cout << "\nLoading changes from " << changes_filename << "..." << std::endl;
            try {
                changes = load_edge_changes(changes_filename);
                num_changes = changes.size();
                std::cout << "Changes loaded: " << num_changes << " total." << std::endl;
            } catch (const std::exception& e) { std::cerr << "Warning: Failed to load changes file '" << changes_filename << "'. Error: " << e.what() << std::endl; num_changes = 0; changes.clear(); }
        } else { std::cout << "\nNo changes file provided." << std::endl; num_changes = 0; }
    }
    MPI_Bcast(&num_changes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (num_changes > 0) {
        if (rank != 0) changes.resize(num_changes);
        MPI_Bcast(changes.data(), num_changes * sizeof(EdgeChange), MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
        changes.clear(); // Ensure empty on all ranks
    }

    // --- Rank 0: Initial SSSP, Partitioning, Initial Affected Calc ---
    std::chrono::duration<double, std::milli> initial_sssp_time_rank0{}; // Timing variable
    if (rank == 0) {
        std::cout << "Rank 0: Performing initial SSSP..." << std::endl;
        auto start_initial_sssp = std::chrono::high_resolution_clock::now();
        initial_sssp_result = dijkstra(graph, start_node); // Run Dijkstra on the full graph
        auto end_initial_sssp = std::chrono::high_resolution_clock::now();
        initial_sssp_time_rank0 = end_initial_sssp - start_initial_sssp;
        std::cout << "Rank 0: Initial SSSP completed in " << initial_sssp_time_rank0.count() << " ms." << std::endl;

        // --- METIS Partitioning (as before) ---
        idx_t objval = 0;
        if (graph.num_vertices > 0 && num_partitions > 1) {
             std::cout << "Rank 0 partitioning graph into " << num_partitions << " parts using METIS..." << std::endl;
             std::vector<idx_t> xadj, adjncy, adjwgt;
             graph.to_metis_csr(xadj, adjncy, adjwgt);
             idx_t nVertices = graph.num_vertices;
             idx_t ncon = 1;
             idx_t* adjwgt_ptr = adjwgt.empty() ? nullptr : adjwgt.data();
             idx_t nParts = num_partitions;
             if (!adjncy.empty()) {
                 int metis_ret = METIS_PartGraphKway(&nVertices, &ncon, xadj.data(), adjncy.data(), nullptr, nullptr, adjwgt_ptr, &nParts, nullptr, nullptr, nullptr, &objval, part.data());
                 if (metis_ret != METIS_OK) { std::cerr << "METIS partitioning failed (Rank 0). Error code: " << metis_ret << ". Aborting." << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
                 else { std::cout << "METIS partitioning successful (Rank 0). Edge cut: " << objval << std::endl; }
             } else { std::cerr << "Warning (Rank 0): Graph has no edges. Assigning vertices sequentially." << std::endl; for(idx_t i = 0; i < nVertices; ++i) part[i] = i % nParts; }
        } else {
             if (rank == 0) std::cout << "Skipping partitioning (num_partitions <= 1 or no vertices). Assigning all to partition 0." << std::endl;
             std::fill(part.begin(), part.end(), 0);
        }

        // --- Initial Affected Calculation (Algorithm 2 on Rank 0) ---
        // IMPORTANT: This modifies a temporary copy of distances/parents for calculation.
        // The actual distributed algorithm needs the *original* state + affected flags.
        std::cout << "Rank 0: Calculating initial affected vertices..." << std::endl;
        std::fill(affected_del.begin(), affected_del.end(), false);
        std::fill(affected.begin(), affected.end(), false);
        SSSPResult temp_sssp_result = initial_sssp_result; // Copy original state for calculation

        // Process Deletions/Increases
        for (const auto& change : changes) {
            // Treat INCREASE like DELETE for initial affected_del calculation
            if (change.type == ChangeType::DECREASE || change.type == ChangeType::INSERT) continue;

            // Check if edge e(u, v) was in the original SSSP tree T
            // Note: This check assumes an undirected graph representation in the parent array
            // or that the edge direction matches parent relationship.
            // If the graph is stored undirected, you might need:
            // bool in_tree = (temp_sssp_result.parent[change.v] == change.u) || (temp_sssp_result.parent[change.u] == change.v);

            if (temp_sssp_result.parent[change.v] == change.u) {
                 int v = change.v;
                 // y = argmax_{x in {u,v}} {Dist[x]}
                 // We mark the child node 'v' as affected if the edge (u,v) from parent u is deleted/increased.
                 int y = v; // The vertex whose distance might become infinite due to parent edge removal
                 if (temp_sssp_result.dist[y] != INFINITY_WEIGHT) { // Only mark if reachable
                     temp_sssp_result.dist[y] = INFINITY_WEIGHT; // Change Dist[y] to infinity in the *temporary* result
                     affected_del[y] = true;
                     affected[y] = true;
                 }
            }
        }

        // Process Insertions/Decreases
        for (const auto& change : changes) {
             // Only handle decrease/insert here
             if (change.type != ChangeType::DECREASE && change.type != ChangeType::INSERT) continue;

             int u = change.u;
             int v = change.v;
             double weight = change.weight;

             // Relax edge based on current *temporary* distances
             if (temp_sssp_result.dist[u] != INFINITY_WEIGHT && temp_sssp_result.dist[v] > temp_sssp_result.dist[u] + weight) {
                 temp_sssp_result.dist[v] = temp_sssp_result.dist[u] + weight; // Update temporary result
                 temp_sssp_result.parent[v] = u;
                 affected[v] = true;
             }
             // Check relaxation in the other direction too
             if (temp_sssp_result.dist[v] != INFINITY_WEIGHT && temp_sssp_result.dist[u] > temp_sssp_result.dist[v] + weight) {
                 temp_sssp_result.dist[u] = temp_sssp_result.dist[v] + weight; // Update temporary result
                 temp_sssp_result.parent[u] = v;
                 affected[u] = true;
             }
        }
         std::cout << "Rank 0: Initial affected calculation done." << std::endl;
    }

    // --- Broadcast Partition Vector ---
    MPI_Bcast(part.data(), graph.num_vertices, MPI_INT, 0, MPI_COMM_WORLD);

    // --- Data Scattering Preparation ---
    Graph local_graph(0); // Initialize on all ranks
    std::vector<int> local_to_global;
    std::vector<int> global_to_local;   // Map for local rank: global_idx -> local_idx (-1 if not local)
    BoundaryEdges local_boundary_edges; // Add map for boundary edges
    std::vector<double> local_dist;
    std::vector<int> local_parent;
    std::vector<bool> local_affected_del;
    std::vector<bool> local_affected;

    // --- Setup Local Data (Rank 0) ---
    if (rank == 0) {
        std::cout << "[Rank 0] Starting setup for own local data..." << std::endl;
        setup_local_data(graph, part, rank,
                         local_graph, local_to_global, global_to_local,
                         local_boundary_edges, // Pass boundary edge map
                         initial_sssp_result, affected_del, affected, // Pass Rank 0's calculated global initial state
                         local_dist, local_parent, local_affected_del, local_affected);
        std::cout << "[Rank 0] Finished setup for own local data." << std::endl;

        // --- Data Scattering (Rank 0 -> Others) ---
        std::cout << "[Rank 0] Starting data scattering to other ranks..." << std::endl;
        for (int dest_rank = 1; dest_rank < size; ++dest_rank) {
            std::cout << "[Rank 0] Preparing data for Rank " << dest_rank << "..." << std::endl;

            // 1. Determine local vertices and mappings for dest_rank
            std::vector<int> dest_local_to_global;
            std::vector<int> dest_global_to_local(graph.num_vertices, -1);
            for (int v_global = 0; v_global < graph.num_vertices; ++v_global) {
                if (part[v_global] == dest_rank) {
                    int v_local = dest_local_to_global.size();
                    dest_local_to_global.push_back(v_global);
                    dest_global_to_local[v_global] = v_local;
                }
            }
            int dest_n_local = dest_local_to_global.size();
            std::cout << "[Rank 0] Rank " << dest_rank << " has " << dest_n_local << " local vertices." << std::endl;

            // 2. Extract local state vectors for dest_rank
            std::vector<double> dest_local_dist(dest_n_local);
            std::vector<int> dest_local_parent(dest_n_local);
            std::vector<bool> dest_local_affected_del(dest_n_local);
            std::vector<bool> dest_local_affected(dest_n_local);

            for (int u_local = 0; u_local < dest_n_local; ++u_local) {
                int u_global = dest_local_to_global[u_local];
                dest_local_dist[u_local] = initial_sssp_result.dist[u_global];
                dest_local_parent[u_local] = initial_sssp_result.parent[u_global]; // Send global parent ID
                dest_local_affected_del[u_local] = affected_del[u_global];
                dest_local_affected[u_local] = affected[u_global];
            }

            // 3. Extract local graph structure AND boundary edges for dest_rank
            std::vector<std::vector<Edge>> adj_list_to_send(dest_n_local);
            BoundaryEdges boundary_edges_to_send; // Map to store boundary edges for dest_rank
            int dest_internal_edges = 0;
            int dest_boundary_edges = 0;

            for (int u_local = 0; u_local < dest_n_local; ++u_local) {
                int u_global = dest_local_to_global[u_local];
                 try {
                    for (const auto& edge : graph.neighbors(u_global)) {
                        int v_global = edge.to;
                        Weight weight = edge.weight;
                        // Check if destination is local to dest_rank
                        if (v_global >= 0 && v_global < graph.num_vertices && dest_global_to_local[v_global] != -1) {
                            int v_local = dest_global_to_local[v_global];
                            adj_list_to_send[u_local].push_back({v_local, weight});
                            dest_internal_edges++;
                        } else if (v_global >= 0 && v_global < graph.num_vertices) { // Check if v_global is valid
                            // It's a boundary edge for dest_rank
                            boundary_edges_to_send[u_local].push_back({v_global, weight}); // Store with global dest ID
                            dest_boundary_edges++;
                        }
                    }
                 } catch ([[maybe_unused]] const std::out_of_range& oor) { /* Ignore */ }
            }
             std::cout << "[Rank 0] Prepared data for Rank " << dest_rank << ". Internal edges: " << dest_internal_edges << ", Boundary edges: " << dest_boundary_edges << std::endl;


            // 4. Send data (Add sends for boundary edges)
            // Communication tags for MPI_Send/MPI_Recv
            std::cout << "[Rank 0] Sending data to Rank " << dest_rank << "..." << std::endl;
            MPI_Send(&dest_n_local, 1, MPI_INT, dest_rank, 0, MPI_COMM_WORLD);
            MPI_Send(dest_local_to_global.data(), dest_n_local, MPI_INT, dest_rank, 1, MPI_COMM_WORLD);
            MPI_Send(dest_local_dist.data(), dest_n_local, MPI_DOUBLE, dest_rank, 2, MPI_COMM_WORLD);
            MPI_Send(dest_local_parent.data(), dest_n_local, MPI_INT, dest_rank, 3, MPI_COMM_WORLD);
            std::vector<char> aff_del_char(dest_local_affected_del.begin(), dest_local_affected_del.end());
            std::vector<char> aff_char(dest_local_affected.begin(), dest_local_affected.end());
            MPI_Send(aff_del_char.data(), dest_n_local, MPI_CHAR, dest_rank, 4, MPI_COMM_WORLD);
            MPI_Send(aff_char.data(), dest_n_local, MPI_CHAR, dest_rank, 5, MPI_COMM_WORLD);

            // Send internal adjacency list
            std::vector<int> adj_sizes(dest_n_local);
            std::vector<Edge> adj_data;
            for(int u_local = 0; u_local < dest_n_local; ++u_local) {
                adj_sizes[u_local] = adj_list_to_send[u_local].size();
                adj_data.insert(adj_data.end(), adj_list_to_send[u_local].begin(), adj_list_to_send[u_local].end());
            }
            MPI_Send(adj_sizes.data(), dest_n_local, MPI_INT, dest_rank, 6, MPI_COMM_WORLD);

            // Create a safer data structure to send Edge data
            int total_adj_entries = adj_data.size();
            std::vector<int> adj_destinations(total_adj_entries);     // Edge destinations
            std::vector<double> adj_weights(total_adj_entries);       // Edge weights

            // Split Edge objects into separate arrays for safer transmission
            for (int i = 0; i < total_adj_entries; i++) {
                adj_destinations[i] = adj_data[i].to;
                adj_weights[i] = adj_data[i].weight;
            }

            // Send the size first
            MPI_Send(&total_adj_entries, 1, MPI_INT, dest_rank, 7, MPI_COMM_WORLD);

            // Only send data if there are edges to send
            if (total_adj_entries > 0) {
                MPI_Send(adj_destinations.data(), total_adj_entries, MPI_INT, dest_rank, 71, MPI_COMM_WORLD);
                MPI_Send(adj_weights.data(), total_adj_entries, MPI_DOUBLE, dest_rank, 72, MPI_COMM_WORLD);
            }

            // Send boundary edges map (more complex serialization)
            int boundary_map_size = boundary_edges_to_send.size();
            MPI_Send(&boundary_map_size, 1, MPI_INT, dest_rank, 8, MPI_COMM_WORLD);

            // Only proceed with boundary edges if there are any
            if (boundary_map_size > 0) {
                std::vector<int> boundary_keys;
                std::vector<int> boundary_counts;
                std::vector<int> boundary_destinations;
                std::vector<double> boundary_weights;

                boundary_keys.reserve(boundary_map_size);
                boundary_counts.reserve(boundary_map_size);

                // Flatten boundary edges into simple arrays
                int total_boundary_edges = 0;
                for(const auto& pair : boundary_edges_to_send) {
                    boundary_keys.push_back(pair.first); // local source index
                    boundary_counts.push_back(pair.second.size());
                    total_boundary_edges += pair.second.size();

                    // Extract destination and weight information
                    for (const auto& edge : pair.second) {
                        boundary_destinations.push_back(edge.to);
                        boundary_weights.push_back(edge.weight);
                    }
                }

                // Send boundary edge metadata
                MPI_Send(boundary_keys.data(), boundary_map_size, MPI_INT, dest_rank, 9, MPI_COMM_WORLD);
                MPI_Send(boundary_counts.data(), boundary_map_size, MPI_INT, dest_rank, 10, MPI_COMM_WORLD);

                // Send the flattened boundary edge data
                MPI_Send(&total_boundary_edges, 1, MPI_INT, dest_rank, 11, MPI_COMM_WORLD);

                if (total_boundary_edges > 0) {
                    MPI_Send(boundary_destinations.data(), total_boundary_edges, MPI_INT, dest_rank, 111, MPI_COMM_WORLD);
                    MPI_Send(boundary_weights.data(), total_boundary_edges, MPI_DOUBLE, dest_rank, 112, MPI_COMM_WORLD);
                }
            }

            std::cout << "[Rank 0] Finished sending data to Rank " << dest_rank << "." << std::endl;
        }
         std::cout << "[Rank 0] Finished data scattering." << std::endl;

    } else {
        // --- Receive Data (Ranks 1 to size-1) ---
        std::cout << "[Rank " << rank << "] Waiting to receive data from Rank 0..." << std::endl;
        int n_local;
        MPI_Status status;
        MPI_Recv(&n_local, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        std::cout << "[Rank " << rank << "] Received n_local = " << n_local << "." << std::endl;

        // Safety check for n_local value
        if (n_local <= 0 || n_local > graph.num_vertices) {
            std::cerr << "[Rank " << rank << "] ERROR: Received invalid n_local value: " << n_local
                      << ". Expected a value between 1 and " << graph.num_vertices
                      << ". Aborting." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        // Resize local vectors with try-catch to handle allocation errors
        try {
            local_to_global.resize(n_local);

            // Use vector constructor instead of assign to avoid potential memory issues
            int n_global = graph.num_vertices;
            std::vector<int> new_global_to_local(n_global, -1);
            global_to_local = std::move(new_global_to_local);

            local_dist.resize(n_local);
            local_parent.resize(n_local);
            local_affected_del.resize(n_local);
            local_affected.resize(n_local);
        } catch (const std::exception& e) {
            std::cerr << "[Rank " << rank << "] Memory allocation error: " << e.what() << ". Aborting." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        std::vector<char> aff_del_char(n_local);
        std::vector<char> aff_char(n_local);
        std::vector<int> adj_sizes(n_local);
        local_boundary_edges.clear(); // Clear boundary map

        // Receive data (Tags 0-5)
        MPI_Recv(local_to_global.data(), n_local, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(local_dist.data(), n_local, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &status);
        MPI_Recv(local_parent.data(), n_local, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
        MPI_Recv(aff_del_char.data(), n_local, MPI_CHAR, 0, 4, MPI_COMM_WORLD, &status);
        MPI_Recv(aff_char.data(), n_local, MPI_CHAR, 0, 5, MPI_COMM_WORLD, &status);

        // Convert char back to bool
        for(int i=0; i<n_local; ++i) local_affected_del[i] = aff_del_char[i];
        for(int i=0; i<n_local; ++i) local_affected[i] = aff_char[i];

        // Rebuild global_to_local map
        for(int u_local = 0; u_local < n_local; ++u_local) {
            if (local_to_global[u_local] >= 0 && local_to_global[u_local] < global_to_local.size()) { // Bounds check
                global_to_local[local_to_global[u_local]] = u_local;
            }
        }

        // Receive internal adjacency list data (Tags 6, 7)
        MPI_Recv(adj_sizes.data(), n_local, MPI_INT, 0, 6, MPI_COMM_WORLD, &status);

        // Receive the size of adjacency data
        int total_adj_entries;
        MPI_Recv(&total_adj_entries, 1, MPI_INT, 0, 7, MPI_COMM_WORLD, &status);

        // Receive edge data as separate arrays
        std::vector<int> adj_destinations;
        std::vector<double> adj_weights;

        if (total_adj_entries > 0) {
            adj_destinations.resize(total_adj_entries);
            adj_weights.resize(total_adj_entries);

            MPI_Recv(adj_destinations.data(), total_adj_entries, MPI_INT, 0, 71, MPI_COMM_WORLD, &status);
            MPI_Recv(adj_weights.data(), total_adj_entries, MPI_DOUBLE, 0, 72, MPI_COMM_WORLD, &status);
        }

        // Reconstruct local graph (internal edges)
        local_graph = Graph(n_local);
        int current_adj_idx = 0;
        for (int u_local = 0; u_local < n_local; ++u_local) {
            for (int i = 0; i < adj_sizes[u_local]; ++i) {
                if (current_adj_idx < total_adj_entries) { // Bounds check
                    int dest = adj_destinations[current_adj_idx];
                    double weight = adj_weights[current_adj_idx];
                    current_adj_idx++;

                    // Add sanity check on destination vertex
                    if (dest >= 0 && dest < n_local) {
                        local_graph.add_edge(u_local, dest, weight);
                    } else {
                        std::cerr << "[Rank " << rank << "] Error: Invalid destination vertex "
                                  << dest << " in adjacency list. Valid range is [0, "
                                  << (n_local-1) << "]. Skipping." << std::endl;
                    }
                } else {
                     std::cerr << "[Rank " << rank << "] Error: Mismatch in adjacency list reconstruction. "
                               << "Tried to access index " << current_adj_idx
                               << " but array size is " << total_adj_entries << std::endl;
                     break; // Avoid further errors
                }
            }
        }

        // Receive boundary edges map (Tags 8, 9, 10, 11)
        int boundary_map_size;
        MPI_Recv(&boundary_map_size, 1, MPI_INT, 0, 8, MPI_COMM_WORLD, &status);

        // Defensive programming - limit boundary map size to avoid potential buffer overflow
        if (boundary_map_size < 0 || boundary_map_size > n_local) {
            std::cerr << "[Rank " << rank << "] ERROR: Invalid boundary map size " << boundary_map_size
                      << ". Setting to 0 to avoid segmentation fault." << std::endl;
            boundary_map_size = 0;
        }

        if (boundary_map_size > 0) {
            std::vector<int> boundary_keys(boundary_map_size);
            std::vector<int> boundary_counts(boundary_map_size);

            // Receive the keys and counts
            MPI_Recv(boundary_keys.data(), boundary_map_size, MPI_INT, 0, 9, MPI_COMM_WORLD, &status);
            MPI_Recv(boundary_counts.data(), boundary_map_size, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);

            // Receive the total number of boundary edges
            int total_boundary_edges;
            MPI_Recv(&total_boundary_edges, 1, MPI_INT, 0, 11, MPI_COMM_WORLD, &status);

            // Apply safety checks
            // Additional bounds check on total size
            if (total_boundary_edges < 0 || total_boundary_edges > 1000000) {
                std::cerr << "[Rank " << rank << "] ERROR: Invalid total boundary edges count "
                          << total_boundary_edges << " (expected range: 0-1000000). Setting to 0." << std::endl;
                total_boundary_edges = 0;
            }

            // Receive the flattened boundary edge data
            std::vector<int> boundary_destinations;
            std::vector<double> boundary_weights;

            if (total_boundary_edges > 0) {
                boundary_destinations.resize(total_boundary_edges);
                boundary_weights.resize(total_boundary_edges);

                MPI_Recv(boundary_destinations.data(), total_boundary_edges, MPI_INT, 0, 111, MPI_COMM_WORLD, &status);
                MPI_Recv(boundary_weights.data(), total_boundary_edges, MPI_DOUBLE, 0, 112, MPI_COMM_WORLD, &status);
            }

            // Validate boundary counts sum matches total_boundary_edges
            int expected_sum = 0;
            for(int i = 0; i < boundary_map_size; i++) {
                if (boundary_counts[i] >= 0) {
                    expected_sum += boundary_counts[i];
                } else {
                    std::cerr << "[Rank " << rank << "] WARNING: Negative boundary count "
                              << boundary_counts[i] << ". Setting to 0." << std::endl;
                    boundary_counts[i] = 0;
                }
            }

            if (expected_sum != total_boundary_edges) {
                std::cerr << "[Rank " << rank << "] WARNING: Boundary counts sum (" << expected_sum
                          << ") doesn't match received total (" << total_boundary_edges
                          << "). Using minimum." << std::endl;
                total_boundary_edges = std::min(expected_sum, total_boundary_edges);
            }

            // Reconstruct boundary edges map with additional safety checks
            int current_boundary_idx = 0;
            for (int i = 0; i < boundary_map_size; ++i) {
                int u_local = boundary_keys[i];
                int count = boundary_counts[i];

                // Bounds check on local vertex index
                if (u_local < 0 || u_local >= n_local) {
                    std::cerr << "[Rank " << rank << "] WARNING: Boundary key " << u_local
                              << " is out of bounds [0, " << n_local - 1 << "]. Skipping." << std::endl;
                    current_boundary_idx += count; // Skip these edges
                    continue;
                }

                std::vector<Edge> edges;
                edges.reserve(count);

                for (int j = 0; j < count; ++j) {
                    if (current_boundary_idx < total_boundary_edges) {
                        int dest = boundary_destinations[current_boundary_idx];
                        double weight = boundary_weights[current_boundary_idx];
                        current_boundary_idx++;

                        // Store the edge
                        edges.push_back({dest, weight});
                    } else {
                        std::cerr << "[Rank " << rank << "] Error: Boundary data index " << current_boundary_idx
                                  << " exceeds array size " << total_boundary_edges << std::endl;
                        break;
                    }
                }

                if (!edges.empty()) {
                    local_boundary_edges[u_local] = std::move(edges);
                }
            }
        }

        std::cout << "[Rank " << rank << "] Received and reconstructed all data. Local graph vertices: " << local_graph.num_vertices << ", Boundary edge sources: " << local_boundary_edges.size() << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD); // Wait for all ranks to finish setup/receive
    if (rank == 0) std::cout << "All ranks finished data setup/scattering." << std::endl;
    // End local data receive/setup section

    // --- Distributed Update ---
    if (rank == 0) std::cout << "Starting distributed update phase..." << std::endl;
    auto start_update = MPI_Wtime();

    // Call the function that implements Algorithm 3/4 (PASS BOUNDARY EDGES)
    Distributed_DynamicSSSP_MPI(
        local_graph,
        local_boundary_edges,   // Pass the boundary edges map
        local_to_global,        // Mapping: local index -> global ID
        global_to_local,        // Mapping: global ID -> local index
        local_dist,          // Local distance array (input/output)
        local_parent,        // Local parent array (input/output)
        rank,                   // Current rank
        part,                   // Global partition vector
        start_node,             // Starting node for SSSP
        local_affected_del,     // Initial affected_del status for local vertices
        local_affected,         // Initial affected status for local vertices
        debug_output            // Pass the debug_output flag
    );

    auto end_update = MPI_Wtime();
    double update_time_ms = (end_update - start_update) * 1000.0;
    if (rank == 0) std::cout << "Distributed update phase finished in " << update_time_ms << " ms." << std::endl;


    // --- Gather Results ---
    int n_global = graph.num_vertices; // Use graph.num_vertices for global size
    std::vector<double> final_global_dist;
    std::vector<int> final_global_parent;
    if (rank == 0) {
        final_global_dist.resize(n_global, INFINITY_WEIGHT);    // Initialize on rank 0
        final_global_parent.resize(n_global, -1);               // Initialize on rank 0

        // Only print debug information when debug_output is enabled
        if (debug_output) {
            std::cout << "[Rank 0 DEBUG] Before Reduce, final_global_dist: [ ";
            for(double d : final_global_dist) std::cout << (d == INFINITY_WEIGHT ? "INF " : std::to_string(d) + " ");
            std::cout << "]" << std::endl;
        }
    }

    // Gather distances
    if (size == 1) {
        // Single-rank: no reduction needed, local_dist covers all vertices
        if (rank == 0) final_global_dist = local_dist;
    } else {
        // Prepare full-length send buffer: map local distances to global positions
        std::vector<double> dist_send(n_global, INFINITY_WEIGHT);
        for (size_t u_local = 0; u_local < local_to_global.size(); ++u_local) {
            int u_global = local_to_global[u_local];
            if (u_global >= 0 && u_global < n_global) {
                dist_send[u_global] = local_dist[u_local];
            }
        }
        MPI_Reduce(dist_send.data(), final_global_dist.data(),
                   n_global, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        // DEBUG: Print final_global_dist after Reduce
        if (debug_output) {
            std::cout << "[Rank 0 DEBUG] After Reduce, final_global_dist: [ ";
            for(double d : final_global_dist) std::cout << (d == INFINITY_WEIGHT ? "INF " : std::to_string(d) + " ");
            std::cout << "]" << std::endl;
        }
    }


    // Gather parents (using Gather)
    std::vector<int> local_parent_out(n_global, -1);
    // Ensure local_to_global is populated before this loop
    if (!local_to_global.empty()) { // Add check to prevent crash if setup is incomplete
         for (int u_local = 0; u_local < static_cast<int>(local_to_global.size()); ++u_local) { // Use local_to_global size
            int u_global = local_to_global[u_local];
             if (u_global >= 0 && u_global < n_global && u_local < local_parent.size()) { // Bounds check for local_parent too
                // local_parent[u_local] already stores the GLOBAL parent ID
                local_parent_out[u_global] = local_parent[u_local]; // Use the global ID directly
            }
        }
    }
    std::vector<int> gathered_parents;
    if (rank == 0) gathered_parents.resize(n_global * size);
    MPI_Gather(local_parent_out.data(), n_global, MPI_INT,
               gathered_parents.data(), n_global, MPI_INT, 0, MPI_COMM_WORLD);

    // Combine gathered parents on Rank 0
    if (rank == 0) {
        // DEBUG: Print gathered_parents after Gather
        if (debug_output) {
            std::cout << "[Rank 0 DEBUG] After Gather, gathered_parents (size " << gathered_parents.size() << "): [ ";
            for(int p : gathered_parents) std::cout << p << " ";
            std::cout << "]" << std::endl;
        }

        // DEBUG: Print partition vector before combining
        if (debug_output) {
            std::cout << "[Rank 0 DEBUG] Partition vector 'part' used for combining: [ ";
            // Need to cast idx_t potentially if printing directly
            for(int i : part) std::cout << static_cast<int>(i) << " ";
            std::cout << "]" << std::endl;
        }


        for(int i=0; i<n_global; ++i) {
            final_global_parent[i] = -1; // Initialize
            int owner_rank = (i >= 0 && i < part.size()) ? part[i] : -1; // Find the rank that owns vertex i
            if (owner_rank != -1 && owner_rank < size) { // Check if owner_rank is valid
                // Calculate the index in the gathered_parents array for the owner rank
                int index = owner_rank * n_global + i;
                if (index >= 0 && index < gathered_parents.size()) { // Bounds check
                    int val = gathered_parents[index];
                    final_global_parent[i] = val; // Use the parent value from the owning rank
                } else {
                     std::cerr << "Warning: Index out of bounds when combining parents for vertex " << i << " from rank " << owner_rank << std::endl;
                }
            } else {
                 std::cerr << "Warning: Could not determine valid owner rank for vertex " << i << " (partition value: " << ((i >= 0 && i < part.size()) ? std::to_string(part[i]) : "N/A") << ")" << std::endl;
            }
        }
        // Special case: Ensure source node parent is -1
        if (start_node >= 0 && start_node < n_global) {
             final_global_parent[start_node] = -1;
        }

        // DEBUG: Print final_global_parent after combining
        if (debug_output) {
            std::cout << "[Rank 0 DEBUG] After Combine, final_global_parent: [ ";
            for(int p : final_global_parent) std::cout << p << " ";
            std::cout << "]" << std::endl;
        }
    }


    // --- Output Results (Rank 0) ---
    if (rank == 0) {
        std::cout << "\n--- Final SSSP Result (MPI) ---" << std::endl;
        for (int i = 0; i < n_global; ++i) {
            std::cout << "Vertex " << i << ": Dist = ";
            if (i < final_global_dist.size() && final_global_dist[i] == INFINITY_WEIGHT) std::cout << "INF"; // Bounds check
            else if (i < final_global_dist.size()) std::cout << final_global_dist[i]; // Bounds check
            else std::cout << "ERR"; // Indicate error if out of bounds
            std::cout << ", Parent = ";
            if (i < final_global_parent.size()) std::cout << final_global_parent[i];  // Bounds check
            else std::cout << "ERR"; // Indicate error if out of bounds
            std::cout << std::endl;
        }

        std::cout << "\n--- Timings (MPI) ---" << std::endl;
        std::cout << "Initial SSSP (Rank 0): " << initial_sssp_time_rank0.count() << " ms" << std::endl;
        std::cout << "Distributed Update Time: " << update_time_ms << " ms" << std::endl;
        std::cout << "\nExecution finished." << std::endl;
    }

    // ===== Cleanup: Let destructors handle resource release =====
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Finalizing MPI..." << std::endl;
    }
    MPI_Finalize();
    return 0;
}

