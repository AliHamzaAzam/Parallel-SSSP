//
// Created by Ali Hamza Azam on 25/04/2025.
//

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept> // For std::runtime_error
#include <numeric>   // For std::accumulate if needed, or other algorithms
#include <algorithm> // For std::fill
#include <omp.h>
#include <metis.h> // Include METIS header

// Include necessary headers
#include "../include/graph.h" // Include graph definitions (includes Weight)
#include "../include/utils.hpp" // Include utility functions

// Forward declarations for SSSP functions
// Sequential
SSSPResult dijkstra(const Graph& g, int source); // Assuming returns SSSPResult
// Updated sequential batch prototype (assuming it takes const changes and modifies SSSPResult)
void process_batch_sequential(Graph& g, SSSPResult& sssp_result, const std::vector<EdgeChange>& batch);

// OpenMP - Corrected Forward Declaration
// Matches the expected signature used later and likely defined in sssp_parallel_openmp.cpp
void BatchUpdate_OpenMP(const Graph& G, SSSPResult& T, const std::vector<EdgeChange>& changes);
void TestDynamicSSSPWorkflow_OpenMP(Graph& G, SSSPResult& T, const std::vector<EdgeChange>& changes);

// OpenCL (Placeholder)
// void BatchUpdate_OpenCL(const Graph& G, SSSPResult& T, const std::vector<EdgeChange>& changes);


// Function to print the current state of distances and parents
void print_sssp_result(const SSSPResult& sssp_result) {
    std::cout << "Current SSSP Result:" << std::endl;
    for (size_t i = 0; i < sssp_result.dist.size(); ++i) {
        std::cout << "Vertex " << i << ": Dist = " << sssp_result.dist[i]
                  << ", Parent = " << sssp_result.parent[i] << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Updated usage message to include mode
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <graph_file.mtx> <start_node> [mode] [num_threads] [num_partitions]" << std::endl;
        std::cerr << "Modes: baseline, sequential, openmp, opencl" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int start_node = std::stoi(argv[2]);
    // Default mode is baseline if not provided
    std::string mode = (argc > 3) ? argv[3] : "baseline"; 
    // Default threads based on whether mode was provided
    int num_threads = (argc > 4) ? std::stoi(argv[4]) : omp_get_max_threads();
    // Default partitions based on whether threads were provided
    idx_t num_partitions = (argc > 5) ? std::max(1, std::stoi(argv[5])) : 1; 

    omp_set_num_threads(num_threads);

    try {
        std::cout << "Loading graph from " << filename << "..." << std::endl;
        // Corrected function name
        Graph graph = load_graph(filename);
        std::cout << "Graph loaded: " << graph.num_vertices << " vertices." << std::endl;

        if (start_node < 0 || start_node >= graph.num_vertices) {
            std::cerr << "Error: Start node " << start_node << " is out of range [0, " << graph.num_vertices - 1 << "]." << std::endl;
            return 1;
        }

        // --- METIS Partitioning --- 
        std::vector<idx_t> part(graph.num_vertices); // To store partition assignments
        idx_t objval = 0; // To store edge cut

        if (graph.num_vertices > 0 && num_partitions > 1) { // Only partition if vertices exist and partitions > 1
             std::cout << "Partitioning graph into " << num_partitions << " parts using METIS..." << std::endl;
             std::vector<idx_t> xadj, adjncy, adjwgt;
             graph.to_metis_csr(xadj, adjncy, adjwgt); // Convert graph to CSR

             idx_t nVertices = graph.num_vertices;
             // Determine ncon based on whether valid edge weights are used
             idx_t ncon = 1; // Default to 1 constraint (vertex balance)
             idx_t* adjwgt_ptr = adjwgt.data();

             // Basic validation for weights
             if (adjwgt.empty() || adjwgt.size() != adjncy.size()) {
                 std::cerr << "Warning: Edge weights missing or size mismatch. Partitioning based on vertex count balance only." << std::endl;
                 adjwgt_ptr = NULL;
                 ncon = 1; // Still balance vertices
             } else {
                 // If weights are present, METIS uses them as the first constraint (ncon=1)
                 ncon = 1; 
             }

             idx_t nParts = num_partitions;

             // Ensure graph has edges before partitioning
             if (adjncy.empty()) {
                 std::cerr << "Warning: Graph has no edges. Assigning vertices sequentially to partitions." << std::endl;
                 for(idx_t i = 0; i < nVertices; ++i) {
                     part[i] = i % nParts;
                 }
             } else {
                 int metis_ret = METIS_PartGraphKway(
                     &nVertices,
                     &ncon,          // Number of balancing constraints (1 for vertex balance or vertex+weight balance)
                     xadj.data(),
                     adjncy.data(),
                     NULL,           // Vertex weights (vwgt) - NULL
                     NULL,           // Vertex sizes (vsize) - NULL
                     adjwgt_ptr,     // Edge weights (adjwgt) - Use if valid
                     &nParts,
                     NULL,           // Target partition weights (tpwgts) - NULL for equal
                     NULL,           // Allowed load imbalance (ubvec) - NULL for default
                     NULL,           // Options - NULL for default
                     &objval,        // Output: Edge cut
                     part.data()     // Output: Partition vector
                 );

                 if (metis_ret != METIS_OK) {
                     std::cerr << "METIS partitioning failed with error code: " << metis_ret << std::endl;
                     // Fallback: Assign vertices sequentially if METIS fails
                     std::cerr << "Falling back to sequential vertex assignment." << std::endl;
                      for(idx_t i = 0; i < nVertices; ++i) {
                         part[i] = i % nParts;
                     }
                     objval = -1; // Indicate partitioning failure
                 } else {
                    std::cout << "METIS partitioning successful. Edge cut: " << objval << std::endl;
                 }
             }
        } else if (graph.num_vertices > 0) {
            std::cout << "Skipping partitioning (num_partitions <= 1 or no vertices). Assigning all to partition 0." << std::endl;
            std::fill(part.begin(), part.end(), 0); // Assign all to partition 0
        } else {
             std::cout << "Graph has no vertices. Skipping partitioning and SSSP." << std::endl;
             return 0; // Exit if no vertices
        }

        // --- Initial SSSP Calculation (using Dijkstra) ---
        std::cout << "\nCalculating initial SSSP from source " << start_node << " using Dijkstra..." << std::endl;
        auto start_time_initial = std::chrono::high_resolution_clock::now();
        SSSPResult sssp_result = dijkstra(graph, start_node); // Get initial result
        auto end_time_initial = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> initial_sssp_time = end_time_initial - start_time_initial;
        std::cout << "Initial SSSP calculated in " << initial_sssp_time.count() << " ms." << std::endl;

        // --- Load Changes (if provided) ---
        std::vector<EdgeChange> changes;
        // Changes file is now potentially the 6th argument if mode, threads, partitions are given
        // Let's adjust logic: Assume changes file is *always* the last argument if present beyond the required ones.
        // A more robust CLI parser (like getopt or cxxopts) would be better.
        // For now, let's assume a fixed structure or check if the last arg looks like a file.
        // Simplified: Let's assume for now the changes file is NOT handled via argv directly,
        // and the mode check logic needs to be independent of argc count beyond the mode itself.
        // We will rely on the mode variable set earlier.
        bool has_changes = false; // Determine if changes should be loaded based on mode? Or a dedicated arg?
        // Let's assume a dedicated changes file argument is needed *if* mode is not baseline.
        std::string update_filename = "";
        if (mode != "baseline" && argc > 6) { // Requires 6 args minimum for a changes file (prog, graph, start, mode, threads, parts, changes_file)
             update_filename = argv[6];
             has_changes = true;
             std::cout << "\nLoading changes from " << update_filename << "..." << std::endl;
             changes = load_edge_changes(update_filename); // Load the combined changes
             std::cout << "Changes loaded: " << changes.size() << " total." << std::endl;
        } else if (mode != "baseline") {
             std::cout << "\nWarning: Mode '" << mode << "' selected, but no changes file provided (expected as 7th argument)." << std::endl;
             // Decide if this is an error or just run initial SSSP
             // Let's treat it as only running initial SSSP for now.
             has_changes = false;
        }

        // Print all loaded changes for debugging
        for (const auto& change : changes) {
            std::cout << (change.is_insertion ? "Insert" : "Delete") << " edge: (" << change.u << ", " << change.v << ")" << std::endl;
        }

        // Apply changes if loaded
        if (has_changes) {
             // Apply changes to the graph structure G itself *before* calling update algorithms
             // This is suitable for the parallel batch algorithms (2 & 3).
             // Sequential algorithm 1 might need adjustment or a different approach.
             std::cout << "Applying structural changes to graph..." << std::endl;
             auto apply_start = std::chrono::high_resolution_clock::now();
             int skipped_changes = 0;
             for (const auto& change : changes) {
                 try {
                     // Validate indices before applying change
                     if (change.u < 0 || change.u >= graph.num_vertices || change.v < 0 || change.v >= graph.num_vertices) {
                         skipped_changes++;
                         continue;
                     }

                     if (change.is_insertion) {
                         graph.add_edge(change.u, change.v, change.weight);
                     } else {
                         std::cout << "Calling remove_edge for (" << change.u << ", " << change.v << ")" << std::endl;
                         graph.remove_edge(change.u, change.v); // remove_edge should handle non-existent edges gracefully
                     }
                 } catch (const std::exception& e) { // Catch potential errors during modification
                      std::cerr << "Warning: Error applying change ("
                                << (change.is_insertion ? "insert" : "delete") << " " << change.u << " " << change.v
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

             std::cout << "\n--- After Applying Changes ---" << std::endl;
             print_sssp_result(sssp_result);

             // Debug: Verify edge removal
             std::cout << "Edge (0, 1) removed: " << !graph.has_edge(0, 1) << std::endl;

        } else if (mode != "baseline") {
             std::cout << "\nMode '" << mode << "' requires a changes file, but none was provided or loaded. Only initial SSSP was computed." << std::endl;
             // If a non-baseline mode was explicitly chosen but no changes file given,
             // should we revert to baseline or just stop after initial?
             // Let's revert to baseline behavior (only report initial time).
             mode = "baseline"; // Treat as baseline run if no changes provided for other modes
        } else {
             std::cout << "\nNo changes file provided (or baseline mode selected)." << std::endl;
        }
        // Validate mode string *after* potential modification above
        if (mode != "baseline" && mode != "sequential" && mode != "openmp" && mode != "opencl") {
            std::cerr << "Error: Invalid mode '" << mode << "' specified." << std::endl;
            return 1; // Exit if mode is invalid
        }


        // --- Perform SSSP Update or Baseline Recomputation ---
        // Mode variable is now std::string, comparisons are correct
        std::cout << "\nRunning mode: " << mode << std::endl;
        auto start_time_update = std::chrono::high_resolution_clock::now();
        bool update_performed = false;

        if (mode == "sequential") {
            if (has_changes) {
                 // NOTE: process_batch_sequential needs to be compatible with the pre-modified graph 'g'.
                 // The current implementation in sssp_sequential.cpp might not be correct.
                 // Assuming it's updated or we accept potential inconsistency for now.
                 std::cout << "Processing batch sequentially..." << std::endl;
                 process_batch_sequential(graph, sssp_result, changes);
                 update_performed = true;
            } else {
                 std::cout << "Sequential mode selected, but no changes file provided. Skipping update." << std::endl;
            }
        } else if (mode == "openmp") {
             if (has_changes) {
                std::cout << "Processing batch using OpenMP (dynamic workflow)..." << std::endl;
                TestDynamicSSSPWorkflow_OpenMP(graph, sssp_result, changes);
                update_performed = true;
             } else {
                 std::cout << "OpenMP mode selected, but no changes file provided. Skipping update." << std::endl;
             }
        } else if (mode == "opencl") {
             if (has_changes) {
                std::cout << "OpenCL mode selected (placeholder - not implemented)." << std::endl;
                // BatchUpdate_OpenCL(g, sssp_result, changes); // Placeholder
                // update_performed = true; // Set to true when implemented
             } else {
                  std::cout << "OpenCL mode selected, but no changes file provided. Skipping update." << std::endl;
             }
        } else if (mode == "baseline") {
            if (has_changes) {
                std::cout << "Recomputing SSSP from scratch (baseline) on modified graph..." << std::endl;
                // Graph 'g' already reflects changes here
                sssp_result = dijkstra(graph, start_node); // Recompute on modified graph
                update_performed = true;
            } else {
                std::cout << "Baseline mode selected, no changes provided. Initial SSSP is the result." << std::endl;
                // No re-computation needed, initial result stands.
            }
        } 

        auto end_time_update = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> update_time = end_time_update - start_time_update;

        if (update_performed) {
            std::cout << "\n--- After Update/Recompute (" << mode << ") ---" << std::endl;
            print_sssp_result(sssp_result);

            // Debug: Check if Parent[1] is still 0
            if (sssp_result.parent[1] == 0) {
                std::cerr << "Error: Parent of Vertex 1 is still 0 after edge (0, 1) deletion!" << std::endl;
            }
        }

        // --- Output Results ---
        std::cout << "\n--- Timings ---" << std::endl;
        std::cout << "Initial Dijkstra: " << initial_sssp_time.count() << " ms" << std::endl;
        if (update_performed) {
             std::cout << "Update/Recompute (" << mode << "): " << update_time.count() << " ms" << std::endl;
        } else if (has_changes) {
             std::cout << "Update (" << mode << "): Skipped (no changes file or mode mismatch)" << std::endl;
        } else {
             std::cout << "Update: Skipped (no changes file provided)" << std::endl;
        }


        // Optional: Verify results or save Dist/Parent arrays
        // Example: Print distance to a specific node if it exists
        // int target_node = std::min(10, g.num_vertices - 1); // Example target
        // if (target_node >= 0) {
        //     std::cout << "\nFinal distance to node " << target_node << ": ";
        //     if (sssp_result.dist[target_node] == INFINITY_WEIGHT) {
        //         std::cout << "INF" << std::endl;
        //     } else {
        //         std::cout << sssp_result.dist[target_node] << std::endl;
        //     }
        // }


    } catch (const std::exception& e) {
        std::cerr << "\nError: An exception occurred: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nExecution finished." << std::endl;
    return 0;
}

// --- Placeholder/Dummy Implementations ---
// These should be removed if the actual implementations exist in other linked files.
// Keeping them temporarily might help isolate build issues if linking fails.

// Assume sssp_sequential.cpp defines these:
// SSSPResult dijkstra(const Graph& g, int source);
// void process_batch_sequential(Graph& g, SSSPResult& sssp_result, const std::vector<EdgeChange>& batch);

// Assume sssp_parallel_openmp.cpp defines this:
// void BatchUpdate_OpenMP(const Graph& G, SSSPResult& T, const std::vector<EdgeChange>& changes);

// Placeholder for OpenCL
// void BatchUpdate_OpenCL(const Graph& G, SSSPResult& T, const std::vector<EdgeChange>& changes) {
//      std::cerr << "BatchUpdate_OpenCL is not implemented." << std::endl;
// }

// Ensure load_graph and load_edge_changes are correctly linked from utils or defined if utils is header-only.
// If utils.hpp contains implementations, no separate linking needed. If it's split into .hpp/.cpp, ensure CMake links it.

