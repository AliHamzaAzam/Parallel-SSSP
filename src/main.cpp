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
#include "../include/graph.hpp" // Include graph definitions (includes Weight)
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

// Forward declarations for the three main entry points
int serial_main(int argc, char* argv[]);
int parallel_main(int argc, char* argv[]);
int mpi_main(int argc, char* argv[]);

int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " --mode [serial|openmp|mpi] <other args...>" << std::endl;
        return 1;
    }
    std::string mode_flag = argv[1];
    if (mode_flag != "--mode") {
        std::cerr << "First argument must be --mode" << std::endl;
        return 1;
    }
    // Ensure there's a mode value AND at least one more argument for the sub-main
    if (argc < 4) {
        std::cerr << "Error: --mode requires an argument (serial|openmp|mpi) and additional arguments for the selected mode." << std::endl;
        return 1;
    }
    std::string mode = argv[2];

    // Shift argv so that sub-mains see their expected arguments
    // Remove executable name, --mode flag, and the mode value itself
    argc -= 3;
    argv += 3;

    // Now argv[0] should be the first argument intended for the sub-main
    // e.g., graph_file for serial/openmp, or potentially different for mpi

    if (mode == "serial") return serial_main(argc, argv);
    if (mode == "openmp" || mode == "parallel") return parallel_main(argc, argv);
    if (mode == "mpi") return mpi_main(argc, argv);
    std::cerr << "Unknown mode: " << mode << std::endl;
    return 1;
}

int serial_main(int argc, char* argv[]) {
    // Argument parsing: prog <graph> <start> [mode] [changes_file]
    if (argc < 3) { // Need at least graph and start node
        std::cerr << "Usage: " << argv[0] << " <graph_file> <start_node> [mode] [changes_file]" << std::endl;
        std::cerr << "Modes: baseline, sequential (default: baseline)" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int start_node = -1;
    std::string mode = "baseline";
    idx_t num_partitions = 1; // Re-introduce and initialize num_partitions
    std::string update_filename = "";
    bool has_changes = false;

    try {
        start_node = std::stoi(argv[2]); // Parse start node
        if (argc > 3) mode = argv[3];   // Parse mode if present

        // Correctly parse changes_file based on position and mode
        if (argc > 4) { // If there's an argument after the mode
            if (mode == "sequential") {
                update_filename = argv[4];
                has_changes = true;
            } else if (mode == "baseline") {
                // Allow changes file for baseline recomputation
                update_filename = argv[4];
                has_changes = true; // Mark that changes should be loaded/applied to graph before recompute
                std::cout << "Warning: Changes file provided (" << update_filename << ") in 'baseline' mode. Changes will be applied for recomputation." << std::endl;
            } else {
                 std::cerr << "Warning: Unexpected argument '" << argv[4] << "' provided after mode '" << mode << "'. Ignoring." << std::endl;
                 // If other serial modes were added, handle them here.
            }
        }

        // Validate mode *after* parsing potential changes file
        if (mode != "baseline" && mode != "sequential") {
             std::cerr << "Error: Invalid mode '" << mode << "' specified for serial execution." << std::endl;
             return 1;
        }


    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        return 1;
    }

    // omp_set_num_threads(omp_get_max_threads()); // No need to set threads explicitly for serial

    try {
        std::cout << "Loading graph from " << filename << "..." << std::endl;
        Graph graph = load_graph(filename);
        std::cout << "Graph loaded: " << graph.num_vertices << " vertices." << std::endl;

        if (start_node < 0 || start_node >= graph.num_vertices) {
            std::cerr << "Error: Start node " << start_node << " is out of range [0, " << graph.num_vertices - 1 << "]." << std::endl;
            return 1;
        }

        // --- METIS Partitioning ---
        // ... (rest of METIS logic remains the same, now uses the initialized num_partitions) ...
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

        // --- Load Changes (if filename provided) ---
        std::vector<EdgeChange> changes;
        if (has_changes && !update_filename.empty()) {
             std::cout << "\nLoading changes from " << update_filename << "..." << std::endl;
             changes = load_edge_changes(update_filename); // Load the combined changes
             std::cout << "Changes loaded: " << changes.size() << " total." << std::endl;
        } else if (mode != "baseline") {
             std::cout << "\nWarning: Mode '" << mode << "' selected, but no changes file provided or loaded." << std::endl;
             has_changes = false; // Ensure no update attempt if loading failed or no file
        }


        // Print all loaded changes for debugging
        // ... (optional: keep this loop if needed) ...
        // for (const auto& change : changes) {
        //     std::cout << (change.is_insertion ? "Insert" : "Delete") << " edge: (" << change.u << ", " << change.v << ")" << std::endl;
        // }


        // Apply changes if loaded
        if (has_changes) {
             // Apply changes to the graph structure G itself *before* calling update algorithms
             std::cout << "Applying structural changes to graph..." << std::endl;
             auto apply_start = std::chrono::high_resolution_clock::now();
             int skipped_changes = 0;
             for (const auto& change : changes) {
                 try {
                     // Validate indices before applying change
                     if (change.u < 0 || change.u >= graph.num_vertices || change.v < 0 || change.v >= graph.num_vertices) {
                         std::cerr << "Warning: Skipping change due to out-of-bounds vertex index: "
                                   << (change.is_insertion ? "i " : "d ") << change.u << " " << change.v << std::endl;
                         skipped_changes++;
                         continue;
                     }

                     if (change.is_insertion) {
                         graph.add_edge(change.u, change.v, change.weight);
                     } else {
                         // std::cout << "Calling remove_edge for (" << change.u << ", " << change.v << ")" << std::endl; // Debug
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

             // std::cout << "\n--- After Applying Changes (Graph Structure) ---" << std::endl; // Debug
             // print_sssp_result(sssp_result); // SSSP result not updated yet

             // Debug: Verify edge removal if specific edge was deleted
             // Example: if edge (0,1) was in changes as deletion:
             // std::cout << "Edge (0, 1) removed check: " << !graph.has_edge(0, 1) << std::endl;

        } else if (mode != "baseline") {
             std::cout << "\nMode '" << mode << "' selected, but no changes file provided/loaded. Only initial SSSP was computed." << std::endl;
             mode = "baseline"; // Treat as baseline if no changes for update modes
        } else {
             std::cout << "\nNo changes file provided (or baseline mode selected)." << std::endl;
        }

        // Validate mode string *after* potential modification above
        if (mode != "baseline" && mode != "sequential" && mode != "openmp" && mode != "opencl") {
            std::cerr << "Error: Invalid mode '" << mode << "' specified." << std::endl;
            return 1; // Exit if mode is invalid
        }


        // --- Perform SSSP Update or Baseline Recomputation ---
        std::cout << "\nRunning mode: " << mode << std::endl;
        auto start_time_update = std::chrono::high_resolution_clock::now();
        bool update_performed = false;

        if (mode == "sequential") {
            if (has_changes) {
                 std::cout << "Processing batch sequentially..." << std::endl;
                 // Ensure process_batch_sequential works correctly with the modified graph
                 process_batch_sequential(graph, sssp_result, changes);
                 update_performed = true;
            } else {
                 std::cout << "Sequential mode selected, but no changes file provided/loaded. Skipping update." << std::endl;
            }
        } else if (mode == "openmp") {
             if (has_changes) {
                std::cout << "Processing batch using OpenMP (dynamic workflow)..." << std::endl;
                // Ensure TestDynamicSSSPWorkflow_OpenMP works correctly with the modified graph
                TestDynamicSSSPWorkflow_OpenMP(graph, sssp_result, changes);
                update_performed = true;
             } else {
                 std::cout << "OpenMP mode selected, but no changes file provided/loaded. Skipping update." << std::endl;
             }
        } else if (mode == "opencl") {
             if (has_changes) {
                std::cout << "OpenCL mode selected (placeholder - not implemented)." << std::endl;
                // BatchUpdate_OpenCL(graph, sssp_result, changes); // Placeholder
             } else {
                  std::cout << "OpenCL mode selected, but no changes file provided/loaded. Skipping update." << std::endl;
             }
        } else if (mode == "baseline") {
            if (has_changes) {
                std::cout << "Recomputing SSSP from scratch (baseline) on modified graph..." << std::endl;
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

            // Debug check (optional)
            // if (sssp_result.parent[1] == 0) { // Example check
            //     std::cerr << "Error: Parent of Vertex 1 is still 0 after edge (0, 1) deletion!" << std::endl;
            // }
        }

        // --- Output Results ---
        std::cout << "\n--- Timings ---" << std::endl;
        std::cout << "Initial Dijkstra: " << initial_sssp_time.count() << " ms" << std::endl;
        if (update_performed) {
             std::cout << "Update/Recompute (" << mode << "): " << update_time.count() << " ms" << std::endl;
        } else if (has_changes && mode != "baseline") { // If changes existed but update wasn't performed (e.g., OpenCL)
             std::cout << "Update (" << mode << "): Skipped (mode not fully implemented or other issue)" << std::endl;
        } else if (!has_changes && mode != "baseline") { // If no changes for update modes
             std::cout << "Update (" << mode << "): Skipped (no changes file provided/loaded)" << std::endl;
        } else if (!has_changes && mode == "baseline") { // Baseline without changes
             // No update time to report
        }


        // Optional: Verify results or save Dist/Parent arrays
        // ... (optional verification code) ...


    } catch (const std::exception& e) {
        std::cerr << "\nError: An exception occurred: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nExecution finished." << std::endl;
    return 0;
}

