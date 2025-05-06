# Parallel-SSSP

Parallel-SSSP implements dynamic Single-Source Shortest Path (SSSP) algorithms for weighted undirected graphs. It supports:

- **Serial** SSSP using Dijkstra’s algorithm and batch dynamic updates.
- **Shared-memory parallel** dynamic updates via OpenMP.
- **Distributed-memory parallel** dynamic updates via MPI (with optional hybrid MPI+OpenMP).

A utility to generate random edge-change sequences and a suite of benchmarking and visualization scripts are included.

---

## Features

- Load graphs in Matrix Market (.mtx) and simple edge list (.edges) formats.
- Compute initial SSSP from a given source node using Dijkstra.
- Apply dynamic edge updates (insertions, deletions, weight changes) incrementally.
- Three execution modes:
  - **serial**: baseline sequential update or full recomputation.
  - **openmp**: multi-threaded dynamic update on shared memory.
  - **mpi**: distributed dynamic update across MPI ranks.
  - **mpi_openmp**: hybrid MPI + OpenMP.
- `generate_updates` tool to produce reproducible test workloads.
- Python scripts for benchmarking, partition visualization, and graph plotting.

---

## Directory Structure

```text
Parallel-SSSP/
├── CMakeLists.txt          # Build configuration
├── include/                # Public headers (graph structures, utils)
│   ├── graph.hpp
│   └── utils.hpp
├── src/                    # Core implementations
│   ├── graph.cpp           # Graph data structures
│   ├── utils.cpp           # I/O and Dijkstra
│   ├── main.cpp            # Common driver (dispatch by mode)
│   ├── serial/             # Serial mode entry point
│   ├── parallel/           # OpenMP-based parallel mode
│   └── mpi/                # MPI-based distributed mode
├── tools/                  # Auxiliary tools
│   └── generate_updates.cpp  # Random update file generator
├── scripts/                # Visualization and benchmarking scripts
│   ├── benchmark.py
│   ├── generate_graph.py
│   └── visualize_graph.sh
├── data/                   # Sample graphs and update files
├── results/                # Sample output graphs and dot files
├── build/                  # Local build artifacts (ignored)
└── README.md               # This file
``` 

---

## Requirements

- C++17 compiler (tested with Clang and GCC)
- [CMake](https://cmake.org/) ≥ 3.10
- [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) library
- [MPI](https://www.open-mpi.org/) implementation (for `mpi` mode)
- OpenMP (for `openmp` and `mpi_openmp` modes)
- Python 3 with `numpy`, `pandas`, `matplotlib` (for scripts)

---

## Building

```bash
mkdir -p build && cd build
cmake ..
make -j
``` 
This produces:
- `SSSP` executable (multi-mode driver)
- `generate_updates` tool

Alternatively, use CLion or VS Code CMake integration.

---

## Usage

All modes are invoked through the single `SSSP` driver with `--mode` flag:

```bash
./SSSP --mode <serial|openmp|mpi|mpi_openmp> <graph_file> <start_node> [options]
```

### Serial Mode

```bash
./SSSP --mode serial <graph.edges|.mtx> <start_node> [baseline|sequential] [<updates_file>]
```
- `baseline`: apply all updates to the graph then recompute SSSP.
- `sequential`: perform incremental dynamic update on each change.

### OpenMP Mode

```bash
./SSSP --mode openmp <graph> <start_node> <num_threads> <num_partitions> [<updates_file>]
```
- `num_threads`: number of OpenMP threads.
- `num_partitions`: number of METIS partitions for work decomposition.

### MPI Mode

```bash
mpirun -n <ranks> ./SSSP --mode mpi <graph> <start_node> [<updates_file>] [<num_partitions>]
```
- `<ranks>`: number of MPI processes.
- `<num_partitions>` (optional): number of METIS partitions (defaults to ranks).

### Hybrid MPI+OpenMP Mode

```bash
mpirun -n <ranks> ./SSSP --mode mpi_openmp <graph> <start_node> [<updates_file>] [<num_partitions>]
export SSSP_DEBUG=true  # Enable verbose debug logs
``` 

---

## Generating Update Workloads

```bash
./generate_updates <input_graph.edges> <output_updates.edges> <num_updates>
```
Produces randomized sequence of insertions and deletions with realistic weights.

---

## Benchmarking

Use the Python script for automated evaluation:

```bash
python3 scripts/benchmark.py --graphs data/*.edges --modes serial openmp mpi --start-node 0 \
    --threads 4 --mpi-ranks 4 --output benchmark_results.csv
```
Plots and CSV reports are generated automatically.

---

## Visualization

- `scripts/visualize_graph.py`: plot graph using `graphviz`.

---

## Contributing

Contributions are welcome! Please open issues or pull requests. Ensure code style consistency and update this README with new features.

---

## License

This project is provided under the MIT License. See [LICENSE](LICENSE) for details, or contact the maintainers if no license file is present.
