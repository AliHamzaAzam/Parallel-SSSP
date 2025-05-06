"""
scripts/benchmark.py - Generate updates, run benchmarks (serial, OpenMP, MPI), and plot results
Usage:
  python3 scripts/benchmark.py --graphs data/*.edges --update-sizes 100 1000 10000 \
       --modes serial openmp mpi --mpi-ranks 4 --threads 8
"""
import argparse
import glob
import os
import shlex
import subprocess
import csv
import time
import re
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd

# Patterns to extract timings from program output
TIMING_RE = {
    'serial': re.compile(r'Initial Dijkstra: ([0-9.]+) ms.*Sequential Update: ([0-9.]+) ms', re.S),
    'openmp': re.compile(r'Initial SSSP completed in ([0-9.]+) ms.*Dynamic update completed in ([0-9.]+) ms', re.S),
    'mpi': re.compile(r'Initial SSSP \(Rank 0\): ([0-9.]+) ms.*Distributed Update Time: ([0-9.]+) ms', re.S)
}

"""Note: Generate_updates is not used; script uses existing *_updates_*.edges files."""

def run_serial(sssp_exe, graph, start, updates):
    cmd = [sssp_exe, '--mode', 'serial', graph, str(start)]
    if updates:
        cmd.append(updates)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout

def run_openmp(sssp_exe, graph, start, threads, updates):
    os.environ['OMP_NUM_THREADS'] = str(threads)
    cmd = [sssp_exe, '--mode', 'openmp', graph, str(start), str(threads), '1']
    if updates:
        cmd.append(updates)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout

def run_mpi(sssp_exe, graph, start, ranks, updates):
    cmd = ['mpirun', '-np', str(ranks), sssp_exe, '--mode', 'mpi', graph, str(start)]
    if updates:
        cmd += [updates, str(ranks)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout

def parse_timings(output, mode):
    m = TIMING_RE[mode].search(output)
    if not m:
        raise ValueError(f"Failed to parse timings for mode {mode}")
    init = float(m.group(1))
    update = float(m.group(2))
    return init, update

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphs', nargs='*', default=None, help='Graph files to test (default: data/*.edges)')
    parser.add_argument('--update-sizes', nargs='*', type=int, default=None, help='List of update counts (default: [100, 1000, 10000])')
    parser.add_argument('--modes', nargs='+', choices=['serial','openmp','mpi'], default=['serial','openmp','mpi'])
    parser.add_argument('--mpi-ranks', type=int, default=4)
    parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--start-node', type=int, default=0)
    parser.add_argument('--output', default='benchmark_results.csv')
    args = parser.parse_args()

    # Set default graphs if none provided (exclude existing update files)
    if not args.graphs:
        all_edges = glob.glob(os.path.join('data', '*.edges'))
        args.graphs = [g for g in all_edges if not g.endswith('_updates.edges')]
        if not args.graphs:
            parser.error('No graph files found in data/*.edges')
    # If update_sizes not provided, will auto-detect from filenames
    if args.update_sizes:
        sizes = set(args.update_sizes)
    else:
        sizes = None

    # Paths to executables
    root = os.getcwd()
    sssp_exe = os.path.join(root, 'cmake-build-debug', 'SSSP')
    gen_exe  = os.path.join(root, 'cmake-build-debug', 'generate_updates')

    # CSV header
    rows = []
    # Discover and test each graph with its update files
    for graph in args.graphs:
        base, _ = os.path.splitext(graph)
        # Find any update files for this graph: base_updates*.edges
        pattern = f"{base}_updates*.edges"
        upd_files = sorted(glob.glob(pattern))
        if not upd_files:
            print(f"Warning: No update files found for {graph} (pattern {pattern}), skipping.")
            continue
        for upd_file in upd_files:
            # Count non-comment lines as number of updates
            try:
                with open(upd_file) as uf:
                    u = sum(1 for line in uf if line.strip() and not line.startswith('#'))
            except Exception as e:
                print(f"Warning: Failed to read {upd_file}: {e}, skipping.")
                continue
            # If user provided specific sizes, filter accordingly
            if sizes and u not in sizes:
                continue
            for mode in args.modes:
                print(f"Running {mode} on {graph}, updates={u}")
                if mode=='serial':
                    out = run_serial(sssp_exe, graph, args.start_node, upd_file)
                elif mode=='openmp':
                    out = run_openmp(sssp_exe, graph, args.start_node, args.threads, upd_file)
                else:
                    out = run_mpi(sssp_exe, graph, args.start_node, args.mpi_ranks, upd_file)
                init, upd_time = parse_timings(out, mode)
                rows.append({'graph': os.path.basename(graph), 'updates': u, 'mode': mode,
                             'init_ms': init, 'update_ms': upd_time})
    # Ensure there is data to write
    if not rows:
        print("Error: No benchmark results collected. Exiting.")
        exit(1)
    # Write CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results written to {args.output}")

    # Plot
    df = pd.DataFrame(rows)
    for phase in ['init_ms','update_ms']:
        plt.figure()
        for mode in args.modes:
            sub = df[df.mode==mode]
            plt.plot(sub.updates, sub[phase], marker='o', label=mode)
        plt.xlabel('Number of updates')
        plt.ylabel(f'{phase} (ms)')
        plt.title(f'{phase} vs updates')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{phase}_benchmark.png')
        print(f"Plot saved to {phase}_benchmark.png")
