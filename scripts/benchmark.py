import argparse
import glob
import os
import subprocess
import csv
import re
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Set up better plot styling
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16
sns.set_style("whitegrid")
sns.set_palette("pastel")
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

# Patterns to extract timings from program output
TIMING_RE = {
    'serial': re.compile(r'Initial Dijkstra: ([0-9.]+) ms.*Update/Recompute \(sequential\): ([0-9.]+) ms', re.S),
    'openmp': re.compile(r'Initial SSSP completed in ([0-9.]+) ms.*Dynamic update completed in ([0-9.]+) ms', re.S),
    'mpi': re.compile(r'--- Timings \(MPI\) ---.*Initial SSSP \(Rank 0\): ([0-9.]+) ms.*Graph distribution \(MPI\): [0-9.]+ ms.*Distributed Update Time \(excluding graph distribution\): ([0-9.]+) ms', re.S),
    'mpi_openmp': re.compile(r'--- Timings \(MPI\) ---.*Initial SSSP \(Rank 0\): ([0-9.]+) ms.*Graph distribution \(MPI\): [0-9.]+ ms.*Distributed Update Time \(excluding graph distribution\): ([0-9.]+) ms', re.S)
}

# Executable paths and MPI hostfile
SSSP_EXE = '/Users/azaleas/Developer/CLionProjects/Parallel-SSSP/cmake-build-debug/SSSP'
HOSTFILE = '/Users/azaleas/Developer/CLionProjects/Parallel-SSSP/hostfile'


def run_serial(graph, start, updates):
    if updates:
        cmd = [SSSP_EXE, '--mode', 'serial', graph, str(start), 'sequential', updates]
    else:
        cmd = [SSSP_EXE, '--mode', 'serial', graph, str(start), 'baseline']
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def run_openmp(graph, start, threads, updates):
    os.environ['OMP_NUM_THREADS'] = str(threads)
    cmd = [SSSP_EXE, '--mode', 'openmp', graph, str(start), str(threads), '1']
    if updates:
        cmd.append(updates)
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def run_mpi(graph, start, ranks, updates):
    cmd = [
        'mpirun', '--hostfile', HOSTFILE, '-np', str(ranks),
        SSSP_EXE, '--mode', 'mpi', graph, str(start)
    ]
    if updates:
        cmd.append(updates)
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def run_mpi_openmp(graph, start, ranks, threads, updates):
    os.environ['OMP_NUM_THREADS'] = str(threads)
    cmd = [
        'mpirun', '--hostfile', HOSTFILE, '-np', str(ranks),
        SSSP_EXE, '--mode', 'mpi_openmp', graph, str(start)
    ]
    if updates:
        cmd.append(updates)
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def parse_timings(output, mode):
    m = TIMING_RE[mode].search(output)
    if not m:
        print(f"[parse_timings] Unexpected output for mode '{mode}':\n{output}")
        raise ValueError(f"Failed to parse timings for mode {mode}")
    # return only update phase time
    return float(m.group(2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphs', nargs='*', default=None,
                        help='Graph files to test (default: data/*.edges)')
    parser.add_argument('--update-sizes', nargs='*', type=int, default=None,
                        help='List of update counts (default: detect from filenames)')
    parser.add_argument('--modes', nargs='+', choices=['serial', 'openmp', 'mpi', 'mpi_openmp'],
                        default=['serial', 'openmp', 'mpi', 'mpi_openmp'])
    parser.add_argument('--mpi-ranks', type=int, default=4)
    parser.add_argument('--thread-counts', nargs='+', type=int, default=[2, 4, 8, 16],
                        help='List of thread counts to test for OpenMP modes')
    parser.add_argument('--start-node', type=int, default=0)
    parser.add_argument('--output', default='benchmark_results.csv')
    parser.add_argument('--test', action='store_true', help='Run built-in test commands instead of scanning graphs')
    args = parser.parse_args()

    # If test mode, run the four example variants on test.edges
    if args.test:
        test_graph = 'data/test.edges'
        test_updates = 'data/test_updates.edges'
        print('== Running built-in tests ==')
        # Serial
        print('Serial test:')
        out = run_serial(test_graph, args.start_node, test_updates)
        print(out.stdout)
        # OpenMP with different thread counts
        for threads in args.thread_counts:
            print(f'OpenMP test with {threads} threads:')
            out = run_openmp(test_graph, args.start_node, threads, test_updates)
            print(out.stdout)
        # MPI
        print('MPI test:')
        out = run_mpi(test_graph, args.start_node, 2, test_updates)
        print(out.stdout)
        # MPI+OpenMP hybrid with different thread counts
        for threads in args.thread_counts:
            print(f'MPI+OpenMP hybrid test with {threads} threads:')
            out = run_mpi_openmp(test_graph, args.start_node, 2, threads, test_updates)
            print(out.stdout)
        exit(0)

    # Discover graphs
    if not args.graphs:
        all_edges = glob.glob(os.path.join('data', '*.edges'))
        args.graphs = [g for g in all_edges if not g.endswith('_updates.edges')]
        if not args.graphs:
            parser.error('No graph files found in data/*.edges')

    sizes = set(args.update_sizes) if args.update_sizes else None

    rows = []
    for graph in args.graphs:
        base, _ = os.path.splitext(graph)
        upd_files = sorted(glob.glob(f"{base}_updates*.edges"))
        if not upd_files:
            print(f"Warning: No update files for {graph}, skipping.")
            continue
        for upd in upd_files:
            try:
                count = sum(1 for line in open(upd) if line.strip() and not line.startswith('#'))
            except Exception as e:
                print(f"Warning: could not read {upd}: {e}")
                continue
            if sizes and count not in sizes:
                continue
            
            # Serial mode (no threads)
            if 'serial' in args.modes:
                print(f"Running serial on {graph} (updates={count})")
                try:
                    t = run_serial(graph, args.start_node, upd)
                    update_ms = parse_timings(t.stdout, 'serial')
                    rows.append({
                        'graph': os.path.basename(graph),
                        'updates': count,
                        'mode': 'serial',
                        'threads': 1,  # Serial mode uses 1 thread
                        'ranks': 1,    # Add ranks field with default value
                        'update_ms': update_ms
                    })
                except subprocess.CalledProcessError as e:
                    print(f"*** SERIAL FAILED on {graph}, updates={count} ***")
                    print("Return code:", e.returncode)
                    print("Stdout:\n", e.stdout)
                    print("Stderr:\n", e.stderr)
            
            # OpenMP mode with different thread counts
            if 'openmp' in args.modes:
                for threads in args.thread_counts:
                    print(f"Running OpenMP with {threads} threads on {graph} (updates={count})")
                    try:
                        t = run_openmp(graph, args.start_node, threads, upd)
                        update_ms = parse_timings(t.stdout, 'openmp')
                        rows.append({
                            'graph': os.path.basename(graph),
                            'updates': count,
                            'mode': 'openmp',
                            'threads': threads,
                            'ranks': 1,  # Add ranks field with default value
                            'update_ms': update_ms
                        })
                    except subprocess.CalledProcessError as e:
                        print(f"*** OPENMP ({threads} threads) FAILED on {graph}, updates={count} ***")
                        print("Return code:", e.returncode)
                        print("Stdout:\n", e.stdout)
                        print("Stderr:\n", e.stderr)
            
            # MPI mode (fixed number of ranks)
            if 'mpi' in args.modes:
                print(f"Running MPI with {args.mpi_ranks} ranks on {graph} (updates={count})")
                try:
                    t = run_mpi(graph, args.start_node, args.mpi_ranks, upd)
                    update_ms = parse_timings(t.stdout, 'mpi')
                    rows.append({
                        'graph': os.path.basename(graph),
                        'updates': count,
                        'mode': 'mpi',
                        'threads': 1,  # MPI uses 1 thread per rank
                        'ranks': args.mpi_ranks,
                        'update_ms': update_ms
                    })
                except subprocess.CalledProcessError as e:
                    print(f"*** MPI FAILED on {graph}, updates={count} ***")
                    print("Return code:", e.returncode)
                    print("Stdout:\n", e.stdout)
                    print("Stderr:\n", e.stderr)
            
            # MPI+OpenMP hybrid mode with different thread counts
            if 'mpi_openmp' in args.modes:
                for threads in args.thread_counts:
                    print(f"Running MPI+OpenMP with {args.mpi_ranks} ranks, {threads} threads on {graph} (updates={count})")
                    try:
                        t = run_mpi_openmp(graph, args.start_node, args.mpi_ranks, threads, upd)
                        update_ms = parse_timings(t.stdout, 'mpi_openmp')
                        rows.append({
                            'graph': os.path.basename(graph),
                            'updates': count,
                            'mode': 'mpi_openmp',
                            'threads': threads,
                            'ranks': args.mpi_ranks,
                            'update_ms': update_ms
                        })
                    except subprocess.CalledProcessError as e:
                        print(f"*** MPI+OPENMP ({threads} threads) FAILED on {graph}, updates={count} ***")
                        print("Return code:", e.returncode)
                        print("Stdout:\n", e.stdout)
                        print("Stderr:\n", e.stderr)

    if not rows:
        print("Error: No results collected.")
        exit(1)

    # Write CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {args.output}")

    # Load into DataFrame
    df = pd.DataFrame(rows)

    # Compute speedup relative to serial for each graph and update count
    serial_times = df[df['mode'] == 'serial'].set_index(['graph', 'updates'])['update_ms']
    df['speedup'] = df.apply(
        lambda row: serial_times.loc[(row['graph'], row['updates'])] / row['update_ms'] 
        if row['mode'] != 'serial' else 1.0, 
        axis=1
    )

    # Define a better color palette with pastel colors
    mode_colors = {
        'serial': '#B3B3B3',    # Light gray
        'openmp': '#A8D8F0',    # Pastel blue
        'mpi': '#A8E6CF',       # Pastel green
        'mpi_openmp': '#FFAAA6' # Pastel red/pink
    }
    
    # Plot 1: Speedup by thread count for OpenMP - Enhanced version
    plt.figure(figsize=(12, 7))
    openmp_df = df[df['mode'] == 'openmp']
    
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*']
    
    for i, update_count in enumerate(sorted(df['updates'].unique())):
        subset = openmp_df[openmp_df['updates'] == update_count]
        if not subset.empty:
            plt.plot(
                subset['threads'], 
                subset['speedup'], 
                marker=markers[i % len(markers)],
                markersize=8,
                linewidth=2.5, 
                label=f"{update_count} updates"
            )
    
    plt.xlabel('Thread Count', fontsize=14, fontweight='bold')
    plt.ylabel('Speedup over Serial', fontsize=14, fontweight='bold')
    plt.title('OpenMP Speedup by Thread Count', fontsize=16, fontweight='bold')
    plt.legend(title='Update Size', fontsize=12, title_fontsize=13, frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    plt.xticks(sorted(df['threads'].unique()))
    for update_count in sorted(df['updates'].unique()):
        subset = openmp_df[openmp_df['updates'] == update_count]
        if not subset.empty:
            max_row = subset.loc[subset['speedup'].idxmax()]
            plt.annotate(f"{max_row['speedup']:.2f}x",
                        (max_row['threads'], max_row['speedup']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontweight='bold')
    plt.savefig('openmp_speedup_by_threads.png', dpi=300, bbox_inches='tight')
    
    # Plot 2: Speedup by thread count for MPI+OpenMP - Enhanced version
    plt.figure(figsize=(12, 7))
    hybrid_df = df[df['mode'] == 'mpi_openmp']
    
    for i, update_count in enumerate(sorted(df['updates'].unique())):
        subset = hybrid_df[hybrid_df['updates'] == update_count]
        if not subset.empty:
            plt.plot(
                subset['threads'], 
                subset['speedup'], 
                marker=markers[i % len(markers)],
                markersize=8,
                linewidth=2.5, 
                label=f"{update_count} updates"
            )
    
    plt.xlabel('Thread Count per Rank', fontsize=14, fontweight='bold')
    plt.ylabel('Speedup over Serial', fontsize=14, fontweight='bold')
    plt.title(f'MPI+OpenMP Hybrid Speedup by Thread Count ({args.mpi_ranks} Ranks)', 
              fontsize=16, fontweight='bold')
    plt.legend(title='Update Size', fontsize=12, title_fontsize=13, frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    plt.xticks(sorted(df['threads'].unique()))
    for update_count in sorted(df['updates'].unique()):
        subset = hybrid_df[hybrid_df['updates'] == update_count]
        if not subset.empty:
            max_row = subset.loc[subset['speedup'].idxmax()]
            plt.annotate(f"{max_row['speedup']:.2f}x",
                        (max_row['threads'], max_row['speedup']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontweight='bold')
    plt.tight_layout()
    plt.savefig('mpi_openmp_speedup_by_threads.png', dpi=300, bbox_inches='tight')
    
    # Plot 3: Compare best performer for each mode - Enhanced version
    plt.figure(figsize=(12, 7))
    
    # Get best speedup for each mode and update count
    best_speedups = df.groupby(['mode', 'updates'])['speedup'].max().reset_index()
    
    for mode in args.modes:
        subset = best_speedups[best_speedups['mode'] == mode]
        if not subset.empty:
            plt.plot(
                subset['updates'], 
                subset['speedup'], 
                marker='o',
                markersize=10,
                linewidth=3,
                color=mode_colors.get(mode, 'black'),
                label=mode
            )
    
    plt.xlabel('Number of Updates', fontsize=14, fontweight='bold')
    plt.ylabel('Best Speedup over Serial', fontsize=14, fontweight='bold')
    plt.title('Best Speedup by Mode and Update Size', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    plt.savefig('best_speedup_by_mode.png', dpi=300, bbox_inches='tight')
    
    # Plot 4: MPI only comparison - Enhanced version
    plt.figure(figsize=(12, 7))
    
    # Get best speedup for each mode and update count, excluding OpenMP
    mpi_only_modes = [m for m in args.modes if m != 'openmp' and m != 'serial']
    
    best_mpi_speedups = df[df['mode'].isin(mpi_only_modes)].groupby(['mode', 'updates'])['speedup'].max().reset_index()
    
    for mode in mpi_only_modes:
        subset = best_mpi_speedups[best_mpi_speedups['mode'] == mode]
        if not subset.empty:
            plt.plot(
                subset['updates'], 
                subset['speedup'], 
                marker='o',
                markersize=10,
                linewidth=3,
                color=mode_colors.get(mode, 'black'),
                label=mode
            )
    
    plt.xlabel('Number of Updates', fontsize=14, fontweight='bold')
    plt.ylabel('Best Speedup over Serial', fontsize=14, fontweight='bold')
    plt.title('MPI vs MPI+OpenMP Best Speedup by Update Size', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('mpi_comparison_speedup.png', dpi=300, bbox_inches='tight')
    
    # NEW PLOT 5: Histogram comparing MPI and MPI+OpenMP with thread count effects
    plt.figure(figsize=(14, 8))
    
    # Filter for MPI and MPI+OpenMP modes only
    mpi_data = df[df['mode'].isin(['mpi', 'mpi_openmp'])].copy()  # Create explicit copy
    
    # Create a new column for plot labels
    mpi_data['config'] = mpi_data.apply(
        lambda row: f"{row['mode']} ({row['threads']} thread{'s' if row['threads'] > 1 else ''})" 
        if row['mode'] == 'mpi_openmp' else f"{row['mode']} ({row['ranks']} ranks)", 
        axis=1
    )
    
    # For each update count, create a grouped bar chart
    update_counts = sorted(mpi_data['updates'].unique())
    
    # Calculate the number of configurations for bar positioning
    configs = sorted(mpi_data['config'].unique())
    n_configs = len(configs)
    
    # Set up x positions
    x = np.arange(len(update_counts))
    width = 0.8 / n_configs
    
    # Create a mapping of config to position
    config_positions = {config: i for i, config in enumerate(configs)}
    
    # Create distinct pastel colors for different thread counts
    # Use more distinct colors for different thread configurations
    thread_colors = {
        'mpi (2 ranks)': '#A8E6CF',       # Pastel green
        'mpi_openmp (1 thread)': '#FFD3B6',   # Pastel orange
        'mpi_openmp (2 threads)': '#FFAAA6',  # Pastel red/pink
        'mpi_openmp (4 threads)': '#FF8B94',  # Deeper pastel pink
        'mpi_openmp (8 threads)': '#D8A1FF',  # Pastel purple
        'mpi_openmp (16 threads)': '#A1C4FF', # Pastel blue
    }
    
    # Plot each configuration as a grouped bar
    for config in configs:
        subset = mpi_data[mpi_data['config'] == config]
        if subset.empty:
            continue
            
        # Collect data points for this configuration
        heights = []
        positions = []
        
        for i, update_count in enumerate(update_counts):
            config_subset = subset[subset['updates'] == update_count]
            if not config_subset.empty:
                heights.append(config_subset['speedup'].max())
                positions.append(i)
        
        # Plot the bars for this configuration
        offset = config_positions[config] - (n_configs - 1) / 2
        bars = plt.bar(
            [p + offset * width for p in positions], 
            heights, 
            width=width * 0.9,
            label=config,
            color=thread_colors.get(config, '#CCCCCC'),  # Use the thread-specific color
            edgecolor='black',
            linewidth=1
        )
        
        # Add value labels above bars
        for bar, height in zip(bars, heights):
            plt.text(
                bar.get_x() + bar.get_width() / 2, 
                height, 
                f'{height:.2f}x', 
                ha='center', 
                va='bottom',
                fontweight='bold',
                fontsize=9
            )
    
    plt.xlabel('Number of Updates', fontsize=14, fontweight='bold')
    plt.ylabel('Speedup over Serial', fontsize=14, fontweight='bold')
    plt.title('MPI vs MPI+OpenMP Speedup Comparison by Configuration', fontsize=16, fontweight='bold')
    plt.xticks(x, update_counts)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    plt.legend(fontsize=11, title='Configuration', title_fontsize=12, 
               frameon=True, fancybox=True, framealpha=0.9, ncol=2)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig('mpi_thread_histogram.png', dpi=300, bbox_inches='tight')
    
    print("Generated plots: openmp_speedup_by_threads.png, mpi_openmp_speedup_by_threads.png, best_speedup_by_mode.png, mpi_comparison_speedup.png, mpi_thread_histogram.png")
