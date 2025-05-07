#!/usr/bin/env python3

import argparse
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import time
import json
import re

# Set up better plot styling
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16
sns.set_style("whitegrid")
sns.set_palette("tab10")

# Constants
SSSP_EXE = '/Users/azaleas/Developer/CLionProjects/Parallel-SSSP/cmake-build-debug/SSSP'
HOSTFILE = '/Users/azaleas/Developer/CLionProjects/Parallel-SSSP/hostfile'
RESULTS_DIR = 'performance_metrics_results'

def ensure_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def run_command_with_time_metrics(cmd, env=None, config_name=None, output_dir=None):
    """Run a command and collect detailed time metrics using /usr/bin/time"""
    if not config_name:
        config_name = f"run_{int(time.time())}"
    
    output_file = os.path.join(output_dir, f"{config_name}.txt") if output_dir else f"{config_name}.txt"
    time_output_file = os.path.join(output_dir, f"{config_name}_time.txt") if output_dir else f"{config_name}_time.txt"
    
    # Prepare time command with resource metrics
    time_cmd = ["/usr/bin/time", "-lp", "-o", time_output_file]
    
    # Combine with the actual command
    full_cmd = time_cmd + cmd
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run the command and collect output
        result = subprocess.run(full_cmd, env=env, check=True, text=True, capture_output=True)
        
        # Save program output to file
        with open(output_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Output:\n{result.stdout}\n")
            f.write(f"Errors:\n{result.stderr}\n")
        
        # Read the time metrics
        with open(time_output_file, 'r') as f:
            time_output = f.read()
        
        # Parse time metrics
        metrics = parse_time_output(time_output)
        
        # Add program output
        metrics['stdout'] = result.stdout
        metrics['stderr'] = result.stderr
        
        return metrics
    
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return {
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr
        }

def parse_time_output(output):
    """Parse the output from /usr/bin/time to extract metrics"""
    metrics = {}
    
    # Regular expressions to extract metrics
    patterns = {
        'real_time': r'real\s+([0-9.]+)',
        'user_time': r'user\s+([0-9.]+)',
        'sys_time': r'sys\s+([0-9.]+)', 
        'max_rss': r'([0-9]+)\s+maximum resident set size',
        'shared_text': r'([0-9]+)\s+average shared memory size',
        'unshared_data': r'([0-9]+)\s+average unshared data size',
        'page_reclaims': r'([0-9]+)\s+page reclaims',
        'page_faults': r'([0-9]+)\s+page faults',
        'voluntary_cs': r'([0-9]+)\s+voluntary context switches',
        'involuntary_cs': r'([0-9]+)\s+involuntary context switches',
    }
    
    for metric, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            value = match.group(1)
            try:
                metrics[metric] = float(value)
            except ValueError:
                metrics[metric] = value
    
    return metrics

def extract_sssp_timings(stdout, mode):
    """Extract timing information from SSSP program output"""
    timing_data = {}
    
    # Patterns for different modes
    patterns = {
        'serial': {
            'initial_time': r'Initial Dijkstra: ([0-9.]+) ms',
            'update_time': r'Update/Recompute \(sequential\): ([0-9.]+) ms'
        },
        'openmp': {
            'initial_time': r'Initial SSSP completed in ([0-9.]+) ms',
            'update_time': r'Dynamic update completed in ([0-9.]+) ms'
        },
        'mpi': {
            'initial_time': r'Initial SSSP \(Rank 0\): ([0-9.]+) ms',
            'distribution_time': r'Graph distribution \(MPI\): ([0-9.]+) ms',
            'update_time': r'Distributed Update Time \(excluding graph distribution\): ([0-9.]+) ms'
        },
        'mpi_openmp': {
            'initial_time': r'Initial SSSP \(Rank 0\): ([0-9.]+) ms',
            'distribution_time': r'Graph distribution \(MPI\): ([0-9.]+) ms',
            'update_time': r'Distributed Update Time \(excluding graph distribution\): ([0-9.]+) ms'
        }
    }
    
    # Extract timings based on mode
    if mode in patterns:
        for key, pattern in patterns[mode].items():
            match = re.search(pattern, stdout)
            if match:
                timing_data[key] = float(match.group(1))
    
    return timing_data

def run_serial_benchmark(graph, start_node, updates, output_dir):
    """Run serial version and collect metrics"""
    # Command line arguments
    if updates:
        cmd = [SSSP_EXE, '--mode', 'serial', graph, str(start_node), 'sequential', updates]
    else:
        cmd = [SSSP_EXE, '--mode', 'serial', graph, str(start_node), 'baseline']
    
    # Config name
    config_name = f"serial_{os.path.basename(graph)}"
    if updates:
        config_name += f"_updates_{os.path.basename(updates)}"
    
    # Run with time metrics
    metrics = run_command_with_time_metrics(cmd, None, config_name, output_dir)
    
    # Extract SSSP-specific timings
    if 'stdout' in metrics:
        sssp_timings = extract_sssp_timings(metrics['stdout'], 'serial')
        metrics.update(sssp_timings)
    
    # Add configuration metadata
    metrics['mode'] = 'serial'
    metrics['graph'] = os.path.basename(graph)
    metrics['ranks'] = 1
    metrics['threads'] = 1
    metrics['updates'] = os.path.basename(updates) if updates else 'none'
    
    return metrics

def run_openmp_benchmark(graph, start_node, threads, updates, output_dir):
    """Run OpenMP version and collect metrics"""
    # Set environment
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    
    # Command line arguments
    cmd = [SSSP_EXE, '--mode', 'openmp', graph, str(start_node), str(threads), '1']
    if updates:
        cmd.append(updates)
    
    # Config name
    config_name = f"openmp_{os.path.basename(graph)}_t{threads}"
    if updates:
        config_name += f"_updates_{os.path.basename(updates)}"
    
    # Run with time metrics
    metrics = run_command_with_time_metrics(cmd, env, config_name, output_dir)
    
    # Extract SSSP-specific timings
    if 'stdout' in metrics:
        sssp_timings = extract_sssp_timings(metrics['stdout'], 'openmp')
        metrics.update(sssp_timings)
    
    # Add configuration metadata
    metrics['mode'] = 'openmp'
    metrics['graph'] = os.path.basename(graph)
    metrics['ranks'] = 1
    metrics['threads'] = threads
    metrics['updates'] = os.path.basename(updates) if updates else 'none'
    
    return metrics

def run_mpi_benchmark(graph, start_node, ranks, updates, output_dir):
    """Run MPI version and collect metrics"""
    # Command line arguments
    cmd = [
        'mpirun', '--hostfile', HOSTFILE, '-np', str(ranks),
        SSSP_EXE, '--mode', 'mpi', graph, str(start_node)
    ]
    if updates:
        cmd.append(updates)
    
    # Config name
    config_name = f"mpi_{os.path.basename(graph)}_r{ranks}"
    if updates:
        config_name += f"_updates_{os.path.basename(updates)}"
    
    # Run with time metrics
    metrics = run_command_with_time_metrics(cmd, None, config_name, output_dir)
    
    # Extract SSSP-specific timings
    if 'stdout' in metrics:
        sssp_timings = extract_sssp_timings(metrics['stdout'], 'mpi')
        metrics.update(sssp_timings)
    
    # Add configuration metadata
    metrics['mode'] = 'mpi'
    metrics['graph'] = os.path.basename(graph)
    metrics['ranks'] = ranks
    metrics['threads'] = 1
    metrics['updates'] = os.path.basename(updates) if updates else 'none'
    
    return metrics

def run_mpi_openmp_benchmark(graph, start_node, ranks, threads, updates, output_dir):
    """Run MPI+OpenMP version and collect metrics"""
    # Set environment
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    
    # Command line arguments
    cmd = [
        'mpirun', '--hostfile', HOSTFILE, '-np', str(ranks),
        SSSP_EXE, '--mode', 'mpi_openmp', graph, str(start_node)
    ]
    if updates:
        cmd.append(updates)
    
    # Config name
    config_name = f"mpi_openmp_{os.path.basename(graph)}_r{ranks}_t{threads}"
    if updates:
        config_name += f"_updates_{os.path.basename(updates)}"
    
    # Run with time metrics
    metrics = run_command_with_time_metrics(cmd, env, config_name, output_dir)
    
    # Extract SSSP-specific timings
    if 'stdout' in metrics:
        sssp_timings = extract_sssp_timings(metrics['stdout'], 'mpi_openmp')
        metrics.update(sssp_timings)
    
    # Add configuration metadata
    metrics['mode'] = 'mpi_openmp'
    metrics['graph'] = os.path.basename(graph)
    metrics['ranks'] = ranks
    metrics['threads'] = threads
    metrics['updates'] = os.path.basename(updates) if updates else 'none'
    
    return metrics

def run_comprehensive_benchmark(graph, start_node, thread_counts, rank_counts, updates, output_dir):
    """Run comprehensive benchmark across all modes and configurations"""
    results = []
    
    # Serial (baseline)
    print("\n=== Running SERIAL baseline ===")
    serial_metrics = run_serial_benchmark(graph, start_node, updates, output_dir)
    results.append(serial_metrics)
    
    # OpenMP (with different thread counts)
    for threads in thread_counts:
        print(f"\n=== Running OPENMP with {threads} threads ===")
        openmp_metrics = run_openmp_benchmark(graph, start_node, threads, updates, output_dir)
        results.append(openmp_metrics)
    
    # MPI (with different rank counts)
    for ranks in rank_counts:
        print(f"\n=== Running MPI with {ranks} ranks ===")
        mpi_metrics = run_mpi_benchmark(graph, start_node, ranks, updates, output_dir)
        results.append(mpi_metrics)
    
    # MPI+OpenMP hybrid (combinations of ranks and threads)
    for ranks in rank_counts:
        for threads in thread_counts:
            print(f"\n=== Running MPI+OPENMP with {ranks} ranks and {threads} threads ===")
            hybrid_metrics = run_mpi_openmp_benchmark(graph, start_node, ranks, threads, updates, output_dir)
            results.append(hybrid_metrics)
    
    return results

def generate_metrics_plots(results, output_dir):
    """Generate comprehensive plots comparing all metrics"""
    if not results:
        print("No results to plot")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    df.fillna(0, inplace=True)  # Replace NaN values
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Calculate total parallelism (threads*ranks)
    df['total_parallelism'] = df['threads'] * df['ranks']
    
    # Save raw data to CSV
    csv_file = os.path.join(output_dir, f"performance_metrics_{timestamp}.csv")
    df.to_csv(csv_file, index=False)
    print(f"Raw metrics saved to {csv_file}")
    
    # Calculate speedup relative to serial
    if 'update_time' in df.columns and len(df[df['mode'] == 'serial']) > 0:
        serial_time = df[df['mode'] == 'serial']['update_time'].values[0]
        df['speedup'] = serial_time / df['update_time']
    
    # Calculate efficiency
    if 'speedup' in df.columns:
        df['efficiency'] = df['speedup'] / df['total_parallelism'] * 100
    
    # Set up plots
    modes = sorted(df['mode'].unique())
    mode_markers = {
        'serial': 'o',
        'openmp': 's',
        'mpi': '^',
        'mpi_openmp': 'D'
    }
    mode_colors = {
        'serial': '#1f77b4',   # Blue
        'openmp': '#ff7f0e',   # Orange
        'mpi': '#2ca02c',      # Green
        'mpi_openmp': '#d62728' # Red
    }
    
    # 1. Update Time by Mode and Thread Count
    plt.figure(figsize=(14, 8))
    
    # Get unique thread counts and rank combinations
    thread_counts = sorted(df['threads'].unique())
    
    # Plot each mode separately
    for mode in modes:
        mode_df = df[df['mode'] == mode]
        
        if mode == 'serial':
            # Plot serial as a horizontal line
            plt.axhline(
                y=mode_df['update_time'].values[0], 
                color=mode_colors.get(mode, 'black'), 
                linestyle='--', 
                linewidth=2, 
                label=f'Serial ({mode_df["update_time"].values[0]:.2f} ms)'
            )
        elif mode == 'openmp':
            # For OpenMP, plot against thread count
            plt.plot(
                mode_df['threads'], 
                mode_df['update_time'], 
                marker=mode_markers.get(mode, 'o'),
                color=mode_colors.get(mode, 'blue'),
                linewidth=2,
                markersize=8,
                label=f'OpenMP'
            )
        elif mode == 'mpi':
            # For MPI, show ranks as separate points
            for rank in sorted(mode_df['ranks'].unique()):
                rank_df = mode_df[mode_df['ranks'] == rank]
                plt.plot(
                    rank_df['total_parallelism'], 
                    rank_df['update_time'], 
                    marker=mode_markers.get(mode, '^'),
                    color=mode_colors.get(mode, 'green'),
                    linewidth=0,
                    markersize=10,
                    label=f'MPI ({rank} ranks)'
                )
        elif mode == 'mpi_openmp':
            # For hybrid, group by ranks
            for rank in sorted(mode_df['ranks'].unique()):
                rank_df = mode_df[mode_df['ranks'] == rank]
                plt.plot(
                    rank_df['threads'], 
                    rank_df['update_time'], 
                    marker=mode_markers.get(mode, 'D'),
                    color=mode_colors.get(mode, 'red'),
                    linewidth=2,
                    markersize=8,
                    label=f'MPI+OpenMP ({rank} ranks)'
                )
    
    plt.xlabel('Thread Count', fontweight='bold')
    plt.ylabel('Update Time (ms)', fontweight='bold')
    plt.title('Update Time by Mode and Thread Count', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Mode', fontsize=10)
    plt.xticks(thread_counts)
    plt.tight_layout()
    
    update_time_file = os.path.join(output_dir, f"update_time_by_mode_{timestamp}.png")
    plt.savefig(update_time_file, dpi=300, bbox_inches='tight')
    print(f"Update time plot saved to {update_time_file}")
    
    # 2. Speedup by Total Parallelism
    if 'speedup' in df.columns:
        plt.figure(figsize=(14, 8))
        
        # Ideal speedup line
        max_parallelism = df['total_parallelism'].max()
        plt.plot(
            [1, max_parallelism], 
            [1, max_parallelism], 
            'k--', 
            alpha=0.7, 
            label='Ideal Speedup'
        )
        
        # Plot each mode
        for mode in modes:
            if mode == 'serial':
                continue  # Skip serial in speedup plot
                
            mode_df = df[df['mode'] == mode]
            
            if mode == 'openmp':
                # For OpenMP, use thread count for x-axis
                plt.plot(
                    mode_df['threads'], 
                    mode_df['speedup'], 
                    marker=mode_markers.get(mode, 'o'),
                    color=mode_colors.get(mode, 'blue'),
                    linewidth=2,
                    markersize=8,
                    label=f'OpenMP'
                )
            elif mode == 'mpi':
                # For MPI, plot points by rank count
                plt.plot(
                    mode_df['ranks'], 
                    mode_df['speedup'], 
                    marker=mode_markers.get(mode, '^'),
                    color=mode_colors.get(mode, 'green'),
                    linewidth=2,
                    markersize=8,
                    label=f'MPI'
                )
            elif mode == 'mpi_openmp':
                # For hybrid, group by ranks
                for rank in sorted(mode_df['ranks'].unique()):
                    rank_df = mode_df[mode_df['ranks'] == rank]
                    plt.plot(
                        rank_df['total_parallelism'], 
                        rank_df['speedup'], 
                        marker=mode_markers.get(mode, 'D'),
                        color=mode_colors.get(mode, 'red'),
                        linewidth=2,
                        markersize=8,
                        label=f'MPI+OpenMP ({rank} ranks)'
                    )
        
        plt.xlabel('Degree of Parallelism (threads × ranks)', fontweight='bold')
        plt.ylabel('Speedup vs Serial', fontweight='bold')
        plt.title('Speedup by Total Degree of Parallelism', fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Mode', fontsize=10)
        plt.tight_layout()
        
        speedup_file = os.path.join(output_dir, f"speedup_by_parallelism_{timestamp}.png")
        plt.savefig(speedup_file, dpi=300, bbox_inches='tight')
        print(f"Speedup plot saved to {speedup_file}")
    
    # 3. Resource Utilization (Memory and CPU Time)
    plt.figure(figsize=(14, 10))
    
    # Create 2x2 subplots for different metrics
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot Memory Usage
    for mode in modes:
        mode_df = df[df['mode'] == mode]
        
        if mode == 'serial':
            # Show serial as horizontal line
            axs[0, 0].axhline(
                y=mode_df['max_rss'].values[0] / 1024, 
                color=mode_colors.get(mode, 'black'), 
                linestyle='--', 
                linewidth=2, 
                label=f'Serial'
            )
        elif mode == 'openmp':
            # For OpenMP, plot against thread count
            axs[0, 0].plot(
                mode_df['threads'], 
                mode_df['max_rss'] / 1024, 
                marker=mode_markers.get(mode, 'o'),
                color=mode_colors.get(mode, 'blue'),
                linewidth=2,
                markersize=8,
                label=f'OpenMP'
            )
        elif mode == 'mpi':
            # For MPI, plot against rank count
            axs[0, 0].plot(
                mode_df['ranks'], 
                mode_df['max_rss'] / 1024, 
                marker=mode_markers.get(mode, '^'),
                color=mode_colors.get(mode, 'green'),
                linewidth=2,
                markersize=8,
                label=f'MPI'
            )
        elif mode == 'mpi_openmp':
            # For hybrid, group by ranks
            for rank in sorted(mode_df['ranks'].unique()):
                rank_df = mode_df[mode_df['ranks'] == rank]
                axs[0, 0].plot(
                    rank_df['threads'], 
                    rank_df['max_rss'] / 1024, 
                    marker=mode_markers.get(mode, 'D'),
                    color=mode_colors.get(mode, 'red'),
                    linewidth=2,
                    markersize=8,
                    label=f'MPI+OpenMP ({rank} ranks)'
                )
    
    axs[0, 0].set_xlabel('Thread Count / Rank Count', fontweight='bold')
    axs[0, 0].set_ylabel('Max Memory Usage (MB)', fontweight='bold')
    axs[0, 0].set_title('Memory Usage by Mode', fontweight='bold')
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    axs[0, 0].legend(title='Mode', fontsize=9)
    
    # Plot CPU User Time
    for mode in modes:
        mode_df = df[df['mode'] == mode]
        
        if mode == 'serial':
            # Show serial as horizontal line
            axs[0, 1].axhline(
                y=mode_df['user_time'].values[0], 
                color=mode_colors.get(mode, 'black'), 
                linestyle='--', 
                linewidth=2, 
                label=f'Serial'
            )
        elif mode == 'openmp':
            # For OpenMP, plot against thread count
            axs[0, 1].plot(
                mode_df['threads'], 
                mode_df['user_time'], 
                marker=mode_markers.get(mode, 'o'),
                color=mode_colors.get(mode, 'blue'),
                linewidth=2,
                markersize=8,
                label=f'OpenMP'
            )
        elif mode == 'mpi':
            # For MPI, plot against rank count
            axs[0, 1].plot(
                mode_df['ranks'], 
                mode_df['user_time'], 
                marker=mode_markers.get(mode, '^'),
                color=mode_colors.get(mode, 'green'),
                linewidth=2,
                markersize=8,
                label=f'MPI'
            )
        elif mode == 'mpi_openmp':
            # For hybrid, group by ranks
            for rank in sorted(mode_df['ranks'].unique()):
                rank_df = mode_df[mode_df['ranks'] == rank]
                axs[0, 1].plot(
                    rank_df['threads'], 
                    rank_df['user_time'], 
                    marker=mode_markers.get(mode, 'D'),
                    color=mode_colors.get(mode, 'red'),
                    linewidth=2,
                    markersize=8,
                    label=f'MPI+OpenMP ({rank} ranks)'
                )
    
    axs[0, 1].set_xlabel('Thread Count / Rank Count', fontweight='bold')
    axs[0, 1].set_ylabel('CPU User Time (s)', fontweight='bold')
    axs[0, 1].set_title('CPU User Time by Mode', fontweight='bold')
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    axs[0, 1].legend(title='Mode', fontsize=9)
    
    # Plot CPU System Time
    for mode in modes:
        mode_df = df[df['mode'] == mode]
        
        if mode == 'serial':
            # Show serial as horizontal line
            axs[1, 0].axhline(
                y=mode_df['sys_time'].values[0], 
                color=mode_colors.get(mode, 'black'), 
                linestyle='--', 
                linewidth=2, 
                label=f'Serial'
            )
        elif mode == 'openmp':
            # For OpenMP, plot against thread count
            axs[1, 0].plot(
                mode_df['threads'], 
                mode_df['sys_time'], 
                marker=mode_markers.get(mode, 'o'),
                color=mode_colors.get(mode, 'blue'),
                linewidth=2,
                markersize=8,
                label=f'OpenMP'
            )
        elif mode == 'mpi':
            # For MPI, plot against rank count
            axs[1, 0].plot(
                mode_df['ranks'], 
                mode_df['sys_time'], 
                marker=mode_markers.get(mode, '^'),
                color=mode_colors.get(mode, 'green'),
                linewidth=2,
                markersize=8,
                label=f'MPI'
            )
        elif mode == 'mpi_openmp':
            # For hybrid, group by ranks
            for rank in sorted(mode_df['ranks'].unique()):
                rank_df = mode_df[mode_df['ranks'] == rank]
                axs[1, 0].plot(
                    rank_df['threads'], 
                    rank_df['sys_time'], 
                    marker=mode_markers.get(mode, 'D'),
                    color=mode_colors.get(mode, 'red'),
                    linewidth=2,
                    markersize=8,
                    label=f'MPI+OpenMP ({rank} ranks)'
                )
    
    axs[1, 0].set_xlabel('Thread Count / Rank Count', fontweight='bold')
    axs[1, 0].set_ylabel('CPU System Time (s)', fontweight='bold')
    axs[1, 0].set_title('CPU System Time by Mode', fontweight='bold')
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    axs[1, 0].legend(title='Mode', fontsize=9)
    
    # Plot Real Wall Time
    for mode in modes:
        mode_df = df[df['mode'] == mode]
        
        if mode == 'serial':
            # Show serial as horizontal line
            axs[1, 1].axhline(
                y=mode_df['real_time'].values[0], 
                color=mode_colors.get(mode, 'black'), 
                linestyle='--', 
                linewidth=2, 
                label=f'Serial'
            )
        elif mode == 'openmp':
            # For OpenMP, plot against thread count
            axs[1, 1].plot(
                mode_df['threads'], 
                mode_df['real_time'], 
                marker=mode_markers.get(mode, 'o'),
                color=mode_colors.get(mode, 'blue'),
                linewidth=2,
                markersize=8,
                label=f'OpenMP'
            )
        elif mode == 'mpi':
            # For MPI, plot against rank count
            axs[1, 1].plot(
                mode_df['ranks'], 
                mode_df['real_time'], 
                marker=mode_markers.get(mode, '^'),
                color=mode_colors.get(mode, 'green'),
                linewidth=2,
                markersize=8,
                label=f'MPI'
            )
        elif mode == 'mpi_openmp':
            # For hybrid, group by ranks
            for rank in sorted(mode_df['ranks'].unique()):
                rank_df = mode_df[mode_df['ranks'] == rank]
                axs[1, 1].plot(
                    rank_df['threads'], 
                    rank_df['real_time'], 
                    marker=mode_markers.get(mode, 'D'),
                    color=mode_colors.get(mode, 'red'),
                    linewidth=2,
                    markersize=8,
                    label=f'MPI+OpenMP ({rank} ranks)'
                )
    
    axs[1, 1].set_xlabel('Thread Count / Rank Count', fontweight='bold')
    axs[1, 1].set_ylabel('Wall Clock Time (s)', fontweight='bold')
    axs[1, 1].set_title('Wall Clock Time by Mode', fontweight='bold')
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    axs[1, 1].legend(title='Mode', fontsize=9)
    
    plt.tight_layout()
    
    resources_file = os.path.join(output_dir, f"resource_usage_{timestamp}.png")
    plt.savefig(resources_file, dpi=300, bbox_inches='tight')
    print(f"Resource usage plots saved to {resources_file}")
    
    # 4. Context Switch Analysis
    if 'voluntary_cs' in df.columns and 'involuntary_cs' in df.columns:
        plt.figure(figsize=(14, 8))
        
        # Set width of bars
        barWidth = 0.35
        
        # Set positions of the bars on X axis
        r1 = np.arange(len(df))
        r2 = [x + barWidth for x in r1]
        
        # Create labels for x-axis
        labels = df.apply(
            lambda row: f"{row['mode']}\n{row['ranks']}r×{row['threads']}t", 
            axis=1
        ).tolist()
        
        # Create bars
        plt.bar(r1, df['voluntary_cs'], width=barWidth, label='Voluntary Context Switches', color='skyblue')
        plt.bar(r2, df['involuntary_cs'], width=barWidth, label='Involuntary Context Switches', color='salmon')
        
        # Add labels and title
        plt.xlabel('Configuration', fontweight='bold')
        plt.ylabel('Number of Context Switches', fontweight='bold')
        plt.title('Context Switches by Configuration', fontweight='bold')
        plt.xticks([r + barWidth/2 for r in range(len(df))], labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        context_switch_file = os.path.join(output_dir, f"context_switches_{timestamp}.png")
        plt.savefig(context_switch_file, dpi=300, bbox_inches='tight')
        print(f"Context switch plot saved to {context_switch_file}")
    
    # 5. User vs System Time by Configuration
    plt.figure(figsize=(14, 8))
    
    # Stack user and system time
    df_plot = df.copy()
    df_plot['labels'] = df_plot.apply(
        lambda row: f"{row['mode']}\n{row['ranks']}r×{row['threads']}t", 
        axis=1
    )
    
    # Sort by mode then by total parallelism
    df_plot = df_plot.sort_values(by=['mode', 'total_parallelism'])
    
    # Create stacked bar chart
    plt.bar(df_plot['labels'], df_plot['user_time'], label='User Time', color='skyblue')
    plt.bar(df_plot['labels'], df_plot['sys_time'], bottom=df_plot['user_time'], label='System Time', color='salmon')
    
    # Add labels and title
    plt.xlabel('Configuration', fontweight='bold')
    plt.ylabel('Time (s)', fontweight='bold')
    plt.title('User vs System Time by Configuration', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    time_breakdown_file = os.path.join(output_dir, f"time_breakdown_{timestamp}.png")
    plt.savefig(time_breakdown_file, dpi=300, bbox_inches='tight')
    print(f"Time breakdown plot saved to {time_breakdown_file}")
    
    # 6. Efficiency Plot
    if 'efficiency' in df.columns:
        plt.figure(figsize=(14, 8))
        
        # Exclude serial from efficiency plot
        df_nonseri = df[df['mode'] != 'serial']
        
        # Plot each mode
        for mode in modes:
            if mode == 'serial':
                continue  # Skip serial
                
            mode_df = df_nonseri[df_nonseri['mode'] == mode]
            
            if mode == 'openmp':
                # For OpenMP, use thread count for x-axis
                plt.plot(
                    mode_df['threads'], 
                    mode_df['efficiency'], 
                    marker=mode_markers.get(mode, 'o'),
                    color=mode_colors.get(mode, 'blue'),
                    linewidth=2,
                    markersize=8,
                    label=f'OpenMP'
                )
            elif mode == 'mpi':
                # For MPI, plot points by rank count
                plt.plot(
                    mode_df['ranks'], 
                    mode_df['efficiency'], 
                    marker=mode_markers.get(mode, '^'),
                    color=mode_colors.get(mode, 'green'),
                    linewidth=2,
                    markersize=8,
                    label=f'MPI'
                )
            elif mode == 'mpi_openmp':
                # For hybrid, group by ranks
                for rank in sorted(mode_df['ranks'].unique()):
                    rank_df = mode_df[mode_df['ranks'] == rank]
                    plt.plot(
                        rank_df['total_parallelism'], 
                        rank_df['efficiency'], 
                        marker=mode_markers.get(mode, 'D'),
                        color=mode_colors.get(mode, 'red'),
                        linewidth=2,
                        markersize=8,
                        label=f'MPI+OpenMP ({rank} ranks)'
                    )
        
        # Add 100% efficiency line
        plt.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Ideal Efficiency (100%)')
        
        plt.xlabel('Degree of Parallelism', fontweight='bold')
        plt.ylabel('Parallel Efficiency (%)', fontweight='bold')
        plt.title('Parallel Efficiency by Configuration', fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Mode', fontsize=10)
        plt.tight_layout()
        
        efficiency_file = os.path.join(output_dir, f"parallel_efficiency_{timestamp}.png")
        plt.savefig(efficiency_file, dpi=300, bbox_inches='tight')
        print(f"Efficiency plot saved to {efficiency_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Performance Profiler for SSSP")
    parser.add_argument("--graph", required=True, help="Graph file to benchmark")
    parser.add_argument("--updates", help="Update file to use")
    parser.add_argument("--start-node", type=int, default=0, help="Start node for SSSP")
    parser.add_argument("--thread-counts", type=int, nargs='+', default=[1, 2, 4, 8, 16], 
                        help="Thread counts to test")
    parser.add_argument("--rank-counts", type=int, nargs='+', default=[1, 2, 4], 
                        help="MPI rank counts to test")
    parser.add_argument("--output-dir", default=RESULTS_DIR, 
                        help="Directory to store results")
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    ensure_directory(output_dir)
    
    print(f"Comprehensive Performance Analysis")
    print(f"Graph: {args.graph}")
    print(f"Updates: {args.updates}")
    print(f"Thread counts: {args.thread_counts}")
    print(f"Rank counts: {args.rank_counts}")
    print(f"Results will be stored in: {output_dir}")
    
    # Run benchmarks
    results = run_comprehensive_benchmark(
        args.graph, 
        args.start_node, 
        args.thread_counts, 
        args.rank_counts, 
        args.updates, 
        output_dir
    )
    
    # Generate plots
    generate_metrics_plots(results, output_dir)
    
    print(f"Analysis complete. Results stored in {output_dir}")