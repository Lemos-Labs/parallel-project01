#!/usr/bin/env python3
"""
Visualize K-means benchmark results.

This script reads benchmark result CSV files from the results directory
and generates various comparison charts in the charts directory.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_plot_style():
    """Configure matplotlib for better-looking plots."""
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def load_results(filename):
    """
    Load benchmark results from CSV file.

    Args:
        filename: Path to the CSV file

    Returns:
        pandas DataFrame with results
    """
    try:
        df = pd.read_csv(filename)
        print(f"✓ Loaded {filename}: {len(df)} results")
        return df
    except Exception as e:
        print(f"✗ Error loading {filename}: {e}")
        return None


def extract_implementation_name(impl_str):
    """Extract clean implementation name (remove thread info)."""
    return impl_str.split('(')[0].strip()


def extract_thread_count(impl_str):
    """Extract thread count from implementation string."""
    if '(' in impl_str:
        try:
            return int(impl_str.split('(')[1].split()[0])
        except:
            pass
    return 1


def plot_execution_time_comparison(df, output_path):
    """
    Create bar chart comparing execution times across implementations.

    Args:
        df: DataFrame with benchmark results
        output_path: Path to save the chart
    """
    # Group by test and implementation
    pivot_df = df.pivot_table(
        values='Avg Time (s)',
        index='Test Description',
        columns='Implementation',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    # Create grouped bar chart
    pivot_df.plot(kind='bar', ax=ax, width=0.8)

    ax.set_title('Execution Time Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Test Dataset', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.legend(title='Implementation', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_speedup_chart(df, output_path, baseline='Sequential'):
    """
    Create speedup chart comparing implementations against baseline.

    Args:
        df: DataFrame with benchmark results
        output_path: Path to save the chart
        baseline: Name of baseline implementation
    """
    # Extract clean implementation names
    df['Clean Implementation'] = df['Implementation'].apply(extract_implementation_name)

    # Filter for baseline
    baseline_df = df[df['Clean Implementation'] == baseline][['Test Description', 'Avg Time (s)']].copy()
    baseline_df.columns = ['Test Description', 'Baseline Time']

    if baseline_df.empty:
        print(f"  ⚠ Warning: No baseline ({baseline}) found for speedup calculation")
        return

    # Merge with all results
    merged_df = df.merge(baseline_df, on='Test Description')

    # Calculate speedup
    merged_df['Speedup'] = merged_df['Baseline Time'] / merged_df['Avg Time (s)']

    # Remove baseline from comparison (speedup = 1.0)
    plot_df = merged_df[merged_df['Clean Implementation'] != baseline]

    if plot_df.empty:
        print(f"  ⚠ Warning: No non-baseline implementations to compare")
        return

    # Pivot for plotting
    pivot_df = plot_df.pivot_table(
        values='Speedup',
        index='Test Description',
        columns='Implementation',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    # Create grouped bar chart
    pivot_df.plot(kind='bar', ax=ax, width=0.8)

    ax.set_title(f'Speedup vs {baseline}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Test Dataset', fontsize=12)
    ax.set_ylabel('Speedup (x times faster)', fontsize=12)
    ax.legend(title='Implementation', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline')

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2fx', padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_scaling_by_dataset_size(df, output_path):
    """
    Create line plot showing performance scaling with dataset size.

    Args:
        df: DataFrame with benchmark results
        output_path: Path to save the chart
    """
    # Define dataset size order (small -> medium -> large -> xlarge)
    size_order = {
        'Small': 0,
        'Medium': 1,
        'Large': 2,
        'XLarge': 3,
        'Xlarge': 3
    }

    # Extract size from test description
    def get_size(desc):
        for size in size_order.keys():
            if size.lower() in desc.lower():
                return size
        return 'Unknown'

    df['Size'] = df['Test Description'].apply(get_size)
    df['Size Order'] = df['Size'].map(size_order)

    # Filter out unknown sizes
    plot_df = df[df['Size Order'].notna()].copy()

    if plot_df.empty:
        print(f"  ⚠ Warning: No data with recognizable dataset sizes")
        return

    # Sort by size
    plot_df = plot_df.sort_values('Size Order')

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot line for each implementation
    for impl in plot_df['Implementation'].unique():
        impl_data = plot_df[plot_df['Implementation'] == impl]

        # Group by size and take mean
        grouped = impl_data.groupby('Size')['Avg Time (s)'].mean()

        ax.plot(grouped.index, grouped.values, marker='o', linewidth=2,
                markersize=8, label=impl)

    ax.set_title('Performance Scaling by Dataset Size', fontsize=16, fontweight='bold')
    ax.set_xlabel('Dataset Size', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.legend(title='Implementation', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_dimensionality_impact(df, output_path):
    """
    Create plot showing impact of dimensionality on performance.

    Args:
        df: DataFrame with benchmark results
        output_path: Path to save the chart
    """
    # Extract dimensions from test description (e.g., "2D", "5D", "10D")
    def get_dimensions(desc):
        import re
        match = re.search(r'(\d+)[dD]', desc)
        if match:
            return int(match.group(1))
        return None

    df['Dimensions'] = df['Test Description'].apply(get_dimensions)

    # Filter out entries without dimension info
    plot_df = df[df['Dimensions'].notna()].copy()

    if plot_df.empty:
        print(f"  ⚠ Warning: No data with dimension information")
        return

    # Sort by dimensions
    plot_df = plot_df.sort_values('Dimensions')

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot line for each implementation
    for impl in plot_df['Implementation'].unique():
        impl_data = plot_df[plot_df['Implementation'] == impl]

        # Group by dimensions and take mean
        grouped = impl_data.groupby('Dimensions')['Avg Time (s)'].mean()

        ax.plot(grouped.index, grouped.values, marker='o', linewidth=2,
                markersize=8, label=impl)

    ax.set_title('Performance vs Dimensionality', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Dimensions', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.legend(title='Implementation', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    # Set x-axis to show only integer dimensions
    ax.set_xticks(sorted(plot_df['Dimensions'].unique()))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_efficiency_heatmap(df, output_path):
    """
    Create heatmap showing efficiency across tests and implementations.

    Args:
        df: DataFrame with benchmark results
        output_path: Path to save the chart
    """
    # Pivot data
    pivot_df = df.pivot_table(
        values='Avg Time (s)',
        index='Test Description',
        columns='Implementation',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    im = ax.imshow(pivot_df.values, cmap='YlOrRd', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticklabels(pivot_df.index)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Execution Time (seconds)', rotation=270, labelpad=20)

    # Add values to cells
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            value = pivot_df.values[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.4f}',
                             ha='center', va='center', color='black', fontsize=8)

    ax.set_title('Execution Time Heatmap', fontsize=16, fontweight='bold')
    ax.set_xlabel('Implementation', fontsize=12)
    ax.set_ylabel('Test Dataset', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_summary_report(df, output_path):
    """
    Create a text summary report.

    Args:
        df: DataFrame with benchmark results
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("K-MEANS BENCHMARK SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total tests run: {len(df)}\n")
        f.write(f"Implementations tested: {df['Implementation'].nunique()}\n")
        f.write(f"Datasets tested: {df['Test Description'].nunique()}\n")
        f.write(f"Success rate: {df['Successful Runs'].sum() / (len(df) * df['Successful Runs'].max()) * 100:.1f}%\n\n")

        # Performance by implementation
        f.write("AVERAGE EXECUTION TIME BY IMPLEMENTATION\n")
        f.write("-" * 70 + "\n")
        impl_stats = df.groupby('Implementation')['Avg Time (s)'].agg(['mean', 'min', 'max', 'std'])
        f.write(impl_stats.to_string())
        f.write("\n\n")

        # Best/worst performance
        f.write("BEST PERFORMANCE (fastest for each test)\n")
        f.write("-" * 70 + "\n")
        idx_min = df.groupby('Test Description')['Avg Time (s)'].idxmin()
        best_df = df.loc[idx_min, ['Test Description', 'Implementation', 'Avg Time (s)']]
        f.write(best_df.to_string(index=False))
        f.write("\n\n")

        # Speedup analysis (if Sequential exists)
        df['Clean Implementation'] = df['Implementation'].apply(extract_implementation_name)
        if 'Sequential' in df['Clean Implementation'].values:
            f.write("SPEEDUP ANALYSIS (vs Sequential)\n")
            f.write("-" * 70 + "\n")

            baseline_df = df[df['Clean Implementation'] == 'Sequential'][['Test Description', 'Avg Time (s)']].copy()
            baseline_df.columns = ['Test Description', 'Baseline Time']

            merged_df = df.merge(baseline_df, on='Test Description')
            merged_df['Speedup'] = merged_df['Baseline Time'] / merged_df['Avg Time (s)']

            speedup_summary = merged_df[merged_df['Clean Implementation'] != 'Sequential'].groupby('Implementation')['Speedup'].agg(['mean', 'min', 'max'])
            f.write(speedup_summary.to_string())
            f.write("\n\n")

        f.write("=" * 70 + "\n")
        f.write("End of Report\n")
        f.write("=" * 70 + "\n")

    print(f"  ✓ Saved: {output_path}")


def visualize_results(results_dir='results', charts_dir='charts', result_file=None):
    """
    Main function to generate all visualizations.

    Args:
        results_dir: Directory containing result CSV files
        charts_dir: Directory to save generated charts
        result_file: Specific result file to visualize (optional)
    """
    # Setup
    setup_plot_style()

    # Create charts directory
    os.makedirs(charts_dir, exist_ok=True)

    # Find result files
    if result_file:
        result_files = [result_file]
    else:
        result_files = sorted(glob.glob(os.path.join(results_dir, 'benchmark_*.txt')))

    if not result_files:
        print(f"✗ No result files found in {results_dir}")
        return

    print(f"\nFound {len(result_files)} result file(s)")
    print("=" * 70)

    # Process each result file
    for result_file in result_files:
        print(f"\nProcessing: {result_file}")
        print("-" * 70)

        # Load data
        df = load_results(result_file)
        if df is None or df.empty:
            continue

        # Create subdirectory for this result
        base_name = Path(result_file).stem
        result_charts_dir = os.path.join(charts_dir, base_name)
        os.makedirs(result_charts_dir, exist_ok=True)

        print(f"Generating charts in: {result_charts_dir}")

        # Generate all charts
        try:
            plot_execution_time_comparison(
                df,
                os.path.join(result_charts_dir, '01_execution_time_comparison.png')
            )
        except Exception as e:
            print(f"  ✗ Error creating execution time chart: {e}")

        try:
            plot_speedup_chart(
                df,
                os.path.join(result_charts_dir, '02_speedup_comparison.png')
            )
        except Exception as e:
            print(f"  ✗ Error creating speedup chart: {e}")

        try:
            plot_scaling_by_dataset_size(
                df,
                os.path.join(result_charts_dir, '03_scaling_by_size.png')
            )
        except Exception as e:
            print(f"  ✗ Error creating scaling chart: {e}")

        try:
            plot_dimensionality_impact(
                df,
                os.path.join(result_charts_dir, '04_dimensionality_impact.png')
            )
        except Exception as e:
            print(f"  ✗ Error creating dimensionality chart: {e}")

        try:
            plot_efficiency_heatmap(
                df,
                os.path.join(result_charts_dir, '05_efficiency_heatmap.png')
            )
        except Exception as e:
            print(f"  ✗ Error creating heatmap: {e}")

        try:
            create_summary_report(
                df,
                os.path.join(result_charts_dir, 'summary_report.txt')
            )
        except Exception as e:
            print(f"  ✗ Error creating summary report: {e}")

        print(f"\n✓ Completed visualizations for {base_name}")

    print("\n" + "=" * 70)
    print(f"✓ All visualizations complete! Check the '{charts_dir}' directory.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize K-means benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all results in the results directory
  python visualize_results.py

  # Visualize a specific result file
  python visualize_results.py -f results/benchmark_20251201_153552.txt

  # Specify custom directories
  python visualize_results.py -r my_results -c my_charts
        """
    )

    parser.add_argument('-r', '--results-dir', default='results',
                        help='Directory containing result CSV files (default: results)')
    parser.add_argument('-c', '--charts-dir', default='charts',
                        help='Directory to save generated charts (default: charts)')
    parser.add_argument('-f', '--file', default=None,
                        help='Specific result file to visualize')

    args = parser.parse_args()

    visualize_results(args.results_dir, args.charts_dir, args.file)


if __name__ == '__main__':
    main()
