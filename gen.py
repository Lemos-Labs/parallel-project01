#!/usr/bin/env python3
"""
Generate test CSV files for K-means benchmark.

This script generates synthetic datasets with configurable parameters:
- N: number of data points
- D: number of dimensions
- K: number of clusters (used to generate realistic clustered data)

The generated data will have K natural clusters, making it suitable for
testing the K-means algorithm.
"""

import argparse
import os

import numpy as np


def generate_clustered_data(n_points, n_dims, n_clusters, cluster_std=1.0, seed=None):
    """
    Generate synthetic data with natural clusters.

    Args:
        n_points: Total number of data points
        n_dims: Number of dimensions
        n_clusters: Number of clusters to generate
        cluster_std: Standard deviation of points around cluster centers
        seed: Random seed for reproducibility

    Returns:
        numpy array of shape (n_points, n_dims)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random cluster centers in the range [0, 100]
    cluster_centers = np.random.uniform(0, 100, size=(n_clusters, n_dims))

    # Assign points to clusters (roughly equally)
    points_per_cluster = n_points // n_clusters
    remainder = n_points % n_clusters

    data = []
    for i in range(n_clusters):
        # Add one extra point to some clusters to handle remainder
        n_cluster_points = points_per_cluster + (1 if i < remainder else 0)

        # Generate points around this cluster center
        cluster_data = np.random.normal(
            loc=cluster_centers[i],
            scale=cluster_std,
            size=(n_cluster_points, n_dims)
        )
        data.append(cluster_data)

    # Combine all clusters and shuffle
    data = np.vstack(data)
    np.random.shuffle(data)

    return data


def generate_random_data(n_points, n_dims, min_val=0, max_val=100, seed=None):
    """
    Generate uniformly random data (no natural clusters).

    Args:
        n_points: Total number of data points
        n_dims: Number of dimensions
        min_val: Minimum value for data
        max_val: Maximum value for data
        seed: Random seed for reproducibility

    Returns:
        numpy array of shape (n_points, n_dims)
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.uniform(min_val, max_val, size=(n_points, n_dims))


def save_csv(data, filename):
    """
    Save data to CSV file.

    Args:
        data: numpy array of shape (n_points, n_dims)
        filename: output filename
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

    # Save with comma separator, no header, no row indices
    np.savetxt(filename, data, delimiter=',', fmt='%.6f')
    print("Generated {}: {} points, {} dimensions".format(filename, data.shape[0], data.shape[1]))


def generate_predefined_datasets(output_dir):
    """
    Generate a set of predefined test datasets of various sizes.

    Args:
        output_dir: Directory to save the datasets
    """
    datasets = [
        # Small datasets for quick testing
        {"name": "small_2d.csv", "n": 1000, "d": 2, "k": 5, "std": 3.0, "seed": 42},
        {"name": "small_5d.csv", "n": 1000, "d": 5, "k": 5, "std": 5.0, "seed": 42},

        # Medium datasets
        {"name": "medium_2d.csv", "n": 50000, "d": 2, "k": 10, "std": 3.0, "seed": 42},
        {"name": "medium_5d.csv", "n": 50000, "d": 5, "k": 10, "std": 5.0, "seed": 42},
        {"name": "medium_10d.csv", "n": 50000, "d": 10, "k": 10, "std": 5.0, "seed": 42},

        # Large datasets for benchmarking
        {"name": "large_2d.csv", "n": 500000, "d": 2, "k": 20, "std": 3.0, "seed": 42},
        {"name": "large_5d.csv", "n": 500000, "d": 5, "k": 20, "std": 5.0, "seed": 42},
        {"name": "large_10d.csv", "n": 500000, "d": 10, "k": 15, "std": 5.0, "seed": 42},

        # Very large dataset for stress testing
        {"name": "xlarge_5d.csv", "n": 2000000, "d": 5, "k": 25, "std": 5.0, "seed": 42},
        {"name": "xlarge_10d.csv", "n": 1000000, "d": 10, "k": 20, "std": 5.0, "seed": 42},
    ]

    print("Generating predefined datasets in '{}'...".format(output_dir))
    print("=" * 70)

    for params in datasets:
        filename = os.path.join(output_dir, params["name"])
        data = generate_clustered_data(
            params["n"],
            params["d"],
            params["k"],
            params["std"],
            params["seed"]
        )
        save_csv(data, filename)

    print("=" * 70)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test CSV files for K-means benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate predefined test datasets
  python generate_test_data.py --preset

  # Generate custom clustered data
  python generate_test_data.py -n 10000 -d 5 -k 8 -o dataset/custom.csv

  # Generate random (non-clustered) data
  python generate_test_data.py -n 5000 -d 3 --random -o dataset/random.csv

  # Generate with specific random seed for reproducibility
  python generate_test_data.py -n 10000 -d 5 -k 10 --seed 123 -o dataset/test.csv
        """
    )

    parser.add_argument('--preset', action='store_true',
                        help='Generate predefined test datasets')
    parser.add_argument('-n', '--points', type=int,
                        help='Number of data points')
    parser.add_argument('-d', '--dims', type=int,
                        help='Number of dimensions')
    parser.add_argument('-k', '--clusters', type=int,
                        help='Number of clusters (for clustered data)')
    parser.add_argument('-o', '--output', type=str,
                        help='Output CSV filename')
    parser.add_argument('--output-dir', type=str, default='dataset',
                        help='Output directory for preset datasets (default: dataset)')
    parser.add_argument('--random', action='store_true',
                        help='Generate random data instead of clustered data')
    parser.add_argument('--std', type=float, default=5.0,
                        help='Standard deviation for clustered data (default: 5.0)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.preset:
        generate_predefined_datasets(args.output_dir)
    elif args.points and args.dims and args.output:
        if args.random:
            data = generate_random_data(args.points, args.dims, seed=args.seed)
        else:
            if not args.clusters:
                parser.error("--clusters is required for clustered data (or use --random)")
            data = generate_clustered_data(
                args.points,
                args.dims,
                args.clusters,
                args.std,
                args.seed
            )
        save_csv(data, args.output)
    else:
        parser.error("Either use --preset or provide -n, -d, and -o arguments")


if __name__ == '__main__':
    main()
