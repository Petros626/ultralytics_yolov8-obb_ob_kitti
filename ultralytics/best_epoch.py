import pandas as pd
import argparse
import os
from pathlib import Path


# Find the epoch in which the best results were achieved
# source: https://github.com/ultralytics/ultralytics/issues/14137


def main(results_dir):
    # Construct the path to the results.csv file
    results_csv_path = Path(results_dir) / 'results.csv'

    # Check if the file exists
    if not results_csv_path.exists():
        print(f"Error: The file {results_csv_path} does not exist.")
        return

    # Load the training log
    results = pd.read_csv(results_csv_path)

    # Strip spaces from column names to avoid issues with leading/trailing spaces
    results.columns = results.columns.str.strip()

    # Calculate fitness (optional, if you want to use a custom fitness metric)
    results["fitness"] = (
        results["metrics/mAP50(B)"] * 0.1
        + results["metrics/mAP70(B)"] * 0.1 
        + results["metrics/mAP50-95(B)"] * 0.9
    )

    # Calculate fitness (optional, if you want to use a custom fitness metric)
    # Fitness is a weighted combination of mAP50, mAP70, and mAP50-95
    best_epoch_50 = results['metrics/mAP50(B)'].idxmax() + 1
    best_epoch_70 = results['metrics/mAP70(B)'].idxmax() + 1
    # Optionally, find the epoch with the highest fitness
    # best_epoch = results['fitness'].idxmax() + 1

    # Print the results
    print(f"Best model for mAP50 was saved at epoch: {best_epoch_50}")
    print(f"Best model for mAP70 was saved at epoch: {best_epoch_70}")
    # print(f"Best model was saved at epoch: {best_epoch}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Find the best epoch from YOLO training results. '
                    'This script analyzes the results.csv file generated during YOLO training '
                    'and identifies the epoch with the highest mAP50 and mAP70 metrics.'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Path to the directory containing the results.csv file. '
             'Example: --results_dir /path/to/training_run',
    )

    args = parser.parse_args()

    # Call the main function with the provided directory
    main(args.results_dir)