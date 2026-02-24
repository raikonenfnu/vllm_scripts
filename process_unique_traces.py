#!/usr/bin/env python3
"""
Script to process JSON trace files and extract unique configurations with counts.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List


def load_json_file(filepath: Path) -> Dict[str, Any]:
    """Load a single JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def json_to_hashable(obj: Any) -> str:
    """Convert JSON object to a hashable string representation."""
    if isinstance(obj, dict):
        return json.dumps(obj, sort_keys=True)
    return str(obj)


def process_directory(input_dir: Path) -> Dict[str, int]:
    """
    Process all JSON files in the input directory and count unique configurations.

    Returns:
        Dictionary mapping configuration to count
    """
    config_counts = defaultdict(int)

    # Get all JSON files in the directory
    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return {}

    print(f"Processing {len(json_files)} JSON files...")

    for json_file in json_files:
        try:
            data = load_json_file(json_file)
            # Convert the entire config to a hashable string
            config_key = json_to_hashable(data)
            config_counts[config_key] += 1
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    return config_counts


def write_unique_configs(config_counts: Dict[str, int], output_dir: Path):
    """
    Write unique configurations to separate JSON files in the output directory.

    Each file contains the configuration and its count.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFound {len(config_counts)} unique configurations")

    # Write a summary file with all unique configs and their counts
    summary = []
    for idx, (config_str, count) in enumerate(sorted(config_counts.items(), key=lambda x: -x[1]), 1):
        config = json.loads(config_str)
        summary.append({
            "id": idx,
            "count": count,
            "configuration": config
        })

    summary_file = output_dir / "summary_all_configs.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Written summary to: {summary_file}")

    # Also write individual files for each unique configuration
    for idx, (config_str, count) in enumerate(sorted(config_counts.items(), key=lambda x: -x[1]), 1):
        config = json.loads(config_str)

        # Create a descriptive filename based on config parameters
        m = config.get('m', 'unknown')
        n = config.get('n', 'unknown')
        k = config.get('k', 'unknown')
        filename = f"unique_config_{idx:03d}_m{m}_n{n}_k{k}_count{count}.json"

        output_data = {
            "count": count,
            "configuration": config
        }

        output_file = output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

    print(f"Written {len(config_counts)} unique configuration files to: {output_dir}")

    # Print statistics
    print("\n--- Statistics ---")
    print(f"Total unique configurations: {len(config_counts)}")
    print(f"Total trace files: {sum(config_counts.values())}")
    print("\nTop 10 most common configurations:")
    for idx, (config_str, count) in enumerate(sorted(config_counts.items(), key=lambda x: -x[1])[:10], 1):
        config = json.loads(config_str)
        print(f"  {idx}. Count: {count:4d} - m={config.get('m')}, n={config.get('n')}, k={config.get('k')}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python process_unique_traces.py <input_directory>")
        print("Example: python process_unique_traces.py moe_json_traces")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"Error: Directory '{input_path}' does not exist")
        sys.exit(1)

    if not input_path.is_dir():
        print(f"Error: '{input_path}' is not a directory")
        sys.exit(1)

    # Create output directory name
    input_dir_name = input_path.name
    output_dir = input_path.parent / f"unique_{input_dir_name}"

    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Process the directory
    config_counts = process_directory(input_path)

    if config_counts:
        # Write results
        write_unique_configs(config_counts, output_dir)
    else:
        print("No configurations to process.")


if __name__ == "__main__":
    main()
