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


ATTN_FIELDS = [
    "seq_lens", "num_heads", "head_size", "sliding_window",
    "dtype", "block_size", "soft_cap", "num_blocks", "q_dtype",
    "k_dtype", "v_dtype", "q_descale_dtype",
    "k_descale_dtype", "v_descale_dtype",
    "q_descale_shape", "k_descale_shape", "v_descale_shape",
    "key_cache_shape", "value_cache_shape",
    "key_cache_strides", "value_cache_strides",
]


def detect_trace_type(config: Dict[str, Any]) -> str:
    """Detect trace type based on config keys."""
    if 'm' in config and 'n' in config and 'k' in config:
        return 'moe'
    elif 'num_heads' in config and 'head_size' in config:
        return 'attn'
    return 'unknown'


def extract_attn_key(config: Dict[str, Any]) -> str:
    """Extract only the attention-relevant fields for uniqueness comparison."""
    key_data = {field: config.get(field) for field in ATTN_FIELDS}
    return json.dumps(key_data, sort_keys=True)


def make_filename(config: Dict[str, Any], trace_type: str, idx: int, count: int) -> str:
    """Generate a descriptive filename based on trace type."""
    if trace_type == 'moe':
        m = config.get('m', 'unknown')
        n = config.get('n', 'unknown')
        k = config.get('k', 'unknown')
        return f"unique_config_{idx:03d}_m{m}_n{n}_k{k}_count{count}.json"
    elif trace_type == 'attn':
        num_heads = config.get('num_heads', [])
        nqh = num_heads[0] if isinstance(num_heads, list) and len(num_heads) > 0 else '?'
        nkvh = num_heads[1] if isinstance(num_heads, list) and len(num_heads) > 1 else '?'
        hs = config.get('head_size', '?')
        bs = config.get('block_size', '?')
        sw = config.get('sliding_window') or 'none'
        kdt = config.get('k_dtype', '')
        kdt_short = kdt.replace('torch.', '').replace('float', 'fp') if kdt else 'none'
        return f"unique_config_{idx:03d}_nqh{nqh}_nkvh{nkvh}_hs{hs}_bs{bs}_sw{sw}_kdt{kdt_short}_count{count}.json"
    else:
        return f"unique_config_{idx:03d}_count{count}.json"


def format_config_summary(config: Dict[str, Any], trace_type: str) -> str:
    """Format a one-line summary of a config for display."""
    if trace_type == 'moe':
        return f"m={config.get('m')}, n={config.get('n')}, k={config.get('k')}"
    elif trace_type == 'attn':
        seq = config.get('seq_lens', [])
        nh = config.get('num_heads', '?')
        hs = config.get('head_size', '?')
        bs = config.get('block_size', '?')
        sw = config.get('sliding_window')
        dt = config.get('dtype', '?')
        sc = config.get('soft_cap')
        nb = config.get('num_blocks', '?')
        qdt = config.get('q_dtype')
        kdt = config.get('k_dtype')
        vdt = config.get('v_dtype')
        kdsc_dt = config.get('k_descale_dtype')
        vdsc_dt = config.get('v_descale_dtype')
        kdsc_sh = config.get('k_descale_shape')
        vdsc_sh = config.get('v_descale_shape')
        kc_sh = config.get('key_cache_shape')
        vc_sh = config.get('value_cache_shape')
        return (f"num_heads={nh}, head_size={hs}, block_size={bs}, "
                f"sliding_window={sw}, dtype={dt}, soft_cap={sc}, "
                f"num_blocks={nb}, q_dtype={qdt}, k_dtype={kdt}, v_dtype={vdt}, "
                f"k_descale_dtype={kdsc_dt}, v_descale_dtype={vdsc_dt}, "
                f"k_descale_shape={kdsc_sh}, v_descale_shape={vdsc_sh}, "
                f"key_cache_shape={kc_sh}, value_cache_shape={vc_sh}, "
                f"seq_lens={seq}")
    else:
        return json.dumps(config, sort_keys=True)[:120]


def process_directory(input_dir: Path) -> Dict[str, int]:
    """
    Process all JSON files in the input directory and count unique configurations.

    Returns:
        Dictionary mapping configuration to count
    """
    config_counts = defaultdict(int)

    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return {}

    print(f"Processing {len(json_files)} JSON files...")

    trace_type = None
    for json_file in json_files:
        try:
            data = load_json_file(json_file)
            if trace_type is None:
                trace_type = detect_trace_type(data)
                print(f"Detected trace type: {trace_type}")

            if trace_type == 'attn':
                config_key = extract_attn_key(data)
            else:
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

    sorted_configs = sorted(config_counts.items(), key=lambda x: -x[1])

    first_config = json.loads(sorted_configs[0][0])
    trace_type = detect_trace_type(first_config)
    print(f"Detected trace type: {trace_type}")

    summary = []
    for idx, (config_str, count) in enumerate(sorted_configs, 1):
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

    for idx, (config_str, count) in enumerate(sorted_configs, 1):
        config = json.loads(config_str)
        filename = make_filename(config, trace_type, idx, count)

        output_data = {
            "count": count,
            "configuration": config
        }

        output_file = output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

    print(f"Written {len(config_counts)} unique configuration files to: {output_dir}")

    print("\n--- Statistics ---")
    print(f"Total unique configurations: {len(config_counts)}")
    print(f"Total trace files: {sum(config_counts.values())}")
    print("\nTop 10 most common configurations:")
    for idx, (config_str, count) in enumerate(sorted_configs[:10], 1):
        config = json.loads(config_str)
        print(f"  {idx}. Count: {count:4d} - {format_config_summary(config, trace_type)}")


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
