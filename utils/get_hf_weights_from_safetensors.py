#!/usr/bin/env python3
"""
Extract weight information from HuggingFace format safetensors files.
Generates a JSON file containing weight names, shapes, and dtypes.

Usage:
  # Extract from a single safetensors file
  python get_hf_weights_from_safetensors.py /path/to/model.safetensors

  # Extract from a folder with model.safetensors.index.json
  python get_hf_weights_from_safetensors.py /path/to/model_folder

  # Specify output file
  python get_hf_weights_from_safetensors.py /path/to/model_folder -o output.json

  # Pretty print JSON output
  python get_hf_weights_from_safetensors.py /path/to/model_folder --pretty
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any


def _require(pkg: str, import_error: ImportError) -> None:
    raise SystemExit(f"Missing dependency {pkg!r}: {import_error}\n"
                     f"Install with: pip install {pkg}") from import_error


try:
    from safetensors import safe_open
except ImportError as e:
    _require('safetensors', e)


def get_dtype_str(dtype_str: str) -> str:
    """Normalize dtype string to standard format."""
    # safetensors returns dtype as string like "F16", "F32", "BF16", etc.
    dtype_map = {
        'F32': 'float32',
        'F16': 'float16',
        'BF16': 'bfloat16',
        'F64': 'float64',
        'I64': 'int64',
        'I32': 'int32',
        'I16': 'int16',
        'I8': 'int8',
        'U8': 'uint8',
        'BOOL': 'bool',
    }
    return dtype_map.get(dtype_str.upper(), dtype_str.lower())


def get_element_size(dtype_str: str) -> int:
    """Get element size in bytes for a given dtype."""
    size_map = {
        'float32': 4,
        'float16': 2,
        'bfloat16': 2,
        'float64': 8,
        'int64': 8,
        'int32': 4,
        'int16': 2,
        'int8': 1,
        'uint8': 1,
        'bool': 1,
    }
    return size_map.get(dtype_str.lower(), 4)  # default to 4 bytes


def find_safetensors_files(model_path: Path) -> tuple[list[Path], Path | None]:
    """
    Find safetensors files in the given path.

    Returns:
        tuple: (list of shard files, optional index file path)
    """
    if model_path.is_file():
        # Single file mode
        if model_path.suffix == '.safetensors':
            return [model_path], None
        else:
            raise ValueError(f"File must be a .safetensors file: {model_path}")

    # Directory mode - look for index file first
    index_file = model_path / 'model.safetensors.index.json'
    if index_file.is_file():
        with open(index_file, encoding='utf-8') as f:
            index_data = json.load(f)

        weight_map = index_data.get('weight_map', {})
        # Get unique shard files
        shard_files = sorted(set(weight_map.values()))
        shard_paths = [model_path / name for name in shard_files]

        # Verify all shard files exist
        for shard_path in shard_paths:
            if not shard_path.is_file():
                raise FileNotFoundError(f"Shard file not found: {shard_path}")

        return shard_paths, index_file

    # Try single model.safetensors file
    single_file = model_path / 'model.safetensors'
    if single_file.is_file():
        return [single_file], None

    raise FileNotFoundError(
        f"No model.safetensors or model.safetensors.index.json found in {model_path}"
    )


def extract_weight_info_from_shard(
        shard_path: Path) -> dict[str, dict[str, Any]]:
    """
    Extract weight information from a single safetensors shard by reading its header.

    Returns:
        dict: Mapping from weight name to weight info dict with shape and dtype.
    """
    weight_info = {}

    with open(shard_path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_bytes = f.read(int(header_len))
        header = json.loads(header_bytes.decode('utf-8'))

    for key, value in header.items():
        if key == '__metadata__':
            continue

        shape = value.get('shape', [])
        dtype_str = value.get('dtype', 'F32')

        weight_info[key] = {
            'shape': shape,
            'dtype': get_dtype_str(dtype_str),
        }

    return weight_info


def extract_weight_map_from_safetensors(model_path: Path) -> dict[str, Any]:
    """
    Extract weight map information from safetensors files.

    Args:
        model_path: Path to a .safetensors file or directory containing safetensors files.

    Returns:
        dict: Dictionary containing metadata and weight_map
    """
    shard_paths, index_file = find_safetensors_files(model_path)

    print(f"Found {len(shard_paths)} safetensors shard(s):")
    for shard_path in shard_paths:
        print(f"  - {shard_path.name}")

    weight_map = {}
    total_size = 0

    # Extract info from all shards
    for shard_path in shard_paths:
        print(f"\nProcessing {shard_path.name}...")
        shard_info = extract_weight_info_from_shard(shard_path)

        for name, info in shard_info.items():
            # Calculate size in bytes
            numel = 1
            for dim in info['shape']:
                numel *= dim
            element_size = get_element_size(info['dtype'])
            size_bytes = numel * element_size
            total_size += size_bytes

            weight_map[name] = info

    return {
        'metadata': {
            'total_size': total_size,
        },
        'weight_map': weight_map,
    }


def main():
    parser = argparse.ArgumentParser(
        description=
        'Extract weight information from HuggingFace format safetensors files')
    parser.add_argument(
        'model_path',
        type=str,
        help=
        'Path to a .safetensors file or directory containing safetensors files',
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='weight_map.json',
        help='Output JSON file path (default: weight_map.json)',
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty print JSON output',
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)

    if not model_path.exists():
        print(f"Error: Path does not exist: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Extract weight map
    print(f"Extracting weight information from: {model_path}\n")

    try:
        result = extract_weight_map_from_safetensors(model_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    total_weights = len(result['weight_map'])
    total_size_gb = result['metadata']['total_size'] / (1024**3)
    print('\nSummary:')
    print(f"  Total weights: {total_weights}")
    print(
        f"  Total size: {total_size_gb:.2f} GB ({result['metadata']['total_size']:,} bytes)"
    )

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        if args.pretty:
            json.dump(result, f, indent=2, ensure_ascii=False)
        else:
            json.dump(result, f, ensure_ascii=False)

    print(f"\nWeight map saved to: {output_path}")

    # Print some example weights
    print('\nExample weights:')
    for i, (name, info) in enumerate(list(result['weight_map'].items())[:10]):
        print(f"  {name}: shape={info['shape']}, dtype={info['dtype']}")
    if len(result['weight_map']) > 10:
        print(f"  ... and {len(result['weight_map']) - 10} more weights")


if __name__ == '__main__':
    main()
