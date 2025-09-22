#!/usr/bin/env python3
"""
Simple CLI for VampRust audio processing.
Processes audio files through Vamp plugins and outputs results as JSON or CSV.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

import vamprust


def load_audio(filepath: str) -> tuple[np.ndarray, float]:
    """Load audio file and return samples and sample rate."""
    try:
        data, sample_rate = sf.read(filepath)
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # ensure c contiguous float32 array
        if not data.flags["C_CONTIGUOUS"] or data.dtype != np.float32:
            data = np.ascontiguousarray(data, dtype=np.float32)
        return data.astype(np.float32), float(sample_rate)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file '{filepath}': {e}")


def find_plugin(
    host: vamprust.VampHost, plugin_id: str
) -> tuple[vamprust.VampLibrary, int]:
    """Find plugin by identifier."""
    libraries = host.find_plugin_libraries()

    for lib_path in libraries:
        library = host.load_library(lib_path)
        if library:
            plugins = library.list_plugins()
            for plugin_info in plugins:
                if plugin_info.identifier == plugin_id:
                    return library, plugin_info.index

    raise RuntimeError(f"Plugin '{plugin_id}' not found")


def process_audio(
    audio_data: np.ndarray, sample_rate: float, plugin_id: str
) -> List[Dict[str, Any]]:
    """Process audio through specified Vamp plugin."""
    host = vamprust.VampHost()
    library, plugin_index = find_plugin(host, plugin_id)

    plugin = library.instantiate_plugin(plugin_index, sample_rate)
    if not plugin:
        raise RuntimeError(f"Failed to instantiate plugin '{plugin_id}'")

    # Convert numpy array to list format expected by the plugin
    audio_list = audio_data.tolist()
    channels = 1  # mono

    # Process the full audio
    results = plugin.process_audio_full(audio_list, sample_rate, channels)

    # Convert results to list of dictionaries
    features = []
    if results:
        for item in results:
            feature = {
                "has_timestamp": item.get("has_timestamp", False),
                "values": item.get("values", []),
                "label": item.get("label", ""),
            }

            if feature["has_timestamp"]:
                feature["sec"] = item.get("sec", 0)
                feature["nsec"] = item.get("nsec", 0)
                # Convert to human-readable timestamp
                feature["timestamp"] = feature["sec"] + feature["nsec"] / 1e9

            features.append(feature)

    return features


def output_json(features: List[Dict[str, Any]], output_file: Optional[str] = None):
    """Output features as JSON."""
    output_data = {"features": features, "feature_count": len(features)}

    if output_file:
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
    else:
        print(json.dumps(output_data, indent=2))


def output_csv(features: List[Dict[str, Any]], output_file: Optional[str] = None):
    """Output features as CSV."""
    if not features:
        if output_file:
            Path(output_file).write_text("# No features extracted\n")
        else:
            print("# No features extracted")
        return

    # Determine all possible fields
    fieldnames = set()
    for feature in features:
        fieldnames.update(feature.keys())
        # Handle values array as separate columns
        if "values" in feature and feature["values"]:
            for i in range(len(feature["values"])):
                fieldnames.add(f"value_{i}")

    fieldnames = sorted(fieldnames)

    # Prepare rows
    rows = []
    for feature in features:
        row = {}
        for field in fieldnames:
            if field.startswith("value_"):
                idx = int(field.split("_")[1])
                values = feature.get("values", [])
                row[field] = values[idx] if idx < len(values) else ""
            elif field == "values":
                # Skip the original values field since we expanded it
                continue
            else:
                row[field] = feature.get(field, "")
        rows.append(row)

    # Remove 'values' from fieldnames if it exists
    if "values" in fieldnames:
        fieldnames.remove("values")

    if output_file:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        import io

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        print(output.getvalue())


def list_plugins():
    """List available Vamp plugins."""
    host = vamprust.VampHost()
    libraries = host.find_plugin_libraries()

    print("Available Vamp plugins:")
    total_plugins = 0

    for lib_path in libraries:
        library = host.load_library(lib_path)
        if library:
            plugins = library.list_plugins()
            if plugins:
                print(f"\n{lib_path}:")
                for plugin in plugins:
                    print(f"  {plugin.identifier} - {plugin.name}")
                    total_plugins += 1

    print(f"\nTotal: {total_plugins} plugins found")


def main():
    parser = argparse.ArgumentParser(
        description="Process audio files through Vamp plugins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process audio with chordino plugin, output as JSON
  %(prog)s audio.wav chordino:chordino

  # Process and save as CSV
  %(prog)s audio.wav chordino:chordino -f csv -o results.csv

  # List available plugins
  %(prog)s --list-plugins
        """,
    )

    parser.add_argument("audio_file", nargs="?", help="Input audio file")
    parser.add_argument(
        "plugin", nargs="?", help="Vamp plugin identifier (e.g., chordino:chordino)"
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument(
        "--list-plugins", action="store_true", help="List available Vamp plugins"
    )

    args = parser.parse_args()

    try:
        if args.list_plugins:
            list_plugins()
            return

        if not args.audio_file or not args.plugin:
            parser.error(
                "Both audio_file and plugin are required unless using --list-plugins"
            )

        # Load and process audio
        audio_data, sample_rate = load_audio(args.audio_file)
        features = process_audio(audio_data, sample_rate, args.plugin)

        # Output results
        if args.format == "json":
            output_json(features, args.output)
        else:
            output_csv(features, args.output)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
