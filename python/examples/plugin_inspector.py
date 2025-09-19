#!/usr/bin/env python3
"""
Plugin Inspector - Comprehensive plugin information viewer for VampRust.

This tool displays detailed information about a specific Vamp plugin including
basic info, parameters, outputs, and more.
"""

import argparse
import sys
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from vamprust import PluginInfo

from vamprust import AudioProcessor


def format_parameter(param: Dict[str, Any], index: int) -> str:
    """Format a parameter descriptor for display."""
    lines = []
    lines.append(f"  Parameter {index}: {param['identifier']}")
    lines.append(f"    Name: {param['name']}")

    if param.get("description"):
        lines.append(f"    Description: {param['description']}")

    lines.append(f"    Range: {param['min_value']} - {param['max_value']}")
    lines.append(f"    Default: {param['default_value']}")

    if param.get("unit"):
        lines.append(f"    Unit: {param['unit']}")

    if param.get("is_quantized"):
        lines.append(f"    Quantized: Yes (step: {param.get('quantize_step', 'N/A')})")
        value_names = param.get("value_names", [])
        if value_names:
            lines.append(f"    Value Names: {', '.join(value_names)}")
    else:
        lines.append("    Quantized: No")

    return "\n".join(lines)


def format_output(output: Dict[str, Any], index: int) -> str:
    """Format an output descriptor for display."""
    lines = []
    lines.append(f"  Output {index}: {output['identifier']}")
    lines.append(f"    Name: {output['name']}")

    if output.get("description"):
        lines.append(f"    Description: {output['description']}")

    if output.get("unit"):
        lines.append(f"    Unit: {output['unit']}")

    lines.append(f"    Sample Type: {output.get('sample_type', 'Unknown')}")

    if output.get("sample_type") == "FixedSampleRate":
        lines.append(f"    Sample Rate: {output.get('sample_rate', 'N/A')} Hz")
    elif output.get("sample_type") == "VariableSampleRate":
        lines.append(f"    Sample Rate Resolution: {output.get('sample_rate', 'N/A')}")

    lines.append(f"    Has Fixed Bin Count: {output.get('has_fixed_bin_count', False)}")
    if output.get("has_fixed_bin_count"):
        lines.append(f"    Bin Count: {output.get('bin_count', 0)}")
        bin_names = output.get("bin_names", [])
        if bin_names:
            lines.append(f"    Bin Names: {', '.join(bin_names)}")

    lines.append(f"    Has Known Extents: {output.get('has_known_extents', False)}")
    if output.get("has_known_extents"):
        lines.append(
            f"    Value Range: {output.get('min_value', 'N/A')} - {output.get('max_value', 'N/A')}"
        )

    lines.append(f"    Is Quantized: {output.get('is_quantized', False)}")
    if output.get("is_quantized"):
        lines.append(f"    Quantize Step: {output.get('quantize_step', 'N/A')}")

    lines.append(f"    Has Duration: {output.get('has_duration', False)}")

    return "\n".join(lines)


def inspect_plugin(plugin_id: str, sample_rate: float = 44100.0) -> None:
    """Inspect and display comprehensive information about a plugin."""
    try:
        processor = AudioProcessor()

        # First, discover plugins to verify the plugin exists
        print(f" Searching for plugin '{plugin_id}'...")
        plugins = processor.discover_plugins()

        plugin_found = None
        for plugin in plugins:
            if plugin.identifier == plugin_id:
                plugin_found = plugin
                break

        if not plugin_found:
            plugins = processor.fuzzy_search_plugins(plugin_id)
            print(" No exact match found. Did you mean:")
            for p in plugins:
                print(f"  • {p.identifier}: {p.name}")
            sys.exit(1)

        print(f" Found plugin: {plugin_found.name}")
        print(f" Library: {plugin_found.library_path}")
        print()

        # Get comprehensive plugin information
        plugin_info = processor.get_plugin_info(plugin_id, sample_rate)
        parameters = processor.get_plugin_parameters(plugin_id, sample_rate)
        outputs = processor.get_plugin_outputs(plugin_id, sample_rate)

        # Display basic plugin information
        print("=" * 60)
        print(" PLUGIN INFORMATION")
        print("=" * 60)
        print(f"Identifier: {plugin_info.get('identifier', 'N/A')}")
        print(f"Name: {plugin_info.get('name', 'N/A')}")
        print(f"Description: {plugin_info.get('description', 'N/A')}")
        print(f"Maker: {plugin_info.get('maker', 'N/A')}")
        print(f"Version: {plugin_info.get('version', 'N/A')}")
        print(f"Copyright: {plugin_info.get('copyright', 'N/A')}")
        print(f"API Version: {plugin_info.get('api_version', 'N/A')}")
        print()

        # Display technical specifications
        print("=" * 60)
        print("️  TECHNICAL SPECIFICATIONS")
        print("=" * 60)
        print(f"Input Domain: {plugin_info.get('input_domain', 'N/A')}")
        print(f"Sample Rate: {plugin_info.get('sample_rate', 'N/A')} Hz")
        print(
            f"Preferred Block Size: {plugin_info.get('preferred_block_size', 'N/A')} samples"
        )
        print(
            f"Preferred Step Size: {plugin_info.get('preferred_step_size', 'N/A')} samples"
        )
        print(f"Parameter Count: {plugin_info.get('parameter_count', 0)}")
        print(f"Program Count: {plugin_info.get('program_count', 0)}")
        print()

        # Display parameters
        print("=" * 60)
        print("️  PARAMETERS")
        print("=" * 60)
        if parameters:
            for i, param in enumerate(parameters):
                print(format_parameter(param, i))
                if i < len(parameters) - 1:
                    print()
        else:
            print("  No parameters available")
        print()

        # Display outputs
        print("=" * 60)
        print(" OUTPUTS")
        print("=" * 60)
        if outputs:
            for i, output in enumerate(outputs):
                print(format_output(output, i))
                if i < len(outputs) - 1:
                    print()
        else:
            print("  No outputs available")
        print()

        # Summary
        print("=" * 60)
        print(" SUMMARY")
        print("=" * 60)
        print(f"Plugin '{plugin_id}' has:")
        print(f"  • {len(parameters)} parameter(s)")
        print(f"  • {len(outputs)} output(s)")
        print(f"  • Input domain: {plugin_info.get('input_domain', 'Unknown')}")
        print(f"  • API version: {plugin_info.get('api_version', 'Unknown')}")

    except ValueError as e:
        print(f" Error loading plugin '{plugin_id}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f" Unexpected error: {e}")
        sys.exit(1)


def list_available_plugins() -> None:
    """List all available plugins."""
    try:
        processor = AudioProcessor()
        plugins = processor.discover_plugins()

        print(f" Available Plugins ({len(plugins)} total):")
        print("=" * 60)

        # Group by library for better organization
        by_library: Dict[str, List[PluginInfo]] = {}
        for plugin in plugins:
            lib_name = plugin.library_path.split("/")[-1]
            if lib_name not in by_library:
                by_library[lib_name] = []
            by_library[lib_name].append(plugin)

        for lib_name in sorted(by_library.keys()):
            print(f"\n {lib_name}:")
            for plugin in sorted(by_library[lib_name], key=lambda p: p.identifier):
                print(f"  • {plugin.identifier}: {plugin.name}")

        print("\n Use --plugin <identifier> to inspect a specific plugin")

    except Exception as e:
        print(f" Error discovering plugins: {e}")
        sys.exit(1)


def main() -> None:
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Inspect Vamp plugins and display comprehensive information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --plugin chordino                    # Inspect the Chordino plugin
  %(prog)s --plugin pyin --sample-rate 48000   # Inspect pYin at 48kHz
  %(prog)s --list                               # List all available plugins
  %(prog)s --search <query>                     # Search for plugins
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--plugin",
        "-p",
        type=str,
        help='Plugin identifier to inspect (e.g., "chordino", "pyin")',
    )
    group.add_argument(
        "--list", "-l", action="store_true", help="List all available plugins"
    )
    group.add_argument(
        "--search", "-q", type=str, help="Fuzzy search for plugins by name or ID"
    )

    parser.add_argument(
        "--sample-rate",
        "-s",
        type=float,
        default=44100.0,
        help="Sample rate for plugin initialization (default: 44100.0)",
    )

    args = parser.parse_args()
    if args.search:
        processor = AudioProcessor()
        plugins = processor.fuzzy_search_plugins(args.search)
        if plugins:
            print(f" Found {len(plugins)} matching plugin(s):")
            for plugin in plugins:
                print(f"  • {plugin.identifier}: {plugin.name}")
            return
        else:
            print(" No matching plugins found.")

    if args.list:
        list_available_plugins()
    else:
        inspect_plugin(args.plugin, args.sample_rate)


if __name__ == "__main__":
    main()
