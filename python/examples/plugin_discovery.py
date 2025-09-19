#!/usr/bin/env python3
"""
Example demonstrating plugin discovery and output inspection in VampRust Python API.
"""

from vamprust import AudioProcessor


def main() -> None:
    # Create audio processor
    processor = AudioProcessor()

    print("=== Discovering all plugins ===")
    plugins = processor.discover_plugins()
    print(f"Found {len(plugins)} plugins total:")

    for plugin in plugins[:10]:  # Display only first 10 plugins for brevity
        print(f"  - {plugin.identifier}: {plugin.name}")

    print("\n=== Chordino Plugin Information ===")
    try:
        # Get detailed info about chordino
        chordino_info = processor.get_plugin_info("chordino")
        print("Plugin Info:")
        for key, value in chordino_info.items():
            print(f"  {key}: {value}")

        print("\nPlugin Outputs:")
        chordino_outputs = processor.get_plugin_outputs("chordino")
        for output in chordino_outputs:
            print(f"  [{output['index']}] {output['identifier']}: {output['name']}")
            print(f"    Unit: {output.get('unit', 'N/A')}")
            print(f"    Sample Type: {output.get('sample_type', 'N/A')}")
            print(f"    Has Fixed Bin Count: {output.get('has_fixed_bin_count', False)}")
            if output.get('has_fixed_bin_count'):
                print(f"    Bin Count: {output.get('bin_count', 0)}")

        print("\nPlugin Parameters:")
        chordino_parameters = processor.get_plugin_parameters("chordino")
        for param in chordino_parameters:
            print(f"  [{param['index']}] {param['identifier']}: {param['name']}")
            print(f"    Range: {param['min_value']} - {param['max_value']} (default: {param['default_value']})")
            if param.get('unit'):
                print(f"    Unit: {param['unit']}")
            if param.get('is_quantized'):
                print(f"    Quantized (step: {param.get('quantize_step', 'N/A')})")

    except ValueError as e:
        print(f"Chordino not found: {e}")

    print("\n=== All Plugin Outputs ===")
    all_outputs = processor.list_plugin_outputs()

    for plugin_id, outputs in all_outputs.items():
        print(f"\n{plugin_id}:")
        for output in outputs:
            print(f"  [{output['index']}] {output['identifier']}: {output['name']}")


if __name__ == "__main__":
    main()
