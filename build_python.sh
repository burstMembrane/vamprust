#!/bin/bash
set -e

echo "Building VampRust Python bindings..."

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

# Initialize submodules if not already done
if [ ! -f "vamp-plugin-sdk/vamp/vamp.h" ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
fi

# Build in development mode
echo "Building Python extension..."
maturin develop --features python

echo "Build completed successfully!"
echo ""
echo "You can now test the bindings:"
echo "  python python/examples/basic_usage.py"
echo ""
echo "Or run the tests:"
echo "  cd python && python -m pytest tests/"