#!/usr/bin/env bash
# scripts/run_gateway.sh
# --------------------------------------------------------------------
# Build + run the hft_gateway with config files.
# Usage:
#   ./scripts/run_gateway.sh [build_dir]
# --------------------------------------------------------------------

set -euo pipefail

# Default build dir is ./build under project root
BUILD_DIR="${1:-build}"

# Move to project root (one level up from scripts/)
cd "$(dirname "$0")/.."

# Ensure build dir exists
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure + build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc || sysctl -n hw.ncpu || echo 4)

# Run the gateway with configs
echo "=== Starting hft_gateway ==="
./hft_gateway --config ./configs/gateway.yaml