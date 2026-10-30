#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
build=${1:-"$root/build/bonsai-web"}
emcmake cmake -S "$root/web/bonsai-native" -B "$build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}" --target bonsai
mkdir -p "$root/web/pkg/bonsai"
cp "$build/bonsai.js" "$build/bonsai.wasm" "$root/web/pkg/bonsai/"
