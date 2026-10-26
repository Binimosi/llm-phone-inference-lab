#!/usr/bin/env bash
# Build llama.cpp (CPU backend) inside Termux -> proot-distro Ubuntu, pinned to the commit used here.
# -j2: a -j6 build crashed gcc (out of memory) on a phone with about 3 GB free RAM.
set -euo pipefail
apt-get update && apt-get install -y build-essential cmake git curl
[ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 36e36e63a54fc8f85b95e9c499a022b597e478dd
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2 --target llama-server llama-bench llama-cli
ls build/bin
