#!/usr/bin/env bash
# Download Gemma 4 E2B-it Q4_0 GGUF (ggml-org) and verify the SHA-256 used in all results here.
set -euo pipefail
DIR=${MODEL_DIR:-$HOME/models}; F=gemma-4-E2B-it-Q4_0.gguf
SHA=8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52
mkdir -p "$DIR"
curl -L -C - -o "$DIR/$F" "https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/$F"
echo "$SHA  $DIR/$F" | sha256sum -c -
