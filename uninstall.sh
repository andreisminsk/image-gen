#!/usr/bin/env bash
set -euo pipefail

BIN_DIR="$HOME/.local/bin"

for cmd in image-gen image-gen-anim i2i-gen i2i-gen-anim remove-object; do
  if [ -f "$BIN_DIR/$cmd" ]; then
    rm "$BIN_DIR/$cmd"
    echo "Removed $BIN_DIR/$cmd"
  else
    echo "Not found: $BIN_DIR/$cmd"
  fi
done

echo "Done."
