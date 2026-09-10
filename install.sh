#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_BIN="$SCRIPT_DIR/venv/bin"
BIN_DIR="$HOME/.local/bin"

mkdir -p "$BIN_DIR"

for cmd in image-gen image-gen-anim i2i-gen i2i-gen-anim remove-object; do
  cat > "$BIN_DIR/$cmd" <<EOF
#!/bin/sh
exec "$VENV_BIN/$cmd" "\$@"
EOF
  chmod +x "$BIN_DIR/$cmd"
  echo "Installed $BIN_DIR/$cmd"
done

# Check if ~/.local/bin is in PATH
case ":$PATH:" in
  *":$BIN_DIR:"*)
    echo ""
    echo "Done! ~/.local/bin is already in your PATH." ;;
  *)
    echo ""
    echo "Add ~/.local/bin to your PATH by adding this to ~/.zshrc (or ~/.bashrc):"
    echo '  export PATH="$HOME/.local/bin:$PATH"'
    echo "Then restart your shell or run: source ~/.zshrc" ;;
esac
