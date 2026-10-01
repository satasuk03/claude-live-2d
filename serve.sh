#!/usr/bin/env bash
# Serve the viewer at http://localhost:8765
cd "$(dirname "$0")/web" && exec python3 -m http.server "${1:-8765}"
