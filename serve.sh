#!/usr/bin/env bash
# Serve the viewer at http://localhost:8765 (no-store, so rebuilt parts and edited JS always reload)
cd "$(dirname "$0")/web" && exec python3 -c '
import sys, http.server
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
http.server.test(HandlerClass=H, port=int(sys.argv[1]))
' "${1:-8765}"
