#!/usr/bin/env python3
"""Static server for the switch-wasm frontend.

python's `http.server` sends no Cache-Control headers, so browsers (notably
Firefox) heuristically cache the wasm/NRO and keep running stale builds. Send
`Cache-Control: no-store` instead so every reload picks up freshly built
assets.
"""
import http.server
import socketserver

PORT = 8000
DIRECTORY = "web"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"serving {DIRECTORY} at http://localhost:{PORT} (no-cache)")
        httpd.serve_forever()
