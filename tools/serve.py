#!/usr/bin/env python3
"""Static file server for web/ with caching turned off.

Firefox and Chromium will both heuristically cache `.wasm` and `.nro` files
served by a plain `http.server`, so a rebuilt core silently keeps running the
previous build. Every response here is marked no-store.

Usage: tools/serve.py [port]
"""

import functools
import http.server
import os
import socketserver
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "web")
DEFAULT_PORT = 8000


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".nro": "application/octet-stream",
        ".js": "text/javascript",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        # One line per request, without the noisy default timestamp block.
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


class ReusableServer(socketserver.TCPServer):
    """Rebinding avoids "address already in use" across quick restarts."""

    allow_reuse_address = True


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    handler = functools.partial(NoCacheHandler, directory=ROOT)
    with ReusableServer(("127.0.0.1", port), handler) as httpd:
        print(f"serving {os.path.realpath(ROOT)} at http://127.0.0.1:{port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
