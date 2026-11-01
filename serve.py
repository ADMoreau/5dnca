"""Serve the NCA viewer locally.

    .venv/bin/python serve.py            # then open http://localhost:8000/viewer/

Serves the project folder on 127.0.0.1 only, plus /api/runs, which lists every
runs/**/growth_rgba.npy so the viewer can offer them in a menu.
"""

from __future__ import annotations

import argparse
import functools
import json
import http.server
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def list_runs() -> list[dict]:
    runs = []
    for f in sorted((ROOT / "runs").glob("**/growth_rgba.npy")):
        shape = np.load(f, mmap_mode="r").shape  # (T, *grid, 4)
        runs.append({"name": str(f.parent.relative_to(ROOT / "runs")),
                     "path": "/" + str(f.relative_to(ROOT)),
                     "grid": list(shape[1:-1]), "frames": shape[0]})
    return runs


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] == "/api/runs":
            body = json.dumps(list_runs()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/viewer/")
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # always show the latest run
        super().end_headers()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    handler = functools.partial(Handler, directory=str(ROOT))
    with http.server.ThreadingHTTPServer((args.host, args.port), handler) as srv:
        print(f"NCA viewer: http://localhost:{args.port}/viewer/", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
