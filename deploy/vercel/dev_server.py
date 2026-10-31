"""Run the Vercel demo locally: the serverless API plus the static page on one port.

python deploy/vercel/dev_server.py        # http://localhost:3000
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import uvicorn
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("neurolens_web", HERE / "api" / "index.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["neurolens_web"] = module
    spec.loader.exec_module(module)
    module.app.mount("/", StaticFiles(directory=HERE / "public", html=True), name="static")
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    uvicorn.run(module.app, host="127.0.0.1", port=port)
