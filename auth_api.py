"""Stable ASGI entry point for the authentication backend.

Run with: uvicorn auth_api:app --host 0.0.0.0 --port $AUTH_PORT
"""

import importlib.util
import sys
from pathlib import Path
# auth_api.py
from authentication_jwt import app


_source = Path(__file__).with_name("authentication_jwt.py")
_spec = importlib.util.spec_from_file_location("gaash_auth_impl", _source)
if _spec is None or _spec.loader is None:  # pragma: no cover - deployment guard
    raise RuntimeError("Could not load the authentication backend module.")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

app = _module.app
