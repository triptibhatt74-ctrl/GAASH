"""Stable ASGI entry point for the chat backend.

Run with: uvicorn bot_api:app --host 0.0.0.0 --port $PORT
"""

import importlib.util
import sys
from pathlib import Path
# bot_api.py
from bot import app


_source = Path(__file__).with_name("bot.py")
_spec = importlib.util.spec_from_file_location("gaash_bot_impl", _source)
if _spec is None or _spec.loader is None:  # pragma: no cover - deployment guard
    raise RuntimeError("Could not load the chat backend module.")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

app = _module.app
