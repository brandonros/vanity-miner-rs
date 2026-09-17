#!/usr/bin/env python3
"""Compatibility entry for the Shallenge Metal builder."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name("build-metal.py")), run_name="__main__")
