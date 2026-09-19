#!/usr/bin/env python3
"""Run a command with a deadline, terminating its process group on failure."""
import importlib.util
from pathlib import Path
import sys
spec = importlib.util.spec_from_file_location('gpu_runner', Path(__file__).with_name('test-gpu.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.run(sys.argv[2:], timeout=int(sys.argv[1]))
