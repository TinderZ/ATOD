"""Select minimal Ray agent modules for this PPU benchmark only.

Ray 2.50's --minimal flag disables its HTTP server, but module discovery
independently checks installed dependencies. Patch that check in this private
agent process so ReporterAgent cannot call unsupported HGGC NVML APIs.
"""
from pathlib import Path
import runpy

import ray.dashboard
import ray.dashboard.utils

ray.dashboard.utils.get_dashboard_dependency_error = lambda: ImportError(
    'PPU benchmark uses minimal agent modules and its own native device monitor'
)
runpy.run_path(str(Path(ray.dashboard.__file__).parent / 'agent.py'), run_name='__main__')
