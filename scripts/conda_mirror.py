"""Run the installed Conda with a mirror-compatible HTTP user agent.

This changes only the current process, never the shared Conda installation.
"""
from conda.base.context import Context
from conda.cli import main

Context.user_agent = property(lambda self: "Mozilla/5.0 ATOD-Conda-Setup")

if __name__ == "__main__":
    raise SystemExit(main())
