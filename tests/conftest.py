"""Pytest configuration file."""

import sys
from pathlib import Path

# Add the repository root to Python path so 'src' module can be imported
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
