"""pytest configuration."""
import sys
from pathlib import Path

# Ensure the package root is on sys.path when running tests from source
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
