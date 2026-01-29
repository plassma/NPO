from pathlib import Path
import sys

# Make project root importable when tests are run directly (e.g. `python tests/foo.py`)
ROOT = Path(__file__).resolve().parent.parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
