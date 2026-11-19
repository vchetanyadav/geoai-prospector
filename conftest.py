"""Let `pytest` run from a bare checkout/unzip, before `pip install -e .`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
