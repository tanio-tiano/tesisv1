import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cec2021_official import FUNCTION_BIASES, OfficialCEC2021Problem as CEC2021Problem
