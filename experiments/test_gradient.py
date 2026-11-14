import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from src.optimization import check_gradient


error = check_gradient()

if error < 1e-5:
    print("Gradient verification PASSED")
else:
    print("Gradient verification FAILED")