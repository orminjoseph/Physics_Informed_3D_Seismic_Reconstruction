"""
=========================================================
Test Path Setup
=========================================================

Ensures the project root is available for imports.

Author: Ormin Joseph
=========================================================
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)