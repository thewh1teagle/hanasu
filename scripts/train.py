"""
Train the model.

Usage:
    uv run scripts/train.py -c src/config.json -m my_model
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.train import main

if __name__ == "__main__":
    main()
