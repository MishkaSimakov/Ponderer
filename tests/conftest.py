import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SCRIPTS = os.path.join(ROOT, "simulator", "Assets", "Scripts")


def read(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return f.read()
