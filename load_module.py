"""
Utility to import numbered lesson files (e.g., '01_tokenizer.py').

Python normally can't import modules whose names start with a digit.
This helper works around that limitation.

Usage in any lesson file:
    from load_module import load
    block1 = load("01_tokenizer.py")
    Tokenizer = block1.Tokenizer
"""

import importlib.util
import os

_DIR = os.path.dirname(os.path.abspath(__file__))


def load(filename):
    """
    Import a lesson file by its filename.

    Args:
        filename (str): e.g., "01_tokenizer.py"

    Returns:
        The imported module object.
    """
    path = os.path.join(_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Lesson file not found: {path}")
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for lesson file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
