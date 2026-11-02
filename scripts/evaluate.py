"""Evaluate a checkpoint on the held-out split (see src/brain_seg/evaluate.py).

Run from the repository root after `pip install -e .`; use --help for options.
"""

from brain_seg.evaluate import main

if __name__ == "__main__":
    main()
