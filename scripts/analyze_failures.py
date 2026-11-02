"""Failure analysis of an evaluation run (see src/brain_seg/analysis.py).

    python scripts/analyze_failures.py --eval-dir outputs/runs/<run>/eval_test
"""

from brain_seg.analysis import main

if __name__ == "__main__":
    main()
