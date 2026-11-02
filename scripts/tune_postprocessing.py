"""Tune post-processing thresholds on the validation set (see src/brain_seg/tuning.py).

    python scripts/tune_postprocessing.py --checkpoint outputs/runs/<run>/checkpoints/best.pt --data-root "$DATA" --limit 60
"""

from brain_seg.tuning import main

if __name__ == "__main__":
    main()
