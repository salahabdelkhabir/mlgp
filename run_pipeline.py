"""
run_pipeline.py
================
Runs all 5 phases in order.  Each phase can be run individually or via this
master script.

Usage
-----
    # Full pipeline
    python run_pipeline.py

    # Single phase
    python run_pipeline.py --phase 1      # EDA only
    python run_pipeline.py --phase 1 2 3  # EDA + features + preprocessing
"""

import sys, argparse, importlib, traceback
from config import logger

PHASES = {
    1: ("01_eda",                    "EDA & data profiling"),
    2: ("02_feature_engineering",    "Feature engineering"),
    3: ("03_preprocessing",          "Preprocessing & encoding"),
    4: ("04_model_selection",        "Model selection & CV"),
    5: ("05_tune_and_evaluate",      "Hyperparameter tuning & final evaluation"),
}


def run_phase(phase_num: int):
    module_name, description = PHASES[phase_num]
    logger.info(f"\n{'█'*60}")
    logger.info(f"  PHASE {phase_num}: {description}")
    logger.info(f"{'█'*60}")
    try:
        mod = importlib.import_module(module_name)
        mod.main()
        logger.info(f"Phase {phase_num} complete ✓")
    except Exception:
        logger.error(f"Phase {phase_num} FAILED:")
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Home Credit Pipeline")
    parser.add_argument(
        "--phase", nargs="+", type=int,
        choices=list(PHASES.keys()),
        help="Phase(s) to run (default: all)",
    )
    args = parser.parse_args()
    phases_to_run = args.phase if args.phase else list(PHASES.keys())

    logger.info("Home Credit Default Risk — ML Pipeline")
    logger.info(f"Phases to run: {phases_to_run}")

    for p in phases_to_run:
        run_phase(p)

    logger.info("\n" + "=" * 60)
    logger.info("ALL PHASES COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
