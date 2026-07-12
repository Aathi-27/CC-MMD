"""
CC-MMD Pipeline: Master Runner

Executes all stages in order. Run individual stages with:
    python run_pipeline.py --stage 2

Or run all:
    python run_pipeline.py --all
"""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_stage(stage_num, stage_name, module_func):
    """Execute a pipeline stage with timing."""
    print(f"\n{'#' * 70}")
    print(f"# STAGE {stage_num}: {stage_name}")
    print(f"{'#' * 70}\n")

    start = time.time()
    try:
        module_func()
        elapsed = time.time() - start
        print(f"\n✓ Stage {stage_num} completed in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n✗ Stage {stage_num} FAILED after {elapsed:.1f}s: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="CC-MMD Pipeline Runner")
    parser.add_argument("--stage", type=int, help="Run specific stage (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all stages")
    parser.add_argument("--from-stage", type=int, default=1, help="Start from this stage")
    args = parser.parse_args()

    stages = {
        1: ("Data Merge & Normalize", lambda: __import__("src.data_merge", fromlist=["main"]).main()),
        2: ("Embedding Extraction", lambda: __import__("src.embedding_extractor", fromlist=["main"]).main()),
        4: ("Train Multitask MLP", lambda: __import__("src.trainer", fromlist=["main"]).main()),
        5: ("Build Cultural Prototypes", lambda: __import__("src.prototypes", fromlist=["main"]).main()),
        "5b": ("Fine-tune Cultural Gates", lambda: __import__("src.gate_finetune", fromlist=["main"]).main()),
        6: ("Threshold Calibration", lambda: __import__("src.calibration", fromlist=["main"]).main()),
        7: ("Inference & Submission", lambda: __import__("src.inference", fromlist=["predict_on_train"]).predict_on_train()),
    }

    if args.stage:
        key = args.stage
        if key in stages:
            name, func = stages[key]
            run_stage(key, name, func)
        else:
            print(f"Unknown stage: {key}. Available: {list(stages.keys())}")
    elif args.all:
        total_start = time.time()
        for key in [1, 2, 4, 5, "5b", 6, 7]:
            if isinstance(key, int) and key < args.from_stage:
                continue
            name, func = stages[key]
            run_stage(key, name, func)

        total_elapsed = time.time() - total_start
        print(f"\n{'=' * 70}")
        print(f"ALL STAGES COMPLETED in {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
        print(f"{'=' * 70}")
    else:
        print("Usage:")
        print("  python run_pipeline.py --stage 2     # Run stage 2 only")
        print("  python run_pipeline.py --all          # Run all stages")
        print("  python run_pipeline.py --all --from-stage 4  # Resume from stage 4")
        print("\nAvailable stages:")
        for key, (name, _) in stages.items():
            print(f"  Stage {key}: {name}")


if __name__ == "__main__":
    main()
