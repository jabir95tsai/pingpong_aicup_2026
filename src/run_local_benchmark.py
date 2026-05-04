"""Run an existing training script against the generated local benchmark dataset.

Example:
    python src/run_local_benchmark.py --script src/train_v8_champion.py --submission-glob "submissions/submission_v8_champion*.csv"
"""
import argparse
import glob
import os
import subprocess
import sys

from config import PROJECT_ROOT


def newest_file(paths):
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, help="Training script to execute, e.g. src/train_v8_champion.py")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(PROJECT_ROOT, "artifacts", "local_benchmark"),
        help="Benchmark dataset directory produced by build_local_benchmark_dataset.py",
    )
    parser.add_argument(
        "--submission-glob",
        required=True,
        help="Glob for the expected submission output, e.g. submissions/submission_v8_champion*.csv",
    )
    parser.add_argument(
        "--script-args",
        default="",
        help="Extra args passed through to the training script.",
    )
    args = parser.parse_args()

    truth_path = os.path.join(args.data_dir, "truth.csv")
    if not os.path.exists(truth_path):
        raise FileNotFoundError(f"Missing truth file: {truth_path}")

    env = os.environ.copy()
    env["PINGPONG_DATA_DIR"] = args.data_dir

    script_path = args.script
    if not os.path.isabs(script_path):
        script_path = os.path.join(PROJECT_ROOT, script_path)

    cmd = [sys.executable, script_path]
    if args.script_args.strip():
        cmd.extend(args.script_args.strip().split())

    before = set(glob.glob(os.path.join(PROJECT_ROOT, args.submission_glob)))
    print("Running:", " ".join(cmd))
    print("PINGPONG_DATA_DIR =", env["PINGPONG_DATA_DIR"])
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)

    after = set(glob.glob(os.path.join(PROJECT_ROOT, args.submission_glob)))
    new_files = sorted(after - before)
    candidate = newest_file(new_files) or newest_file(sorted(after))
    if candidate is None:
        raise FileNotFoundError(f"No submission found for glob: {args.submission_glob}")

    eval_cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "src", "evaluate_submission.py"),
        "--truth-path",
        truth_path,
        candidate,
    ]
    print("Evaluating:", candidate)
    subprocess.run(eval_cmd, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
