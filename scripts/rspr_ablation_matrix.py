#!/usr/bin/env python3
"""Print the canonical RSPR ablation arguments without launching training."""

import argparse
import shlex

ABLATIONS = {
    "A0": ["--rspr_mode", "off"],
    "A1": ["--rspr_mode", "mean", "--rspr_sample_count", "1", "--rspr_top_r", "0"],
    "A2": ["--rspr_mode", "stochastic", "--rspr_detach_samples"],
    "A3": ["--rspr_mode", "stochastic"],
    "A4": ["--rspr_mode", "legacy"],
    "A5": ["--rspr_mode", "stochastic", "--rspr_match_mode", "hard"],
    "A6": ["--rspr_mode", "stochastic", "--rspr_match_mode", "soft"],
    "A7": ["--rspr_mode", "stochastic", "--rspr_match_mode", "soft", "--rspr_rank_weight", "0"],
    "A8": ["--rspr_mode", "stochastic", "--rspr_match_mode", "soft", "--rspr_anchor_weight", "0"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print canonical RSPR ablation arguments; never launches training."
    )
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), required=True)
    parser.add_argument(
        "--print-shell-args",
        action="store_true",
        help="Print shell-safe arguments for the selected ablation.",
    )
    args = parser.parse_args()
    if not args.print_shell_args:
        parser.error("--print-shell-args is required; this utility never launches training")
    return args


def main() -> None:
    args = parse_args()
    print(shlex.join(ABLATIONS[args.ablation]))


if __name__ == "__main__":
    main()
