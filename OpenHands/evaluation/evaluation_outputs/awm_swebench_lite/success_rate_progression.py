#!/usr/bin/env python3
"""
Success Rate Progression Analysis

Analyzes how the resolve rate evolves as more workflows are accumulated
during AWM online learning.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple


def load_results_chronologically(results_path: Path) -> List[Dict]:
    """Load results and sort by timestamp."""
    results = []

    with open(results_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                results.append(entry)
            except json.JSONDecodeError:
                continue

    # Sort by timestamp
    def parse_timestamp(entry):
        ts = entry.get("timestamp", "")
        if ts:
            try:
                return datetime.fromisoformat(ts)
            except:
                pass
        return datetime.min

    results.sort(key=parse_timestamp)
    return results


def compute_progression(results: List[Dict]) -> List[Tuple[int, int, int, float]]:
    """
    Compute cumulative success rate progression.

    Returns list of tuples: (instance_num, cumulative_success, workflows_count, success_rate)
    """
    progression = []
    cumulative_success = 0

    for i, entry in enumerate(results):
        instance_num = i + 1
        is_pass = entry.get("test_result") == "PASS"

        if is_pass:
            cumulative_success += 1

        # Each success triggers workflow induction, so workflows = cumulative successes
        workflows_count = cumulative_success
        success_rate = cumulative_success / instance_num

        progression.append((instance_num, cumulative_success, workflows_count, success_rate))

    return progression


def compute_windowed_rate(results: List[Dict], window_size: int = 10) -> List[Tuple[int, float]]:
    """
    Compute moving average success rate over a sliding window.

    Returns list of tuples: (instance_num, windowed_rate)
    """
    windowed = []
    window = []

    for i, entry in enumerate(results):
        is_pass = 1 if entry.get("test_result") == "PASS" else 0
        window.append(is_pass)

        if len(window) > window_size:
            window.pop(0)

        rate = sum(window) / len(window)
        windowed.append((i + 1, rate))

    return windowed


def print_progression_table(progression: List[Tuple], step: int = 10):
    """Print progression at regular intervals."""
    print("\n=== Success Rate Progression (sampled every {} instances) ===\n".format(step))
    print("| Instance | Cumulative Success | Workflows | Success Rate |")
    print("|----------|-------------------|-----------|--------------|")

    for i, (inst, cum_success, workflows, rate) in enumerate(progression):
        if i == 0 or (inst % step == 0) or inst == len(progression):
            print(f"| {inst:8d} | {cum_success:17d} | {workflows:9d} | {rate*100:10.1f}% |")

    # Always print the last one
    inst, cum_success, workflows, rate = progression[-1]
    if inst % step != 0:
        print(f"| {inst:8d} | {cum_success:17d} | {workflows:9d} | {rate*100:10.1f}% |")


def print_latex_table(progression: List[Tuple], step: int = 20):
    """Print LaTeX formatted table for paper."""
    print("\n=== LaTeX Table (sampled every {} instances) ===\n".format(step))
    print("$$")
    print(r"\begin{array}{|c|c|c|c|}")
    print(r"\hline")
    print(r"\textbf{Instance} & \textbf{Cumulative Success} & \textbf{Workflows} & \textbf{Success Rate} \\")
    print(r"\hline")

    for i, (inst, cum_success, workflows, rate) in enumerate(progression):
        if inst == 1 or (inst % step == 0) or inst == len(progression):
            print(f"{inst} & {cum_success} & {workflows} & {rate*100:.1f}\\% \\\\")

    # Always print the last one if not already printed
    inst, cum_success, workflows, rate = progression[-1]
    if inst % step != 0 and inst != 1:
        print(f"{inst} & {cum_success} & {workflows} & {rate*100:.1f}\\% \\\\")

    print(r"\hline")
    print(r"\end{array}")
    print("$$")


def analyze_trend(progression: List[Tuple]) -> Dict:
    """Analyze the trend in success rate."""
    n = len(progression)

    # Split into quarters
    q1_end = n // 4
    q2_end = n // 2
    q3_end = 3 * n // 4

    def avg_rate(start, end):
        rates = [p[3] for p in progression[start:end]]
        return sum(rates) / len(rates) if rates else 0

    q1_rate = avg_rate(0, q1_end)
    q2_rate = avg_rate(q1_end, q2_end)
    q3_rate = avg_rate(q2_end, q3_end)
    q4_rate = avg_rate(q3_end, n)

    # First 10 vs last 10
    first_10_passes = sum(1 for p in progression[:10] if progression.index(p) < 10 and results[progression.index(p)].get("test_result") == "PASS")

    return {
        "q1_rate": q1_rate,
        "q2_rate": q2_rate,
        "q3_rate": q3_rate,
        "q4_rate": q4_rate,
        "initial_rate": progression[9][3] if len(progression) >= 10 else progression[-1][3],  # Rate at instance 10
        "final_rate": progression[-1][3],
        "total_workflows": progression[-1][2],
    }


def print_analysis(trend: Dict, windowed: List[Tuple]):
    """Print trend analysis."""
    print("\n=== Trend Analysis ===\n")
    print(f"Initial success rate (first 10): {trend['initial_rate']*100:.1f}%")
    print(f"Final success rate: {trend['final_rate']*100:.1f}%")
    print(f"Total workflows learned: {trend['total_workflows']}")
    print()
    print("Success rate by quarter:")
    print(f"  Q1 (instances 1-25):   {trend['q1_rate']*100:.1f}%")
    print(f"  Q2 (instances 26-50):  {trend['q2_rate']*100:.1f}%")
    print(f"  Q3 (instances 51-75):  {trend['q3_rate']*100:.1f}%")
    print(f"  Q4 (instances 76-100): {trend['q4_rate']*100:.1f}%")

    # Calculate improvement
    improvement = trend['q4_rate'] - trend['q1_rate']
    print(f"\nImprovement Q4 vs Q1: {improvement*100:+.1f} percentage points")


def print_quarter_latex(trend: Dict):
    """Print LaTeX table for quarterly analysis."""
    print("\n=== LaTeX Quarter Analysis Table ===\n")
    print("$$")
    print(r"\begin{array}{|l|c|c|}")
    print(r"\hline")
    print(r"\textbf{Period} & \textbf{Instances} & \textbf{Avg Success Rate} \\")
    print(r"\hline")
    print(f"\\text{{Q1 (Early)}} & 1-25 & {trend['q1_rate']*100:.1f}\\% \\\\")
    print(f"\\text{{Q2}} & 26-50 & {trend['q2_rate']*100:.1f}\\% \\\\")
    print(f"\\text{{Q3}} & 51-75 & {trend['q3_rate']*100:.1f}\\% \\\\")
    print(f"\\text{{Q4 (Late)}} & 76-100 & {trend['q4_rate']*100:.1f}\\% \\\\")
    print(r"\hline")
    print(r"\end{array}")
    print("$$")


# Global results for trend analysis
results = []


def main():
    global results

    script_dir = Path(__file__).parent
    results_path = script_dir / "results.jsonl"

    if not results_path.exists():
        print(f"Error: {results_path} not found")
        return

    print("Loading results...")
    results = load_results_chronologically(results_path)
    print(f"Loaded {len(results)} instances")

    # Compute progression
    progression = compute_progression(results)
    windowed = compute_windowed_rate(results, window_size=10)

    # Print tables
    print_progression_table(progression, step=10)
    print_latex_table(progression, step=20)

    # Trend analysis
    trend = analyze_trend(progression)
    print_analysis(trend, windowed)
    print_quarter_latex(trend)


if __name__ == "__main__":
    main()
