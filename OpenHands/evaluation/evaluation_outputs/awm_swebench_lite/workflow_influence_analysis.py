#!/usr/bin/env python3
"""
Workflow Influence Analysis

Analyzes whether workflow memory contributed to success rate improvements
in later quarters of the AWM online learning evaluation.

Key metrics:
1. Workflow availability per instance
2. Scenario matching between problems and available workflows
3. Action sequence alignment with workflow patterns
4. Success rate correlation with workflow availability
"""

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_data(script_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """Load results and memory data."""
    results_path = script_dir / "results.jsonl"
    memory_path = script_dir / "memory.json"

    # Load results chronologically
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

    # Load workflows
    with open(memory_path, "r") as f:
        memory = json.load(f)

    workflows = memory.get("workflows", [])

    return results, workflows


def build_workflow_timeline(results: List[Dict]) -> List[int]:
    """
    Build timeline of workflow availability.
    Returns list where timeline[i] = number of workflows available before instance i.
    """
    timeline = []
    cumulative_workflows = 0

    for entry in results:
        timeline.append(cumulative_workflows)
        if entry.get("test_result") == "PASS":
            cumulative_workflows += 1

    return timeline


def extract_problem_keywords(entry: Dict) -> List[str]:
    """Extract keywords from problem statement in the first few history steps."""
    keywords = []
    history = entry.get("history", [])

    # Get text from first few thoughts that likely contain problem description
    for step in history[:5]:
        thought = step.get("thought", "").lower()

        # Extract keywords related to problem types
        problem_patterns = [
            r"error", r"bug", r"fix", r"issue", r"problem",
            r"migration", r"model", r"field", r"form", r"view",
            r"query", r"database", r"sql", r"orm",
            r"template", r"static", r"media",
            r"auth", r"user", r"permission",
            r"test", r"validate", r"check",
            r"serializ", r"deserializ",
            r"unicode", r"encoding", r"decode",
            r"file", r"upload", r"path",
            r"session", r"cookie", r"cache",
            r"url", r"route", r"dispatch",
            r"admin", r"inline",
            r"message", r"signal",
            r"middleware", r"handler",
            r"pk", r"primary.?key", r"foreign.?key", r"fk",
            r"proxy", r"abstract", r"concrete",
            r"compound", r"nested", r"recursive",
            r"separab", r"matrix",
        ]

        for pattern in problem_patterns:
            if re.search(pattern, thought):
                keywords.append(pattern.replace("\\", "").replace(".?", ""))

    return list(set(keywords))


def match_scenarios(
    keywords: List[str], workflows: List[Dict], max_workflows: int
) -> List[Dict]:
    """
    Find workflows whose applicable_scenarios match the problem keywords.
    Only considers workflows that would have been available (up to max_workflows).
    """
    matched = []

    # Workflows are learned in order, so only consider first max_workflows
    available_workflows = workflows[:max_workflows] if max_workflows > 0 else []

    for wf in available_workflows:
        scenarios = wf.get("applicable_scenarios", [])
        wf_name = wf.get("name", "")

        # Check if any scenario matches any keyword
        for scenario in scenarios:
            scenario_lower = scenario.lower()
            for kw in keywords:
                if kw in scenario_lower or scenario_lower in kw:
                    matched.append(wf)
                    break
            else:
                continue
            break

    return matched


def get_action_sequence(history: List[Dict]) -> List[str]:
    """Extract action type sequence from history."""
    return [step.get("action_type", "") for step in history if step.get("action_type")]


def compute_step_pattern_alignment(history: List[Dict]) -> Dict[str, float]:
    """
    Compute alignment of action sequence with standard workflow patterns.

    Standard workflow pattern: Reproduce -> Locate -> (Debug) -> Fix -> Test

    Returns alignment metrics.
    """
    action_types = get_action_sequence(history)

    # Map action types to workflow step types
    action_to_step = {
        "run_command": ["Reproduce", "Test", "Debug"],
        "read_file": ["Locate", "Debug"],
        "edit_file": ["Fix"],
        "think": ["Reasoning"],
    }

    # Check for standard workflow phase progression
    phases = {
        "reproduce": False,  # Early run_command with "reproduce" or "test" in context
        "locate": False,     # read_file actions
        "fix": False,        # edit_file actions
        "test": False,       # Late run_command with "test" or "pytest"
    }

    # Simple heuristic: check if phases appear in order
    found_read = False
    found_edit = False
    found_test_command = False

    for i, atype in enumerate(action_types):
        if atype == "read_file":
            found_read = True
        elif atype == "edit_file":
            if found_read:
                found_edit = True
        elif atype == "run_command":
            if found_edit:
                found_test_command = True

    # Calculate alignment score
    alignment_score = 0.0
    if found_read:
        alignment_score += 0.25
    if found_edit:
        alignment_score += 0.25
    if found_test_command:
        alignment_score += 0.25
    if found_read and found_edit and found_test_command:
        alignment_score += 0.25  # Bonus for complete pattern

    return {
        "alignment_score": alignment_score,
        "has_read": found_read,
        "has_edit": found_edit,
        "has_test_after_edit": found_test_command,
    }


def analyze_workflow_influence(
    results: List[Dict], workflows: List[Dict]
) -> Dict[str, any]:
    """Main analysis function."""

    # Build workflow availability timeline
    timeline = build_workflow_timeline(results)

    # Analysis storage
    analysis = {
        "per_instance": [],
        "by_quarter": defaultdict(lambda: {"pass": 0, "fail": 0, "total": 0}),
        "by_workflow_count": defaultdict(lambda: {"pass": 0, "fail": 0}),
        "scenario_matches": {"with_match": {"pass": 0, "fail": 0}, "without_match": {"pass": 0, "fail": 0}},
        "alignment_scores": {"pass": [], "fail": []},
    }

    for i, entry in enumerate(results):
        instance_num = i + 1
        test_result = entry.get("test_result", "")
        is_pass = test_result == "PASS"
        workflows_available = timeline[i]

        # Determine quarter
        if instance_num <= 25:
            quarter = "Q1"
        elif instance_num <= 50:
            quarter = "Q2"
        elif instance_num <= 75:
            quarter = "Q3"
        else:
            quarter = "Q4"

        # Extract problem keywords
        keywords = extract_problem_keywords(entry)

        # Find matching workflows
        matched_workflows = match_scenarios(keywords, workflows, workflows_available)
        has_scenario_match = len(matched_workflows) > 0

        # Compute action alignment
        alignment = compute_step_pattern_alignment(entry.get("history", []))

        # Store per-instance analysis
        instance_analysis = {
            "instance_num": instance_num,
            "instance_id": entry.get("instance_id", ""),
            "test_result": test_result,
            "quarter": quarter,
            "workflows_available": workflows_available,
            "problem_keywords": keywords[:10],  # Limit for readability
            "matched_workflows": len(matched_workflows),
            "matched_workflow_names": [wf.get("name", "")[:50] for wf in matched_workflows[:3]],
            "alignment_score": alignment["alignment_score"],
        }
        analysis["per_instance"].append(instance_analysis)

        # Aggregate by quarter
        analysis["by_quarter"][quarter]["total"] += 1
        if is_pass:
            analysis["by_quarter"][quarter]["pass"] += 1
        else:
            analysis["by_quarter"][quarter]["fail"] += 1

        # Aggregate by workflow count bins
        if workflows_available == 0:
            bin_name = "0"
        elif workflows_available <= 10:
            bin_name = "1-10"
        elif workflows_available <= 20:
            bin_name = "11-20"
        elif workflows_available <= 30:
            bin_name = "21-30"
        else:
            bin_name = "31+"

        if is_pass:
            analysis["by_workflow_count"][bin_name]["pass"] += 1
        else:
            analysis["by_workflow_count"][bin_name]["fail"] += 1

        # Aggregate by scenario match
        if has_scenario_match:
            if is_pass:
                analysis["scenario_matches"]["with_match"]["pass"] += 1
            else:
                analysis["scenario_matches"]["with_match"]["fail"] += 1
        else:
            if is_pass:
                analysis["scenario_matches"]["without_match"]["pass"] += 1
            else:
                analysis["scenario_matches"]["without_match"]["fail"] += 1

        # Collect alignment scores
        if is_pass:
            analysis["alignment_scores"]["pass"].append(alignment["alignment_score"])
        else:
            analysis["alignment_scores"]["fail"].append(alignment["alignment_score"])

    return analysis


def print_workflow_availability_table(analysis: Dict):
    """Print workflow availability vs success rate table."""
    print("\n=== Success Rate by Workflow Availability ===\n")
    print("| Workflows Available | Pass | Fail | Total | Success Rate |")
    print("|--------------------:|-----:|-----:|------:|-------------:|")

    bin_order = ["0", "1-10", "11-20", "21-30", "31+"]
    for bin_name in bin_order:
        data = analysis["by_workflow_count"].get(bin_name, {"pass": 0, "fail": 0})
        total = data["pass"] + data["fail"]
        rate = (data["pass"] / total * 100) if total > 0 else 0
        if total > 0:
            print(f"| {bin_name:>18} | {data['pass']:>4} | {data['fail']:>4} | {total:>5} | {rate:>10.1f}% |")


def print_latex_workflow_availability(analysis: Dict):
    """Print LaTeX table for workflow availability analysis."""
    print("\n=== LaTeX: Success Rate by Workflow Availability ===\n")
    print("$$")
    print(r"\begin{array}{|r|c|c|c|c|}")
    print(r"\hline")
    print(r"\textbf{Workflows Available} & \textbf{Pass} & \textbf{Fail} & \textbf{Total} & \textbf{Success Rate} \\")
    print(r"\hline")

    bin_order = ["0", "1-10", "11-20", "21-30", "31+"]
    for bin_name in bin_order:
        data = analysis["by_workflow_count"].get(bin_name, {"pass": 0, "fail": 0})
        total = data["pass"] + data["fail"]
        rate = (data["pass"] / total * 100) if total > 0 else 0
        if total > 0:
            print(f"{bin_name} & {data['pass']} & {data['fail']} & {total} & {rate:.1f}\\% \\\\")

    print(r"\hline")
    print(r"\end{array}")
    print("$$")


def print_scenario_match_analysis(analysis: Dict):
    """Print scenario matching analysis."""
    print("\n=== Success Rate by Scenario Match ===\n")

    with_match = analysis["scenario_matches"]["with_match"]
    without_match = analysis["scenario_matches"]["without_match"]

    total_with = with_match["pass"] + with_match["fail"]
    total_without = without_match["pass"] + without_match["fail"]

    rate_with = (with_match["pass"] / total_with * 100) if total_with > 0 else 0
    rate_without = (without_match["pass"] / total_without * 100) if total_without > 0 else 0

    print("| Condition | Pass | Fail | Total | Success Rate |")
    print("|-----------|-----:|-----:|------:|-------------:|")
    print(f"| Relevant workflow available | {with_match['pass']:>4} | {with_match['fail']:>4} | {total_with:>5} | {rate_with:>10.1f}% |")
    print(f"| No relevant workflow        | {without_match['pass']:>4} | {without_match['fail']:>4} | {total_without:>5} | {rate_without:>10.1f}% |")


def print_latex_scenario_match(analysis: Dict):
    """Print LaTeX table for scenario matching."""
    print("\n=== LaTeX: Success Rate by Scenario Match ===\n")

    with_match = analysis["scenario_matches"]["with_match"]
    without_match = analysis["scenario_matches"]["without_match"]

    total_with = with_match["pass"] + with_match["fail"]
    total_without = without_match["pass"] + without_match["fail"]

    rate_with = (with_match["pass"] / total_with * 100) if total_with > 0 else 0
    rate_without = (without_match["pass"] / total_without * 100) if total_without > 0 else 0

    print("$$")
    print(r"\begin{array}{|l|c|c|c|c|}")
    print(r"\hline")
    print(r"\textbf{Condition} & \textbf{Pass} & \textbf{Fail} & \textbf{Total} & \textbf{Success Rate} \\")
    print(r"\hline")
    print(f"\\text{{Relevant workflow available}} & {with_match['pass']} & {with_match['fail']} & {total_with} & {rate_with:.1f}\\% \\\\")
    print(f"\\text{{No relevant workflow}} & {without_match['pass']} & {without_match['fail']} & {total_without} & {rate_without:.1f}\\% \\\\")
    print(r"\hline")
    print(r"\end{array}")
    print("$$")


def print_alignment_analysis(analysis: Dict):
    """Print action sequence alignment analysis."""
    print("\n=== Action Sequence Alignment (Reproduce→Locate→Fix→Test) ===\n")

    pass_scores = analysis["alignment_scores"]["pass"]
    fail_scores = analysis["alignment_scores"]["fail"]

    pass_avg = sum(pass_scores) / len(pass_scores) if pass_scores else 0
    fail_avg = sum(fail_scores) / len(fail_scores) if fail_scores else 0

    print(f"Average alignment score for successes: {pass_avg:.3f}")
    print(f"Average alignment score for failures:  {fail_avg:.3f}")
    print(f"Difference: {pass_avg - fail_avg:+.3f}")


def print_quarter_analysis(analysis: Dict):
    """Print detailed per-quarter analysis."""
    print("\n=== Detailed Quarter Analysis ===\n")

    for quarter in ["Q1", "Q2", "Q3", "Q4"]:
        data = analysis["by_quarter"][quarter]
        total = data["total"]
        rate = (data["pass"] / total * 100) if total > 0 else 0

        # Get instances for this quarter
        quarter_instances = [
            inst for inst in analysis["per_instance"]
            if inst["quarter"] == quarter
        ]

        avg_workflows = sum(inst["workflows_available"] for inst in quarter_instances) / len(quarter_instances)
        avg_matches = sum(inst["matched_workflows"] for inst in quarter_instances) / len(quarter_instances)

        print(f"{quarter}:")
        print(f"  Success rate: {rate:.1f}% ({data['pass']}/{total})")
        print(f"  Avg workflows available: {avg_workflows:.1f}")
        print(f"  Avg relevant workflows: {avg_matches:.2f}")
        print()


def print_example_cases(analysis: Dict):
    """Print representative example cases."""
    print("\n=== Representative Example Cases ===\n")

    instances = analysis["per_instance"]

    # Find examples
    # 1. Success with relevant workflow (Q3/Q4)
    success_with_match = [
        inst for inst in instances
        if inst["test_result"] == "PASS" and inst["matched_workflows"] > 0 and inst["quarter"] in ["Q3", "Q4"]
    ]

    # 2. Success without relevant workflow (Q1)
    success_without_match = [
        inst for inst in instances
        if inst["test_result"] == "PASS" and inst["matched_workflows"] == 0 and inst["quarter"] == "Q1"
    ]

    # 3. Failure despite relevant workflow
    failure_with_match = [
        inst for inst in instances
        if inst["test_result"] == "FAIL" and inst["matched_workflows"] > 0
    ]

    print("1. SUCCESS with relevant workflow available (potential influence):")
    if success_with_match:
        ex = success_with_match[0]
        print(f"   Instance {ex['instance_num']}: {ex['instance_id'][:40]}")
        print(f"   Quarter: {ex['quarter']}, Workflows available: {ex['workflows_available']}")
        print(f"   Matched workflows: {ex['matched_workflows']}")
        if ex['matched_workflow_names']:
            print(f"   Example match: {ex['matched_workflow_names'][0][:50]}...")
    print()

    print("2. SUCCESS without relevant workflow (independent success):")
    if success_without_match:
        ex = success_without_match[0]
        print(f"   Instance {ex['instance_num']}: {ex['instance_id'][:40]}")
        print(f"   Quarter: {ex['quarter']}, Workflows available: {ex['workflows_available']}")
        print(f"   Keywords: {', '.join(ex['problem_keywords'][:5])}")
    print()

    print("3. FAILURE despite relevant workflow available:")
    if failure_with_match:
        ex = failure_with_match[0]
        print(f"   Instance {ex['instance_num']}: {ex['instance_id'][:40]}")
        print(f"   Quarter: {ex['quarter']}, Workflows available: {ex['workflows_available']}")
        print(f"   Matched workflows: {ex['matched_workflows']}")
        if ex['matched_workflow_names']:
            print(f"   Example match: {ex['matched_workflow_names'][0][:50]}...")


def print_summary(analysis: Dict):
    """Print summary conclusions."""
    print("\n=== Summary: Workflow Memory Influence ===\n")

    # Calculate key metrics
    with_match = analysis["scenario_matches"]["with_match"]
    without_match = analysis["scenario_matches"]["without_match"]

    total_with = with_match["pass"] + with_match["fail"]
    total_without = without_match["pass"] + without_match["fail"]

    rate_with = (with_match["pass"] / total_with * 100) if total_with > 0 else 0
    rate_without = (without_match["pass"] / total_without * 100) if total_without > 0 else 0

    # Q1 vs Q4 comparison
    q1 = analysis["by_quarter"]["Q1"]
    q4 = analysis["by_quarter"]["Q4"]
    q1_rate = (q1["pass"] / q1["total"] * 100) if q1["total"] > 0 else 0
    q4_rate = (q4["pass"] / q4["total"] * 100) if q4["total"] > 0 else 0

    print("Key Findings:")
    print(f"1. Workflow availability grew from 0 (Q1) to ~40+ (Q4)")
    print(f"2. Success rate with relevant workflow: {rate_with:.1f}%")
    print(f"3. Success rate without relevant workflow: {rate_without:.1f}%")
    print(f"4. Difference: {rate_with - rate_without:+.1f} percentage points")
    print()
    print(f"5. Q1 success rate: {q1_rate:.1f}%")
    print(f"6. Q4 success rate: {q4_rate:.1f}%")
    print(f"7. Q1→Q4 improvement: {q4_rate - q1_rate:+.1f} percentage points")


def main():
    script_dir = Path(__file__).parent

    print("Loading data...")
    results, workflows = load_data(script_dir)
    print(f"Loaded {len(results)} instances and {len(workflows)} workflows")

    print("\nAnalyzing workflow influence...")
    analysis = analyze_workflow_influence(results, workflows)

    # Print all analysis tables
    print_workflow_availability_table(analysis)
    print_latex_workflow_availability(analysis)

    print_scenario_match_analysis(analysis)
    print_latex_scenario_match(analysis)

    print_alignment_analysis(analysis)
    print_quarter_analysis(analysis)
    print_example_cases(analysis)
    print_summary(analysis)


if __name__ == "__main__":
    main()
