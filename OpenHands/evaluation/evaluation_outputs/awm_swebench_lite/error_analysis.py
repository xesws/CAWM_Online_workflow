#!/usr/bin/env python3
"""
AWM SWE-bench Lite Error Analysis Script

Analyzes evaluation results and generates a paper-ready error analysis report.
"""

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Dict


@dataclass
class HistoryAnalysis:
    """Analysis of an instance's action history."""
    total_steps: int = 0
    total_text_length: int = 0
    action_counts: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)


@dataclass
class InstanceAnalysis:
    """Complete analysis of a single instance."""
    instance_id: str
    test_result: str
    error_category: str
    history_analysis: HistoryAnalysis
    diff_patch_length: int
    problem_statement_preview: str
    test_output_preview: str
    raw_entry: dict = field(default_factory=dict)


# Error category constants
CATEGORY_SUCCESS = "success"
CATEGORY_API_ERROR = "api_error"
CATEGORY_RUNTIME_ERROR = "runtime_error"
CATEGORY_EMPTY_GENERATION = "empty_generation"
CATEGORY_PATCH_FAILED = "patch_failed"
CATEGORY_TEST_TIMEOUT = "test_timeout"
CATEGORY_TEST_FAILED = "test_failed"

# Category display names for report
CATEGORY_NAMES = {
    CATEGORY_SUCCESS: "Success",
    CATEGORY_API_ERROR: "API/Infrastructure Error",
    CATEGORY_RUNTIME_ERROR: "Runtime Error",
    CATEGORY_EMPTY_GENERATION: "Empty Patch Generation",
    CATEGORY_PATCH_FAILED: "Patch Application Failure",
    CATEGORY_TEST_TIMEOUT: "Test Timeout",
    CATEGORY_TEST_FAILED: "Test Failure (Incorrect Fix)",
}


def classify_error(entry: dict) -> str:
    """Classify the type of error for a failed instance."""
    test_result = entry.get("test_result", "")
    test_output = entry.get("test_output", "") or ""
    diff_patch = entry.get("diff_patch", "") or ""

    if test_result == "PASS":
        return CATEGORY_SUCCESS

    # Check for API/infrastructure errors (Error: prefix indicates exception in loop)
    if test_output.startswith("Error:"):
        error_lower = test_output.lower()
        api_keywords = [
            "connection", "api", "openrouter", "rate limit", "rate_limit",
            "timeout", "timed out", "network", "socket", "ssl", "certificate",
            "503", "502", "500", "429", "unauthorized", "forbidden"
        ]
        if any(kw in error_lower for kw in api_keywords):
            return CATEGORY_API_ERROR
        return CATEGORY_RUNTIME_ERROR

    # Check for empty generation
    if not diff_patch or len(diff_patch.strip()) < 10:
        return CATEGORY_EMPTY_GENERATION

    # Check for patch application failure
    if "APPLY_PATCH_FAIL" in test_output:
        return CATEGORY_PATCH_FAILED

    # Check for timeout
    test_output_lower = test_output.lower()
    if "timeout" in test_output_lower or "timed out" in test_output_lower:
        return CATEGORY_TEST_TIMEOUT

    # Default: actual test failure (patch applied, tests ran, but failed)
    return CATEGORY_TEST_FAILED


def analyze_history(history: list) -> HistoryAnalysis:
    """Analyze the action history for patterns and issues."""
    analysis = HistoryAnalysis()

    if not history:
        return analysis

    analysis.total_steps = len(history)
    analysis.total_text_length = sum(len(json.dumps(step)) for step in history)

    # Count action types
    action_types = [step.get("action_type", "unknown") for step in history]
    analysis.action_counts = dict(Counter(action_types))

    # Detect issues
    issues = []

    # High step count (approaching max_iterations=100)
    if analysis.total_steps > 70:
        issues.append("high_step_count")

    # Context explosion (very large history)
    if analysis.total_text_length > 150000:  # ~150KB
        issues.append("context_explosion")

    # Repetitive actions (lost focus indicator)
    if len(history) >= 10:
        recent_actions = [str(s.get("action", ""))[:200] for s in history[-10:]]
        unique_actions = len(set(recent_actions))
        if unique_actions < len(recent_actions) * 0.6:  # >40% repetition
            issues.append("repetitive_actions")

    # Excessive commands at the end (stuck in loop)
    if len(history) >= 5:
        final_action_types = [s.get("action_type", "") for s in history[-5:]]
        if final_action_types.count("run_command") >= 4:
            issues.append("excessive_final_commands")
        if final_action_types.count("edit_file") == 0 and final_action_types.count("run_command") >= 3:
            issues.append("no_edits_final_steps")

    # Check if reached max iterations
    if analysis.total_steps >= 95:  # Close to max_iterations=100
        issues.append("reached_max_iterations")

    analysis.issues = issues
    return analysis


def load_results(results_path: Path) -> List[InstanceAnalysis]:
    """Load and analyze all results from results.jsonl."""
    analyses = []

    with open(results_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())

                # Classify error
                error_category = classify_error(entry)

                # Analyze history
                history = entry.get("history", [])
                history_analysis = analyze_history(history)

                # Create analysis object
                analysis = InstanceAnalysis(
                    instance_id=entry.get("instance_id", f"unknown_{line_num}"),
                    test_result=entry.get("test_result", "UNKNOWN"),
                    error_category=error_category,
                    history_analysis=history_analysis,
                    diff_patch_length=len(entry.get("diff_patch", "") or ""),
                    problem_statement_preview=(entry.get("problem_statement", "") or "")[:500],
                    test_output_preview=(entry.get("test_output", "") or "")[:2000],
                    raw_entry=entry,
                )
                analyses.append(analysis)

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue

    return analyses


def compute_step_pattern_stats(analyses: List[InstanceAnalysis]) -> Dict:
    """Compute step pattern statistics for success vs failure."""
    success_stats = defaultdict(list)
    failure_stats = defaultdict(list)

    for a in analyses:
        stats = success_stats if a.test_result == "PASS" else failure_stats

        stats["total_steps"].append(a.history_analysis.total_steps)
        stats["total_text_length"].append(a.history_analysis.total_text_length)

        # Per action type
        for action_type, count in a.history_analysis.action_counts.items():
            stats[f"action_{action_type}"].append(count)

    def compute_mean_std(values: list) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, std

    result = {
        "success": {},
        "failure": {},
    }

    # Get all metric keys
    all_keys = set(success_stats.keys()) | set(failure_stats.keys())

    for key in all_keys:
        result["success"][key] = compute_mean_std(success_stats.get(key, []))
        result["failure"][key] = compute_mean_std(failure_stats.get(key, []))

    return result


def get_representative_example(analyses: List[InstanceAnalysis], category: str) -> Optional[InstanceAnalysis]:
    """Get a representative example for a given error category."""
    category_instances = [a for a in analyses if a.error_category == category]
    if not category_instances:
        return None

    # For test_failed, prefer instances with more steps (more effort, still failed)
    if category == CATEGORY_TEST_FAILED:
        category_instances.sort(key=lambda x: x.history_analysis.total_steps, reverse=True)

    # Return the first (or most representative)
    return category_instances[0]


def summarize_history(analysis: InstanceAnalysis) -> str:
    """Create a brief summary of what the agent did."""
    history = analysis.raw_entry.get("history", [])
    if not history:
        return "No history available."

    # Get action sequence summary
    action_types = [s.get("action_type", "unknown") for s in history]
    action_summary = dict(Counter(action_types))

    # Get first and last few thoughts
    first_thoughts = []
    for step in history[:3]:
        thought = step.get("thought", "")
        if thought:
            first_thoughts.append(thought[:200] + "..." if len(thought) > 200 else thought)

    last_thoughts = []
    for step in history[-3:]:
        thought = step.get("thought", "")
        if thought:
            last_thoughts.append(thought[:200] + "..." if len(thought) > 200 else thought)

    summary_parts = [
        f"**Total steps:** {len(history)}",
        f"**Action distribution:** {action_summary}",
    ]

    if first_thoughts:
        summary_parts.append(f"**Initial approach:** {first_thoughts[0]}")

    if last_thoughts and last_thoughts != first_thoughts:
        summary_parts.append(f"**Final state:** {last_thoughts[-1]}")

    if analysis.history_analysis.issues:
        summary_parts.append(f"**Detected issues:** {', '.join(analysis.history_analysis.issues)}")

    return "\n".join(summary_parts)


def generate_report(analyses: List[InstanceAnalysis], output_path: Path, config: Dict) -> None:
    """Generate the paper-ready markdown report."""

    # Compute statistics
    total = len(analyses)
    success_count = sum(1 for a in analyses if a.test_result == "PASS")
    failure_count = total - success_count

    # Category breakdown
    category_counts = Counter(a.error_category for a in analyses)

    # Step pattern stats
    step_stats = compute_step_pattern_stats(analyses)

    # History issues across all failures
    all_issues = []
    for a in analyses:
        if a.test_result != "PASS":
            all_issues.extend(a.history_analysis.issues)
    issue_counts = Counter(all_issues)

    # Build report
    report_lines = []

    # Header
    report_lines.append("# AWM SWE-bench Lite Error Analysis Report")
    report_lines.append("")
    report_lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    report_lines.append("")

    # 1. Executive Summary
    report_lines.append("## 1. Executive Summary")
    report_lines.append("")
    report_lines.append(f"- **Total instances evaluated:** {total}")
    report_lines.append(f"- **Overall pass rate:** {success_count/total*100:.1f}% ({success_count}/{total})")
    report_lines.append(f"- **Model:** {config.get('llm_config_name', 'unknown')}")
    report_lines.append(f"- **Max iterations:** {config.get('max_iterations', 'unknown')}")
    report_lines.append("")

    # Key findings summary
    main_failure_categories = [(cat, cnt) for cat, cnt in category_counts.most_common() if cat != CATEGORY_SUCCESS]
    if main_failure_categories:
        top_category = main_failure_categories[0]
        report_lines.append(f"**Key Finding:** The primary failure mode is **{CATEGORY_NAMES.get(top_category[0], top_category[0])}**, "
                          f"accounting for {top_category[1]} ({top_category[1]/failure_count*100:.1f}%) of all failures. ")
        if issue_counts:
            top_issue = issue_counts.most_common(1)[0]
            report_lines.append(f"Among failed instances, **{top_issue[0].replace('_', ' ')}** was detected in {top_issue[1]} cases.")
    report_lines.append("")

    # 2. Results Overview
    report_lines.append("## 2. Results Overview")
    report_lines.append("")
    report_lines.append("### 2.1 Overall Statistics")
    report_lines.append("")
    report_lines.append("| Metric | Value |")
    report_lines.append("|--------|-------|")
    report_lines.append(f"| Total Instances | {total} |")
    report_lines.append(f"| Resolved (PASS) | {success_count} ({success_count/total*100:.1f}%) |")
    report_lines.append(f"| Failed (FAIL) | {failure_count} ({failure_count/total*100:.1f}%) |")
    report_lines.append("")

    report_lines.append("### 2.2 Error Category Breakdown")
    report_lines.append("")
    report_lines.append("| Error Category | Count | % of Failures | % of Total |")
    report_lines.append("|----------------|-------|---------------|------------|")

    for category in [CATEGORY_SUCCESS, CATEGORY_API_ERROR, CATEGORY_RUNTIME_ERROR,
                     CATEGORY_EMPTY_GENERATION, CATEGORY_PATCH_FAILED,
                     CATEGORY_TEST_TIMEOUT, CATEGORY_TEST_FAILED]:
        count = category_counts.get(category, 0)
        pct_failure = count / failure_count * 100 if failure_count > 0 and category != CATEGORY_SUCCESS else 0
        pct_total = count / total * 100 if total > 0 else 0

        if category == CATEGORY_SUCCESS:
            report_lines.append(f"| {CATEGORY_NAMES[category]} | {count} | - | {pct_total:.1f}% |")
        elif count > 0:
            report_lines.append(f"| {CATEGORY_NAMES[category]} | {count} | {pct_failure:.1f}% | {pct_total:.1f}% |")
    report_lines.append("")

    # 3. Step Pattern Analysis
    report_lines.append("## 3. Step Pattern Analysis")
    report_lines.append("")
    report_lines.append("### 3.1 Successful vs Failed: Action Distribution")
    report_lines.append("")
    report_lines.append("| Metric | Success (mean +/- std) | Failure (mean +/- std) |")
    report_lines.append("|--------|----------------------|----------------------|")

    # Key metrics to show
    metrics_to_show = ["total_steps", "total_text_length", "action_read_file",
                       "action_edit_file", "action_run_command", "action_think"]

    for metric in metrics_to_show:
        success_mean, success_std = step_stats["success"].get(metric, (0, 0))
        failure_mean, failure_std = step_stats["failure"].get(metric, (0, 0))

        # Format metric name
        display_name = metric.replace("action_", "").replace("_", " ").title()
        if metric == "total_text_length":
            display_name = "Total Context (chars)"

        report_lines.append(f"| {display_name} | {success_mean:.1f} +/- {success_std:.1f} | {failure_mean:.1f} +/- {failure_std:.1f} |")
    report_lines.append("")

    report_lines.append("### 3.2 Common Patterns in Failures")
    report_lines.append("")
    if issue_counts:
        report_lines.append("| Pattern | Count | % of Failures |")
        report_lines.append("|---------|-------|---------------|")
        for issue, count in issue_counts.most_common():
            display_name = issue.replace("_", " ").title()
            report_lines.append(f"| {display_name} | {count} | {count/failure_count*100:.1f}% |")
    else:
        report_lines.append("No specific patterns detected in failures.")
    report_lines.append("")

    # 4. Deep Analysis by Error Category
    report_lines.append("## 4. Deep Analysis by Error Category")
    report_lines.append("")

    section_num = 1
    for category in [CATEGORY_API_ERROR, CATEGORY_RUNTIME_ERROR, CATEGORY_EMPTY_GENERATION,
                     CATEGORY_PATCH_FAILED, CATEGORY_TEST_TIMEOUT, CATEGORY_TEST_FAILED]:
        count = category_counts.get(category, 0)
        if count == 0:
            continue

        category_instances = [a for a in analyses if a.error_category == category]

        report_lines.append(f"### 4.{section_num} {CATEGORY_NAMES[category]} ({count} instances)")
        report_lines.append("")

        # Root cause analysis based on category
        if category == CATEGORY_API_ERROR:
            report_lines.append("**Root Cause Analysis:**")
            report_lines.append("API/Infrastructure errors occur when the LLM API (OpenRouter) fails during inference. "
                              "Common causes include:")
            report_lines.append("- Rate limiting (429 errors)")
            report_lines.append("- Service unavailability (5xx errors)")
            report_lines.append("- Network connectivity issues")
            report_lines.append("- Request timeouts")
        elif category == CATEGORY_RUNTIME_ERROR:
            report_lines.append("**Root Cause Analysis:**")
            report_lines.append("Runtime errors occur during agent execution, separate from API issues. "
                              "These may include Docker container issues, file system errors, or agent exceptions.")
        elif category == CATEGORY_EMPTY_GENERATION:
            report_lines.append("**Root Cause Analysis:**")
            report_lines.append("Empty patch generation occurs when the agent fails to produce a valid solution. "
                              "Possible causes include:")
            report_lines.append("- Agent unable to understand the problem")
            report_lines.append("- Agent got stuck in exploration without making edits")
            report_lines.append("- Context limitations preventing solution formulation")
        elif category == CATEGORY_PATCH_FAILED:
            report_lines.append("**Root Cause Analysis:**")
            report_lines.append("Patch application failures occur when the generated diff cannot be applied to the repository. "
                              "Causes include:")
            report_lines.append("- Incorrect file paths in the patch")
            report_lines.append("- Context mismatch (edited file changed since agent read it)")
            report_lines.append("- Malformed patch format")
        elif category == CATEGORY_TEST_TIMEOUT:
            report_lines.append("**Root Cause Analysis:**")
            report_lines.append("Test timeouts occur when the evaluation exceeds the maximum allowed time (30 minutes). "
                              "This can happen due to:")
            report_lines.append("- Tests entering infinite loops")
            report_lines.append("- Resource-intensive test suites")
            report_lines.append("- Deadlocks introduced by the patch")
        elif category == CATEGORY_TEST_FAILED:
            report_lines.append("**Root Cause Analysis:**")
            report_lines.append("Test failures indicate the patch was applied but did not correctly fix the issue. "
                              "Common reasons include:")
            report_lines.append("- Incomplete fix (addresses only part of the problem)")
            report_lines.append("- Wrong fix location (modified incorrect code)")
            report_lines.append("- Incorrect logic (misunderstood the requirement)")
            report_lines.append("- Edge cases not handled")
        report_lines.append("")

        # Representative example
        example = get_representative_example(analyses, category)
        if example:
            report_lines.append("**Representative Example:**")
            report_lines.append("")
            report_lines.append(f"- **Instance:** `{example.instance_id}`")
            report_lines.append(f"- **Steps taken:** {example.history_analysis.total_steps}")
            report_lines.append(f"- **Patch length:** {example.diff_patch_length} chars")
            report_lines.append("")

            # Show relevant output based on category
            if category in [CATEGORY_API_ERROR, CATEGORY_RUNTIME_ERROR]:
                report_lines.append("**Error message:**")
                report_lines.append("```")
                report_lines.append(example.test_output_preview[:1000])
                report_lines.append("```")
            elif category == CATEGORY_PATCH_FAILED:
                report_lines.append("**Patch application output:**")
                report_lines.append("```")
                # Extract just the patch failure part
                output = example.test_output_preview
                if "APPLY_PATCH_FAIL" in output:
                    start = max(0, output.find("APPLY_PATCH_FAIL") - 200)
                    end = min(len(output), output.find("APPLY_PATCH_FAIL") + 500)
                    report_lines.append(output[start:end])
                else:
                    report_lines.append(output[:800])
                report_lines.append("```")
            else:
                report_lines.append("**Agent history summary:**")
                report_lines.append("")
                report_lines.append(summarize_history(example))
            report_lines.append("")

        # List affected instances
        report_lines.append("**Affected Instances:**")
        instance_ids = [a.instance_id for a in category_instances]
        # Show first 10, then "and N more"
        if len(instance_ids) <= 10:
            report_lines.append(", ".join(f"`{id}`" for id in instance_ids))
        else:
            report_lines.append(", ".join(f"`{id}`" for id in instance_ids[:10]) + f", and {len(instance_ids)-10} more")
        report_lines.append("")

        section_num += 1

    # 5. Context/Focus Issues Analysis
    report_lines.append("## 5. Context and Focus Issues Analysis")
    report_lines.append("")
    report_lines.append("This section analyzes instances where context size or agent focus may have contributed to failure.")
    report_lines.append("")

    # Find instances with issues
    instances_with_issues = [a for a in analyses if a.test_result != "PASS" and a.history_analysis.issues]

    report_lines.append("### 5.1 Issue Distribution")
    report_lines.append("")
    report_lines.append("| Issue Type | Count | Description |")
    report_lines.append("|------------|-------|-------------|")

    issue_descriptions = {
        "high_step_count": "Instance used >70 steps (approaching max_iterations=100)",
        "context_explosion": "History exceeded 150KB of text",
        "repetitive_actions": "Agent repeated similar actions (>40% repetition in last 10 steps)",
        "excessive_final_commands": "Agent ran 4+ commands in final 5 steps without edits",
        "no_edits_final_steps": "No edit actions in final 5 steps despite running commands",
        "reached_max_iterations": "Instance used 95+ steps (likely hit max_iterations)",
    }

    for issue, count in issue_counts.most_common():
        desc = issue_descriptions.get(issue, "Unknown issue type")
        display_name = issue.replace("_", " ").title()
        report_lines.append(f"| {display_name} | {count} | {desc} |")
    report_lines.append("")

    if instances_with_issues:
        report_lines.append("### 5.2 Instances with Context/Focus Issues")
        report_lines.append("")
        report_lines.append("| Instance ID | Steps | Context Size | Issues |")
        report_lines.append("|-------------|-------|--------------|--------|")
        for a in sorted(instances_with_issues, key=lambda x: len(x.history_analysis.issues), reverse=True)[:20]:
            issues_str = ", ".join(a.history_analysis.issues)
            report_lines.append(f"| `{a.instance_id}` | {a.history_analysis.total_steps} | {a.history_analysis.total_text_length:,} | {issues_str} |")
        if len(instances_with_issues) > 20:
            report_lines.append(f"| ... | ... | ... | ({len(instances_with_issues)-20} more instances) |")
    report_lines.append("")

    # 6. Full Instance List
    report_lines.append("## 6. Full Instance List")
    report_lines.append("")
    report_lines.append("| Instance ID | Result | Error Category | Steps | Issues |")
    report_lines.append("|-------------|--------|----------------|-------|--------|")

    for a in sorted(analyses, key=lambda x: (x.test_result != "PASS", x.instance_id)):
        issues_str = ", ".join(a.history_analysis.issues) if a.history_analysis.issues else "-"
        cat_name = CATEGORY_NAMES.get(a.error_category, a.error_category)
        if a.test_result == "PASS":
            cat_name = "-"
        report_lines.append(f"| `{a.instance_id}` | {a.test_result} | {cat_name} | {a.history_analysis.total_steps} | {issues_str} |")
    report_lines.append("")

    # 7. Insights and Recommendations
    report_lines.append("## 7. Insights and Recommendations")
    report_lines.append("")

    insights = []

    # Generate insights based on data
    if category_counts.get(CATEGORY_TEST_FAILED, 0) > failure_count * 0.5:
        insights.append("**High rate of incorrect fixes:** Over half of failures are due to incorrect solutions. "
                       "Consider improving the agent's understanding of test requirements or providing better debugging feedback.")

    if issue_counts.get("high_step_count", 0) > failure_count * 0.3:
        insights.append("**Many instances approach step limit:** A significant portion of failures involve high step counts. "
                       "Consider increasing max_iterations or improving early termination logic.")

    if issue_counts.get("repetitive_actions", 0) > failure_count * 0.2:
        insights.append("**Repetitive action patterns detected:** The agent sometimes gets stuck in loops. "
                       "Consider adding loop detection or diversifying exploration strategies.")

    if category_counts.get(CATEGORY_EMPTY_GENERATION, 0) > failure_count * 0.1:
        insights.append("**Empty patch generation:** Some instances produce no solution. "
                       "The agent may need better guidance to commit to an approach and make edits.")

    if category_counts.get(CATEGORY_API_ERROR, 0) > 0:
        insights.append(f"**API reliability:** {category_counts[CATEGORY_API_ERROR]} instances failed due to API issues. "
                       "Consider implementing retry logic or using more reliable API endpoints.")

    for insight in insights:
        report_lines.append(f"- {insight}")
        report_lines.append("")

    if not insights:
        report_lines.append("No specific recommendations based on current data distribution.")
        report_lines.append("")

    # Appendix: Methodology
    report_lines.append("## Appendix: Methodology")
    report_lines.append("")
    report_lines.append("### Error Classification")
    report_lines.append("")
    report_lines.append("Errors are classified in the following priority order:")
    report_lines.append("1. **API/Infrastructure Error**: test_output starts with 'Error:' and contains API-related keywords")
    report_lines.append("2. **Runtime Error**: test_output starts with 'Error:' (other exceptions)")
    report_lines.append("3. **Empty Patch Generation**: diff_patch is empty or <10 characters")
    report_lines.append("4. **Patch Application Failure**: test_output contains 'APPLY_PATCH_FAIL'")
    report_lines.append("5. **Test Timeout**: test_output contains 'timeout' or 'timed out'")
    report_lines.append("6. **Test Failure**: Default category for all other failures")
    report_lines.append("")
    report_lines.append("### Context/Focus Issue Detection")
    report_lines.append("")
    report_lines.append("| Issue | Threshold |")
    report_lines.append("|-------|-----------|")
    report_lines.append("| High Step Count | >70 steps |")
    report_lines.append("| Context Explosion | >150,000 characters in history |")
    report_lines.append("| Repetitive Actions | >40% duplicate actions in last 10 steps |")
    report_lines.append("| Excessive Final Commands | 4+ run_command in last 5 steps |")
    report_lines.append("| Reached Max Iterations | >=95 steps |")
    report_lines.append("")

    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Report generated: {output_path}")


def main():
    """Main entry point."""
    # Paths
    script_dir = Path(__file__).parent
    results_path = script_dir / "results.jsonl"
    config_path = script_dir / "config.json"
    output_path = script_dir / "error_analysis_report.md"

    # Verify results exist
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return

    # Load config
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    print(f"Loading results from: {results_path}")
    print(f"File size: {results_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Load and analyze
    analyses = load_results(results_path)
    print(f"Loaded {len(analyses)} instances")

    # Quick stats
    success_count = sum(1 for a in analyses if a.test_result == "PASS")
    print(f"Pass rate: {success_count}/{len(analyses)} ({success_count/len(analyses)*100:.1f}%)")

    # Generate report
    generate_report(analyses, output_path, config)

    print("Done!")


if __name__ == "__main__":
    main()
