#!/usr/bin/env python3
"""
Analyze incomplete instances from SWE-bench evaluation
"""

import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("/Users/tangyiq/dev/OpenHands/OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgent/kimi-k2-0905_maxiter_100_N_v0.62.0-no-hint-run_1")
OUTPUT_FILE = BASE_DIR / "output.jsonl"

def load_report():
    """Load the main report.json"""
    with open(BASE_DIR / "report.json", 'r') as f:
        return json.load(f)

def analyze_incomplete_instances():
    """Analyze incomplete instances in detail"""
    report = load_report()
    incomplete_ids = set(report['incomplete_ids'])

    # Load all outputs
    outputs = {}
    with open(OUTPUT_FILE, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                outputs[data['instance_id']] = data

    # Categorize incomplete instances
    categories = {
        'not_in_output': [],  # Not even attempted
        'llm_error': [],      # LLM service errors
        'agent_error': [],    # Agent execution errors
        'timeout': [],        # Hit max iterations
        'early_exit': [],     # Exited early for some reason
        'other': []
    }

    error_details = defaultdict(list)

    for inst_id in incomplete_ids:
        if inst_id not in outputs:
            categories['not_in_output'].append(inst_id)
            continue

        data = outputs[inst_id]

        # Check for errors
        error = data.get('error', '')
        if 'LLM' in error or 'SERVICE_UNAVAILABLE' in error:
            categories['llm_error'].append(inst_id)
            error_details['llm_errors'].append({
                'instance_id': inst_id,
                'error': error
            })
        elif error:
            categories['agent_error'].append(inst_id)
            error_details['agent_errors'].append({
                'instance_id': inst_id,
                'error': error
            })

        # Check metrics
        metrics = data.get('metrics', {})
        if metrics.get('git_patch'):
            # Has a patch but still incomplete?
            categories['other'].append(inst_id)

        # Check test results
        test_result = data.get('test_result', {})
        if test_result.get('result') == 'timeout':
            categories['timeout'].append(inst_id)

    return categories, error_details

def print_analysis(categories, error_details):
    """Print analysis results"""
    print("\n" + "="*80)
    print("INCOMPLETE INSTANCES DETAILED ANALYSIS")
    print("="*80)

    total = sum(len(v) for v in categories.values())
    print(f"\nTotal incomplete: {total}")

    print("\n## BREAKDOWN BY CATEGORY")
    for category, instances in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
        count = len(instances)
        pct = count / total * 100 if total > 0 else 0
        print(f"\n{category}: {count} instances ({pct:.1f}%)")

        # Show a few examples
        for inst_id in instances[:3]:
            print(f"  - {inst_id}")

    print("\n## ERROR DETAILS")

    # Analyze LLM errors
    if error_details['llm_errors']:
        print(f"\nLLM Errors ({len(error_details['llm_errors'])} instances):")
        error_types = defaultdict(int)
        for err in error_details['llm_errors']:
            error_types[err['error']] += 1

        for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {error}: {count} instances")

    # Analyze agent errors
    if error_details['agent_errors']:
        print(f"\nAgent Errors ({len(error_details['agent_errors'])} instances):")
        error_types = defaultdict(int)
        for err in error_details['agent_errors'][:20]:  # Sample first 20
            error_msg = err['error']
            # Truncate long errors
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + "..."
            error_types[error_msg] += 1

        for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - ({count}x) {error}")

if __name__ == "__main__":
    categories, error_details = analyze_incomplete_instances()
    print_analysis(categories, error_details)
