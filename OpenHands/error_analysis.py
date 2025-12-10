#!/usr/bin/env python3
"""
SWE-bench Evaluation Error Analysis Script
Analyzes evaluation outputs to identify common error patterns
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# Base paths
BASE_DIR = Path("/Users/tangyiq/dev/OpenHands/OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgent/kimi-k2-0905_maxiter_100_N_v0.62.0-no-hint-run_1")
EVAL_OUTPUTS_DIR = BASE_DIR / "eval_outputs"
OUTPUT_FILE = BASE_DIR / "output.jsonl"

def load_report():
    """Load the main report.json"""
    with open(BASE_DIR / "report.json", 'r') as f:
        return json.load(f)

def load_instance_report(instance_id):
    """Load report.json for a specific instance"""
    report_path = EVAL_OUTPUTS_DIR / instance_id / "report.json"
    if report_path.exists():
        with open(report_path, 'r') as f:
            return json.load(f)
    return None

def read_test_output(instance_id):
    """Read test_output.txt for a specific instance"""
    output_path = EVAL_OUTPUTS_DIR / instance_id / "test_output.txt"
    if output_path.exists():
        with open(output_path, 'r') as f:
            return f.read()
    return None

def analyze_unresolved_instances(unresolved_ids):
    """Analyze all unresolved instances"""
    analysis = {
        'total_unresolved': len(unresolved_ids),
        'instances': {},
        'failure_categories': defaultdict(list),
        'test_failure_counts': defaultdict(int)
    }

    for instance_id in unresolved_ids:
        print(f"Analyzing {instance_id}...")
        report = load_instance_report(instance_id)
        test_output = read_test_output(instance_id)

        if not report:
            analysis['instances'][instance_id] = {
                'error': 'No report found',
                'status': 'missing_report'
            }
            analysis['failure_categories']['missing_report'].append(instance_id)
            continue

        instance_data = report.get(instance_id, {})
        tests_status = instance_data.get('tests_status', {})

        # Count failed FAIL_TO_PASS tests (tests that should have passed)
        fail_to_pass = tests_status.get('FAIL_TO_PASS', {})
        failed_tests = fail_to_pass.get('failure', [])
        successful_fixes = fail_to_pass.get('success', [])

        # Check for test regressions (PASS_TO_FAIL)
        pass_to_fail = tests_status.get('PASS_TO_FAIL', {})
        regressions = pass_to_fail.get('failure', [])

        # Categorize the failure
        category = categorize_failure(instance_data, failed_tests, regressions, test_output)

        analysis['instances'][instance_id] = {
            'failed_tests_count': len(failed_tests),
            'failed_tests': failed_tests[:5],  # Store first 5 for brevity
            'successful_fixes': len(successful_fixes),
            'regressions': len(regressions),
            'regression_tests': regressions[:5],
            'patch_applied': instance_data.get('patch_successfully_applied', False),
            'category': category,
            'repo': instance_id.split('__')[0]
        }

        analysis['failure_categories'][category].append(instance_id)
        analysis['test_failure_counts'][len(failed_tests)] += 1

    return analysis

def categorize_failure(instance_data, failed_tests, regressions, test_output):
    """Categorize the type of failure"""

    # Check if patch was applied successfully
    if not instance_data.get('patch_successfully_applied', False):
        return 'patch_application_failed'

    # Check for test regressions
    if regressions:
        return 'test_regression'

    # Check if all tests failed
    if failed_tests:
        num_failed = len(failed_tests)
        if num_failed == 1:
            return 'single_test_failed'
        elif num_failed <= 5:
            return 'few_tests_failed'
        else:
            return 'many_tests_failed'

    return 'unknown'

def analyze_empty_patches(empty_patch_ids, output_file):
    """Analyze instances with empty patches"""
    analysis = {
        'total_empty': len(empty_patch_ids),
        'instances': {}
    }

    # Load output.jsonl to get more context
    with open(output_file, 'r') as f:
        outputs = {json.loads(line)['instance_id']: json.loads(line)
                  for line in f if line.strip()}

    for instance_id in empty_patch_ids:
        print(f"Analyzing empty patch: {instance_id}...")
        output_data = outputs.get(instance_id, {})

        analysis['instances'][instance_id] = {
            'has_output': instance_id in outputs,
            'test_result': output_data.get('test_result', {}).get('result', 'unknown'),
            'metrics': output_data.get('metrics', {}),
            'repo': instance_id.split('__')[0]
        }

    return analysis

def sample_incomplete_instances(incomplete_ids, output_file, sample_size=20):
    """Sample and analyze incomplete instances"""
    import random

    analysis = {
        'total_incomplete': len(incomplete_ids),
        'sample_size': min(sample_size, len(incomplete_ids)),
        'instances': {},
        'error_categories': defaultdict(list)
    }

    # Load output.jsonl
    with open(output_file, 'r') as f:
        outputs = {json.loads(line)['instance_id']: json.loads(line)
                  for line in f if line.strip()}

    # Sample instances
    sampled_ids = random.sample(incomplete_ids, min(sample_size, len(incomplete_ids)))

    for instance_id in sampled_ids:
        print(f"Sampling incomplete: {instance_id}...")
        output_data = outputs.get(instance_id, {})

        # Categorize why it's incomplete
        category = 'not_attempted'
        if instance_id in outputs:
            test_result = output_data.get('test_result', {})
            if test_result.get('result') == 'error':
                category = 'execution_error'
            elif 'error' in output_data:
                category = 'agent_error'
            else:
                category = 'no_patch_generated'

        analysis['instances'][instance_id] = {
            'has_output': instance_id in outputs,
            'category': category,
            'repo': instance_id.split('__')[0],
            'metrics': output_data.get('metrics', {})
        }

        analysis['error_categories'][category].append(instance_id)

    return analysis

def generate_report(report_data, unresolved_analysis, empty_analysis, incomplete_analysis):
    """Generate comprehensive error analysis report"""

    print("\n" + "="*80)
    print("SWE-BENCH EVALUATION ERROR ANALYSIS REPORT")
    print("="*80)

    print(f"\n## OVERALL METRICS")
    print(f"Total instances in dataset: {report_data['total_instances']}")
    print(f"Instances submitted: {report_data['submitted_instances']}")
    print(f"Instances completed: {report_data['completed_instances']}")
    print(f"Instances resolved: {report_data['resolved_instances']}")
    print(f"Instances unresolved: {report_data['unresolved_instances']}")
    print(f"Empty patch instances: {report_data['empty_patch_instances']}")
    print(f"Incomplete instances: {len(report_data['incomplete_ids'])}")
    print(f"\nResolution rate: {report_data['resolved_instances']/report_data['completed_instances']*100:.1f}% ({report_data['resolved_instances']}/{report_data['completed_instances']})")
    print(f"Submission rate: {report_data['submitted_instances']/report_data['total_instances']*100:.1f}% ({report_data['submitted_instances']}/{report_data['total_instances']})")

    print(f"\n## UNRESOLVED INSTANCES ANALYSIS ({unresolved_analysis['total_unresolved']} instances)")
    print("\nFailure Categories:")
    for category, instances in sorted(unresolved_analysis['failure_categories'].items()):
        print(f"  - {category}: {len(instances)} instances")
        for inst_id in instances[:3]:  # Show first 3 examples
            inst_data = unresolved_analysis['instances'][inst_id]
            print(f"    * {inst_id} ({inst_data['failed_tests_count']} failed tests)")

    print("\nTest Failure Distribution:")
    for count, num_instances in sorted(unresolved_analysis['test_failure_counts'].items()):
        print(f"  - {count} failed tests: {num_instances} instances")

    print("\nTop Failing Instances (by number of failed tests):")
    sorted_instances = sorted(
        unresolved_analysis['instances'].items(),
        key=lambda x: x[1].get('failed_tests_count', 0),
        reverse=True
    )
    for inst_id, data in sorted_instances[:5]:
        print(f"  - {inst_id}: {data['failed_tests_count']} failed tests")
        if data.get('failed_tests'):
            print(f"    Tests: {', '.join(data['failed_tests'][:2])}...")

    print(f"\n## EMPTY PATCH INSTANCES ({empty_analysis['total_empty']} instances)")
    for inst_id, data in empty_analysis['instances'].items():
        print(f"  - {inst_id}")
        print(f"    Repo: {data['repo']}")
        print(f"    Has output: {data['has_output']}")

    print(f"\n## INCOMPLETE INSTANCES SAMPLE ({incomplete_analysis['sample_size']}/{incomplete_analysis['total_incomplete']} sampled)")
    print("\nError Categories:")
    for category, instances in sorted(incomplete_analysis['error_categories'].items()):
        print(f"  - {category}: {len(instances)} instances")
        for inst_id in instances[:2]:
            print(f"    * {inst_id}")

    print("\nRepository Distribution (all incomplete):")
    repo_counts = defaultdict(int)
    for inst_id in report_data['incomplete_ids']:
        repo = inst_id.split('__')[0]
        repo_counts[repo] += 1

    for repo, count in sorted(repo_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {repo}: {count} incomplete instances")

    print("\n" + "="*80)
    print("KEY FINDINGS & RECOMMENDATIONS")
    print("="*80)

    print("\n1. LOW SUBMISSION RATE (11%)")
    print(f"   - Only {report_data['submitted_instances']}/{report_data['total_instances']} instances were submitted")
    print("   - Investigate why 267 instances were never completed")
    print("   - Common causes: agent errors, timeouts, early termination")

    print("\n2. MODERATE RESOLUTION RATE (41.9%)")
    print(f"   - {report_data['resolved_instances']}/{report_data['completed_instances']} completed instances passed all tests")
    print("   - 18 instances had patches that didn't fix the issue")

    print("\n3. COMMON FAILURE PATTERNS")
    top_category = max(unresolved_analysis['failure_categories'].items(),
                      key=lambda x: len(x[1]))
    print(f"   - Most common: {top_category[0]} ({len(top_category[1])} instances)")
    print("   - Many instances have multiple failed FAIL_TO_PASS tests")
    print("   - Indicates incomplete or incorrect fixes")

    print("\n4. NEXT STEPS")
    print("   - Deep dive into incomplete instances to identify agent failures")
    print("   - Analyze test output for unresolved instances to understand why fixes failed")
    print("   - Review empty patch instances to understand termination conditions")
    print("   - Consider increasing max iterations or improving agent debugging capabilities")

    print("\n" + "="*80)

def save_detailed_report(report_data, unresolved_analysis, empty_analysis, incomplete_analysis):
    """Save detailed JSON report"""
    detailed_report = {
        'summary': {
            'total_instances': report_data['total_instances'],
            'submitted_instances': report_data['submitted_instances'],
            'completed_instances': report_data['completed_instances'],
            'resolved_instances': report_data['resolved_instances'],
            'unresolved_instances': report_data['unresolved_instances'],
            'empty_patch_instances': report_data['empty_patch_instances'],
            'incomplete_instances': len(report_data['incomplete_ids']),
            'resolution_rate': report_data['resolved_instances']/report_data['completed_instances'],
            'submission_rate': report_data['submitted_instances']/report_data['total_instances']
        },
        'unresolved_analysis': unresolved_analysis,
        'empty_patch_analysis': empty_analysis,
        'incomplete_analysis': incomplete_analysis
    }

    output_path = BASE_DIR / "detailed_error_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(detailed_report, f, indent=2)

    print(f"\nDetailed report saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    print("Loading main report...")
    report_data = load_report()

    print("\n" + "="*80)
    print("PHASE 1: Analyzing Unresolved Instances")
    print("="*80)
    unresolved_analysis = analyze_unresolved_instances(report_data['unresolved_ids'])

    print("\n" + "="*80)
    print("PHASE 2: Analyzing Empty Patch Instances")
    print("="*80)
    empty_analysis = analyze_empty_patches(report_data['empty_patch_ids'], OUTPUT_FILE)

    print("\n" + "="*80)
    print("PHASE 3: Sampling Incomplete Instances")
    print("="*80)
    incomplete_analysis = sample_incomplete_instances(
        report_data['incomplete_ids'],
        OUTPUT_FILE,
        sample_size=20
    )

    print("\n" + "="*80)
    print("PHASE 4: Generating Comprehensive Report")
    print("="*80)
    generate_report(report_data, unresolved_analysis, empty_analysis, incomplete_analysis)

    # Save detailed JSON report
    report_path = save_detailed_report(report_data, unresolved_analysis, empty_analysis, incomplete_analysis)

    print("\n✓ Error analysis complete!")
