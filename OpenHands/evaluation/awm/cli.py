"""
Command Line Interface for AWM Online Learning

Provides a convenient way to run AWM experiments from the command line.
"""

import argparse
import resource
import sys
import os
from typing import Optional

from evaluation.awm.config import AWMConfig
from evaluation.awm.token_utils import DEFAULT_TRUNCATION_LIMIT
from evaluation.awm.loop import AWMOnlineLoop, load_django_issues
from openhands.core.config import get_llm_config_arg
from openhands.core.logger import openhands_logger as logger


# Minimum recommended file descriptor limit for AWM
# Each sample leaks ~20 FDs, so for 300 samples we need ~6000 + baseline
MIN_RECOMMENDED_FD_LIMIT = 8192


def ensure_sufficient_file_descriptors(num_samples: int = 300) -> None:
    """
    Ensure sufficient file descriptors are available for AWM evaluation.

    Each inference sample may use up to ~20 file descriptors due to:
    - EventStream (ThreadPoolExecutor, asyncio loops)
    - LLM connections (httpx to API providers)
    - Docker runtime (container connections)
    - Port locks and misc resources

    Args:
        num_samples: Expected number of samples to process
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

        # Calculate minimum needed: baseline (50) + samples * 25 (with margin)
        min_needed = 50 + (num_samples * 25)
        recommended = max(min_needed, MIN_RECOMMENDED_FD_LIMIT)

        if soft < recommended:
            # Try to increase to recommended limit
            new_soft = min(recommended, hard) if hard > 0 else recommended
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
                new_soft_actual, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
                print(f"[FD Limit] Increased file descriptor limit: {soft} -> {new_soft_actual}")

                if new_soft_actual < min_needed:
                    print(f"[FD Limit] WARNING: Current limit ({new_soft_actual}) may be insufficient for {num_samples} samples.")
                    print(f"[FD Limit] Recommend running: ulimit -n {recommended}")
            except (ValueError, OSError) as e:
                print(f"[FD Limit] WARNING: Could not increase file descriptor limit: {e}")
                print(f"[FD Limit] Current limit: {soft}, recommended: {recommended}")
                print(f"[FD Limit] Run 'ulimit -n {recommended}' before starting if you encounter 'Too many open files' errors.")
        else:
            print(f"[FD Limit] File descriptor limit OK: {soft} (recommended: {recommended})")

    except Exception as e:
        print(f"[FD Limit] Could not check file descriptor limits: {e}")


def parse_truncation_value(value: str) -> Optional[int]:
    """
    解析 truncation 参数值

    Args:
        value: 参数值字符串

    Returns:
        None 表示禁用, int 表示 token 限制
    """
    if value.lower() == 'true':
        return DEFAULT_TRUNCATION_LIMIT
    elif value.lower() == 'false':
        return None
    else:
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid truncation value: {value}. "
                "Use 'true', 'false', or a number."
            )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="AWM Online Learning for OpenHands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # LLM配置
    parser.add_argument(
        "--llm-config",
        type=str,
        required=True,
        help="LLM config name from config.toml (e.g., llm.eval_gpt4)",
    )

    # 数据集配置
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench",
        help="SWE-bench dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--repo-filter",
        type=str,
        default="django/django",
        help="Filter by repository",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of issues to process",
    )

    # Agent配置
    parser.add_argument(
        "--agent-class",
        type=str,
        default="CodeActAgent",
        help="Agent class name",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum iterations per issue",
    )

    # Induction配置
    parser.add_argument(
        "--induction-trigger",
        type=int,
        default=10,
        help="Trigger induction every N successful experiences",
    )
    parser.add_argument(
        "--max-workflows",
        type=int,
        default=50,
        help="Maximum workflows to keep in memory",
    )
    parser.add_argument(
        "--truncation",
        type=parse_truncation_value,
        default=None,
        nargs='?',
        const=DEFAULT_TRUNCATION_LIMIT,
        help="Truncation limit for workflow memory injection. "
             f"Use 'true' for default ({DEFAULT_TRUNCATION_LIMIT} tokens), "
             "'false' to disable, or a specific number.",
    )

    # 输出配置
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/evaluation_outputs/awm",
        help="Output directory",
    )

    # 运行配置
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from checkpoint file",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation step (for debugging)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    return parser.parse_args()


def main():
    """主入口函数"""
    args = parse_args()

    # 创建配置
    config = AWMConfig(
        llm_config_name=args.llm_config,
        agent_class=args.agent_class,
        max_iterations=args.max_iterations,
        dataset_name=args.dataset,
        split=args.split,
        repo_filter=args.repo_filter,
        induction_trigger_count=args.induction_trigger,
        max_workflows=args.max_workflows,
        truncation_limit=args.truncation,
        output_dir=args.output_dir,
        skip_evaluation=args.skip_evaluation,
        verbose=not args.quiet,
    )

    # 打印配置
    print("\n" + "=" * 60)
    print("AWM Online Learning Configuration")
    print("=" * 60)
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")

    # 加载数据集
    print("Loading dataset...")
    issues = load_django_issues(
        dataset_name=config.dataset_name,
        split=config.split,
        repo_filter=config.repo_filter,
        limit=args.limit,
    )
    print(f"Loaded {len(issues)} issues")

    if len(issues) == 0:
        print("No issues found. Exiting.")
        sys.exit(1)

    # Ensure sufficient file descriptors for the evaluation
    ensure_sufficient_file_descriptors(num_samples=len(issues))

    # 创建并运行loop
    print("\nInitializing AWM Online Loop...")
    loop = AWMOnlineLoop(config)

    print("\nStarting online learning...\n")
    result = loop.run(issues, resume_from=args.resume_from)

    # 打印最终统计
    print("\n" + "=" * 60)
    print("AWM Online Learning Complete")
    print("=" * 60)
    stats = result.statistics
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Total success: {stats['total_success']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Workflows learned: {stats['workflows_learned']}")
    print(f"  Elapsed time: {stats['elapsed_seconds']:.1f}s")
    print("=" * 60)

    # 保存结果
    result_path = os.path.join(config.output_dir, "result.json")
    result.save(result_path)
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
