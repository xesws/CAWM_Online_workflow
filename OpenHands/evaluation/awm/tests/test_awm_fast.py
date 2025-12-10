#!/usr/bin/env python
"""
Fast AWM Test Script

使用较小的项目（sympy, requests）进行快速测试，避免 Django 等大型项目的长时间等待。

优点：
- Docker 镜像较小，构建/拉取速度快
- 测试用例简单，成功率高
- 可以快速验证 AWM 工作流学习功能

使用方法：
    # 跳过评估，只测试推理和工作流学习
    poetry run python evaluation/awm/tests/test_awm_fast.py --skip-eval

    # 完整测试（包括评估）
    poetry run python evaluation/awm/tests/test_awm_fast.py

    # 自定义数量
    poetry run python evaluation/awm/tests/test_awm_fast.py --limit 3

    # 使用特定 LLM
    poetry run python evaluation/awm/tests/test_awm_fast.py --llm-config llm.kimi-k2
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


# 小型项目的 SWE-bench 实例（Docker 镜像小，成功率高）
FAST_TEST_INSTANCES = [
    # SymPy - 纯 Python，Docker 镜像小 (~500MB)，成功率高
    "sympy__sympy-13480",
    "sympy__sympy-13647",
    "sympy__sympy-15011",
    "sympy__sympy-18189",
    "sympy__sympy-22005",
    "sympy__sympy-24066",
    "sympy__sympy-12419",
    "sympy__sympy-14024",
    "sympy__sympy-14396",
    "sympy__sympy-14817",

    # Requests - 简单的 HTTP 库，镜像小
    "psf__requests-2317",
    "psf__requests-2674",
    "psf__requests-3362",

    # Astropy - 天文学库，镜像中等
    "astropy__astropy-6938",
    "astropy__astropy-7746",
]

# 中等大小项目（如果需要更多测试用例）
MEDIUM_TEST_INSTANCES = [
    # scikit-learn - 镜像较大但比 Django 小
    "scikit-learn__scikit-learn-13779",
    "scikit-learn__scikit-learn-13496",
    "scikit-learn__scikit-learn-14092",

    # xarray - 数据分析库
    "pydata__xarray-4094",
    "pydata__xarray-4248",
]


def get_test_instances(limit: int = 5, include_medium: bool = False):
    """获取测试实例列表"""
    instances = FAST_TEST_INSTANCES.copy()
    if include_medium:
        instances.extend(MEDIUM_TEST_INSTANCES)
    return instances[:limit]


def run_fast_test(
    llm_config_name: str = "llm.kimi-k2",
    limit: int = 5,
    skip_evaluation: bool = False,
    output_dir: str = None,
    include_medium: bool = False,
    api_key: str = None,
):
    """运行快速 AWM 测试"""
    from evaluation.awm.config import AWMConfig
    from evaluation.awm.loop import AWMOnlineLoop
    from evaluation.awm.types import DjangoIssue
    from openhands.core.config import get_llm_config_arg
    from openhands.core.logger import openhands_logger as logger

    # 设置输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"evaluation/evaluation_outputs/awm_fast_test_{timestamp}"

    print("=" * 60)
    print("AWM Fast Test")
    print("=" * 60)
    print(f"LLM Config: {llm_config_name}")
    print(f"Test Instances: {limit}")
    print(f"Skip Evaluation: {skip_evaluation}")
    print(f"Output Dir: {output_dir}")
    print("=" * 60)

    # 加载 LLM 配置
    llm_config = get_llm_config_arg(llm_config_name)
    if llm_config is None:
        print(f"ERROR: LLM config '{llm_config_name}' not found in config.toml")
        print("Please add the config or use --llm-config to specify a valid config name")
        return False

    # 覆盖 API key (如果通过命令行提供)
    if api_key:
        from pydantic import SecretStr
        llm_config = llm_config.model_copy(update={"api_key": SecretStr(api_key)})
        print(f"Using custom API key: {api_key[:20]}...")

    # 获取测试实例
    test_instance_ids = get_test_instances(limit, include_medium)
    print(f"\nTest Instances:")
    for i, inst_id in enumerate(test_instance_ids, 1):
        print(f"  {i}. {inst_id}")
    print()

    # 从 SWE-bench 加载实例数据
    print("Loading SWE-bench dataset...")
    from datasets import load_dataset

    dataset = load_dataset("princeton-nlp/SWE-bench", split="test")
    instance_map = {item["instance_id"]: item for item in dataset}

    # 构建 DjangoIssue 列表（名字有误，但保持兼容）
    issues = []
    for inst_id in test_instance_ids:
        if inst_id not in instance_map:
            print(f"WARNING: Instance {inst_id} not found in dataset, skipping")
            continue

        item = instance_map[inst_id]
        issues.append(DjangoIssue(
            instance_id=item["instance_id"],
            problem_statement=item["problem_statement"],
            repo=item["repo"],
            base_commit=item.get("base_commit"),
            version=item.get("version"),
        ))

    if not issues:
        print("ERROR: No valid test instances found")
        return False

    print(f"Loaded {len(issues)} test instances")

    # 创建 AWM 配置
    config = AWMConfig(
        llm_config_name=llm_config_name,
        output_dir=output_dir,
        max_iterations=50,  # 减少迭代次数以加速
        induction_trigger_count=1,  # 每次成功都触发归纳
        skip_evaluation=skip_evaluation,
        verbose=True,
    )

    # 创建并运行 AWM Loop
    print("\nStarting AWM Online Loop...")
    loop = AWMOnlineLoop(config, llm_config)

    try:
        result = loop.run(issues)

        # 打印结果
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)

        stats = result.statistics
        print(f"Total Processed: {stats['total_processed']}")
        print(f"Total Success: {stats['total_success']}")
        print(f"Success Rate: {stats['success_rate']:.1%}")
        print(f"Workflows Learned: {stats['workflows_learned']}")
        print(f"Elapsed Time: {stats['elapsed_seconds']:.1f}s")

        # 显示学到的 workflows
        if result.workflows:
            print("\n" + "-" * 40)
            print("LEARNED WORKFLOWS:")
            print("-" * 40)
            for i, wf in enumerate(result.workflows, 1):
                print(f"\n{i}. {wf.name}")
                print(f"   Description: {wf.description[:100]}...")
                print(f"   Steps: {len(wf.steps)}")

        print("\n" + "=" * 60)
        print(f"Results saved to: {output_dir}")
        print("=" * 60)

        return stats['total_success'] > 0

    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fast AWM Test using small repos (sympy, requests)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with evaluation skipped (fastest)
  python test_awm_fast.py --skip-eval --limit 3

  # Full test with evaluation
  python test_awm_fast.py --limit 5

  # Include medium-sized projects
  python test_awm_fast.py --include-medium --limit 8
        """
    )

    parser.add_argument(
        "--llm-config",
        default="llm.kimi-k2",
        help="LLM config name from config.toml (default: llm.kimi-k2)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of test instances to run (default: 5)"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation step (faster, only test inference and workflow learning)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: auto-generated with timestamp)"
    )
    parser.add_argument(
        "--include-medium",
        action="store_true",
        help="Include medium-sized projects (scikit-learn, xarray)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        type=str,
        help="Override LLM API key (e.g., OpenRouter key: sk-or-v1-xxx)"
    )

    args = parser.parse_args()

    success = run_fast_test(
        llm_config_name=args.llm_config,
        limit=args.limit,
        skip_evaluation=args.skip_eval,
        output_dir=args.output_dir,
        include_medium=args.include_medium,
        api_key=args.api_key,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
