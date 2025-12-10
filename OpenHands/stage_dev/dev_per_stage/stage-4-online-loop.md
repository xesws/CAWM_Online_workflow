# Stage 4: 完整AWM Online Loop (P3)

> **优先级**: P3
> **复杂度**: 低
> **依赖**: Stage 1, Stage 2, Stage 3 全部完成

---

## 1. 核心目标

整合所有模块，实现完整的online learning循环，并提供命令行接口。

---

## 2. 文件结构

```
evaluation/awm/
├── loop.py               # AWM Online Loop主控制器
├── cli.py                # 命令行入口
├── config.py             # AWM配置
└── utils.py              # 工具函数
```

---

## 3. 详细任务

### 3.1 Task: 实现AWM配置模块

**文件**: `evaluation/awm/config.py`

```python
"""
Configuration for AWM Online Learning

Provides configuration classes and utilities for AWM components.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
import json


@dataclass
class AWMConfig:
    """
    AWM Online Learning配置
    """
    # LLM配置
    llm_config_name: str = "llm.eval"
    agent_class: str = "CodeActAgent"
    max_iterations: int = 100

    # 数据集配置
    dataset_name: str = "princeton-nlp/SWE-bench"
    split: str = "test"
    repo_filter: str = "django/django"

    # Induction配置
    induction_trigger_count: int = 10
    max_workflows: int = 50

    # 输出配置
    output_dir: str = "evaluation/evaluation_outputs/awm"
    checkpoint_interval: int = 5  # 每N个样本保存一次checkpoint

    # 其他选项
    skip_evaluation: bool = False
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "llm_config_name": self.llm_config_name,
            "agent_class": self.agent_class,
            "max_iterations": self.max_iterations,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "repo_filter": self.repo_filter,
            "induction_trigger_count": self.induction_trigger_count,
            "max_workflows": self.max_workflows,
            "output_dir": self.output_dir,
            "checkpoint_interval": self.checkpoint_interval,
            "skip_evaluation": self.skip_evaluation,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AWMConfig":
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: str):
        """保存配置到文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AWMConfig":
        """从文件加载配置"""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


@dataclass
class AWMRunResult:
    """
    AWM运行结果
    """
    experiences: list = field(default_factory=list)
    workflows: list = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiences": [exp.to_dict() for exp in self.experiences],
            "workflows": [wf.to_dict() for wf in self.workflows],
            "statistics": self.statistics,
        }

    def save(self, path: str):
        """保存结果到文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
```

### 3.2 Task: 实现AWM Online Loop主控制器

**文件**: `evaluation/awm/loop.py`

```python
"""
AWM Online Learning Loop

The main controller that orchestrates the online learning process:
1. Process issues one by one
2. Evaluate results immediately
3. Store successful experiences
4. Trigger workflow induction periodically
5. Inject workflows into agent memory
"""

import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from evaluation.awm.config import AWMConfig, AWMRunResult
from evaluation.awm.experience import CodingExperience
from evaluation.awm.pipeline import OnlineEvaluationPipeline
from evaluation.awm.buffer import ExperienceBuffer
from evaluation.awm.induction import WorkflowInductionModule
from evaluation.awm.memory import MemoryManager
from evaluation.awm.workflow import Workflow
from evaluation.awm.types import DjangoIssue
from openhands.core.config import LLMConfig, get_llm_config_arg
from openhands.core.logger import openhands_logger as logger


class AWMOnlineLoop:
    """
    AWM Online Learning Loop - 完整实现

    Orchestrates the entire online learning process following the AWM paper.
    """

    def __init__(
        self,
        config: AWMConfig,
        llm_config: Optional[LLMConfig] = None,
    ):
        """
        初始化AWM Online Loop

        Args:
            config: AWM配置
            llm_config: LLM配置（如果不提供，从config_name加载）
        """
        self.config = config

        # 加载LLM配置
        if llm_config is None:
            llm_config = get_llm_config_arg(config.llm_config_name)
        self.llm_config = llm_config

        # 初始化组件
        self._init_components()

        # 统计
        self.total_processed = 0
        self.total_success = 0
        self.start_time = None

    def _init_components(self):
        """初始化所有组件"""
        # Pipeline
        self.pipeline = OnlineEvaluationPipeline(
            llm_config=self.llm_config,
            agent_class=self.config.agent_class,
            max_iterations=self.config.max_iterations,
            dataset_name=self.config.dataset_name,
            split=self.config.split,
            eval_output_dir=self.config.output_dir,
        )

        # Experience Buffer
        buffer_path = os.path.join(self.config.output_dir, "buffer.json")
        self.experience_buffer = ExperienceBuffer(
            induction_trigger_count=self.config.induction_trigger_count,
            persistence_path=buffer_path,
        )

        # Induction Module
        self.induction_module = WorkflowInductionModule(self.llm_config)

        # Memory Manager
        memory_path = os.path.join(self.config.output_dir, "memory.json")
        self.memory_manager = MemoryManager(
            max_workflows=self.config.max_workflows,
            persistence_path=memory_path,
        )

    def run(
        self,
        issues: List[DjangoIssue],
        resume_from: Optional[str] = None,
    ) -> AWMRunResult:
        """
        运行完整的online learning loop

        Args:
            issues: Django issues列表，按默认顺序处理
            resume_from: 可选的checkpoint路径，用于断点续跑

        Returns:
            AWMRunResult: 运行结果
        """
        self.start_time = datetime.now()

        # 恢复checkpoint
        if resume_from:
            self._restore_checkpoint(resume_from)

        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)

        # 保存配置
        self.config.save(os.path.join(self.config.output_dir, "config.json"))

        results = []

        for issue in issues:
            self.total_processed += 1
            self._print_progress(issue, len(issues))

            try:
                # Step 1: 获取当前augmented prompt（包含已学习的workflows）
                current_prompt = self.memory_manager.get_augmented_prompt()

                # Step 2: Process single sample
                experience = self.pipeline.process_single_sample(
                    instance_id=issue.instance_id,
                    problem_statement=issue.problem_statement,
                    workflow_memory=self.memory_manager.workflows,
                    custom_system_prompt=current_prompt if self.memory_manager.workflows else None,
                    skip_evaluation=self.config.skip_evaluation,
                )

                results.append(experience)

                # Step 3: 如果成功，加入buffer
                if experience.test_result == "PASS":
                    self.total_success += 1
                    self._print_success(experience)

                    should_induce = self.experience_buffer.add(experience)

                    # Step 4: 触发workflow induction
                    if should_induce:
                        self._run_induction()
                else:
                    self._print_failure(experience)

                # Step 5: 定期保存checkpoint
                if self.total_processed % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()

            except Exception as e:
                logger.error(f"Error processing {issue.instance_id}: {e}")
                # 创建失败的experience记录
                error_experience = CodingExperience(
                    instance_id=issue.instance_id,
                    problem_statement=issue.problem_statement,
                    diff_patch="",
                    history=[],
                    test_result="FAIL",
                    test_output=f"Error: {str(e)}",
                )
                results.append(error_experience)

        # 最终保存
        self._save_checkpoint()

        return AWMRunResult(
            experiences=results,
            workflows=self.memory_manager.workflows,
            statistics=self.get_statistics(),
        )

    def _run_induction(self):
        """运行workflow induction"""
        logger.info("\n" + "=" * 50)
        logger.info("Triggering Workflow Induction")
        logger.info("=" * 50)

        all_experiences = self.experience_buffer.get_all_experiences()

        # 归纳新workflows
        new_workflows = self.induction_module.induce(all_experiences)

        if new_workflows:
            logger.info(f"Induced {len(new_workflows)} new workflows:")
            for wf in new_workflows:
                logger.info(f"  - {wf.name}")

            # 添加到memory
            self.memory_manager.add_workflows(new_workflows)
        else:
            logger.info("No new workflows induced")

        logger.info("=" * 50 + "\n")

    def _save_checkpoint(self):
        """保存checkpoint用于断点续跑"""
        checkpoint_path = os.path.join(self.config.output_dir, "checkpoint.json")

        checkpoint = {
            "total_processed": self.total_processed,
            "total_success": self.total_success,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_checkpoint": datetime.now().isoformat(),
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def _restore_checkpoint(self, path: str):
        """恢复checkpoint"""
        if not os.path.exists(path):
            logger.warning(f"Checkpoint file not found: {path}")
            return

        try:
            with open(path, "r") as f:
                checkpoint = json.load(f)

            self.total_processed = checkpoint.get("total_processed", 0)
            self.total_success = checkpoint.get("total_success", 0)

            logger.info(
                f"Restored from checkpoint: "
                f"processed={self.total_processed}, success={self.total_success}"
            )

        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")

    def _print_progress(self, issue: DjangoIssue, total: int):
        """打印进度"""
        if self.config.verbose:
            print(f"\n{'=' * 60}")
            print(
                f"Processing issue {self.total_processed}/{total}: "
                f"{issue.instance_id}"
            )
            print(f"Success rate so far: {self.total_success}/{self.total_processed - 1}")
            print(f"Workflows learned: {self.memory_manager.get_workflow_count()}")

    def _print_success(self, experience: CodingExperience):
        """打印成功信息"""
        if self.config.verbose:
            print(f"✓ PASSED - {experience.instance_id}")

    def _print_failure(self, experience: CodingExperience):
        """打印失败信息"""
        if self.config.verbose:
            print(f"✗ FAILED - {experience.instance_id}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取运行统计"""
        elapsed = None
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()

        return {
            "total_processed": self.total_processed,
            "total_success": self.total_success,
            "success_rate": self.total_success / max(1, self.total_processed),
            "workflows_learned": self.memory_manager.get_workflow_count(),
            "buffer_size": len(self.experience_buffer.get_all_experiences()),
            "elapsed_seconds": elapsed,
            "config": self.config.to_dict(),
        }


def load_django_issues(
    dataset_name: str = "princeton-nlp/SWE-bench",
    split: str = "test",
    repo_filter: str = "django/django",
    limit: Optional[int] = None,
) -> List[DjangoIssue]:
    """
    从SWE-bench数据集加载Django相关的issues

    Args:
        dataset_name: 数据集名称
        split: 数据集split
        repo_filter: 仓库过滤器
        limit: 最大加载数量

    Returns:
        List[DjangoIssue]: Django issues列表
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)

    django_issues = []
    for item in dataset:
        if item["repo"] == repo_filter:
            django_issues.append(DjangoIssue(
                instance_id=item["instance_id"],
                problem_statement=item["problem_statement"],
                repo=item["repo"],
                base_commit=item.get("base_commit"),
                version=item.get("version"),
            ))

            if limit and len(django_issues) >= limit:
                break

    logger.info(f"Loaded {len(django_issues)} Django issues from {dataset_name}")
    return django_issues
```

### 3.3 Task: 实现命令行入口

**文件**: `evaluation/awm/cli.py`

```python
"""
Command Line Interface for AWM Online Learning

Provides a convenient way to run AWM experiments from the command line.
"""

import argparse
import sys
import os

from evaluation.awm.config import AWMConfig
from evaluation.awm.loop import AWMOnlineLoop, load_django_issues
from openhands.core.config import get_llm_config_arg
from openhands.core.logger import openhands_logger as logger


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
```

### 3.4 Task: 创建运行脚本

**文件**: `evaluation/awm/scripts/run_awm.sh`

```bash
#!/usr/bin/env bash
set -eo pipefail

# AWM Online Learning Runner Script
# Usage: ./run_awm.sh [llm_config] [options...]

LLM_CONFIG=${1:-"llm.eval"}
shift || true

echo "Running AWM Online Learning with LLM config: $LLM_CONFIG"

# Default options
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/evaluation_outputs/awm}"
REPO_FILTER="${REPO_FILTER:-django/django}"
INDUCTION_TRIGGER="${INDUCTION_TRIGGER:-10}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"

poetry run python -m evaluation.awm.cli \
    --llm-config "$LLM_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --repo-filter "$REPO_FILTER" \
    --induction-trigger "$INDUCTION_TRIGGER" \
    --max-iterations "$MAX_ITERATIONS" \
    "$@"
```

---

## 4. 完整使用示例

### 4.1 命令行运行

```bash
# 基本运行
./evaluation/awm/scripts/run_awm.sh llm.eval_gpt4

# 带参数运行
poetry run python -m evaluation.awm.cli \
    --llm-config llm.eval_gpt4 \
    --dataset princeton-nlp/SWE-bench_Lite \
    --repo-filter django/django \
    --limit 50 \
    --induction-trigger 5 \
    --output-dir evaluation/evaluation_outputs/awm_experiment_1

# 断点续跑
poetry run python -m evaluation.awm.cli \
    --llm-config llm.eval_gpt4 \
    --resume-from evaluation/evaluation_outputs/awm/checkpoint.json
```

### 4.2 Python API

```python
from evaluation.awm.config import AWMConfig
from evaluation.awm.loop import AWMOnlineLoop, load_django_issues

# 创建配置
config = AWMConfig(
    llm_config_name="llm.eval_gpt4",
    induction_trigger_count=5,
    max_workflows=30,
)

# 加载数据
issues = load_django_issues(limit=20)

# 运行
loop = AWMOnlineLoop(config)
result = loop.run(issues)

# 查看结果
print(f"Success rate: {result.statistics['success_rate']:.2%}")
print(f"Workflows: {len(result.workflows)}")
for wf in result.workflows:
    print(f"  - {wf.name}")
```

---

## 5. 验收标准

| 验收项 | 描述 | 状态 |
|-------|------|------|
| 1 | 可以通过CLI启动完整的AWM online learning流程 | ⬜ |
| 2 | 支持断点续跑 | ⬜ |
| 3 | 正确记录和输出统计信息 | ⬜ |
| 4 | 所有组件正确协同工作 | ⬜ |
| 5 | 端到端测试通过（处理至少10个issues） | ⬜ |

---

## 6. 输出文件结构

运行完成后，输出目录结构如下：

```
evaluation/evaluation_outputs/awm/
├── config.json           # 运行配置
├── checkpoint.json       # 断点续跑信息
├── buffer.json           # Experience buffer
├── memory.json           # Workflow memory
├── result.json           # 最终结果
└── logs/                 # 日志文件
    └── instance_*.log    # 每个instance的日志
```

---

## 7. 后续优化建议

1. **并行处理**: 当前是串行处理，可以考虑并行inference
2. **Workflow选择**: 目前是全量塞入prompt，可以增加检索机制
3. **增量Induction**: 每次只用新的experiences进行induction
4. **Workflow验证**: 添加workflow质量评估机制
5. **可视化**: 添加实时进度可视化

---

## 8. 完成！

恭喜！您已经完成了AWM迁移到OpenHands的全部四个阶段。

**总结**:
- Stage 1: 基础设施（单样本Inference/Evaluation + 数据结构）
- Stage 2: Pipeline + Buffer（在线处理流程 + 经验存储）
- Stage 3: Induction + Memory（Workflow归纳 + 记忆注入）
- Stage 4: Online Loop（完整集成 + CLI）

现在可以开始实际编码了！
