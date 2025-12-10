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
import logging
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


class LogTruncatingFilter(logging.Filter):
    """
    Filter to truncate verbose action and observation logs in console output.
    Truncates both ACTION and OBSERVATION messages to reduce terminal clutter.
    """

    # Configurable truncation limits
    ACTION_MAX_CHARS = 50
    OBSERVATION_MAX_CHARS = 50

    def filter(self, record: logging.LogRecord) -> bool:
        msg_type = getattr(record, 'msg_type', '')
        msg_str = str(record.msg)

        if msg_type == 'ACTION' and len(msg_str) > self.ACTION_MAX_CHARS:
            record.msg = msg_str[:self.ACTION_MAX_CHARS] + "... [truncated]"
        elif msg_type == 'OBSERVATION' and len(msg_str) > self.OBSERVATION_MAX_CHARS:
            record.msg = msg_str[:self.OBSERVATION_MAX_CHARS] + "... [truncated]"

        return True


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

        # Setup log truncation
        self._setup_logging()

        # 初始化组件
        self._init_components()

        # 统计
        self.total_processed = 0
        self.total_success = 0
        self.start_time = None

    def _setup_logging(self):
        """Setup logging filters"""
        # Add truncation filter to global logger
        # Note: This affects global state, but acceptable for the runner process
        logger.addFilter(LogTruncatingFilter())

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
            truncation_limit=self.config.truncation_limit,
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

        # 初始化实时结果日志 (results.jsonl)
        results_path = os.path.join(self.config.output_dir, "results.jsonl")
        # 如果是从头开始，清空旧文件；如果是resume，保留（append模式始终有效，但需要避免重复？）
        # 简单起见，如果是 resume_from，我们假设用户负责清理或我们只追加
        if not resume_from:
            # 创建/清空文件
            with open(results_path, "w") as f:
                pass

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

                # 实时写入 results.jsonl
                with open(results_path, "a") as f:
                    f.write(json.dumps(experience.to_dict()) + "\n")

                # Step 3: 如果成功，加入buffer
                if experience.test_result == "PASS":
                    self.total_success += 1
                    self._print_success(experience)

                    self.experience_buffer.add(experience)

                    # Step 4: 触发增量 workflow induction
                    self._run_induction(experience)
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
                
                # 记录错误到 jsonl
                with open(results_path, "a") as f:
                    f.write(json.dumps(error_experience.to_dict()) + "\n")

        # 最终保存
        self._save_checkpoint()

        return AWMRunResult(
            experiences=results,
            workflows=self.memory_manager.workflows,
            statistics=self.get_statistics(),
        )

    def _run_induction(self, latest_experience: CodingExperience):
        """运行增量 workflow induction"""
        logger.info("\n" + "=" * 50)
        logger.info("Triggering Incremental Workflow Induction")
        logger.info("=" * 50)

        # 只使用最新的单个 experience 进行增量归纳
        updated_workflows = self.induction_module.induce_from_single(
            experience=latest_experience,
            existing_workflows=self.memory_manager.workflows,
        )

        # 更新 memory
        self.memory_manager.workflows = updated_workflows
        # 持久化 memory
        self.memory_manager.save()

        logger.info(f"Updated workflows: {len(updated_workflows)} total")
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
            print(f"\n" + "=" * 60)
            print(
                f"Processing issue {self.total_processed}/{total}: "
                f"{issue.instance_id}"
            )
            print(f"Success rate so far: {self.total_success}/{max(1, self.total_processed - 1)}")
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
        # Empty repo_filter means load all repos
        if not repo_filter or item["repo"] == repo_filter:
            django_issues.append(DjangoIssue(
                instance_id=item["instance_id"],
                problem_statement=item["problem_statement"],
                repo=item["repo"],
                base_commit=item.get("base_commit"),
                version=item.get("version"),
            ))

            if limit and len(django_issues) >= limit:
                break

    repo_desc = repo_filter if repo_filter else "all repos"
    logger.info(f"Loaded {len(django_issues)} issues from {dataset_name} ({repo_desc})")
    return django_issues
