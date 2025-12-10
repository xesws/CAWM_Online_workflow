"""
Online Evaluation Pipeline for AWM

Provides a unified pipeline for single-sample inference and evaluation,
supporting online learning workflows.
"""

from typing import List, Optional, Any, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.runtime.base import Runtime
import pandas as pd

from evaluation.awm.experience import (
    CodingExperience,
    InferenceOutput,
    EvaluationResult,
    HistoryStep,
)
from evaluation.awm.single_inference import SingleInferenceRunner, load_instance_from_dataset
from evaluation.awm.single_evaluation import SingleEvaluationRunner
from openhands.core.config import LLMConfig
from openhands.core.logger import openhands_logger as logger


class OnlineEvaluationPipeline:
    """
    支持单样本的即时 inference → evaluation 流程

    Usage:
        pipeline = OnlineEvaluationPipeline(llm_config)
        experience = pipeline.process_single_sample(
            instance_id="django__django-11039",
            problem_statement="...",
        )
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        agent_class: str = "CodeActAgent",
        max_iterations: int = 100,
        dataset_name: str = "princeton-nlp/SWE-bench",
        split: str = "test",
        eval_output_dir: str = "evaluation/evaluation_outputs/awm",
    ):
        """
        初始化Pipeline

        Args:
            llm_config: LLM配置
            agent_class: Agent类名
            max_iterations: Agent最大迭代次数
            dataset_name: SWE-bench数据集名称
            split: 数据集split
            eval_output_dir: 输出目录
        """
        self.llm_config = llm_config
        self.agent_class = agent_class
        self.max_iterations = max_iterations
        self.dataset_name = dataset_name
        self.split = split
        self.eval_output_dir = eval_output_dir

        # 初始化runners
        self.inference_runner = SingleInferenceRunner(
            llm_config=llm_config,
            agent_class=agent_class,
            max_iterations=max_iterations,
            eval_output_dir=eval_output_dir,
        )

        self.evaluation_runner = SingleEvaluationRunner(
            dataset_name=dataset_name,
            split=split,
        )

    def process_single_sample(
        self,
        instance_id: str,
        problem_statement: Optional[str] = None,
        workflow_memory: Optional[List[Any]] = None,
        custom_system_prompt: Optional[str] = None,
        skip_evaluation: bool = False,
    ) -> CodingExperience:
        """
        完整处理单个sample: inference → evaluation → experience

        Args:
            instance_id: SWE-bench instance ID
            problem_statement: 问题描述（可选，如果不提供会从数据集加载）
            workflow_memory: 可选的workflow memory列表
            custom_system_prompt: 可选的自定义system prompt
            skip_evaluation: 是否跳过evaluation步骤（用于调试）

        Returns:
            CodingExperience: 完整的experience记录
        """
        logger.info(f"Processing instance: {instance_id}")

        # Step 1: 加载instance数据
        instance = load_instance_from_dataset(
            instance_id,
            self.dataset_name,
            self.split,
        )

        if problem_statement is None:
            problem_statement = instance.get("problem_statement", "")

        # Step 2: Run inference
        # Keep runtime alive if we're going to run evaluation (for reuse)
        keep_runtime = not skip_evaluation
        runtime = None  # Initialize to None for finally block safety

        try:
            logger.info(f"Running inference for {instance_id}...")
            inference_output, runtime = self._run_inference(
                instance,
                workflow_memory,
                custom_system_prompt,
                keep_runtime=keep_runtime,
            )
            logger.info(f"Inference complete. Patch length: {len(inference_output.diff_patch)}")

            # Step 3: Run evaluation (可选)
            if skip_evaluation:
                eval_result = EvaluationResult(
                    instance_id=instance_id,
                    resolved=False,
                    test_output="Evaluation skipped",
                    report={"skipped": True},
                )
            else:
                logger.info(f"Running evaluation for {instance_id} (reusing runtime)...")
                eval_result = self._run_evaluation(
                    instance_id,
                    inference_output.diff_patch,
                    runtime=runtime,  # Pass runtime for reuse
                )
                logger.info(f"Evaluation complete. Resolved: {eval_result.resolved}")
        finally:
            # Always close runtime after both phases complete
            if runtime is not None:
                runtime.close()
                logger.info("Runtime closed after evaluation")

        # Step 4: Construct experience
        experience = self._construct_experience(
            inference_output,
            eval_result,
            problem_statement,
        )

        return experience

    def _run_inference(
        self,
        instance: pd.Series,
        workflow_memory: Optional[List[Any]] = None,
        custom_system_prompt: Optional[str] = None,
        keep_runtime: bool = False,
    ) -> Tuple[InferenceOutput, Optional["Runtime"]]:
        """运行inference阶段"""
        return self.inference_runner.run(
            instance=instance,
            workflow_memory=workflow_memory,
            custom_system_prompt=custom_system_prompt,
            keep_runtime=keep_runtime,
        )

    def _run_evaluation(
        self,
        instance_id: str,
        diff_patch: str,
        runtime: Optional["Runtime"] = None,
    ) -> EvaluationResult:
        """运行evaluation阶段"""
        return self.evaluation_runner.run(
            instance_id=instance_id,
            diff_patch=diff_patch,
            runtime=runtime,
        )

    def _construct_experience(
        self,
        inference_output: InferenceOutput,
        eval_result: EvaluationResult,
        problem_statement: str,
    ) -> CodingExperience:
        """构建完整的experience"""
        return CodingExperience(
            instance_id=inference_output.instance_id,
            problem_statement=problem_statement,
            diff_patch=inference_output.diff_patch,
            history=inference_output.history,
            test_result="PASS" if eval_result.resolved else "FAIL",
            test_output=eval_result.test_output,
            model_name=inference_output.metadata.get("model_name", ""),
        )

    def run_inference_only(
        self,
        instance_id: str,
        workflow_memory: Optional[List[Any]] = None,
    ) -> InferenceOutput:
        """
        只运行inference，不进行evaluation

        Args:
            instance_id: Instance ID
            workflow_memory: 可选的workflow memory

        Returns:
            InferenceOutput: Inference输出
        """
        instance = load_instance_from_dataset(
            instance_id,
            self.dataset_name,
            self.split,
        )
        # Don't keep runtime since we're not doing evaluation
        inference_output, _ = self._run_inference(instance, workflow_memory, keep_runtime=False)
        return inference_output

    def run_evaluation_only(
        self,
        instance_id: str,
        diff_patch: str,
    ) -> EvaluationResult:
        """
        只运行evaluation

        Args:
            instance_id: Instance ID
            diff_patch: 要评估的patch

        Returns:
            EvaluationResult: 评估结果
        """
        return self._run_evaluation(instance_id, diff_patch)
