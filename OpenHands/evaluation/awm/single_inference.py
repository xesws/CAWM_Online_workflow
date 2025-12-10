"""
Single Sample Inference Module for AWM

Provides the ability to run inference on a single SWE-bench instance,
independent of the batch evaluation pipeline.
"""

import asyncio
import os
from typing import List, Optional, Any, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.runtime.base import Runtime

import pandas as pd

from evaluation.awm.experience import InferenceOutput, HistoryStep
from evaluation.awm.history_parser import parse_openhands_history
from evaluation.utils.shared import (
    EvalMetadata,
    make_metadata,
    codeact_user_response,
)
from openhands.core.config import LLMConfig, get_llm_config_arg
from openhands.core.logger import openhands_logger as logger
from openhands.events.serialization.event import event_to_dict


class SingleInferenceRunner:
    """
    单样本Inference执行器

    通过import复用现有run_infer.py的逻辑，不修改原文件。
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        agent_class: str = "CodeActAgent",
        max_iterations: int = 100,
        eval_output_dir: str = "evaluation/evaluation_outputs/awm",
    ):
        """
        初始化Inference Runner

        Args:
            llm_config: LLM配置
            agent_class: Agent类名
            max_iterations: 最大迭代次数
            eval_output_dir: 输出目录
        """
        self.llm_config = llm_config
        self.agent_class = agent_class
        self.max_iterations = max_iterations
        self.eval_output_dir = eval_output_dir

    def run(
        self,
        instance: pd.Series,
        workflow_memory: Optional[List[Any]] = None,
        custom_system_prompt: Optional[str] = None,
        keep_runtime: bool = False,
    ) -> Tuple[InferenceOutput, Optional["Runtime"]]:
        """
        对单个instance进行inference

        Args:
            instance: SWE-bench instance (pandas Series)
            workflow_memory: 可选的workflow memory列表
            custom_system_prompt: 可选的自定义system prompt
            keep_runtime: If True, don't close runtime after inference.
                         Caller is responsible for closing it.

        Returns:
            Tuple of (InferenceOutput, Optional[Runtime])
            - InferenceOutput: 包含 diff_patch, history, metadata
            - Runtime: The runtime instance if keep_runtime=True, else None
        """
        # 延迟导入，避免循环依赖
        from evaluation.benchmarks.swe_bench.run_infer import (
            get_config,
            initialize_runtime,
            complete_runtime,
            get_instruction,
            AGENT_CLS_TO_FAKE_USER_RESPONSE_FN,
            set_dataset_type,
        )
        from openhands.core.main import create_runtime, run_controller
        from openhands.utils.async_utils import call_async_from_sync
        from openhands.controller.state.state import State

        # 设置dataset类型
        set_dataset_type("princeton-nlp/SWE-bench")

        # 创建metadata
        metadata = self._create_metadata(instance)

        # 获取配置
        config = get_config(instance, metadata)

        # 创建runtime
        runtime = create_runtime(config)
        try:
            call_async_from_sync(runtime.connect)
        except Exception:
            # If connect fails, close the runtime to avoid leaking file descriptors
            runtime.close()
            raise

        try:
            # 初始化runtime
            initialize_runtime(runtime, instance, metadata)

            # 获取instruction
            message_action = get_instruction(instance, metadata)

            # 注入 Workflow Memory 到 instruction (AWM "M + W" 方法)
            if custom_system_prompt:
                message_action = self._inject_workflow_to_instruction(
                    message_action,
                    custom_system_prompt
                )

            # 运行controller
            state: State | None = asyncio.run(
                run_controller(
                    config=config,
                    initial_user_action=message_action,
                    runtime=runtime,
                    fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get(
                        self.agent_class, codeact_user_response
                    ),
                )
            )

            # 获取git patch
            return_val = complete_runtime(runtime, instance)
            git_patch = return_val.get("git_patch", "")

            # 解析history
            if state is not None:
                raw_history = [event_to_dict(event) for event in state.history]
                parsed_history = parse_openhands_history(raw_history)
            else:
                raw_history = []
                parsed_history = []

            inference_output = InferenceOutput(
                instance_id=instance["instance_id"],
                diff_patch=git_patch,
                history=parsed_history,
                problem_statement=instance.get("problem_statement", ""),
                metadata={
                    "model_name": self.llm_config.model,
                    "max_iterations": self.max_iterations,
                    "agent_class": self.agent_class,
                    "raw_history": raw_history,  # 保留原始history
                },
            )

            # Return runtime for reuse if requested
            if keep_runtime:
                return inference_output, runtime
            else:
                runtime.close()
                return inference_output, None

        except Exception:
            # On error, always close runtime
            runtime.close()
            raise

    def _create_metadata(self, instance: pd.Series) -> EvalMetadata:
        """创建EvalMetadata"""
        return make_metadata(
            llm_config=self.llm_config,
            dataset_name="awm_single_inference",
            agent_class=self.agent_class,
            max_iterations=self.max_iterations,
            eval_note="awm",
            eval_output_dir=self.eval_output_dir,
            details={"mode": "swe"},
        )

    def _inject_workflow_to_instruction(
        self,
        message_action,
        workflow_prompt: str,
    ):
        """
        将 workflow memory 注入到 instruction message 中

        遵循 AWM 论文的 "M + W" 方法：
        - M: 原始任务描述 (instruction)
        - W: 已学习的 workflows

        Args:
            message_action: 原始的 MessageAction
            workflow_prompt: 已学习的 workflows 格式化字符串

        Returns:
            新的 MessageAction，包含注入的 workflows
        """
        from openhands.events.action import MessageAction

        # 构建增强后的 instruction
        augmented_content = f"""## Learned Workflows (Use these patterns when applicable)

{workflow_prompt}

---

## Your Task

{message_action.content}
"""

        logger.info("Workflow memory injected into instruction")

        # 返回新的 MessageAction
        return MessageAction(
            content=augmented_content,
            image_urls=message_action.image_urls if hasattr(message_action, 'image_urls') else None,
        )


def load_instance_from_dataset(
    instance_id: str,
    dataset_name: str = "princeton-nlp/SWE-bench",
    split: str = "test",
) -> pd.Series:
    """
    从SWE-bench数据集加载单个instance

    Args:
        instance_id: Instance ID
        dataset_name: 数据集名称
        split: 数据集split

    Returns:
        pd.Series: Instance数据
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    df = dataset.to_pandas()

    instance = df[df["instance_id"] == instance_id]
    if len(instance) == 0:
        raise ValueError(f"Instance {instance_id} not found in {dataset_name}/{split}")

    return instance.iloc[0]
