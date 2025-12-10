# Stage 1: 基础设施改造 (P0)

> **优先级**: P0 (最高)
> **复杂度**: 高
> **依赖**: 无

---

## 1. 核心目标

将现有的batch-based evaluation改造为支持单样本处理，为AWM online learning奠定基础。

**关键原则**: 完全独立模块，通过import复用现有代码，不修改现有文件。

---

## 2. 文件结构

```
evaluation/awm/                    # 全新独立模块
├── __init__.py
├── experience.py                  # Experience数据结构定义
├── types.py                       # 共享类型定义
├── single_inference.py            # 单样本Inference (复用run_infer.py逻辑)
├── single_evaluation.py           # 单样本Evaluation (复用eval_infer.py逻辑)
└── history_parser.py              # History解析工具
```

---

## 3. 详细任务

### 3.1 Task: 创建AWM模块初始化文件

**文件**: `evaluation/awm/__init__.py`

```python
"""
AWM (Agentic Working Memory) Module for OpenHands

This module implements online learning capabilities for the OpenHands agent,
allowing it to learn from successful experiences and improve over time.
"""

from evaluation.awm.experience import (
    CodingExperience,
    HistoryStep,
    InferenceOutput,
    EvaluationResult,
)
from evaluation.awm.single_inference import SingleInferenceRunner
from evaluation.awm.single_evaluation import SingleEvaluationRunner

__all__ = [
    "CodingExperience",
    "HistoryStep",
    "InferenceOutput",
    "EvaluationResult",
    "SingleInferenceRunner",
    "SingleEvaluationRunner",
]
```

### 3.2 Task: 定义Experience数据结构

**文件**: `evaluation/awm/experience.py`

```python
"""
Experience Data Structures for AWM

Defines the core data structures for storing and processing coding experiences.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal, Any, Dict
from datetime import datetime
import json


@dataclass
class HistoryStep:
    """
    Agent单步调用记录，对应OpenHands的history格式
    """
    step_id: int
    observation: str               # 当前观察到的状态
    thought: str                   # Agent的思考过程
    action: str                    # 采取的action
    action_type: str               # action类型: edit_file, run_command, search, etc.

    # 可选的详细信息
    file_path: Optional[str] = None       # 如果是文件操作，记录路径
    command: Optional[str] = None         # 如果是命令执行，记录命令

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryStep":
        return cls(**data)


@dataclass
class CodingExperience:
    """
    完整的coding experience记录

    严格按照SWE-bench/OpenHands的inference输出格式定义
    """

    # ===== 问题描述 =====
    instance_id: str               # SWE-bench instance ID, e.g., "django__django-11039"
    problem_statement: str         # GitHub issue的描述

    # ===== Agent产出 (Inference阶段输出) =====
    diff_patch: str                # Agent给出的solution (difference patch)
    history: List[HistoryStep]     # Agent的完整调用轨迹

    # ===== 测试结果 (Evaluation阶段输出) =====
    test_result: Literal["PASS", "FAIL"]  # 测试通过与否
    test_output: Optional[str] = None     # 测试的详细输出（可选）

    # ===== 元信息 =====
    task_type: Optional[str] = None       # 任务类型标签 (用于分组)
    timestamp: datetime = field(default_factory=datetime.now)
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["history"] = [step.to_dict() for step in self.history]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodingExperience":
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["history"] = [HistoryStep.from_dict(h) for h in data["history"]]
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "CodingExperience":
        return cls.from_dict(json.loads(json_str))


@dataclass
class InferenceOutput:
    """
    Inference阶段的输出
    """
    instance_id: str
    diff_patch: str
    history: List[HistoryStep]
    problem_statement: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["history"] = [step.to_dict() for step in self.history]
        return data


@dataclass
class EvaluationResult:
    """
    Evaluation阶段的结果
    """
    instance_id: str
    resolved: bool
    test_output: str
    report: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
```

### 3.3 Task: 定义共享类型

**文件**: `evaluation/awm/types.py`

```python
"""
Shared Type Definitions for AWM Module
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DjangoIssue:
    """Django issue数据结构"""
    instance_id: str           # e.g., "django__django-11039"
    problem_statement: str     # GitHub issue描述
    repo: str = "django/django"
    base_commit: Optional[str] = None

    # 其他可选的SWE-bench元数据
    hints_text: Optional[str] = None
    created_at: Optional[str] = None
    version: Optional[str] = None
```

### 3.4 Task: 实现History解析工具

**文件**: `evaluation/awm/history_parser.py`

```python
"""
History Parser for OpenHands Events

Converts OpenHands history format to AWM HistoryStep format.
"""

from typing import List, Dict, Any, Optional
from evaluation.awm.experience import HistoryStep


def parse_openhands_history(history: List[Dict[str, Any]]) -> List[HistoryStep]:
    """
    将OpenHands的history格式转换为HistoryStep格式。

    OpenHands history格式来自 event_to_dict(event)，包含：
    - action类型 (CmdRunAction, FileEditAction, etc.)
    - observation内容
    - thought/reasoning

    Args:
        history: OpenHands事件历史列表

    Returns:
        List[HistoryStep]: 转换后的步骤列表
    """
    steps = []
    step_id = 0

    for i, event in enumerate(history):
        if not isinstance(event, dict):
            continue

        # 检查是否是action事件
        action_type = _extract_action_type(event)
        if action_type is None:
            continue

        step_id += 1

        # 提取各字段
        thought = _extract_thought(event)
        action = _extract_action_content(event)
        observation = _extract_observation(history, i)
        file_path = _extract_file_path(event)
        command = _extract_command(event)

        step = HistoryStep(
            step_id=step_id,
            observation=observation,
            thought=thought,
            action=action,
            action_type=action_type,
            file_path=file_path,
            command=command,
        )
        steps.append(step)

    return steps


def _extract_action_type(event: Dict[str, Any]) -> Optional[str]:
    """提取action类型"""
    # OpenHands事件格式
    if "action" in event:
        action_name = event.get("action", "")

        # 映射到简化的类型名
        type_mapping = {
            "run": "run_command",
            "run_ipython": "run_ipython",
            "read": "read_file",
            "write": "write_file",
            "edit": "edit_file",
            "browse": "browse",
            "finish": "finish",
            "message": "message",
            "think": "think",
        }

        for key, mapped_type in type_mapping.items():
            if key in action_name.lower():
                return mapped_type

        return action_name

    return None


def _extract_thought(event: Dict[str, Any]) -> str:
    """提取思考内容"""
    # 尝试多个可能的字段
    for field in ["thought", "reasoning", "args.thought", "message"]:
        if "." in field:
            parts = field.split(".")
            value = event
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break
            if value:
                return str(value)
        elif field in event:
            return str(event[field])

    return ""


def _extract_action_content(event: Dict[str, Any]) -> str:
    """提取action内容"""
    if "args" in event:
        args = event["args"]
        if isinstance(args, dict):
            # 根据不同action类型提取相关内容
            if "command" in args:
                return args["command"]
            if "code" in args:
                return args["code"]
            if "path" in args:
                return f"path: {args['path']}"
            if "content" in args:
                return args.get("content", "")[:500]  # 截断
        return str(args)[:500]

    return str(event.get("action", ""))


def _extract_observation(history: List[Dict[str, Any]], action_index: int) -> str:
    """提取对应的observation"""
    # 查找action之后的第一个observation
    for i in range(action_index + 1, min(action_index + 3, len(history))):
        event = history[i]
        if isinstance(event, dict) and "observation" in event:
            obs = event.get("content", event.get("observation", ""))
            return str(obs)[:1000]  # 截断

    return ""


def _extract_file_path(event: Dict[str, Any]) -> Optional[str]:
    """提取文件路径"""
    args = event.get("args", {})
    if isinstance(args, dict):
        return args.get("path") or args.get("file_path")
    return None


def _extract_command(event: Dict[str, Any]) -> Optional[str]:
    """提取命令"""
    args = event.get("args", {})
    if isinstance(args, dict):
        return args.get("command") or args.get("code")
    return None


def format_history_for_display(steps: List[HistoryStep], max_steps: int = 20) -> str:
    """
    格式化history用于显示

    Args:
        steps: 步骤列表
        max_steps: 最大显示步骤数

    Returns:
        格式化的字符串
    """
    lines = []
    display_steps = steps[:max_steps]

    for step in display_steps:
        lines.append(f"Step {step.step_id}:")
        if step.thought:
            lines.append(f"  Thought: {step.thought[:200]}...")
        lines.append(f"  Action [{step.action_type}]: {step.action[:100]}...")
        if step.observation:
            lines.append(f"  Observation: {step.observation[:100]}...")
        lines.append("")

    if len(steps) > max_steps:
        lines.append(f"... ({len(steps) - max_steps} more steps)")

    return "\n".join(lines)
```

### 3.5 Task: 实现单样本Inference模块

**文件**: `evaluation/awm/single_inference.py`

```python
"""
Single Sample Inference Module for AWM

Provides the ability to run inference on a single SWE-bench instance,
independent of the batch evaluation pipeline.
"""

import asyncio
import os
from typing import List, Optional, Any, Dict

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
    ) -> InferenceOutput:
        """
        对单个instance进行inference

        Args:
            instance: SWE-bench instance (pandas Series)
            workflow_memory: 可选的workflow memory列表
            custom_system_prompt: 可选的自定义system prompt

        Returns:
            InferenceOutput: 包含 diff_patch, history, metadata
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

        # 如果有custom system prompt，需要特殊处理
        if custom_system_prompt:
            metadata = self._inject_custom_prompt(metadata, custom_system_prompt)

        # 获取配置
        config = get_config(instance, metadata)

        # 创建runtime
        runtime = create_runtime(config)
        call_async_from_sync(runtime.connect)

        try:
            # 初始化runtime
            initialize_runtime(runtime, instance, metadata)

            # 获取instruction
            message_action = get_instruction(instance, metadata)

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

            return InferenceOutput(
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

        finally:
            runtime.close()

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

    def _inject_custom_prompt(
        self,
        metadata: EvalMetadata,
        custom_prompt: str
    ) -> EvalMetadata:
        """
        注入自定义system prompt

        Note: 这需要通过环境变量或配置来实现，
        因为system prompt是在agent初始化时加载的。
        """
        # TODO: 实现custom prompt注入
        # 可能的方案：
        # 1. 使用自定义的system_prompt.j2模板
        # 2. 通过AgentConfig设置
        # 3. 环境变量 INSTRUCTION_TEMPLATE_NAME
        logger.warning(
            "Custom system prompt injection not yet implemented. "
            "Using default system prompt."
        )
        return metadata


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
```

### 3.6 Task: 实现单样本Evaluation模块

**文件**: `evaluation/awm/single_evaluation.py`

```python
"""
Single Sample Evaluation Module for AWM

Provides the ability to evaluate a single patch against SWE-bench tests,
independent of the batch evaluation pipeline.
"""

import os
import tempfile
import time
from typing import Optional, Dict, Any

import pandas as pd

from evaluation.awm.experience import EvaluationResult
from openhands.core.logger import openhands_logger as logger


class SingleEvaluationRunner:
    """
    单样本Evaluation执行器

    通过import复用现有eval_infer.py的逻辑，不修改原文件。
    """

    def __init__(
        self,
        dataset_name: str = "princeton-nlp/SWE-bench",
        split: str = "test",
    ):
        """
        初始化Evaluation Runner

        Args:
            dataset_name: SWE-bench数据集名称
            split: 数据集split
        """
        self.dataset_name = dataset_name
        self.split = split
        self._instance_cache: Dict[str, Any] = {}
        self._test_spec_cache: Dict[str, Any] = {}
        self._load_harness()

    def _load_harness(self):
        """加载SWE-bench harness"""
        try:
            from swebench.harness.grading import get_eval_report
            from swebench.harness.run_evaluation import (
                APPLY_PATCH_FAIL,
                APPLY_PATCH_PASS,
            )
            from swebench.harness.test_spec.test_spec import (
                SWEbenchInstance,
                make_test_spec,
            )
            from swebench.harness.utils import load_swebench_dataset

            self.get_eval_report = get_eval_report
            self.APPLY_PATCH_FAIL = APPLY_PATCH_FAIL
            self.APPLY_PATCH_PASS = APPLY_PATCH_PASS
            self.make_test_spec = make_test_spec
            self.load_swebench_dataset = load_swebench_dataset

            # 预加载数据集
            self._load_dataset()

        except ImportError as e:
            logger.error(f"Failed to import swebench harness: {e}")
            raise

    def _load_dataset(self):
        """加载数据集到缓存"""
        full_dataset = self.load_swebench_dataset(self.dataset_name, self.split)
        for instance in full_dataset:
            self._instance_cache[instance["instance_id"]] = instance

    def run(
        self,
        instance_id: str,
        diff_patch: str,
        timeout: int = 1800,
    ) -> EvaluationResult:
        """
        对单个instance的patch进行evaluation

        Args:
            instance_id: Instance ID
            diff_patch: 要评估的diff patch
            timeout: 测试超时时间（秒）

        Returns:
            EvaluationResult: 评估结果
        """
        # 延迟导入
        from evaluation.benchmarks.swe_bench.eval_infer import (
            process_git_patch,
            get_config,
        )
        from evaluation.benchmarks.swe_bench.run_infer import get_instance_docker_image
        from evaluation.utils.shared import get_default_sandbox_config_for_eval
        from openhands.core.main import create_runtime
        from openhands.events.action import CmdRunAction
        from openhands.events.observation import CmdOutputObservation
        from openhands.utils.async_utils import call_async_from_sync

        # 处理patch
        processed_patch = process_git_patch(diff_patch)

        if not processed_patch.strip():
            return EvaluationResult(
                instance_id=instance_id,
                resolved=False,
                test_output="",
                report={"empty_generation": True, "resolved": False},
            )

        # 获取instance信息
        instance = self._get_instance(instance_id)
        test_spec = self._get_test_spec(instance_id, instance)

        # 准备Docker环境
        base_container_image = get_instance_docker_image(instance_id)
        sandbox_config = get_default_sandbox_config_for_eval()
        sandbox_config.base_container_image = base_container_image

        # 创建runtime config (minimal)
        from openhands.core.config import OpenHandsConfig, SandboxConfig
        config = OpenHandsConfig(
            sandbox=sandbox_config,
            runtime=os.environ.get("RUNTIME", "docker"),
        )

        # 创建runtime
        runtime = create_runtime(config)
        call_async_from_sync(runtime.connect)

        try:
            # 复制patch和eval脚本到容器
            with tempfile.TemporaryDirectory() as temp_dir:
                # 写patch文件
                patch_file = os.path.join(temp_dir, "patch.diff")
                with open(patch_file, "w") as f:
                    f.write(processed_patch)
                runtime.copy_to(patch_file, "/tmp")

                # 写eval脚本
                eval_script = os.path.join(temp_dir, "eval.sh")
                with open(eval_script, "w") as f:
                    f.write(test_spec.eval_script)
                runtime.copy_to(eval_script, "/tmp")

            # 设置执行权限
            action = CmdRunAction(command="chmod +x /tmp/eval.sh")
            action.set_hard_timeout(600)
            obs = runtime.run_action(action)

            # 应用patch
            apply_cmd = (
                "cd /testbed && "
                "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                "(echo 'Trying patch command...' && "
                "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                "echo 'APPLY_PATCH_FAIL')))"
            )
            action = CmdRunAction(command=apply_cmd)
            action.set_hard_timeout(600)
            obs = runtime.run_action(action)

            apply_output = obs.content if isinstance(obs, CmdOutputObservation) else ""

            if "APPLY_PATCH_FAIL" in apply_output:
                return EvaluationResult(
                    instance_id=instance_id,
                    resolved=False,
                    test_output=apply_output,
                    report={"failed_apply_patch": True, "resolved": False},
                )

            # 运行测试
            log_file = "/tmp/eval_output.log"
            action = CmdRunAction(command=f"/tmp/eval.sh > {log_file} 2>&1 & echo $!")
            action.set_hard_timeout(300)
            obs = runtime.run_action(action)

            if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
                pid = obs.content.split()[-1].strip()

                # 等待测试完成
                start_time = time.time()
                while True:
                    if time.time() - start_time > timeout:
                        return EvaluationResult(
                            instance_id=instance_id,
                            resolved=False,
                            test_output="Test timeout",
                            report={"test_timeout": True, "resolved": False},
                        )

                    check_action = CmdRunAction(command=f"ps -p {pid} > /dev/null; echo $?")
                    check_action.set_hard_timeout(300)
                    check_obs = runtime.run_action(check_action)

                    if (isinstance(check_obs, CmdOutputObservation) and
                        check_obs.content.split()[-1].strip() == "1"):
                        break

                    time.sleep(30)

                # 读取测试输出
                cat_action = CmdRunAction(command=f"cat {log_file}")
                cat_action.set_hard_timeout(300)
                cat_obs = runtime.run_action(cat_action)

                if isinstance(cat_obs, CmdOutputObservation) and cat_obs.exit_code == 0:
                    test_output = cat_obs.content

                    # 评估结果
                    resolved = self._grade_result(
                        instance_id, processed_patch, test_output, test_spec
                    )

                    return EvaluationResult(
                        instance_id=instance_id,
                        resolved=resolved,
                        test_output=test_output,
                        report={"resolved": resolved},
                    )

            return EvaluationResult(
                instance_id=instance_id,
                resolved=False,
                test_output="Evaluation failed",
                report={"error_eval": True, "resolved": False},
            )

        finally:
            runtime.close()

    def _get_instance(self, instance_id: str) -> Dict[str, Any]:
        """获取instance数据"""
        if instance_id not in self._instance_cache:
            raise ValueError(f"Instance {instance_id} not found in dataset")
        return self._instance_cache[instance_id]

    def _get_test_spec(self, instance_id: str, instance: Dict[str, Any]) -> Any:
        """获取或创建test_spec"""
        if instance_id not in self._test_spec_cache:
            self._test_spec_cache[instance_id] = self.make_test_spec(instance)
        return self._test_spec_cache[instance_id]

    def _grade_result(
        self,
        instance_id: str,
        model_patch: str,
        test_output: str,
        test_spec: Any,
    ) -> bool:
        """评估测试结果"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                log_dir = os.path.join(temp_dir, "logs", instance_id.lower())
                os.makedirs(log_dir, exist_ok=True)

                test_output_path = os.path.join(log_dir, "test_output.txt")
                with open(test_output_path, "w") as f:
                    f.write(test_output)

                report = self.get_eval_report(
                    test_spec=test_spec,
                    prediction={
                        "model_patch": model_patch,
                        "instance_id": instance_id,
                    },
                    include_tests_status=True,
                    test_log_path=test_output_path,
                )

                return report.get(instance_id, {}).get("resolved", False)

        except Exception as e:
            logger.error(f"Error grading result for {instance_id}: {e}")
            return False
```

---

## 4. 验收标准

| 验收项 | 描述 | 状态 |
|-------|------|------|
| 1 | `SingleInferenceRunner.run()` 可以处理单个instance并返回 `InferenceOutput` | ⬜ |
| 2 | `SingleEvaluationRunner.run()` 可以评估单个patch并返回 `EvaluationResult` | ⬜ |
| 3 | `CodingExperience` 可以正确序列化/反序列化 (JSON) | ⬜ |
| 4 | `parse_openhands_history()` 可以正确解析OpenHands history | ⬜ |
| 5 | 现有batch evaluation功能不受影响（向后兼容） | ⬜ |

---

## 5. 依赖的现有文件 (通过import复用)

| 文件 | 复用的函数/类 |
|------|-------------|
| `evaluation/benchmarks/swe_bench/run_infer.py` | `get_config`, `initialize_runtime`, `complete_runtime`, `get_instruction` |
| `evaluation/benchmarks/swe_bench/eval_infer.py` | `process_git_patch`, `get_config` |
| `evaluation/utils/shared.py` | `EvalMetadata`, `make_metadata`, `codeact_user_response` |
| `openhands/core/main.py` | `create_runtime`, `run_controller` |
| `swebench/harness/*` | `get_eval_report`, `make_test_spec`, etc. |

---

## 6. 下一步

完成Stage 1后，进入 [Stage 2: Online Pipeline + Experience Buffer](./stage-2-pipeline-buffer.md)
