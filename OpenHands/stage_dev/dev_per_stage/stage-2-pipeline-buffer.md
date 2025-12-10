# Stage 2: Online Pipeline + Experience Buffer (P1)

> **优先级**: P1
> **复杂度**: 中
> **依赖**: Stage 1 完成

---

## 1. 核心目标

实现单样本的 inference → evaluation 即时流程，以及成功经验的存储机制。

---

## 2. 文件结构

```
evaluation/awm/
├── pipeline.py           # Online Evaluation Pipeline
├── buffer.py             # Experience Buffer
└── task_classifier.py    # 任务分类器
```

---

## 3. 详细任务

### 3.1 Task: 实现Online Evaluation Pipeline

**文件**: `evaluation/awm/pipeline.py`

```python
"""
Online Evaluation Pipeline for AWM

Provides a unified pipeline for single-sample inference and evaluation,
supporting online learning workflows.
"""

from typing import List, Optional, Any, Dict
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
        logger.info(f"Running inference for {instance_id}...")
        inference_output = self._run_inference(
            instance,
            workflow_memory,
            custom_system_prompt,
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
            logger.info(f"Running evaluation for {instance_id}...")
            eval_result = self._run_evaluation(
                instance_id,
                inference_output.diff_patch,
            )
            logger.info(f"Evaluation complete. Resolved: {eval_result.resolved}")

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
    ) -> InferenceOutput:
        """运行inference阶段"""
        return self.inference_runner.run(
            instance=instance,
            workflow_memory=workflow_memory,
            custom_system_prompt=custom_system_prompt,
        )

    def _run_evaluation(
        self,
        instance_id: str,
        diff_patch: str,
    ) -> EvaluationResult:
        """运行evaluation阶段"""
        return self.evaluation_runner.run(
            instance_id=instance_id,
            diff_patch=diff_patch,
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
        return self._run_inference(instance, workflow_memory)

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
```

### 3.2 Task: 实现Experience Buffer

**文件**: `evaluation/awm/buffer.py`

```python
"""
Experience Buffer for AWM

Stores successful experiences and provides functionality for:
- Grouping by task type
- Triggering workflow induction
- Persistence and recovery
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Callable

from evaluation.awm.experience import CodingExperience
from evaluation.awm.task_classifier import classify_task, TaskType
from openhands.core.logger import openhands_logger as logger


class ExperienceBuffer:
    """
    存储成功的experiences，支持按任务类型分组

    遵循AWM原论文：按website（这里是task_type）分组存储
    """

    def __init__(
        self,
        induction_trigger_count: int = 10,
        persistence_path: Optional[str] = None,
        auto_persist: bool = True,
    ):
        """
        初始化Experience Buffer

        Args:
            induction_trigger_count: 触发induction的成功experience数量阈值
            persistence_path: 持久化文件路径（可选）
            auto_persist: 是否自动持久化
        """
        # 按任务类型分组存储
        self.buffer: Dict[str, List[CodingExperience]] = defaultdict(list)

        # 触发induction的阈值
        self.induction_trigger_count = induction_trigger_count

        # 已处理的成功experience计数
        self.success_count = 0

        # 持久化设置
        self.persistence_path = persistence_path
        self.auto_persist = auto_persist

        # 如果存在持久化文件，加载
        if persistence_path and os.path.exists(persistence_path):
            self.load()

    def add(self, experience: CodingExperience) -> bool:
        """
        添加experience到buffer

        只接受成功的experience (test_result == "PASS")

        Args:
            experience: 要添加的experience

        Returns:
            bool: 是否应该触发workflow induction
        """
        if experience.test_result != "PASS":
            logger.debug(f"Skipping failed experience: {experience.instance_id}")
            return False

        # 分类任务类型
        if experience.task_type is None:
            experience.task_type = classify_task(experience)

        task_type = experience.task_type

        # 添加到对应分组
        self.buffer[task_type].append(experience)
        self.success_count += 1

        logger.info(
            f"Added experience {experience.instance_id} to buffer "
            f"(type: {task_type}, total: {self.success_count})"
        )

        # 自动持久化
        if self.auto_persist and self.persistence_path:
            self._persist()

        # 判断是否触发induction
        return self.success_count % self.induction_trigger_count == 0

    def get_all_experiences(self) -> List[CodingExperience]:
        """获取所有experience用于induction"""
        all_exp = []
        for task_type, exps in self.buffer.items():
            all_exp.extend(exps)
        return all_exp

    def get_experiences_by_type(self, task_type: str) -> List[CodingExperience]:
        """获取特定类型的experience"""
        return self.buffer.get(task_type, [])

    def get_recent_experiences(self, n: int = 10) -> List[CodingExperience]:
        """获取最近的n个experience"""
        all_exp = self.get_all_experiences()
        # 按时间戳排序
        sorted_exp = sorted(all_exp, key=lambda x: x.timestamp, reverse=True)
        return sorted_exp[:n]

    def get_statistics(self) -> Dict:
        """获取buffer统计信息"""
        type_counts = {t: len(exps) for t, exps in self.buffer.items()}
        return {
            "total_count": self.success_count,
            "type_counts": type_counts,
            "types": list(self.buffer.keys()),
            "induction_trigger_count": self.induction_trigger_count,
            "next_induction_at": (
                (self.success_count // self.induction_trigger_count + 1)
                * self.induction_trigger_count
            ),
        }

    def clear(self):
        """清空buffer"""
        self.buffer = defaultdict(list)
        self.success_count = 0
        if self.auto_persist and self.persistence_path:
            self._persist()

    def _persist(self):
        """持久化buffer到文件"""
        if not self.persistence_path:
            return

        data = {
            "success_count": self.success_count,
            "buffer": {
                task_type: [exp.to_dict() for exp in exps]
                for task_type, exps in self.buffer.items()
            },
            "last_updated": datetime.now().isoformat(),
        }

        # 确保目录存在
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)

        with open(self.persistence_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Buffer persisted to {self.persistence_path}")

    def load(self):
        """从文件加载buffer"""
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return

        try:
            with open(self.persistence_path, "r") as f:
                data = json.load(f)

            self.success_count = data.get("success_count", 0)

            self.buffer = defaultdict(list)
            for task_type, exps in data.get("buffer", {}).items():
                self.buffer[task_type] = [
                    CodingExperience.from_dict(exp) for exp in exps
                ]

            logger.info(
                f"Buffer loaded from {self.persistence_path} "
                f"({self.success_count} experiences)"
            )

        except Exception as e:
            logger.error(f"Failed to load buffer from {self.persistence_path}: {e}")

    def save(self, path: Optional[str] = None):
        """
        手动保存buffer到指定路径

        Args:
            path: 保存路径（如果不指定，使用默认路径）
        """
        original_path = self.persistence_path
        if path:
            self.persistence_path = path
        self._persist()
        self.persistence_path = original_path


class ExperienceBufferWithCallbacks(ExperienceBuffer):
    """
    带回调的Experience Buffer

    支持在特定事件发生时触发回调函数
    """

    def __init__(
        self,
        induction_trigger_count: int = 10,
        persistence_path: Optional[str] = None,
        on_induction_trigger: Optional[Callable[[List[CodingExperience]], None]] = None,
        on_experience_added: Optional[Callable[[CodingExperience], None]] = None,
    ):
        super().__init__(induction_trigger_count, persistence_path)

        self.on_induction_trigger = on_induction_trigger
        self.on_experience_added = on_experience_added

    def add(self, experience: CodingExperience) -> bool:
        """添加experience并触发回调"""
        should_induce = super().add(experience)

        # 触发添加回调
        if experience.test_result == "PASS" and self.on_experience_added:
            self.on_experience_added(experience)

        # 触发induction回调
        if should_induce and self.on_induction_trigger:
            self.on_induction_trigger(self.get_all_experiences())

        return should_induce
```

### 3.3 Task: 实现任务分类器

**文件**: `evaluation/awm/task_classifier.py`

```python
"""
Task Classifier for AWM

Classifies coding tasks into categories for better experience organization.
"""

from enum import Enum
from typing import Optional
import re

from evaluation.awm.experience import CodingExperience


class TaskType(str, Enum):
    """任务类型枚举"""
    BUG_FIX = "bug_fix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    TEST = "test"
    DOCS = "docs"
    PERFORMANCE = "performance"
    GENERAL = "general"


# 关键词映射
TASK_TYPE_KEYWORDS = {
    TaskType.BUG_FIX: [
        "bug", "fix", "error", "exception", "crash", "issue",
        "broken", "fail", "wrong", "incorrect", "unexpected",
        "traceback", "regression", "typo",
    ],
    TaskType.FEATURE: [
        "add", "implement", "feature", "support", "new",
        "enhancement", "introduce", "enable", "allow",
    ],
    TaskType.REFACTOR: [
        "refactor", "cleanup", "clean up", "simplify",
        "restructure", "reorganize", "improve", "optimize code",
    ],
    TaskType.TEST: [
        "test", "unittest", "pytest", "coverage",
        "mock", "fixture", "assertion",
    ],
    TaskType.DOCS: [
        "doc", "documentation", "docstring", "readme",
        "comment", "typo in doc",
    ],
    TaskType.PERFORMANCE: [
        "performance", "slow", "speed", "optimize",
        "memory", "efficient", "cache", "fast",
    ],
}


def classify_task(experience: CodingExperience) -> str:
    """
    对任务进行分类

    Args:
        experience: CodingExperience对象

    Returns:
        str: 任务类型
    """
    problem = experience.problem_statement.lower()

    # 计算每种类型的匹配分数
    scores = {}
    for task_type, keywords in TASK_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in problem)
        if score > 0:
            scores[task_type] = score

    # 返回分数最高的类型
    if scores:
        best_type = max(scores, key=scores.get)
        return best_type.value

    return TaskType.GENERAL.value


def classify_task_with_llm(
    experience: CodingExperience,
    llm_client,
) -> str:
    """
    使用LLM进行更准确的任务分类

    Args:
        experience: CodingExperience对象
        llm_client: LLM客户端

    Returns:
        str: 任务类型
    """
    prompt = f"""Classify the following GitHub issue into one of these categories:
- bug_fix: Fixing bugs, errors, exceptions
- feature: Adding new functionality
- refactor: Code restructuring without changing behavior
- test: Test-related changes
- docs: Documentation changes
- performance: Performance optimization
- general: Other

Issue description:
{experience.problem_statement[:1000]}

Respond with just the category name, nothing else.
"""

    try:
        response = llm_client.completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
        )
        category = response.choices[0].message.content.strip().lower()

        # 验证返回的类型是否有效
        valid_types = [t.value for t in TaskType]
        if category in valid_types:
            return category

    except Exception as e:
        pass

    # 回退到关键词分类
    return classify_task(experience)


def get_task_type_description(task_type: str) -> str:
    """获取任务类型的描述"""
    descriptions = {
        TaskType.BUG_FIX.value: "Bug fixes and error corrections",
        TaskType.FEATURE.value: "New feature implementations",
        TaskType.REFACTOR.value: "Code refactoring and cleanup",
        TaskType.TEST.value: "Test-related changes",
        TaskType.DOCS.value: "Documentation updates",
        TaskType.PERFORMANCE.value: "Performance optimizations",
        TaskType.GENERAL.value: "General changes",
    }
    return descriptions.get(task_type, "Unknown task type")
```

---

## 4. 使用示例

```python
from openhands.core.config import get_llm_config_arg
from evaluation.awm.pipeline import OnlineEvaluationPipeline
from evaluation.awm.buffer import ExperienceBuffer

# 初始化
llm_config = get_llm_config_arg("llm.eval_gpt4")
pipeline = OnlineEvaluationPipeline(llm_config)
buffer = ExperienceBuffer(
    induction_trigger_count=10,
    persistence_path="evaluation/evaluation_outputs/awm/buffer.json",
)

# 处理单个样本
experience = pipeline.process_single_sample(
    instance_id="django__django-11039",
)

# 如果成功，加入buffer
if experience.test_result == "PASS":
    should_induce = buffer.add(experience)
    if should_induce:
        print("Should trigger workflow induction!")

# 查看统计
print(buffer.get_statistics())
```

---

## 5. 验收标准

| 验收项 | 描述 | 状态 |
|-------|------|------|
| 1 | `OnlineEvaluationPipeline.process_single_sample()` 正常工作 | ⬜ |
| 2 | `ExperienceBuffer.add()` 可以正确存储成功的experience | ⬜ |
| 3 | Buffer可以在达到阈值时返回 `True` 触发induction | ⬜ |
| 4 | Buffer支持持久化和恢复 | ⬜ |
| 5 | 任务分类器 `classify_task()` 工作正常 | ⬜ |

---

## 6. 下一步

完成Stage 2后，进入 [Stage 3: Induction Module + Memory Integration](./stage-3-induction-memory.md)
