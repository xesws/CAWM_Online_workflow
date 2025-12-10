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
