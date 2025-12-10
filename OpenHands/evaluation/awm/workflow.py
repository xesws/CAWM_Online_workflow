"""
Workflow Data Structures for AWM

Defines the structure for representing reusable coding workflows
extracted from successful experiences.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json
import re


@dataclass
class WorkflowStep:
    """
    Workflow单步数据结构

    代表workflow中的一个具体步骤
    """
    step_type: str           # e.g., "Understand", "Locate", "Fix", "Verify"
    reasoning: str           # 为什么要做这一步
    action_template: str     # action模板，包含placeholder如 {{target_file}}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowStep":
        return cls(**data)

    def __str__(self) -> str:
        return f"[{self.step_type}] {self.reasoning}\n   Action: {self.action_template}"


@dataclass
class Workflow:
    """
    完整的Workflow数据结构

    代表一个可复用的编码工作流
    """
    name: str
    description: str
    applicable_scenarios: List[str]
    steps: List[WorkflowStep]

    # 元数据
    source_experiences: List[str] = field(default_factory=list)  # instance_ids
    created_at: Optional[str] = None
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["steps"] = [step.to_dict() for step in self.steps]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        data["steps"] = [WorkflowStep.from_dict(s) for s in data.get("steps", [])]
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Workflow":
        return cls.from_dict(json.loads(json_str))

    def to_prompt_string(self) -> str:
        """
        转换为可插入prompt的字符串格式

        遵循AWM原论文的格式要求
        """
        lines = [
            f"### {self.name}",
            f"Description: {self.description}",
            f"When to use: {', '.join(self.applicable_scenarios)}",
            "",
            "Steps:"
        ]

        for i, step in enumerate(self.steps, 1):
            lines.append(f"  {i}. [{step.step_type}] {step.reasoning}")
            lines.append(f"     Action: {step.action_template}")

        return "\n".join(lines)

    def validate(self) -> bool:
        """
        验证workflow是否符合要求

        Requirements (from AWM paper):
        - 3-8 steps
        - Each step has type, reasoning, and action
        """
        if not (3 <= len(self.steps) <= 8):
            return False

        for step in self.steps:
            if not step.step_type or not step.reasoning or not step.action_template:
                return False

        return True


def create_workflow(
    name: str,
    description: str,
    scenarios: List[str],
    steps: List[Dict[str, str]],
    source_experiences: Optional[List[str]] = None,
) -> Workflow:
    """
    创建Workflow的便捷函数

    Args:
        name: Workflow名称
        description: 描述
        scenarios: 适用场景列表
        steps: 步骤列表，每个步骤是包含 step_type, reasoning, action_template 的字典
        source_experiences: 来源的experience IDs

    Returns:
        Workflow: 创建的workflow对象
    """
    from datetime import datetime

    workflow_steps = [
        WorkflowStep(
            step_type=s.get("step_type", "Action"),
            reasoning=s.get("reasoning", ""),
            action_template=s.get("action_template", s.get("action", "")),
        )
        for s in steps
    ]

    return Workflow(
        name=name,
        description=description,
        applicable_scenarios=scenarios,
        steps=workflow_steps,
        source_experiences=source_experiences or [],
        created_at=datetime.now().isoformat(),
    )
