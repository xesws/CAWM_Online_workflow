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

    # Truncation配置
    truncation_limit: Optional[int] = None  # None = 禁用, int = token 限制

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
            "truncation_limit": self.truncation_limit,
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
