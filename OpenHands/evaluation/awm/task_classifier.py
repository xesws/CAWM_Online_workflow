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
