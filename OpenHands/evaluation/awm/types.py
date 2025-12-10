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
