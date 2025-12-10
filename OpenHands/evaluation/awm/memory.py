"""
Memory Manager for AWM

Manages the agent's memory, including:
- Base system prompt
- Learned workflows from successful experiences
"""

import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime

from evaluation.awm.workflow import Workflow
from openhands.core.logger import openhands_logger as logger


class MemoryManager:
    """
    管理Agent的Memory，包括base memory和workflow memory

    遵循AWM原论文: 直接append所有workflow到memory
    """

    def __init__(
        self,
        base_system_prompt: Optional[str] = None,
        max_workflows: int = 50,
        persistence_path: Optional[str] = None,
        truncation_limit: Optional[int] = None,
    ):
        """
        初始化Memory Manager

        Args:
            base_system_prompt: 基础system prompt（如果不提供，使用默认）
            max_workflows: 最大保存的workflow数量
            persistence_path: 持久化路径
            truncation_limit: workflow memory 的 token 限制（None 表示禁用）
        """
        if base_system_prompt is None:
            base_system_prompt = self._load_default_system_prompt()

        self.base_system_prompt = base_system_prompt
        self.workflows: List[Workflow] = []
        self.max_workflows = max_workflows
        self.persistence_path = persistence_path
        self.truncation_limit = truncation_limit

        # 如果存在持久化文件，加载
        if persistence_path and os.path.exists(persistence_path):
            self.load()

    def _load_default_system_prompt(self) -> str:
        """加载默认的system prompt"""
        try:
            # 尝试从CodeActAgent加载
            prompt_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "openhands",
                "agenthub",
                "codeact_agent",
                "prompts",
                "system_prompt.j2",
            )
            if os.path.exists(prompt_path):
                with open(prompt_path, "r") as f:
                    return f.read()
        except Exception:
            pass

        return "You are a helpful coding assistant."

    def add_workflows(self, new_workflows: List[Workflow]):
        """
        添加新的workflows到memory

        遵循原论文: 直接append，不做selection/retrieval
        超过max_workflows时，移除最旧的

        Args:
            new_workflows: 要添加的workflows
        """
        self.workflows.extend(new_workflows)

        # 如果超过限制，移除最旧的
        if len(self.workflows) > self.max_workflows:
            removed_count = len(self.workflows) - self.max_workflows
            self.workflows = self.workflows[removed_count:]
            logger.info(f"Removed {removed_count} old workflows due to limit")

        logger.info(
            f"Added {len(new_workflows)} workflows. "
            f"Total: {len(self.workflows)}"
        )

        # 持久化
        if self.persistence_path:
            self.save()

    def get_augmented_prompt(self) -> str:
        """
        获取包含workflows的完整system prompt

        如果启用了 truncation，会截断 workflows 部分以满足 token 限制。
        遵循原论文: M + W

        Returns:
            str: 增强后的system prompt
        """
        if not self.workflows:
            return self.base_system_prompt

        workflow_section = self._format_workflows_with_truncation()

        return f"""
{self.base_system_prompt}

## Learned Workflows from Past Successful Experiences

The following workflows have been extracted from previously solved Django issues.
Use them as guidance when solving similar problems. Each workflow represents a
proven approach that has successfully resolved similar issues.

{workflow_section}

**Important**: These workflows are guidelines, not rigid rules. Adapt them to the
specific problem at hand. If a workflow doesn't seem applicable, feel free to
use your own approach.
"""

    def _format_workflows(self) -> str:
        """格式化所有workflows为prompt string"""
        formatted = []

        for i, wf in enumerate(self.workflows, 1):
            formatted.append(f"--- Workflow {i} ---")
            formatted.append(wf.to_prompt_string())
            formatted.append("")

        return "\n".join(formatted)

    def _format_workflows_with_truncation(self) -> str:
        """
        格式化 workflows，应用 truncation（如果启用）

        策略: FIFO - 保留最旧的 workflows，截断最新的
        """
        if self.truncation_limit is None:
            # 不启用 truncation，返回所有 workflows
            return self._format_workflows()

        from evaluation.awm.token_utils import count_tokens

        # 计算 base prompt 的 token 数量（预留空间）
        base_overhead = count_tokens(self.base_system_prompt) + 200  # 200 for section headers
        available_tokens = self.truncation_limit - base_overhead

        if available_tokens <= 0:
            logger.warning(
                f"Truncation limit ({self.truncation_limit}) is too small "
                f"for base prompt ({base_overhead} tokens). Returning empty workflows."
            )
            return ""

        # FIFO: 从最旧的 workflow 开始添加
        formatted_parts = []
        current_tokens = 0
        workflows_included = 0

        for i, wf in enumerate(self.workflows):
            wf_text = f"--- Workflow {i + 1} ---\n{wf.to_prompt_string()}\n"
            wf_tokens = count_tokens(wf_text)

            if current_tokens + wf_tokens > available_tokens:
                # 达到限制，停止添加
                logger.info(
                    f"Truncation applied: included {workflows_included}/{len(self.workflows)} "
                    f"workflows ({current_tokens} tokens, limit: {available_tokens})"
                )
                break

            formatted_parts.append(wf_text)
            current_tokens += wf_tokens
            workflows_included += 1

        if workflows_included < len(self.workflows):
            formatted_parts.append(
                f"\n[Note: {len(self.workflows) - workflows_included} additional workflows "
                f"truncated due to token limit]\n"
            )

        return "\n".join(formatted_parts)

    def get_workflow_count(self) -> int:
        """获取当前workflow数量"""
        return len(self.workflows)

    def get_workflow_names(self) -> List[str]:
        """获取所有workflow名称"""
        return [wf.name for wf in self.workflows]

    def clear_workflows(self):
        """清空所有workflows"""
        self.workflows = []
        if self.persistence_path:
            self.save()

    def save(self, path: Optional[str] = None):
        """
        保存workflows到文件

        Args:
            path: 保存路径（如果不指定，使用默认路径）
        """
        save_path = path or self.persistence_path
        if not save_path:
            return

        data = {
            "workflows": [wf.to_dict() for wf in self.workflows],
            "last_updated": datetime.now().isoformat(),
            "total_count": len(self.workflows),
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Memory saved to {save_path}")

    def load(self, path: Optional[str] = None):
        """
        从文件加载workflows

        Args:
            path: 加载路径（如果不指定，使用默认路径）
        """
        load_path = path or self.persistence_path
        if not load_path or not os.path.exists(load_path):
            return

        try:
            with open(load_path, "r") as f:
                data = json.load(f)

            self.workflows = [
                Workflow.from_dict(wf) for wf in data.get("workflows", [])
            ]

            logger.info(
                f"Memory loaded from {load_path} "
                f"({len(self.workflows)} workflows)"
            )

        except Exception as e:
            logger.error(f"Failed to load memory from {load_path}: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取memory统计信息"""
        from evaluation.awm.token_utils import count_tokens

        augmented_prompt = self.get_augmented_prompt()
        return {
            "workflow_count": len(self.workflows),
            "workflow_names": self.get_workflow_names(),
            "max_workflows": self.max_workflows,
            "truncation_limit": self.truncation_limit,
            "base_prompt_length": len(self.base_system_prompt),
            "augmented_prompt_length": len(augmented_prompt),
            "augmented_prompt_tokens": count_tokens(augmented_prompt),
        }
