"""
Chunk Summarizer for Experience Log Handler

对 history 的每个 chunk 进行 LLM 摘要
"""

from typing import List, Dict, Any
from jinja2 import Environment, FileSystemLoader
import os

from evaluation.awm.experience import HistoryStep
from openhands.core.config import LLMConfig
from openhands.llm.llm import LLM
from openhands.core.logger import openhands_logger as logger


class ChunkSummarizer:
    """
    使用 LLM 对 history chunk 进行摘要
    """

    def __init__(self, llm_config: LLMConfig):
        self.llm = LLM(llm_config, service_id="chunk_summarizer")
        self._load_prompt_template()

    def _load_prompt_template(self):
        """加载 prompt 模板"""
        template_dir = os.path.join(os.path.dirname(__file__), "prompts")
        env = Environment(loader=FileSystemLoader(template_dir))
        self.template = env.get_template("chunk_summary_prompt.j2")

    def summarize_chunk(
        self,
        chunk: List[HistoryStep],
        problem_statement: str,
    ) -> Dict[str, Any]:
        """
        对单个 chunk 进行摘要

        Args:
            chunk: 步骤列表 (5-10 步)
            problem_statement: 问题描述 (用于上下文)

        Returns:
            Dict 包含:
            - phase: 当前阶段 (understanding/locating/fixing/testing)
            - action_summary: 简化的 action 描述
            - key_reasoning: 关键推理
            - files_involved: 涉及的文件
            - outcome: 结果 (如果有)
        """
        # 格式化 chunk 为文本
        chunk_text = self._format_chunk(chunk)

        # 构建 prompt
        prompt = self.template.render(
            problem_statement=problem_statement[:500],
            chunk_steps=chunk_text,
            step_count=len(chunk),
        )

        # 调用 LLM
        try:
            response = self.llm.completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Failed to summarize chunk: {e}")
            return self._default_response()

    def _format_chunk(self, chunk: List[HistoryStep]) -> str:
        """格式化 chunk 为可读文本"""
        lines = []
        for i, step in enumerate(chunk, 1):
            lines.append(f"Step {i}:")
            lines.append(f"  Action: [{step.action_type}] {step.action[:200]}")
            if step.thought:
                lines.append(f"  Thought: {step.thought[:300]}")
            if step.observation:
                lines.append(f"  Result: {step.observation[:200]}")
            lines.append("")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 响应"""
        result = self._default_response()

        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Phase:"):
                result["phase"] = line.split(":", 1)[1].strip().lower()
            elif line.startswith("Action:"):
                result["action_summary"] = line.split(":", 1)[1].strip()
            elif line.startswith("Reasoning:"):
                result["key_reasoning"] = line.split(":", 1)[1].strip()
            elif line.startswith("Files:"):
                files_str = line.split(":", 1)[1].strip()
                if files_str.lower() != "none":
                    result["files_involved"] = [
                        f.strip() for f in files_str.split(",") if f.strip()
                    ]
            elif line.startswith("Outcome:"):
                outcome = line.split(":", 1)[1].strip()
                if outcome.lower() != "none":
                    result["outcome"] = outcome

        return result

    def _default_response(self) -> Dict[str, Any]:
        """返回默认响应"""
        return {
            "phase": "unknown",
            "action_summary": "",
            "key_reasoning": "",
            "files_involved": [],
            "outcome": None,
        }
