"""
Experience Log Handler for AWM

将完整的 experience (~50K tokens) 压缩为适合 induction 的格式 (<20K tokens)
通过分段 LLM 提炼，保留关键 action 和核心 reasoning
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from evaluation.awm.experience import CodingExperience, HistoryStep
from evaluation.awm.chunk_summarizer import ChunkSummarizer
from openhands.core.config import LLMConfig


@dataclass
class CompressedStep:
    """压缩后的步骤"""
    phase: str                    # 阶段：understanding, locating, fixing, testing
    action_summary: str           # 简化的 action：read_file, edit_file, run_test
    key_reasoning: str            # 关键推理：为什么做这个决策
    files_involved: List[str]     # 涉及的文件
    outcome: Optional[str] = None # 结果：success/failure/error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "action_summary": self.action_summary,
            "key_reasoning": self.key_reasoning,
            "files_involved": self.files_involved,
            "outcome": self.outcome,
        }


@dataclass
class CompressedExperience:
    """压缩后的 Experience，用于 Induction"""
    instance_id: str
    problem_summary: str          # 问题摘要 (从 problem_statement 提取)

    # 压缩后的轨迹
    phases: List[CompressedStep]  # 按阶段组织的关键步骤

    # 最终结果
    solution_summary: str         # 解决方案摘要
    test_result: str              # PASS/FAIL

    # 元信息
    original_step_count: int      # 原始步骤数
    compressed_step_count: int    # 压缩后步骤数

    def to_induction_format(self) -> str:
        """转换为 Induction Module 可用的格式"""
        lines = [
            f"## Experience: {self.instance_id}",
            f"",
            f"### Problem",
            f"{self.problem_summary}",
            f"",
            f"### Solution Trajectory ({self.compressed_step_count} key steps from {self.original_step_count} total)",
            f"",
        ]

        for i, phase in enumerate(self.phases, 1):
            lines.append(f"**Phase {i}: {phase.phase}**")
            lines.append(f"- Action: {phase.action_summary}")
            lines.append(f"- Reasoning: {phase.key_reasoning}")
            if phase.files_involved:
                lines.append(f"- Files: {', '.join(phase.files_involved)}")
            if phase.outcome:
                lines.append(f"- Outcome: {phase.outcome}")
            lines.append("")

        lines.extend([
            f"### Solution",
            f"{self.solution_summary}",
            f"",
            f"### Result: {self.test_result}",
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "problem_summary": self.problem_summary,
            "phases": [p.to_dict() for p in self.phases],
            "solution_summary": self.solution_summary,
            "test_result": self.test_result,
            "original_step_count": self.original_step_count,
            "compressed_step_count": self.compressed_step_count,
        }


class ExperienceLogHandler:
    """
    Experience 日志处理器

    将完整的 CodingExperience 压缩为 CompressedExperience
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        chunk_size: int = 10,           # 每个 chunk 包含的步骤数
        target_tokens: int = 15000,     # 目标 token 数 (<20K)
    ):
        self.chunk_summarizer = ChunkSummarizer(llm_config)
        self.chunk_size = chunk_size
        self.target_tokens = target_tokens

    def compress(self, experience: CodingExperience) -> CompressedExperience:
        """
        压缩单个 experience

        流程：
        1. 将 history 分成 chunks (每 chunk_size 步)
        2. 对每个 chunk 用 LLM 提炼关键信息
        3. 识别阶段边界 (understanding → locating → fixing → testing)
        4. 合并相邻同阶段的 chunks
        5. 生成最终的 CompressedExperience
        """
        history = experience.history
        original_count = len(history)

        # Step 1: 分 chunk
        chunks = self._split_into_chunks(history)

        # Step 2: 对每个 chunk 进行 LLM 提炼
        chunk_summaries = []
        for chunk in chunks:
            summary = self.chunk_summarizer.summarize_chunk(
                chunk=chunk,
                problem_statement=experience.problem_statement,
            )
            chunk_summaries.append(summary)

        # Step 3: 识别阶段并合并
        phases = self._identify_and_merge_phases(chunk_summaries)

        # Step 4: 生成问题摘要和解决方案摘要
        problem_summary = self._summarize_problem(experience.problem_statement)
        solution_summary = self._summarize_solution(experience.diff_patch)

        return CompressedExperience(
            instance_id=experience.instance_id,
            problem_summary=problem_summary,
            phases=phases,
            solution_summary=solution_summary,
            test_result=experience.test_result,
            original_step_count=original_count,
            compressed_step_count=len(phases),
        )

    def _split_into_chunks(self, history: List[HistoryStep]) -> List[List[HistoryStep]]:
        """将 history 分成固定大小的 chunks"""
        chunks = []
        for i in range(0, len(history), self.chunk_size):
            chunk = history[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    def _identify_and_merge_phases(
        self,
        chunk_summaries: List[Dict[str, Any]]
    ) -> List[CompressedStep]:
        """
        识别阶段边界并合并相邻同阶段的 chunks

        阶段类型：
        - understanding: 阅读问题、搜索代码
        - locating: 定位相关文件
        - fixing: 修改代码
        - testing: 运行测试
        - debugging: 调试错误（可选）
        """
        phases = []
        current_phase = None
        current_steps = []

        for summary in chunk_summaries:
            phase_type = summary.get("phase", "unknown")

            if phase_type != current_phase and current_steps:
                # 阶段切换，保存之前的阶段
                merged = self._merge_steps(current_steps, current_phase)
                phases.append(merged)
                current_steps = []

            current_phase = phase_type
            current_steps.append(summary)

        # 保存最后一个阶段
        if current_steps:
            merged = self._merge_steps(current_steps, current_phase)
            phases.append(merged)

        return phases

    def _merge_steps(
        self,
        steps: List[Dict[str, Any]],
        phase: str
    ) -> CompressedStep:
        """合并同阶段的多个 chunk summaries"""
        # 合并 actions
        actions = [s.get("action_summary", "") for s in steps]
        action_summary = " → ".join(filter(None, actions))

        # 合并 reasoning (取最重要的)
        reasonings = [s.get("key_reasoning", "") for s in steps]
        key_reasoning = " | ".join(filter(None, reasonings))

        # 合并文件列表
        files = set()
        for s in steps:
            files.update(s.get("files_involved", []))

        # 取最后一个 outcome
        outcome = steps[-1].get("outcome") if steps else None

        return CompressedStep(
            phase=phase,
            action_summary=action_summary,
            key_reasoning=key_reasoning,
            files_involved=list(files),
            outcome=outcome,
        )

    def _summarize_problem(self, problem_statement: str) -> str:
        """提取问题的核心描述"""
        if len(problem_statement) <= 500:
            return problem_statement
        return problem_statement[:500] + "..."

    def _summarize_solution(self, diff_patch: str) -> str:
        """提取解决方案的核心描述"""
        lines = diff_patch.split('\n')
        files_changed = []
        for line in lines:
            if line.startswith('diff --git'):
                parts = line.split()
                if len(parts) >= 4:
                    files_changed.append(parts[3].lstrip('b/'))

        if files_changed:
            return f"Modified files: {', '.join(files_changed[:5])}"
        return "Patch applied"
