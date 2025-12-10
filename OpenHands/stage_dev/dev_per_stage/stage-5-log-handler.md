# Stage 5: Experience Log Handler (P1)

> **优先级**: P1 (解决上下文爆炸问题)
> **复杂度**: 中
> **依赖**: Stage 1-4 已完成

---

## 1. 问题背景

当前 AWM 实现存在严重的上下文爆炸问题：

| 指标 | 当前值 | 问题 |
|------|--------|------|
| 单个 history | ~50 步 × 1,000 tokens = **50K tokens** | 单个 experience 就很大 |
| 累积 10 个 | **500K tokens** | 远超 200K 上下文限制 |
| 累积方式 | 全量累积 (10, 20, 30...) | 越来越无法处理 |

---

## 2. 解决方案概述

创建 **Experience Log Handler** 模块，在调用 Induction Module 前对 experience 进行智能压缩：

```
原始 Experience (~50K tokens)
         ↓
   [Experience Log Handler]
   - 分段提炼 (每 5-10 步)
   - 提取关键 action
   - 保留核心 reasoning
         ↓
压缩后 Experience (<20K tokens)
         ↓
   [Induction Module]
   - 只处理最新 1 个 experience
   - 增量归纳 workflow
```

---

## 3. 核心设计原则

1. **分段 LLM 提炼**：将 history 分成 5-10 步的 chunks，用 LLM 逐段提炼
2. **简化 action + 保留 reasoning**：
   - Action: `read → edit → test` (简化)
   - Reasoning: 为什么做这个决策 (详细保留)
3. **单次 Induction**：每次只处理最新 1 个 experience（历史的已归纳成 workflow）
4. **目标大小**：< 20K tokens/compressed experience

---

## 4. 文件结构

```
evaluation/awm/
├── log_handler.py              # (新建) Experience Log Handler 主模块
├── chunk_summarizer.py         # (新建) 分段摘要器
├── prompts/
│   ├── induction_prompt.j2     # (已有) Induction prompt
│   └── chunk_summary_prompt.j2 # (新建) 分段摘要 prompt
├── induction.py                # (修改) 添加增量归纳方法
├── loop.py                     # (修改) 更新调用逻辑
└── experience.py               # (修改) 添加压缩数据结构
```

---

## 5. 详细任务

### 5.1 Task: 实现 Experience Log Handler

**文件**: `evaluation/awm/log_handler.py`

```python
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
```

### 5.2 Task: 实现 Chunk Summarizer

**文件**: `evaluation/awm/chunk_summarizer.py`

```python
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
        self.llm = LLM(llm_config)
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
```

### 5.3 Task: 创建 Chunk 摘要 Prompt

**文件**: `evaluation/awm/prompts/chunk_summary_prompt.j2`

```jinja2
You are analyzing a segment of an agent's problem-solving trajectory.

## Problem Context
{{ problem_statement }}

## Trajectory Segment ({{ step_count }} steps)
{{ chunk_steps }}

## Task
Analyze this segment and extract the key information. Respond in the following format:

Phase: [one of: understanding, locating, fixing, testing, debugging]
Action: [brief summary of what actions were taken, e.g., "read config files, search for error pattern"]
Reasoning: [the key reasoning that led to these actions - WHY the agent made these decisions]
Files: [comma-separated list of files involved, or "none" if no files]
Outcome: [success/failure/error/ongoing, or "none" if not applicable]

Guidelines:
- Phase should reflect the PRIMARY activity in this segment
- Action should be concise (under 50 words)
- Reasoning should capture the decision-making logic (under 100 words)
- Focus on WHAT the agent learned or decided, not just WHAT it did
```

### 5.4 Task: 修改 Induction Module

**文件**: `evaluation/awm/induction.py` (添加方法)

```python
# 在 WorkflowInductionModule 类中添加以下方法

def induce_from_single(
    self,
    experience: CodingExperience,
    existing_workflows: List[Workflow],
) -> List[Workflow]:
    """
    从单个 experience 增量归纳 workflow

    Args:
        experience: 新的成功 experience
        existing_workflows: 已有的 workflows

    Returns:
        更新后的 workflows 列表
    """
    from evaluation.awm.log_handler import ExperienceLogHandler

    # Step 1: 压缩 experience
    log_handler = ExperienceLogHandler(self.llm_config)
    compressed = log_handler.compress(experience)

    logger.info(
        f"Compressed experience: {compressed.original_step_count} steps → "
        f"{compressed.compressed_step_count} phases"
    )

    # Step 2: 构建包含已有 workflows 的 prompt
    prompt = self._build_incremental_prompt(
        compressed_experience=compressed,
        existing_workflows=existing_workflows,
    )

    # Step 3: 调用 LLM
    response = self.llm.completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
    )

    # Step 4: 解析并返回更新后的 workflows
    return self._parse_incremental_response(
        response.choices[0].message.content,
        existing_workflows,
    )

def _build_incremental_prompt(
    self,
    compressed_experience,  # CompressedExperience
    existing_workflows: List[Workflow],
) -> str:
    """构建增量归纳的 prompt"""
    existing_wf_str = ""
    if existing_workflows:
        existing_wf_str = "\n\n".join([
            wf.to_prompt_string() for wf in existing_workflows
        ])

    return f"""You are updating a collection of reusable coding workflows based on a new successful experience.

## Existing Workflows
{existing_wf_str if existing_wf_str else "(No existing workflows yet)"}

## New Experience
{compressed_experience.to_induction_format()}

## Task
Based on this new experience:
1. If the experience demonstrates a NEW pattern not covered by existing workflows, create a new workflow
2. If the experience REINFORCES an existing workflow, note which one (no change needed)
3. If the experience suggests IMPROVING an existing workflow, provide the updated version

Output format:
ACTION: [NEW/REINFORCE/IMPROVE]
TARGET: [workflow name if REINFORCE/IMPROVE, or "new" if NEW]

[If NEW or IMPROVE, provide the workflow in standard format:]
## Workflow: [Name]
Description: [Brief description]
Applicable scenarios: [comma-separated list]

Steps:
1. [StepType] Reasoning
   Action: action_template
...
"""

def _parse_incremental_response(
    self,
    response: str,
    existing_workflows: List[Workflow],
) -> List[Workflow]:
    """解析增量归纳的响应"""
    # 解析 ACTION 类型
    action_match = re.search(r"ACTION:\s*(\w+)", response, re.IGNORECASE)
    action_type = action_match.group(1).upper() if action_match else "REINFORCE"

    if action_type == "REINFORCE":
        # 无变化，返回原有 workflows
        return existing_workflows

    elif action_type == "NEW":
        # 添加新的 workflow
        new_workflows = self._parse_workflows(response)
        return existing_workflows + new_workflows

    elif action_type == "IMPROVE":
        # 更新现有 workflow
        target_match = re.search(r"TARGET:\s*(.+)", response, re.IGNORECASE)
        target_name = target_match.group(1).strip() if target_match else ""

        updated_workflows = self._parse_workflows(response)
        if not updated_workflows:
            return existing_workflows

        # 替换匹配的 workflow
        result = []
        updated = False
        for wf in existing_workflows:
            if wf.name.lower() == target_name.lower():
                result.append(updated_workflows[0])
                updated = True
            else:
                result.append(wf)

        # 如果没找到匹配的，添加为新的
        if not updated:
            result.extend(updated_workflows)

        return result

    return existing_workflows
```

### 5.5 Task: 修改 AWM Loop

**文件**: `evaluation/awm/loop.py` (修改方法)

```python
# 修改 _run_induction 方法

def _run_induction(self, latest_experience: CodingExperience):
    """运行增量 workflow induction"""
    logger.info("\n" + "=" * 50)
    logger.info("Triggering Incremental Workflow Induction")
    logger.info("=" * 50)

    # 只使用最新的单个 experience 进行增量归纳
    updated_workflows = self.induction_module.induce_from_single(
        experience=latest_experience,
        existing_workflows=self.memory_manager.workflows,
    )

    # 更新 memory
    self.memory_manager.workflows = updated_workflows

    logger.info(f"Updated workflows: {len(updated_workflows)} total")
    logger.info("=" * 50 + "\n")

# 修改主循环中的调用逻辑
# 在 run() 方法中：
if experience.test_result == "PASS":
    self.total_success += 1
    self.experience_buffer.add(experience)  # 仍然存储完整日志

    # 每次成功都触发增量 induction
    self._run_induction(experience)  # 传入最新的 experience
```

---

## 6. 验收标准

| 验收项 | 描述 | 状态 |
|-------|------|------|
| 1 | `ExperienceLogHandler.compress()` 可以将 ~50K tokens 压缩到 <20K tokens | ⬜ |
| 2 | `ChunkSummarizer` 正确识别阶段 (understanding/locating/fixing/testing) | ⬜ |
| 3 | 压缩后的 experience 保留关键 reasoning | ⬜ |
| 4 | `induce_from_single()` 支持增量 workflow 更新 | ⬜ |
| 5 | 整体流程：每次成功 → 压缩 → 单次 induction → 更新 memory | ⬜ |

---

## 7. 文件修改清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `evaluation/awm/log_handler.py` | 新建 | Experience Log Handler 主模块 |
| `evaluation/awm/chunk_summarizer.py` | 新建 | 分段摘要器 |
| `evaluation/awm/prompts/chunk_summary_prompt.j2` | 新建 | 分段摘要 prompt |
| `evaluation/awm/induction.py` | 修改 | 添加 `induce_from_single()` 方法 |
| `evaluation/awm/loop.py` | 修改 | 更新 `_run_induction()` 调用逻辑 |

---

## 8. 处理流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Experience Log Handler Flow                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Original Experience                                                         │
│  (~50K tokens, ~50 steps)                                                   │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────┐                                                        │
│  │ Split to Chunks │  (每 10 步一个 chunk)                                   │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │  Chunk 1   │  Chunk 2   │  Chunk 3   │  Chunk 4   │ ... │                │
│  │ (10 steps) │ (10 steps) │ (10 steps) │ (10 steps) │     │                │
│  └─────┬──────┴─────┬──────┴─────┬──────┴─────┬──────┴─────┘                │
│        │            │            │            │                              │
│        ▼            ▼            ▼            ▼                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                        │
│  │   LLM    │ │   LLM    │ │   LLM    │ │   LLM    │  (ChunkSummarizer)     │
│  │ Summarize│ │ Summarize│ │ Summarize│ │ Summarize│                        │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘                        │
│       │            │            │            │                               │
│       ▼            ▼            ▼            ▼                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                        │
│  │ Phase:   │ │ Phase:   │ │ Phase:   │ │ Phase:   │                        │
│  │understand│ │ locating │ │ fixing   │ │ testing  │                        │
│  │ Action:  │ │ Action:  │ │ Action:  │ │ Action:  │                        │
│  │ Reason:  │ │ Reason:  │ │ Reason:  │ │ Reason:  │                        │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘                        │
│       │            │            │            │                               │
│       └────────────┴────────────┴────────────┘                               │
│                         │                                                    │
│                         ▼                                                    │
│              ┌─────────────────────┐                                        │
│              │ Identify & Merge    │                                        │
│              │ Phases              │                                        │
│              └──────────┬──────────┘                                        │
│                         │                                                    │
│                         ▼                                                    │
│              ┌─────────────────────┐                                        │
│              │ CompressedExperience│  (<20K tokens)                         │
│              │ - problem_summary   │                                        │
│              │ - phases[]          │                                        │
│              │ - solution_summary  │                                        │
│              └──────────┬──────────┘                                        │
│                         │                                                    │
│                         ▼                                                    │
│              ┌─────────────────────┐                                        │
│              │ Induction Module    │                                        │
│              │ (Single Experience) │                                        │
│              └──────────┬──────────┘                                        │
│                         │                                                    │
│                         ▼                                                    │
│              ┌─────────────────────┐                                        │
│              │ Updated Workflows   │                                        │
│              └─────────────────────┘                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. 与现有 Stage 的关系

- **依赖 Stage 1-4**：使用已有的 `CodingExperience`, `HistoryStep`, `Workflow` 等数据结构
- **修改 Stage 3 (induction.py)**：添加增量归纳方法
- **修改 Stage 4 (loop.py)**：更改触发逻辑为每次成功触发

---

## 10. 后续优化方向

1. **并行 Chunk 处理**：多个 chunks 可以并行调用 LLM
2. **Adaptive Chunk Size**：根据步骤复杂度动态调整 chunk 大小
3. **Reasoning 质量评估**：添加机制评估提取的 reasoning 质量
4. **缓存机制**：对相似 pattern 的 chunks 进行缓存
