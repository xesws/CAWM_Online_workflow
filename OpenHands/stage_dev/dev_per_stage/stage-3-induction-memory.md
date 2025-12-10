# Stage 3: Induction Module + Memory Integration (P2)

> **优先级**: P2
> **复杂度**: 中
> **依赖**: Stage 1, Stage 2 完成

---

## 1. 核心目标

实现从成功experiences中归纳workflows的能力，以及将workflows注入agent memory的机制。

---

## 2. 文件结构

```
evaluation/awm/
├── workflow.py           # Workflow数据结构
├── induction.py          # Workflow Induction Module
├── memory.py             # Memory Manager
└── prompts/
    └── induction_prompt.j2    # Induction的prompt模板
```

---

## 3. 详细任务

### 3.1 Task: 定义Workflow数据结构

**文件**: `evaluation/awm/workflow.py`

```python
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
```

### 3.2 Task: 实现Induction Module

**文件**: `evaluation/awm/induction.py`

```python
"""
Workflow Induction Module for AWM

Extracts reusable workflows from successful coding experiences
using LLM-based analysis.
"""

import os
import re
from typing import List, Optional
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

from evaluation.awm.experience import CodingExperience
from evaluation.awm.workflow import Workflow, WorkflowStep
from evaluation.awm.history_parser import format_history_for_display
from openhands.core.config import LLMConfig
from openhands.core.logger import openhands_logger as logger


class WorkflowInductionModule:
    """
    从成功的experiences中归纳可复用的workflows

    遵循AWM原论文的LM-based induction方法
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        prompt_template_path: Optional[str] = None,
    ):
        """
        初始化Induction Module

        Args:
            llm_config: LLM配置
            prompt_template_path: Prompt模板路径（可选）
        """
        from openhands.llm.llm import LLM

        self.llm = LLM(llm_config)
        self.llm_config = llm_config

        # 加载prompt模板
        if prompt_template_path is None:
            prompt_template_path = os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "induction_prompt.j2"
            )

        self.prompt_template_path = prompt_template_path
        self._load_template()

    def _load_template(self):
        """加载Jinja2模板"""
        template_dir = os.path.dirname(self.prompt_template_path)
        template_name = os.path.basename(self.prompt_template_path)

        env = Environment(loader=FileSystemLoader(template_dir))
        self.template = env.get_template(template_name)

    def induce(
        self,
        experiences: List[CodingExperience],
        max_workflows: int = 5,
        min_experiences: int = 3,
    ) -> List[Workflow]:
        """
        从experiences中归纳workflows

        遵循原论文:
        - 提取跨task重复的action子集
        - 每个workflow 3-8步
        - 用变量名替换具体值

        Args:
            experiences: 成功的experiences列表
            max_workflows: 最多归纳的workflow数量
            min_experiences: 最少需要的experience数量

        Returns:
            List[Workflow]: 归纳出的workflows列表
        """
        if len(experiences) < min_experiences:
            logger.warning(
                f"Not enough experiences for induction "
                f"({len(experiences)} < {min_experiences})"
            )
            return []

        # 1. 格式化experiences
        formatted_experiences = self._format_experiences(experiences)

        # 2. 构建prompt
        prompt = self._build_prompt(formatted_experiences, max_workflows)

        # 3. 调用LLM
        logger.info(f"Inducing workflows from {len(experiences)} experiences...")
        response = self.llm.completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )

        response_text = response.choices[0].message.content

        # 4. 解析输出
        workflows = self._parse_workflows(response_text)

        # 5. 验证和过滤
        valid_workflows = []
        for wf in workflows:
            if wf.validate():
                # 记录来源
                wf.source_experiences = [exp.instance_id for exp in experiences]
                wf.created_at = datetime.now().isoformat()
                valid_workflows.append(wf)
            else:
                logger.warning(f"Invalid workflow skipped: {wf.name}")

        logger.info(f"Induced {len(valid_workflows)} valid workflows")
        return valid_workflows

    def _format_experiences(
        self,
        experiences: List[CodingExperience],
        max_problem_length: int = 500,
        max_steps: int = 15,
    ) -> str:
        """
        将experiences格式化为LLM可理解的格式

        Args:
            experiences: Experience列表
            max_problem_length: 问题描述最大长度
            max_steps: 每个experience显示的最大步骤数

        Returns:
            格式化的字符串
        """
        formatted = []

        for i, exp in enumerate(experiences):
            # 截断problem statement
            problem = exp.problem_statement
            if len(problem) > max_problem_length:
                problem = problem[:max_problem_length] + "..."

            # 格式化history
            history_str = format_history_for_display(exp.history, max_steps)

            exp_str = f"""
=== Experience {i+1} ===
Instance: {exp.instance_id}
Task Type: {exp.task_type or 'unknown'}

Problem:
{problem}

Trajectory:
{history_str}

Solution (Diff Patch):
{exp.diff_patch[:500]}{'...' if len(exp.diff_patch) > 500 else ''}
"""
            formatted.append(exp_str)

        return "\n\n".join(formatted)

    def _build_prompt(
        self,
        formatted_experiences: str,
        max_workflows: int,
    ) -> str:
        """构建归纳prompt"""
        return self.template.render(
            experiences=formatted_experiences,
            max_workflows=max_workflows,
        )

    def _parse_workflows(self, response: str) -> List[Workflow]:
        """
        解析LLM输出为Workflow结构

        Args:
            response: LLM的响应文本

        Returns:
            List[Workflow]: 解析出的workflows
        """
        workflows = []

        # 按 "---" 或 "## Workflow:" 分割不同workflow
        workflow_texts = re.split(r'\n---\n|(?=## Workflow:)', response)

        for text in workflow_texts:
            text = text.strip()
            if not text or "Workflow:" not in text:
                continue

            workflow = self._parse_single_workflow(text)
            if workflow:
                workflows.append(workflow)

        return workflows

    def _parse_single_workflow(self, text: str) -> Optional[Workflow]:
        """
        解析单个workflow文本

        Args:
            text: Workflow文本

        Returns:
            Workflow对象或None
        """
        try:
            # 提取workflow名称
            name_match = re.search(r"(?:##\s*)?Workflow:\s*(.+)", text)
            name = name_match.group(1).strip() if name_match else "Unnamed Workflow"

            # 提取描述
            desc_match = re.search(r"Description:\s*(.+?)(?=\n|$)", text)
            description = desc_match.group(1).strip() if desc_match else ""

            # 提取适用场景
            scenario_match = re.search(
                r"(?:Applicable scenarios|When to use):\s*(.+?)(?=\n\n|\nSteps:)",
                text,
                re.IGNORECASE | re.DOTALL
            )
            scenarios = []
            if scenario_match:
                scenario_str = scenario_match.group(1).strip()
                scenarios = [s.strip() for s in scenario_str.split(",")]

            # 提取步骤
            steps = self._parse_steps(text)

            if not steps:
                return None

            return Workflow(
                name=name,
                description=description,
                applicable_scenarios=scenarios,
                steps=steps,
            )

        except Exception as e:
            logger.error(f"Failed to parse workflow: {e}")
            return None

    def _parse_steps(self, text: str) -> List[WorkflowStep]:
        """
        解析workflow步骤

        Args:
            text: 包含步骤的文本

        Returns:
            WorkflowStep列表
        """
        steps = []

        # 匹配 "数字. [StepType] reasoning" 格式
        step_pattern = r"(\d+)\.\s*\[([^\]]+)\]\s*(.+?)(?=\n\s*Action:|(?=\n\d+\.)|$)"
        action_pattern = r"Action:\s*(.+?)(?=\n\d+\.|\n---|$)"

        step_matches = list(re.finditer(step_pattern, text, re.DOTALL))

        for match in step_matches:
            step_num = match.group(1)
            step_type = match.group(2).strip()
            reasoning = match.group(3).strip()

            # 查找对应的action
            remaining_text = text[match.end():]
            action_match = re.search(action_pattern, remaining_text, re.DOTALL)
            action_template = action_match.group(1).strip() if action_match else ""

            # 清理action中的多余内容
            action_template = action_template.split("\n")[0].strip()

            steps.append(WorkflowStep(
                step_type=step_type,
                reasoning=reasoning,
                action_template=action_template,
            ))

        return steps


def create_default_induction_prompt_template() -> str:
    """
    创建默认的induction prompt模板

    Returns:
        str: Jinja2模板内容
    """
    return '''Given a list of successful Django bug-fixing/feature-implementation trajectories, extract common reusable workflows.

Each trajectory contains:
- Problem statement (GitHub issue description)
- Sequence of (thought, action) steps taken by the agent
- Final solution (diff patch)

Requirements:
1. Extract SUB-ROUTINES that appear across multiple tasks, not full task solutions
2. Abstract specific values: replace concrete paths with {{target_file}}, error messages with {{error_pattern}}, etc.
3. Focus on PATTERNS and REASONING, not specific code
4. Each workflow should be 3-8 steps
5. Include the REASONING for each step, explaining WHY to do it

Example workflow format:

## Workflow: Debug a Failing Test in Django
Description: A common workflow for identifying and fixing test failures in Django codebase.
Applicable scenarios: Test failure, assertion error, unexpected behavior in tests

Steps:
1. [Understand] Read the error message and stack trace to identify the failing test location.
   Action: search_code("{{error_keyword}}")

2. [Locate] Find the test file and the specific test method.
   Action: read_file("{{test_file}}")

3. [Trace] Understand what the test is checking and trace to the source code.
   Action: read_file("{{source_file}}")

4. [Analyze] Identify the root cause of the failure.
   Action: think("The issue is {{analysis}}")

5. [Fix] Apply the fix to the source code.
   Action: edit_file("{{source_file}}", {{changes}})

6. [Verify] Run the specific test to confirm the fix.
   Action: run_command("python -m pytest {{test_file}}::{{test_method}}")

---

Now, analyze the following successful experiences and extract common workflows:

{{ experiences }}

Extract up to {{ max_workflows }} common workflows from the above experiences.
Use the same format as the example. Separate each workflow with "---".
'''
```

### 3.3 Task: 实现Memory Manager

**文件**: `evaluation/awm/memory.py`

```python
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
    ):
        """
        初始化Memory Manager

        Args:
            base_system_prompt: 基础system prompt（如果不提供，使用默认）
            max_workflows: 最大保存的workflow数量
            persistence_path: 持久化路径
        """
        if base_system_prompt is None:
            base_system_prompt = self._load_default_system_prompt()

        self.base_system_prompt = base_system_prompt
        self.workflows: List[Workflow] = []
        self.max_workflows = max_workflows
        self.persistence_path = persistence_path

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

        遵循原论文: M + W，完全塞入

        Returns:
            str: 增强后的system prompt
        """
        if not self.workflows:
            return self.base_system_prompt

        workflow_section = self._format_workflows()

        return f"""{self.base_system_prompt}

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
        return {
            "workflow_count": len(self.workflows),
            "workflow_names": self.get_workflow_names(),
            "max_workflows": self.max_workflows,
            "base_prompt_length": len(self.base_system_prompt),
            "augmented_prompt_length": len(self.get_augmented_prompt()),
        }
```

### 3.4 Task: 创建Induction Prompt模板

**文件**: `evaluation/awm/prompts/induction_prompt.j2`

```jinja2
Given a list of successful Django bug-fixing/feature-implementation trajectories, extract common reusable workflows.

Each trajectory contains:
- Problem statement (GitHub issue description)
- Sequence of (thought, action) steps taken by the agent
- Final solution (diff patch)

Requirements:
1. Extract SUB-ROUTINES that appear across multiple tasks, not full task solutions
2. Abstract specific values: replace concrete file paths with {{target_file}}, specific error messages with {{error_pattern}}, etc.
3. Focus on PATTERNS and REASONING, not specific code
4. Each workflow should be 3-8 steps
5. Include the REASONING for each step, explaining WHY to do it
6. Workflows should be general enough to apply to multiple similar problems

Example workflow format:

## Workflow: Debug a Failing Test in Django
Description: A common workflow for identifying and fixing test failures in Django codebase.
Applicable scenarios: Test failure, assertion error, unexpected behavior in tests

Steps:
1. [Understand] Read the error message and stack trace to identify the failing test location.
   Action: search_code("{{error_keyword}}")

2. [Locate] Find the test file and the specific test method.
   Action: read_file("{{test_file}}")

3. [Trace] Understand what the test is checking and trace to the source code.
   Action: read_file("{{source_file}}")

4. [Analyze] Identify the root cause of the failure.
   Action: think("The issue is {{analysis}}")

5. [Fix] Apply the fix to the source code.
   Action: edit_file("{{source_file}}", {{changes}})

6. [Verify] Run the specific test to confirm the fix.
   Action: run_command("python -m pytest {{test_file}}::{{test_method}}")

---

Now, analyze the following successful experiences and extract common workflows:

{{ experiences }}

Extract up to {{ max_workflows }} common workflows from the above experiences.
Follow these guidelines:
- Use the exact format shown in the example
- Separate each workflow with "---"
- Make workflows specific enough to be useful but general enough to apply to similar problems
- Use placeholder variables like {{variable_name}} for values that change between instances
- Focus on the reasoning and decision-making process, not just the actions
```

---

## 4. 使用示例

```python
from openhands.core.config import get_llm_config_arg
from evaluation.awm.induction import WorkflowInductionModule
from evaluation.awm.memory import MemoryManager
from evaluation.awm.buffer import ExperienceBuffer

# 初始化
llm_config = get_llm_config_arg("llm.eval_gpt4")
induction_module = WorkflowInductionModule(llm_config)
memory_manager = MemoryManager(
    persistence_path="evaluation/evaluation_outputs/awm/memory.json"
)

# 假设buffer中已有足够的experiences
buffer = ExperienceBuffer()
buffer.load()

# 运行induction
experiences = buffer.get_all_experiences()
workflows = induction_module.induce(experiences, max_workflows=5)

# 添加到memory
memory_manager.add_workflows(workflows)

# 获取增强后的prompt
augmented_prompt = memory_manager.get_augmented_prompt()
print(f"Augmented prompt length: {len(augmented_prompt)}")
```

---

## 5. 验收标准

| 验收项 | 描述 | 状态 |
|-------|------|------|
| 1 | `WorkflowInductionModule.induce()` 可以从experiences中归纳workflows | ⬜ |
| 2 | 归纳出的workflows符合3-8步的要求 | ⬜ |
| 3 | `MemoryManager.get_augmented_prompt()` 正确注入workflows | ⬜ |
| 4 | Workflows可以正确序列化/反序列化 | ⬜ |
| 5 | Memory支持持久化和恢复 | ⬜ |

---

## 6. 下一步

完成Stage 3后，进入 [Stage 4: 完整AWM Online Loop](./stage-4-online-loop.md)
