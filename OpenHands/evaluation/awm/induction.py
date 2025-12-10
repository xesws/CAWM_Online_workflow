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

        self.llm = LLM(llm_config, service_id="workflow_induction")
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
            desc_match = re.search(r"Description:\s*(.+?)(?:\n|$)", text)
            description = desc_match.group(1).strip() if desc_match else ""

            # 提取适用场景
            scenario_match = re.search(
                r"(?:Applicable scenarios|When to use):\s*(.+?)(?:\n\n|\nSteps:)",
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
        # 支持可选的 "Reasoning:" 前缀
        step_pattern = r"(\d+)\.\s*\[([^\]]+)\]\s*(?:Reasoning:\s*)?(.+?)(?=\n\s*Action:|(?=\n\d+\.)|$)"
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
