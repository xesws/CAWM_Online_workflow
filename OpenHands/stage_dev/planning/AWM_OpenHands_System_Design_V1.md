# AWM迁移到OpenHands Coding Agent的系统设计 (V1.0)

> **设计原则**：1.0版本严格按照AWM原论文的方法实现，最小化改动

---

## 1. 核心约束条件

| 约束项 | 决策 |
|--------|------|
| Memory Integration | 完整append所有成功经验（遵循原论文） |
| Context长度 | 完全塞入prompt（遵循原论文） |
| 任务范围 | 仅Django issues（SWE-bench子集） |
| 分组方式 | 按任务类型分组（非语言） |
| 成功判断 | 基于SWE-bench test脚本 |
| Workflow粒度 | 按原论文实现（3-8步） |
| 样本顺序 | 默认顺序（遵循原论文online setting） |

---

## 2. 整体架构设计

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OpenHands + AWM Online System                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     Django Issue Queue (SWE-bench)                     │  │
│  │              [issue_1, issue_2, ..., issue_n] (默认顺序)               │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │ streaming (one by one)                   │
│                                  ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         INFERENCE 阶段                                 │  │
│  │  ┌─────────────┐    ┌─────────────────────────────────────────────┐   │  │
│  │  │   Issue     │    │              Agent Core                      │   │  │
│  │  │   Input     │───▶│  L(q, M + W, o) → a                         │   │  │
│  │  │  (query q)  │    │  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │   │  │
│  │  └─────────────┘    │  │Base Mem │ +│Workflow │ +│Observation  │  │   │  │
│  │                     │  │   M     │  │   W     │  │    o        │  │   │  │
│  │                     │  └─────────┘  └─────────┘  └─────────────┘  │   │  │
│  │                     └───────────────────┬─────────────────────────┘   │  │
│  │                                         │                              │  │
│  │                                         ▼                              │  │
│  │                     ┌───────────────────────────────────────────┐     │  │
│  │                     │         OpenHands Environment              │     │  │
│  │                     │    (Docker: sweb.eval.x86_64.{instance})   │     │  │
│  │                     └───────────────────┬───────────────────────┘     │  │
│  │                                         │                              │  │
│  │                                         ▼                              │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │  │
│  │  │                      Inference Output                             │ │  │
│  │  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │ │  │
│  │  │  │  Diff Patch    │  │    History     │  │ Problem Statement  │  │ │  │
│  │  │  │  (solution)    │  │  (trajectory)  │  │   + Ground Truth   │  │ │  │
│  │  │  └────────────────┘  └────────────────┘  └────────────────────┘  │ │  │
│  │  └──────────────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │ 立即传递 (非batch)                       │
│                                  ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        EVALUATION 阶段                                 │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │  │
│  │  │              SWE-bench Test Runner (魔改版)                       │ │  │
│  │  │         支持单个sample测试 (非batch testset)                      │ │  │
│  │  │                                                                   │ │  │
│  │  │    Input: single diff patch + instance_id                         │ │  │
│  │  │    Process: 在Docker中应用patch并运行测试                          │ │  │
│  │  │    Output: test_result ∈ {PASS, FAIL}                             │ │  │
│  │  └──────────────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │                                          │
│                      ┌───────────┴───────────┐                              │
│                      ▼                       ▼                              │
│                PASS (成功)              FAIL (失败)                         │
│                      │                       │                              │
│                      ▼                       ▼                              │
│  ┌─────────────────────────────┐    ┌─────────────────────┐                │
│  │    Experience Buffer E      │    │       Discard       │                │
│  │    存储成功的experience      │    │                     │                │
│  └──────────────┬──────────────┘    └─────────────────────┘                │
│                 │                                                           │
│                 │ 每N个成功experience触发                                   │
│                 ▼                                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Induction Module I                               │  │
│  │                         I(E) → W                                      │  │
│  │           (LLM + Prompt: 提取可复用的coding workflows)                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                 │                                                           │
│                 ▼                                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Workflow Memory Store                              │  │
│  │                   W = {w₁, w₂, ..., wₖ}                               │  │
│  │                 直接append到Agent的Memory                              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 与现有OpenHands流程的对比

```
现有OpenHands流程 (Batch模式):
═══════════════════════════════════════════════════════════════
Step 1: Batch Inference
    [issue_1, issue_2, ..., issue_n] 
           ↓ 批量处理
    [output_1.jsonl, output_2.jsonl, ..., output_n.jsonl]
           ↓ 全部完成后
Step 2: Batch Evaluation  
    [output_1, output_2, ..., output_n]
           ↓ 批量测试
    [result_1, result_2, ..., result_n]
═══════════════════════════════════════════════════════════════

我们需要的流程 (Online模式):
═══════════════════════════════════════════════════════════════
For each issue_t in [issue_1, issue_2, ..., issue_n]:
    
    Step 1: Single Inference
        issue_t → Agent(M + W_t) → output_t
    
    Step 2: Immediate Evaluation
        output_t → SWE-bench Test → result_t
    
    Step 3: Experience Update
        if result_t == PASS:
            E ← E ∪ {experience_t}
            
    Step 4: Workflow Induction (每N个成功时触发)
        if len(E) % N == 0:
            W_new = I(E)
            W_{t+1} = W_t ∪ W_new
            
    → 进入下一个 issue_{t+1}，使用更新后的 W_{t+1}
═══════════════════════════════════════════════════════════════
```

---

## 3. Experience定义 (遵循SWE-bench流程)

### 3.1 Experience结构

```python
@dataclass
class CodingExperience:
    """
    严格按照SWE-bench/OpenHands的inference输出格式定义
    """
    
    # ===== 问题描述 =====
    instance_id: str               # SWE-bench instance ID, e.g., "django__django-11039"
    problem_statement: str         # GitHub issue的描述
    
    # ===== Agent产出 (Inference阶段输出) =====
    diff_patch: str                # Agent给出的solution (difference patch)
                                   # 这是核心输出，格式短小
    
    history: List[HistoryStep]     # Agent的完整调用轨迹
                                   # 包含每一步的observation, action, reasoning
    
    # ===== 测试结果 (Evaluation阶段输出) =====
    test_result: Literal["PASS", "FAIL"]  # 测试通过与否
    test_output: Optional[str]     # 测试的详细输出（可选，用于debug）
    
    # ===== 元信息 =====
    task_type: Optional[str]       # 任务类型标签 (用于分组，可后续通过LLM标注)
    timestamp: datetime            # 处理时间
    model_name: str                # 使用的LLM模型
    

@dataclass
class HistoryStep:
    """
    Agent单步调用记录，对应OpenHands的history格式
    """
    step_id: int
    observation: str               # 当前观察到的状态
    thought: str                   # Agent的思考过程
    action: str                    # 采取的action
    action_type: str               # action类型: edit_file, run_command, search, etc.
    
    # 可选的详细信息
    file_path: Optional[str]       # 如果是文件操作，记录路径
    command: Optional[str]         # 如果是命令执行，记录命令
```

### 3.2 Diff Patch示例

```diff
# diff_patch 示例 (来自SWE-bench)
# 这是Agent给出的solution，格式紧凑

--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -1234,7 +1234,10 @@ class QuerySet:
     def explain(self, *, format=None, **options):
-        return self.query.explain(using=self.db, format=format, **options)
+        if format is not None:
+            options['format'] = format
+        return self.query.explain(using=self.db, **options)
```

### 3.3 History/Trajectory示例

```json
{
  "history": [
    {
      "step_id": 1,
      "observation": "Current directory: /workspace/django. Files: django/, tests/, setup.py...",
      "thought": "I need to first understand the bug. Let me read the issue and find the relevant file.",
      "action": "search_code('explain format')",
      "action_type": "search"
    },
    {
      "step_id": 2,
      "observation": "Found matches in django/db/models/query.py:1234...",
      "thought": "The issue is in the explain method. Let me read this file.",
      "action": "read_file('django/db/models/query.py', start=1230, end=1250)",
      "action_type": "read_file"
    },
    {
      "step_id": 3,
      "observation": "def explain(self, *, format=None, **options): ...",
      "thought": "I see the bug. The format parameter is not being passed correctly. I need to modify this.",
      "action": "edit_file('django/db/models/query.py', ...)",
      "action_type": "edit_file"
    },
    {
      "step_id": 4,
      "observation": "File edited successfully.",
      "thought": "Let me run the tests to verify the fix.",
      "action": "run_command('python -m pytest tests/queries/test_explain.py')",
      "action_type": "run_command"
    },
    {
      "step_id": 5,
      "observation": "All tests passed.",
      "thought": "The fix is verified. I can now submit.",
      "action": "submit()",
      "action_type": "submit"
    }
  ]
}
```

---

## 4. 核心模块设计

### 4.1 Online Evaluation Pipeline (需要重构)

**问题**：现有OpenHands是batch inference + batch evaluation，无法支持online workflow更新。

**解决方案**：重构为single-sample pipeline。

```python
class OnlineEvaluationPipeline:
    """
    重构SWE-bench测试流程，支持单个sample的即时evaluation
    """
    
    def __init__(self, swebench_config: dict):
        self.docker_client = docker.from_env()
        self.swebench_config = swebench_config
        
    def run_single_inference(
        self, 
        instance_id: str,
        problem_statement: str,
        agent: OpenHandsAgent,
        workflow_memory: List[Workflow]
    ) -> InferenceOutput:
        """
        对单个issue进行inference
        
        Returns:
            InferenceOutput包含: diff_patch, history, metadata
        """
        # 1. 准备Docker环境
        container = self._setup_container(instance_id)
        
        try:
            # 2. 注入workflow memory到agent
            agent.update_memory(workflow_memory)
            
            # 3. 运行agent
            result = agent.solve(
                problem_statement=problem_statement,
                container=container
            )
            
            # 4. 提取输出
            return InferenceOutput(
                instance_id=instance_id,
                diff_patch=result.patch,
                history=result.history,
                problem_statement=problem_statement
            )
        finally:
            container.stop()
            container.remove()
    
    def run_single_evaluation(
        self, 
        inference_output: InferenceOutput
    ) -> EvaluationResult:
        """
        对单个inference结果进行evaluation
        
        这是需要魔改SWE-bench脚本的核心部分
        """
        instance_id = inference_output.instance_id
        diff_patch = inference_output.diff_patch
        
        # 1. 准备测试环境 (使用官方SWE-bench评估镜像)
        eval_container = self._setup_eval_container(instance_id)
        
        try:
            # 2. 应用patch
            self._apply_patch(eval_container, diff_patch)
            
            # 3. 运行测试
            test_result = self._run_tests(eval_container, instance_id)
            
            # 4. 返回结果
            return EvaluationResult(
                instance_id=instance_id,
                passed=test_result.all_passed,
                test_output=test_result.output
            )
        finally:
            eval_container.stop()
            eval_container.remove()
    
    def process_single_sample(
        self,
        instance_id: str,
        problem_statement: str,
        agent: OpenHandsAgent,
        workflow_memory: List[Workflow]
    ) -> CodingExperience:
        """
        完整处理单个sample: inference → evaluation → experience
        """
        # Step 1: Inference
        inference_output = self.run_single_inference(
            instance_id, problem_statement, agent, workflow_memory
        )
        
        # Step 2: Immediate Evaluation
        eval_result = self.run_single_evaluation(inference_output)
        
        # Step 3: Construct Experience
        experience = CodingExperience(
            instance_id=instance_id,
            problem_statement=problem_statement,
            diff_patch=inference_output.diff_patch,
            history=inference_output.history,
            test_result="PASS" if eval_result.passed else "FAIL",
            test_output=eval_result.test_output,
            timestamp=datetime.now(),
            model_name=agent.model_name
        )
        
        return experience
```

### 4.2 魔改SWE-bench脚本的要点

```python
"""
需要修改的SWE-bench脚本位置和内容
"""

# 原始脚本: evaluation/benchmarks/swe_bench/scripts/eval_infer.sh
# 原始行为: 读取整个output.jsonl，批量评估所有instances

# 需要的修改:

# 1. 新增单sample评估接口
def eval_single_instance(
    instance_id: str,
    diff_patch: str,
    swebench_config: dict
) -> dict:
    """
    评估单个instance，而非整个testset
    
    Args:
        instance_id: e.g., "django__django-11039"
        diff_patch: Agent生成的patch
        swebench_config: SWE-bench配置
        
    Returns:
        {"resolved": bool, "test_output": str}
    """
    pass

# 2. 复用现有的Docker镜像逻辑
# 现有脚本已经有 sweb.eval.x86_64.{instance_id} 的镜像处理
# 需要抽取这部分逻辑为可复用的函数

# 3. 修改测试执行逻辑
# 原始: 遍历output.jsonl中所有instance
# 修改: 接受单个instance作为输入

# 关键文件:
# - evaluation/benchmarks/swe_bench/run_infer.py  (inference逻辑)
# - evaluation/benchmarks/swe_bench/eval_infer.py (evaluation逻辑)
# - swebench/harness/run_evaluation.py           (SWE-bench原始评估)
```

### 4.3 Experience Buffer设计

```python
class ExperienceBuffer:
    """
    存储成功的experiences，用于后续的workflow induction
    遵循AWM原论文：按website（这里是task_type）分组
    """
    
    def __init__(self, induction_trigger_count: int = 10):
        # 按任务类型分组存储
        self.buffer: Dict[str, List[CodingExperience]] = defaultdict(list)
        
        # 触发induction的阈值
        self.induction_trigger_count = induction_trigger_count
        
        # 已处理的experience计数
        self.success_count = 0
        
    def add(self, experience: CodingExperience) -> bool:
        """
        添加成功的experience到buffer
        
        Returns:
            bool: 是否应该触发workflow induction
        """
        if experience.test_result != "PASS":
            return False
            
        # 确定任务类型 (可以通过LLM标注，或简单规则)
        task_type = self._classify_task(experience)
        experience.task_type = task_type
        
        # 添加到对应分组
        self.buffer[task_type].append(experience)
        self.success_count += 1
        
        # 判断是否触发induction
        return self.success_count % self.induction_trigger_count == 0
    
    def _classify_task(self, experience: CodingExperience) -> str:
        """
        对任务进行分类
        
        Django常见任务类型:
        - bug_fix: 修复bug
        - feature: 新功能
        - refactor: 重构
        - test: 测试相关
        - docs: 文档相关
        - performance: 性能优化
        """
        # 简单实现：基于关键词匹配
        problem = experience.problem_statement.lower()
        
        if any(kw in problem for kw in ['bug', 'fix', 'error', 'exception', 'crash']):
            return 'bug_fix'
        elif any(kw in problem for kw in ['add', 'implement', 'feature', 'support']):
            return 'feature'
        elif any(kw in problem for kw in ['refactor', 'cleanup', 'simplify']):
            return 'refactor'
        elif any(kw in problem for kw in ['test', 'unittest', 'pytest']):
            return 'test'
        else:
            return 'general'
    
    def get_all_experiences(self) -> List[CodingExperience]:
        """获取所有experience用于induction"""
        all_exp = []
        for task_type, exps in self.buffer.items():
            all_exp.extend(exps)
        return all_exp
    
    def get_experiences_by_type(self, task_type: str) -> List[CodingExperience]:
        """获取特定类型的experience"""
        return self.buffer.get(task_type, [])
```

### 4.4 Induction Module设计

```python
class WorkflowInductionModule:
    """
    从成功的experiences中归纳可复用的workflows
    遵循AWM原论文的LM-based induction方法
    """
    
    def __init__(self, llm_client, model_name: str = "kimi-k2"):
        self.llm_client = llm_client
        self.model_name = model_name
        
    def induce(self, experiences: List[CodingExperience]) -> List[Workflow]:
        """
        从experiences中归纳workflows
        
        遵循原论文:
        - 提取跨task重复的action子集
        - 每个workflow至少3-8步
        - 用变量名替换具体值
        """
        # 1. 准备输入：将experiences格式化为prompt
        formatted_experiences = self._format_experiences(experiences)
        
        # 2. 调用LLM进行归纳
        prompt = self._build_induction_prompt(formatted_experiences)
        response = self.llm_client.generate(prompt)
        
        # 3. 解析输出为workflow结构
        workflows = self._parse_workflows(response)
        
        return workflows
    
    def _format_experiences(self, experiences: List[CodingExperience]) -> str:
        """
        将experiences格式化为LLM可理解的格式
        """
        formatted = []
        for i, exp in enumerate(experiences):
            exp_str = f"""
=== Experience {i+1} ===
Instance: {exp.instance_id}
Problem: {exp.problem_statement[:500]}...  # 截断避免过长

Trajectory:
{self._format_history(exp.history)}

Solution (Diff Patch):
{exp.diff_patch}
"""
            formatted.append(exp_str)
        
        return "\n\n".join(formatted)
    
    def _format_history(self, history: List[HistoryStep]) -> str:
        """格式化history为可读格式"""
        lines = []
        for step in history:
            lines.append(f"Step {step.step_id}:")
            lines.append(f"  Thought: {step.thought}")
            lines.append(f"  Action: {step.action_type} - {step.action[:100]}")
        return "\n".join(lines)
    
    def _build_induction_prompt(self, formatted_experiences: str) -> str:
        """
        构建归纳prompt，遵循AWM原论文的设计
        """
        return f"""Given a list of successful Django bug-fixing/feature-implementation trajectories, extract common reusable workflows.

Each trajectory contains:
- Problem statement (GitHub issue description)
- Sequence of (thought, action) steps taken by the agent
- Final solution (diff patch)

Requirements:
1. Extract SUB-ROUTINES that appear across multiple tasks, not full task solutions
2. Abstract specific values: replace "fix bug in models/query.py" with "fix bug in {{target_file}}"
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


Now, analyze the following successful experiences and extract common workflows:

{formatted_experiences}

Extract 3-5 common workflows from the above experiences. Use the same format as the example.
Separate each workflow with "---".
"""

    def _parse_workflows(self, response: str) -> List[Workflow]:
        """
        解析LLM输出为Workflow结构
        """
        workflows = []
        
        # 按 "---" 分割不同workflow
        workflow_texts = response.split("---")
        
        for text in workflow_texts:
            text = text.strip()
            if not text or "## Workflow:" not in text:
                continue
                
            workflow = self._parse_single_workflow(text)
            if workflow:
                workflows.append(workflow)
        
        return workflows
    
    def _parse_single_workflow(self, text: str) -> Optional[Workflow]:
        """解析单个workflow文本"""
        try:
            # 提取workflow名称
            name_match = re.search(r"## Workflow: (.+)", text)
            name = name_match.group(1) if name_match else "Unnamed Workflow"
            
            # 提取描述
            desc_match = re.search(r"Description: (.+)", text)
            description = desc_match.group(1) if desc_match else ""
            
            # 提取适用场景
            scenario_match = re.search(r"Applicable scenarios: (.+)", text)
            scenarios = scenario_match.group(1).split(", ") if scenario_match else []
            
            # 提取步骤
            steps = self._parse_steps(text)
            
            return Workflow(
                name=name,
                description=description,
                applicable_scenarios=scenarios,
                steps=steps
            )
        except Exception as e:
            print(f"Failed to parse workflow: {e}")
            return None
    
    def _parse_steps(self, text: str) -> List[WorkflowStep]:
        """解析workflow步骤"""
        steps = []
        # 匹配 "数字. [xxx]" 格式的步骤
        step_pattern = r"(\d+)\.\s+\[(\w+)\]\s+(.+?)(?=\n\s*Action:|$)"
        action_pattern = r"Action:\s+(.+?)(?=\n\d+\.|\n---|$)"
        
        step_matches = re.findall(step_pattern, text, re.DOTALL)
        action_matches = re.findall(action_pattern, text, re.DOTALL)
        
        for i, (step_num, step_type, reasoning) in enumerate(step_matches):
            action = action_matches[i].strip() if i < len(action_matches) else ""
            steps.append(WorkflowStep(
                step_type=step_type,
                reasoning=reasoning.strip(),
                action_template=action
            ))
        
        return steps


@dataclass
class Workflow:
    """Workflow数据结构"""
    name: str
    description: str
    applicable_scenarios: List[str]
    steps: List[WorkflowStep]
    
    def to_prompt_string(self) -> str:
        """转换为可插入prompt的字符串格式"""
        lines = [
            f"### {self.name}",
            f"Description: {self.description}",
            f"When to use: {', '.join(self.applicable_scenarios)}",
            "Steps:"
        ]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"  {i}. [{step.step_type}] {step.reasoning}")
            lines.append(f"     Action: {step.action_template}")
        
        return "\n".join(lines)


@dataclass
class WorkflowStep:
    """Workflow单步数据结构"""
    step_type: str           # e.g., "Understand", "Locate", "Fix"
    reasoning: str           # 为什么要做这一步
    action_template: str     # action模板，包含placeholder
```

### 4.5 Memory Integration设计

```python
class MemoryManager:
    """
    管理Agent的Memory，包括base memory和workflow memory
    遵循AWM原论文: 直接append所有workflow到memory
    """
    
    def __init__(self, base_memory: str):
        self.base_memory = base_memory
        self.workflows: List[Workflow] = []
        
    def add_workflows(self, new_workflows: List[Workflow]):
        """
        添加新的workflows到memory
        遵循原论文: 直接append，不做selection/retrieval
        """
        self.workflows.extend(new_workflows)
    
    def get_full_memory(self) -> str:
        """
        获取完整的memory string，用于注入到agent prompt
        遵循原论文: M + W，完全塞入
        """
        workflow_section = self._format_workflows()
        
        return f"""{self.base_memory}

## Learned Workflows from Past Successful Experiences

The following workflows have been extracted from previously solved Django issues.
Use them as guidance when solving similar problems.

{workflow_section}
"""

    def _format_workflows(self) -> str:
        """格式化所有workflows为prompt string"""
        if not self.workflows:
            return "(No workflows learned yet)"
        
        formatted = []
        for i, wf in enumerate(self.workflows, 1):
            formatted.append(f"--- Workflow {i} ---")
            formatted.append(wf.to_prompt_string())
            formatted.append("")
        
        return "\n".join(formatted)
```

---

## 5. 完整Online Loop

```python
class AWMOnlineLoop:
    """
    AWM Online Learning Loop
    完整实现online workflow learning
    """
    
    def __init__(
        self,
        agent: OpenHandsAgent,
        pipeline: OnlineEvaluationPipeline,
        induction_module: WorkflowInductionModule,
        induction_trigger_count: int = 10
    ):
        self.agent = agent
        self.pipeline = pipeline
        self.induction_module = induction_module
        
        self.experience_buffer = ExperienceBuffer(induction_trigger_count)
        self.memory_manager = MemoryManager(agent.base_system_prompt)
        
        # 统计
        self.total_processed = 0
        self.total_success = 0
        
    def run(self, django_issues: List[DjangoIssue]):
        """
        运行完整的online learning loop
        
        Args:
            django_issues: Django issues列表，按默认顺序处理
        """
        results = []
        
        for issue in django_issues:
            self.total_processed += 1
            print(f"\n{'='*60}")
            print(f"Processing issue {self.total_processed}/{len(django_issues)}: {issue.instance_id}")
            
            # Step 1: 获取当前memory (包含已学习的workflows)
            current_memory = self.memory_manager.get_full_memory()
            self.agent.set_system_prompt(current_memory)
            
            # Step 2: Inference + Immediate Evaluation
            experience = self.pipeline.process_single_sample(
                instance_id=issue.instance_id,
                problem_statement=issue.problem_statement,
                agent=self.agent,
                workflow_memory=self.memory_manager.workflows
            )
            
            results.append(experience)
            
            # Step 3: 如果成功，加入buffer
            if experience.test_result == "PASS":
                self.total_success += 1
                print(f"✓ PASSED - Success rate: {self.total_success}/{self.total_processed}")
                
                should_induce = self.experience_buffer.add(experience)
                
                # Step 4: 触发workflow induction
                if should_induce:
                    print(f"\n>>> Triggering workflow induction (buffer size: {len(self.experience_buffer.get_all_experiences())})")
                    self._run_induction()
            else:
                print(f"✗ FAILED - Success rate: {self.total_success}/{self.total_processed}")
        
        return results
    
    def _run_induction(self):
        """运行workflow induction"""
        all_experiences = self.experience_buffer.get_all_experiences()
        
        # 归纳新workflows
        new_workflows = self.induction_module.induce(all_experiences)
        
        print(f"Induced {len(new_workflows)} new workflows:")
        for wf in new_workflows:
            print(f"  - {wf.name}")
        
        # 添加到memory (直接append，遵循原论文)
        self.memory_manager.add_workflows(new_workflows)
        
    def get_statistics(self) -> dict:
        """获取运行统计"""
        return {
            "total_processed": self.total_processed,
            "total_success": self.total_success,
            "success_rate": self.total_success / max(1, self.total_processed),
            "workflows_learned": len(self.memory_manager.workflows),
            "buffer_size": len(self.experience_buffer.get_all_experiences())
        }
```

---

## 6. Django数据集准备

```python
@dataclass
class DjangoIssue:
    """Django issue数据结构"""
    instance_id: str           # e.g., "django__django-11039"
    problem_statement: str     # GitHub issue描述
    # 其他SWE-bench元数据...


def load_django_issues_from_swebench(
    swebench_path: str,
    split: str = "test"
) -> List[DjangoIssue]:
    """
    从SWE-bench数据集加载Django相关的issues
    
    假设我们已经有了Django issues的index列表
    """
    # 加载完整数据集
    dataset = load_dataset("princeton-nlp/SWE-bench", split=split)
    
    # 过滤Django issues
    django_issues = []
    for item in dataset:
        if item["repo"] == "django/django":
            django_issues.append(DjangoIssue(
                instance_id=item["instance_id"],
                problem_statement=item["problem_statement"]
            ))
    
    return django_issues


# Django issues在SWE-bench中的大致数量:
# - SWE-bench full: ~300个Django issues
# - SWE-bench lite: ~30个Django issues
```

---

## 7. 需要魔改的关键文件

### 7.1 OpenHands相关

```
openhands/
├── evaluation/
│   └── benchmarks/
│       └── swe_bench/
│           ├── scripts/
│           │   ├── run_infer.sh      # 需要修改：支持single sample
│           │   └── eval_infer.sh     # 需要修改：支持single sample
│           ├── run_infer.py          # 需要修改：抽取为可复用函数
│           └── eval_infer.py         # 需要修改：抽取为可复用函数
```

### 7.2 核心修改点

```python
# 1. run_infer.py 修改
# 原始: def run_infer(instances: List[Instance], ...):
# 修改: def run_single_infer(instance: Instance, ...):

# 2. eval_infer.py 修改  
# 原始: def eval_infer(output_file: str, ...):
# 修改: def eval_single_instance(instance_id: str, diff_patch: str, ...):

# 3. 新增文件
# openhands/evaluation/awm/
# ├── __init__.py
# ├── experience.py         # Experience数据结构
# ├── buffer.py             # ExperienceBuffer
# ├── induction.py          # WorkflowInductionModule
# ├── memory.py             # MemoryManager
# ├── pipeline.py           # OnlineEvaluationPipeline
# └── loop.py               # AWMOnlineLoop
```

---

## 8. 实现优先级

| 优先级 | 任务 | 描述 |
|--------|------|------|
| P0 | 魔改SWE-bench脚本 | 支持单个sample的inference和evaluation |
| P0 | Experience数据结构 | 定义清晰的experience格式 |
| P1 | Online Pipeline | 实现inference→evaluation的即时流程 |
| P1 | Experience Buffer | 存储成功的experiences |
| P2 | Induction Module | LLM-based workflow归纳 |
| P2 | Memory Integration | 将workflows注入agent prompt |
| P3 | 完整Loop | 整合所有模块运行online learning |

---

## 9. 待确认的问题

1. **Docker资源管理**：每个sample都需要启动/关闭Docker容器，是否会有性能瓶颈？是否可以复用容器？

2. **Induction触发频率**：每10个成功experience触发一次是否合适？是否需要动态调整？

3. **任务类型分类**：简单的关键词匹配是否足够？是否需要LLM-based分类？

4. **Context长度**：完全塞入所有workflows后，context是否会超限？Django ~300个issue能产生多少workflows？

5. **错误处理**：如果Docker容器启动失败、测试超时等情况如何处理？

---

*文档版本: V1.0*
*更新日期: 2024*
