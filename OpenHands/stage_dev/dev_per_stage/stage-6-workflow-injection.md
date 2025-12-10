# Stage 6: Workflow Memory Injection

> 任务: 实现 `single_inference.py` 中缺失的 `_inject_custom_prompt()` 方法，使学习到的 Workflows 能够注入到 Agent 的决策过程中
> 状态: **已实现** ✅
> 优先级: **高** (阻塞 AWM 核心功能)

---

## 1. 问题诊断

### 1.1 当前状态

`single_inference.py:160-180` 中的 `_inject_custom_prompt()` 是一个 TODO:

```python
def _inject_custom_prompt(self, metadata, custom_prompt):
    # TODO: 实现custom prompt注入
    logger.warning(
        "Custom system prompt injection not yet implemented. "
        "Using default system prompt."
    )
    return metadata
```

### 1.2 影响范围

```
loop.py: memory_manager.get_augmented_prompt()     ✅ 生成 augmented prompt
    ↓
pipeline.py: custom_system_prompt 参数              ✅ 传递 prompt
    ↓
single_inference.py: _inject_custom_prompt()       ❌ TODO - 未实现
    ↓
Agent 实际使用 prompt                               ❌ 没有收到 workflow
```

**结论**: 学到的 Workflows 没有被实际注入到 Agent 的决策过程中。

---

## 2. OpenHands Prompt 架构分析

### 2.1 System Prompt 生成流程

```
1. CodeActAgent.__init__()
   ↓
2. PromptManager(prompt_dir, system_prompt_filename='system_prompt.j2')
   ↓
3. conversation_memory._ensure_system_message()
   ↓
4. prompt_manager.get_system_message(cli_mode=...)
   ↓
5. SystemMessageAction(content=system_prompt) → 插入到 events[0]
```

### 2.2 Instruction Message 生成流程 (run_infer.py)

```
get_instruction(instance, metadata)
    ↓
template = env.get_template(template_name)  # e.g., 'swe_default.j2'
    ↓
instruction = template.render(context)
    ↓
return MessageAction(content=instruction)  # 这是用户的任务描述
```

### 2.3 关键配置点

| 配置项 | 位置 | 作用 |
|--------|------|------|
| `AgentConfig.system_prompt_filename` | `agent_config.py:22` | 选择 system prompt 模板 |
| `EvalMetadata.instruction_template_name` | `shared.py:56` | 选择 instruction 模板 |
| `RuntimeInfo.additional_agent_instructions` | `prompt.py:16` | 追加到 workspace context |

---

## 3. 实现方案

### 推荐方案: 修改 Instruction Message (最简单)

**原理**: 在 `single_inference.py` 中，获取 instruction 后，将 workflow memory 追加到 instruction 内容中。

**优点**:
- 不需要修改 OpenHands 核心代码
- 不需要创建自定义模板文件
- 完全在 AWM 模块内部实现
- 符合 AWM 论文的 "M + W" 方法（在任务描述中注入 workflows）

**实现位置**: `evaluation/awm/single_inference.py`

### 3.1 修改后的代码

```python
# single_inference.py

def run(
    self,
    instance: pd.Series,
    workflow_memory: Optional[List[Any]] = None,
    custom_system_prompt: Optional[str] = None,
) -> InferenceOutput:
    """..."""
    from evaluation.benchmarks.swe_bench.run_infer import (
        get_config,
        initialize_runtime,
        complete_runtime,
        get_instruction,
        AGENT_CLS_TO_FAKE_USER_RESPONSE_FN,
        set_dataset_type,
    )
    from openhands.core.main import create_runtime, run_controller
    from openhands.utils.async_utils import call_async_from_sync

    # ... 原有代码 ...

    # 获取instruction
    message_action = get_instruction(instance, metadata)

    # 🔥 新增: 注入 Workflow Memory 到 instruction
    if custom_system_prompt:
        message_action = self._inject_workflow_to_instruction(
            message_action,
            custom_system_prompt
        )

    # 运行controller
    state = asyncio.run(run_controller(...))
    # ...

def _inject_workflow_to_instruction(
    self,
    message_action: MessageAction,
    workflow_prompt: str,
) -> MessageAction:
    """
    将 workflow memory 注入到 instruction message 中

    遵循 AWM 论文的 "M + W" 方法：
    - M: 原始任务描述 (instruction)
    - W: 已学习的 workflows
    """
    # 构建增强后的 instruction
    augmented_content = f"""## Learned Workflows (Use these patterns when applicable)

{workflow_prompt}

---

## Your Task

{message_action.content}
"""

    # 返回新的 MessageAction
    from openhands.events.action import MessageAction as MA
    return MA(
        content=augmented_content,
        image_urls=message_action.image_urls if hasattr(message_action, 'image_urls') else None,
    )
```

### 3.2 替代方案对比

| 方案 | 复杂度 | 侵入性 | 推荐度 |
|------|--------|--------|--------|
| **修改 Instruction (推荐)** | 低 | 无 | ⭐⭐⭐⭐⭐ |
| 自定义 System Prompt 模板 | 中 | 需创建模板文件 | ⭐⭐⭐ |
| 修改 AgentConfig | 高 | 需改 OpenHands 核心 | ⭐⭐ |
| 通过 RuntimeInfo | 中 | 需理解 RecallObservation | ⭐⭐ |

---

## 4. 实现步骤

### Step 1: 修改 `single_inference.py`

1. 找到 `get_instruction()` 调用位置 (约 line 106)
2. 在调用后添加 workflow 注入逻辑
3. 实现 `_inject_workflow_to_instruction()` 方法
4. 删除原来的 `_inject_custom_prompt()` TODO 方法

### Step 2: 验证修改

```python
# 测试代码
from evaluation.awm.single_inference import SingleInferenceRunner

runner = SingleInferenceRunner(llm_config)
output = runner.run(
    instance=test_instance,
    custom_system_prompt="## Workflow: Debug Test\n1. Read error log..."
)

# 检查 instruction 是否包含 workflow
print(output.metadata['instruction_preview'])  # 需要添加此字段用于调试
```

### Step 3: 端到端测试

```bash
poetry run python -m evaluation.awm.cli \
  --llm-config llm.kimi-k2 \
  --limit 3 \
  --induction-trigger 1 \
  --output-dir evaluation/evaluation_outputs/awm_injection_test
```

观察日志是否显示:
- `Augmented prompt injected to instruction` (新增日志)
- 第二个任务开始时应该看到 workflows 被注入

---

## 5. 文件修改清单

| 文件 | 操作 | 修改内容 |
|------|------|----------|
| `evaluation/awm/single_inference.py` | 修改 | 实现 `_inject_workflow_to_instruction()` |

---

## 6. 验收标准

| 项目 | 预期 |
|------|------|
| Workflow 注入 | 第二个任务的 instruction 包含已学习的 workflows |
| 日志输出 | 显示 "Workflow memory injected" |
| Agent 行为 | Agent 在处理任务时参考注入的 workflows |
| 不破坏现有功能 | 当没有 workflows 时，行为与原来一致 |

---

## 7. 与其他 Stage 的关系

```
Stage 1: Infrastructure (single_inference.py 创建)
    ↓
Stage 3: Induction + Memory (memory_manager.get_augmented_prompt())
    ↓
Stage 4: Online Loop (调用 pipeline.process_single_sample())
    ↓
Stage 5: Log Handler (压缩 experience)
    ↓
Stage 6: Workflow Injection (本阶段 - 实际注入 workflows) ← 当前
```

**依赖关系**:
- Stage 6 是 Stage 1 的补丁，修复了 Stage 1 遗留的 TODO
- Stage 3/4 已经准备好了 workflow prompt，只等 Stage 6 实现注入
- Stage 5 的压缩功能与 Stage 6 独立，可并行开发
