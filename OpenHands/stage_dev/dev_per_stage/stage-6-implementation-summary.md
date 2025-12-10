# Stage 6: Workflow Memory Injection - Implementation Summary

> 状态: **已完成** ✅
> 完成日期: 2025-11-29

---

## 1. 实现目标

修复 `single_inference.py` 中的 `_inject_custom_prompt()` TODO，使学习到的 Workflows 能够实际注入到 Agent 的决策过程中。

---

## 2. 问题诊断

### 2.1 原始问题

`single_inference.py` 中的 `_inject_custom_prompt()` 是一个未实现的 TODO：

```python
def _inject_custom_prompt(self, metadata, custom_prompt):
    # TODO: 实现custom prompt注入
    logger.warning(
        "Custom system prompt injection not yet implemented. "
        "Using default system prompt."
    )
    return metadata
```

### 2.2 影响范围

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

## 3. 解决方案

### 3.1 方案选择

采用 **修改 Instruction Message** 方案（最简单、最符合 AWM 论文）：

| 方案 | 复杂度 | 侵入性 | 选择 |
|------|--------|--------|------|
| **修改 Instruction (选择)** | 低 | 无 | ✅ |
| 自定义 System Prompt 模板 | 中 | 需创建模板文件 | - |
| 修改 AgentConfig | 高 | 需改 OpenHands 核心 | - |

### 3.2 实现原理

遵循 AWM 论文的 "M + W" 方法：
- **M**: 原始任务描述 (instruction)
- **W**: 已学习的 workflows

在 `get_instruction()` 返回后，将 workflow memory 前置到 instruction 内容中。

---

## 4. 代码变更

### 4.1 文件修改

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `evaluation/awm/single_inference.py` | 修改 | 实现 workflow 注入 |

### 4.2 具体变更

#### 删除旧代码 (lines 90-92)

```python
# 删除：
if custom_system_prompt:
    metadata = self._inject_custom_prompt(metadata, custom_system_prompt)
```

#### 新增注入逻辑 (lines 104-109)

```python
# 获取instruction
message_action = get_instruction(instance, metadata)

# 注入 Workflow Memory 到 instruction (AWM "M + W" 方法)
if custom_system_prompt:
    message_action = self._inject_workflow_to_instruction(
        message_action,
        custom_system_prompt
    )
```

#### 新增方法 (lines 163-202)

```python
def _inject_workflow_to_instruction(
    self,
    message_action,
    workflow_prompt: str,
):
    """
    将 workflow memory 注入到 instruction message 中

    遵循 AWM 论文的 "M + W" 方法：
    - M: 原始任务描述 (instruction)
    - W: 已学习的 workflows
    """
    from openhands.events.action import MessageAction

    # 构建增强后的 instruction
    augmented_content = f"""## Learned Workflows (Use these patterns when applicable)

{workflow_prompt}

---

## Your Task

{message_action.content}
"""

    logger.info("Workflow memory injected into instruction")

    return MessageAction(
        content=augmented_content,
        image_urls=message_action.image_urls if hasattr(message_action, 'image_urls') else None,
    )
```

---

## 5. 数据流（修复后）

```
loop.py: memory_manager.get_augmented_prompt()     ✅ 生成 augmented prompt
    ↓
pipeline.py: custom_system_prompt 参数              ✅ 传递 prompt
    ↓
single_inference.py: _inject_workflow_to_instruction() ✅ 注入到 instruction
    ↓
Agent 实际使用 prompt                               ✅ 收到 workflows
```

---

## 6. 验证方法

### 6.1 日志验证

运行 AWM 时，第二个任务开始时应该看到：

```
INFO: Workflow memory injected into instruction
```

### 6.2 端到端测试

```bash
poetry run python -m evaluation.awm.cli \
  --llm-config llm.your-model \
  --limit 3 \
  --induction-trigger 1 \
  --output-dir evaluation/evaluation_outputs/awm_injection_test
```

观察：
1. 第一个任务：无 workflow 注入（因为还没学到任何 workflow）
2. 第一个任务成功后：触发 induction，学习 workflow
3. 第二个任务：应该看到 "Workflow memory injected" 日志

---

## 7. 注入后的 Instruction 格式

Agent 收到的 instruction 格式如下：

```markdown
## Learned Workflows (Use these patterns when applicable)

## Workflow: Debug Django Test Failure
Description: A workflow for debugging failing Django tests
Applicable scenarios: test failure, assertion error

Steps:
1. [Understand] Read the error message...
   Action: search_code("{{error_keyword}}")
...

---

## Your Task

[原始的 problem_statement 内容]
```

---

## 8. 与其他 Stage 的关系

```
Stage 1: Infrastructure (single_inference.py 创建)
    ↓
Stage 3: Induction + Memory (memory_manager.get_augmented_prompt())
    ↓
Stage 4: Online Loop (调用 pipeline.process_single_sample())
    ↓
Stage 5: Log Handler (压缩 experience)
    ↓
Stage 6: Workflow Injection (本阶段 - 实际注入 workflows) ✅
```

**依赖关系**:
- Stage 6 是 Stage 1 的补丁，修复了 Stage 1 遗留的 TODO
- Stage 3/4 已经准备好了 workflow prompt，Stage 6 完成了最后一环
- Stage 5 的压缩功能与 Stage 6 独立

---

## 9. 总结

| 项目 | 状态 |
|------|------|
| 问题诊断 | ✅ 识别 TODO 未实现 |
| 方案设计 | ✅ 选择修改 Instruction 方案 |
| 代码实现 | ✅ `_inject_workflow_to_instruction()` |
| 集成完成 | ✅ 调用点正确 |
| 文档更新 | ✅ 本文件 |

**AWM 核心功能现已完整**: Workflows 可以被学习、存储、并注入到后续任务中。
