# Stage 11: AWM Workflow Memory Truncation - 实现总结

## 功能概述

实现了 `--truncation` CLI 参数，用于控制 workflow memory 注入的 token 数量限制，防止 context overflow。

## 参数格式

```bash
# 启用 truncation（默认 50000 tokens）
python -m evaluation.awm.cli --llm-config llm.eval --truncation

# 指定具体的 token 限制
python -m evaluation.awm.cli --llm-config llm.eval --truncation=30000

# 禁用 truncation（默认行为）
python -m evaluation.awm.cli --llm-config llm.eval --truncation=false
```

## 设计决策

- **单位**: Token 数量（使用 tiktoken）
- **保留策略**: FIFO（保留最旧的 workflows，截断最新的）
- **默认值**: 50000 tokens

## 修改的文件

### 1. 新建 `evaluation/awm/token_utils.py`

Token counting 工具函数：
- `count_tokens(text, model)` - 计算文本的 token 数量
- `truncate_to_token_limit(text, max_tokens, model, keep_start)` - 截断文本到指定 token 数量
- `DEFAULT_TRUNCATION_LIMIT = 50000` - 默认限制

### 2. 修改 `evaluation/awm/config.py`

在 `AWMConfig` 中添加：
```python
truncation_limit: Optional[int] = None  # None = 禁用, int = token 限制
```

### 3. 修改 `evaluation/awm/cli.py`

- 添加 `parse_truncation_value()` 函数解析参数
- 添加 `--truncation` 参数（支持 true/false/数字）
- 在创建 `AWMConfig` 时传入 `truncation_limit`

### 4. 修改 `evaluation/awm/memory.py`

- `__init__` 添加 `truncation_limit` 参数
- `get_augmented_prompt()` 调用新方法 `_format_workflows_with_truncation()`
- 新增 `_format_workflows_with_truncation()` 方法：
  - 计算 base prompt token 开销
  - FIFO 策略：从最旧的 workflow 开始添加
  - 达到 token 限制时停止并记录日志
  - 添加截断提示信息
- `get_statistics()` 添加 `truncation_limit` 和 `augmented_prompt_tokens` 字段

### 5. 修改 `evaluation/awm/loop.py`

在 `_init_components()` 中传递 `truncation_limit` 到 `MemoryManager`

## 数据流

```
CLI (--truncation)
    ↓
AWMConfig (truncation_limit: int | None)
    ↓
AWMOnlineLoop
    ↓
MemoryManager (truncation_limit)
    ↓
get_augmented_prompt() → _format_workflows_with_truncation()
    ↓
SingleInferenceRunner._inject_workflow_to_instruction()
```

## 测试验证

```
=== 无 truncation ===
Workflows: 5
Prompt tokens: 1202

=== 有 truncation (1000 tokens) ===
Workflows: 5
Prompt tokens: 771
日志: Truncation applied: included 3/5 workflows (666 tokens, limit: 794)
```

## 依赖

`tiktoken` 已作为 litellm 的传递依赖存在于 poetry.lock 中，无需额外添加。
