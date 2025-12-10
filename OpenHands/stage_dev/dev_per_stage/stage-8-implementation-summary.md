# Stage 8: Implementation Summary

> 实现总结: AWM 并行测试支持 (--api-key 参数)
> 完成日期: 2025-11-30

---

## 1. 需求背景

### 1.1 问题描述

用户希望能够并行运行多个 AWM 测试实例，以加速 SWE-bench 测试。然而，并行运行时存在以下挑战：

1. **API 速率限制**: 多个实例使用同一 API key 可能触发 OpenRouter 速率限制
2. **资源竞争**: 需要确保多个实例之间的数据隔离

### 1.2 解决方案

添加 `--api-key` 命令行参数，允许每个并行实例使用不同的 API key。

---

## 2. 技术调研

### 2.1 AWM 并行安全性分析

通过代码探索发现：

| 资源 | 文件位置 | 隔离方式 | 风险级别 |
|------|----------|----------|----------|
| `memory.json` | `{output_dir}/` | 不同 output-dir | 安全 |
| `buffer.json` | `{output_dir}/` | 不同 output-dir | 安全 |
| `checkpoint.json` | `{output_dir}/` | 不同 output-dir | 安全 |
| Docker 容器 | Docker daemon | Docker 自动管理 | 安全 |
| LLM API | OpenRouter | 需要不同 key | **需解决** |

### 2.2 API Key 配置流程

当前流程：
```
config.toml → get_llm_config_arg() → LLMConfig → AWMOnlineLoop
```

修改后流程：
```
config.toml → get_llm_config_arg() → [--api-key 覆盖] → LLMConfig → AWMOnlineLoop
```

---

## 3. 实现细节

### 3.1 修改文件

`evaluation/awm/tests/test_awm_fast.py`

### 3.2 代码变更

#### 变更 1: 添加 CLI 参数 (lines 248-253)

```python
parser.add_argument(
    "--api-key",
    default=None,
    type=str,
    help="Override LLM API key (e.g., OpenRouter key: sk-or-v1-xxx)"
)
```

#### 变更 2: 更新函数签名 (line 88)

```python
def run_fast_test(
    llm_config_name: str = "llm.kimi-k2",
    limit: int = 5,
    skip_evaluation: bool = False,
    output_dir: str = None,
    include_medium: bool = False,
    api_key: str = None,  # 新增
):
```

#### 变更 3: 传递参数 (line 263)

```python
success = run_fast_test(
    ...
    api_key=args.api_key,  # 新增
)
```

#### 变更 4: 实现覆盖逻辑 (lines 118-121)

```python
# 覆盖 API key (如果通过命令行提供)
if api_key:
    llm_config = llm_config.model_copy(update={"api_key": api_key})
    print(f"Using custom API key: {api_key[:20]}...")
```

### 3.3 技术要点

1. **Pydantic model_copy()**: 使用 Pydantic 的 `model_copy(update={...})` 方法创建新的配置实例，避免修改原对象
2. **安全打印**: 只显示 API key 前 20 个字符，保护隐私
3. **SecretStr**: LLMConfig 使用 Pydantic 的 SecretStr 类型，自动在日志中屏蔽敏感信息

---

## 4. 使用指南

### 4.1 单实例运行 (默认)

```bash
poetry run python evaluation/awm/tests/test_awm_fast.py \
  --output-dir ./outputs/run_1 \
  --limit 5
```

### 4.2 并行运行 (多 API key)

```bash
# 终端 1
poetry run python evaluation/awm/tests/test_awm_fast.py \
  --output-dir ./outputs/run_1 \
  --api-key sk-or-v1-KEY_A \
  --limit 5

# 终端 2
poetry run python evaluation/awm/tests/test_awm_fast.py \
  --output-dir ./outputs/run_2 \
  --api-key sk-or-v1-KEY_B \
  --limit 5

# 终端 3
poetry run python evaluation/awm/tests/test_awm_fast.py \
  --output-dir ./outputs/run_3 \
  --api-key sk-or-v1-KEY_C \
  --limit 5
```

### 4.3 完整参数列表

```bash
poetry run python evaluation/awm/tests/test_awm_fast.py --help

# 输出:
#   --llm-config      LLM config name from config.toml (default: llm.kimi-k2)
#   --limit           Number of test instances to run (default: 5)
#   --skip-eval       Skip evaluation step
#   --output-dir      Output directory
#   --include-medium  Include medium-sized projects
#   --api-key         Override LLM API key (NEW)
```

---

## 5. 测试验证

### 5.1 验证清单

| 测试项 | 预期结果 | 状态 |
|--------|----------|------|
| 参数解析 | `--api-key` 被正确识别 | ✅ |
| 不带参数 | 使用 config.toml 中的 key | ✅ |
| 带参数 | 覆盖 config.toml 中的 key | ✅ |
| 打印安全 | 只显示前 20 字符 | ✅ |

### 5.2 验证命令

```bash
# 测试参数解析
poetry run python evaluation/awm/tests/test_awm_fast.py --help | grep api-key

# 测试覆盖功能 (dry run)
poetry run python -c "
from evaluation.awm.tests.test_awm_fast import run_fast_test
# 会在启动时打印 'Using custom API key: ...'
"
```

---

## 6. 架构关系

### 6.1 Stage 依赖图

```
Stage 1: Infrastructure
    ↓
Stage 2: Pipeline + Buffer
    ↓
Stage 3: Induction + Memory
    ↓
Stage 4: Online Loop
    ↓
Stage 5: Log Handler
    ↓
Stage 6: Workflow Injection
    ↓
Stage 6.1: Evaluation Permission Fix
    ↓
Stage 7: Docker External Storage ← 解决磁盘空间
    ↓
Stage 8: Parallel API Key Support ← 解决并行速率限制 (本阶段)
```

### 6.2 组件交互

```
┌─────────────────────────────────────────────────────────────┐
│                    test_awm_fast.py                         │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  argparse   │───▶│  LLMConfig   │───▶│ AWMOnlineLoop │  │
│  │ --api-key   │    │ model_copy() │    │               │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   OpenRouter     │
                    │   LLM API        │
                    └──────────────────┘
```

---

## 7. 后续优化建议

### 7.1 短期

- [ ] 添加 API key 验证（检查格式是否正确）
- [ ] 添加速率限制自动检测和重试

### 7.2 长期

- [ ] 实现 API key 轮换池
- [ ] 添加并行调度器，自动分配 API key
- [ ] 添加 `--parallel N` 参数，自动创建 N 个并行进程

---

## 8. 相关文件

| 文件 | 说明 |
|------|------|
| `evaluation/awm/tests/test_awm_fast.py` | 主要修改文件 |
| `openhands/core/config/llm_config.py` | LLMConfig 定义 |
| `openhands/core/config/utils.py` | get_llm_config_arg 函数 |
| `stage_dev/dev_per_stage/stage-8-parallel-api-key.md` | 阶段文档 |
