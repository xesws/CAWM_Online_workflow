# Stage 8: AWM 并行测试支持 (--api-key 参数)

> 任务: 为 AWM 测试脚本添加 `--api-key` 命令行参数，支持并行测试时使用不同的 API key
> 状态: **已完成** ✅
> 创建日期: 2025-11-30
> 完成日期: 2025-11-30

---

## 1. 背景

### 1.1 问题

在并行运行多个 AWM 测试实例时，如果使用同一个 API key，可能会遇到：
- OpenRouter API 速率限制
- 请求配额耗尽
- 请求被拒绝

### 1.2 解决方案

添加 `--api-key` 命令行参数，允许每个并行实例使用不同的 API key。

---

## 2. 并行测试原理

AWM 没有内置并行支持，但可以通过以下方式安全并行：

| 资源 | 隔离方式 | 说明 |
|------|----------|------|
| `memory.json` | 不同 output-dir | 每个实例独立的工作流记忆 |
| `buffer.json` | 不同 output-dir | 每个实例独立的经验缓冲 |
| `checkpoint.json` | 不同 output-dir | 每个实例独立的进度 |
| Docker 容器 | Docker 管理 | Docker 自动处理并行 |
| LLM API | 不同 api-key | 避免速率限制 |

---

## 3. 实现方案

### 修改文件
`evaluation/awm/tests/test_awm_fast.py`

### 3.1 添加 CLI 参数

```python
parser.add_argument(
    "--api-key",
    default=None,
    type=str,
    help="Override LLM API key (e.g., OpenRouter key: sk-or-v1-xxx)"
)
```

### 3.2 函数签名更新

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

### 3.3 API Key 覆盖逻辑

```python
llm_config = get_llm_config_arg(llm_config_name)
if api_key:
    llm_config = llm_config.model_copy(update={"api_key": api_key})
    print(f"Using custom API key: {api_key[:20]}...")
```

---

## 4. 使用示例

### 4.1 单实例 (使用 config.toml 中的 key)

```bash
poetry run python evaluation/awm/tests/test_awm_fast.py \
  --output-dir ./outputs/run_1 \
  --limit 5
```

### 4.2 并行实例 (不同 API key)

```bash
# 终端 1
poetry run python evaluation/awm/tests/test_awm_fast.py \
  --output-dir ./outputs/run_1 \
  --api-key sk-or-v1-AAAA... \
  --limit 5

# 终端 2
poetry run python evaluation/awm/tests/test_awm_fast.py \
  --output-dir ./outputs/run_2 \
  --api-key sk-or-v1-BBBB... \
  --limit 5

# 终端 3
poetry run python evaluation/awm/tests/test_awm_fast.py \
  --output-dir ./outputs/run_3 \
  --api-key sk-or-v1-CCCC... \
  --limit 5
```

---

## 5. 资源建议

### 5.1 并行数量

基于你的硬件配置 (20 CPU, 32GB RAM):
- 推荐: 2-3 个并行实例
- 最大: 4-5 个并行实例

### 5.2 瓶颈分析

| 资源 | 使用情况 | 说明 |
|------|----------|------|
| CPU | 中等 | Docker 容器运行测试时 CPU 密集 |
| 内存 | 中等 | 每个容器约 2-4GB |
| 磁盘 | 低 | 已配置外置硬盘 (Stage 7) |
| 网络 | 高 | LLM API 调用 |

---

## 6. 安全考虑

- API key 会在命令行历史中显示
- 打印时只显示前 20 个字符
- LLMConfig 使用 Pydantic SecretStr，不会泄露到日志

---

## 7. 与其他 Stage 的关系

```
Stage 1: Infrastructure (single_inference.py, single_evaluation.py)
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
Stage 7: Docker External Storage ✅
    ↓
Stage 8: Parallel API Key Support (本阶段)
```

**依赖关系**:
- Stage 8 依赖 Stage 7 (外置硬盘存储，避免多实例磁盘空间不足)
- Stage 8 扩展 Stage 1 的测试脚本功能

---

## 8. 验收标准

| 项目 | 预期结果 |
|------|----------|
| CLI 参数解析 | `--api-key` 参数被正确识别 |
| API key 覆盖 | 使用 `--api-key` 时覆盖 config.toml 中的值 |
| 并行运行 | 多个实例可以同时运行，互不干扰 |
| 日志安全 | API key 不会完整显示在日志中 |
