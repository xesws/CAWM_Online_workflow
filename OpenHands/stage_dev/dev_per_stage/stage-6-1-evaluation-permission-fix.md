# Stage 6.1: Evaluation Permission Fix

> 任务: 修复 `single_evaluation.py` 中 `/tmp/eval.sh` 执行权限问题
> 状态: **已完成** ✅
> 完成日期: 2025-11-30

---

## 1. 问题描述

### 1.1 症状

运行 AWM 测试时，所有 5 个测试用例都失败，错误信息相同：

```json
{
  "test_result": "FAIL",
  "test_output": "bash: /tmp/eval.sh: Permission denied"
}
```

### 1.2 影响范围

- 所有 AWM 评估任务都无法正确执行测试
- Agent 的 patch 虽然生成成功，但无法验证是否正确

---

## 2. 根因分析

### 2.1 问题代码位置

**文件**: `evaluation/awm/single_evaluation.py`

**原始代码 (lines 152-155)**:
```python
# 设置执行权限
action = CmdRunAction(command="chmod +x /tmp/eval.sh")
action.set_hard_timeout(600)
obs = runtime.run_action(action)  # ❌ 没有检查 exit_code!
```

**执行测试 (line 181)**:
```python
action = CmdRunAction(command=f"/tmp/eval.sh > {log_file} 2>&1 & echo $!")
```

### 2.2 与原始代码对比

**原始 `eval_infer.py` (lines 197-203)** 有正确的错误检查：
```python
# Set +x
action = CmdRunAction(command='chmod +x /tmp/eval.sh')
obs = runtime.run_action(action)
assert obs.exit_code == 0  # ✅ 有错误检查
```

### 2.3 根因

1. `chmod +x /tmp/eval.sh` 命令执行失败（可能原因：容器权限、noexec 挂载等）
2. AWM 版本没有检查 `chmod` 的退出码
3. 后续直接执行 `/tmp/eval.sh` 需要文件有执行权限
4. 因为文件没有执行权限，bash 报错 "Permission denied"

---

## 3. 解决方案

### 3.1 修复策略

采用双重保险：
1. 添加 `chmod` 失败的日志警告
2. 使用 `bash /tmp/eval.sh` 代替 `/tmp/eval.sh`（不需要执行权限）

### 3.2 代码变更

**变更 1: 添加错误检查 (line 157-159)**

```python
# 检查 chmod 是否成功
if isinstance(obs, CmdOutputObservation) and obs.exit_code != 0:
    logger.warning(f"chmod +x failed: {obs.content}, will use bash to run script")
```

**变更 2: 使用 bash 执行脚本 (line 185)**

```python
# Before:
action = CmdRunAction(command=f"/tmp/eval.sh > {log_file} 2>&1 & echo $!")

# After:
action = CmdRunAction(command=f"bash /tmp/eval.sh > {log_file} 2>&1 & echo $!")
```

### 3.3 为什么 `bash /tmp/eval.sh` 有效

| 执行方式 | 需要执行权限 | 说明 |
|----------|-------------|------|
| `/tmp/eval.sh` | ✅ 需要 | 操作系统需要文件有 +x 权限才能执行 |
| `bash /tmp/eval.sh` | ❌ 不需要 | bash 读取文件内容并解释执行，只需要读权限 |

---

## 4. 文件修改清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `evaluation/awm/single_evaluation.py` | 修改 | 添加 chmod 检查 + 使用 bash 执行 |

---

## 5. 验证方法

重新运行 AWM 测试：

```bash
poetry run python -m evaluation.awm.cli \
  --llm-config llm.kimi-k2 \
  --limit 5 \
  --output-dir evaluation/evaluation_outputs/awm_test
```

### 预期结果

- 不再出现 "Permission denied" 错误
- 测试能够正常执行并返回 PASS/FAIL 结果

---

## 6. 与其他 Stage 的关系

```
Stage 1: Infrastructure (single_inference.py)
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
Stage 6.1: Evaluation Permission Fix (本阶段) ← Bug Fix
```

**依赖关系**:
- Stage 6.1 是 Stage 1 创建的 `single_evaluation.py` 的 Bug Fix
- 不影响其他 Stage 的功能

---

## 7. 经验总结

### 7.1 Bug 来源

在从 `eval_infer.py` 复制代码到 `single_evaluation.py` 时，遗漏了关键的错误检查：
- 原始代码: `assert obs.exit_code == 0`
- AWM 版本: 没有任何检查

### 7.2 最佳实践

1. **所有命令执行后都应检查退出码**
2. **使用 `bash script.sh` 比依赖 `chmod +x` 更可靠**
3. **复制代码时要特别注意错误处理逻辑**

---

## 8. 完整 Diff

```diff
--- a/evaluation/awm/single_evaluation.py
+++ b/evaluation/awm/single_evaluation.py
@@ -152,6 +152,10 @@ class SingleEvaluationRunner:
             action = CmdRunAction(command="chmod +x /tmp/eval.sh")
             action.set_hard_timeout(600)
             obs = runtime.run_action(action)
+
+            # 检查 chmod 是否成功
+            if isinstance(obs, CmdOutputObservation) and obs.exit_code != 0:
+                logger.warning(f"chmod +x failed: {obs.content}, will use bash to run script")

             # 应用patch
             apply_cmd = (
@@ -178,8 +182,8 @@ class SingleEvaluationRunner:
                     report={"failed_apply_patch": True, "resolved": False},
                 )

-            # 运行测试
+            # 运行测试 (使用 bash 执行脚本，避免权限问题)
             log_file = "/tmp/eval_output.log"
-            action = CmdRunAction(command=f"/tmp/eval.sh > {log_file} 2>&1 & echo $!")
+            action = CmdRunAction(command=f"bash /tmp/eval.sh > {log_file} 2>&1 & echo $!")
             action.set_hard_timeout(300)
             obs = runtime.run_action(action)
```
