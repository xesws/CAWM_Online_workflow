# AWM Online Learning Module Running Guide

> AWM (Agentic Working Memory) online learning module for OpenHands agents to learn and improve from successful experiences.

---

## 1. Prerequisites

### 1.1 LLM Configuration (config.toml)

Configure your LLM in `OpenHands/config.toml`:

```toml
[core]
workspace_base="./workspace"

# Example: Using OpenRouter
[llm.kimi-k2]
model = "openrouter/moonshotai/kimi-k2-0905"
api_key = "your-api-key"
base_url = "https://openrouter.ai/api/v1"

# Example: Using OpenAI
[llm.gpt4]
model = "gpt-4"
api_key = "your-openai-api-key"

# Example: Using Anthropic
[llm.claude]
model = "claude-3-sonnet-20240229"
api_key = "your-anthropic-api-key"
```

### 1.2 Environment Requirements

| Requirement | Description |
|-------------|-------------|
| **Docker** | Must be installed and running (for SWE-bench evaluation containers) |
| **Python Dependencies** | Install via `poetry install` |
| **Network Connection** | First run requires downloading SWE-bench dataset |
| **Disk Space** | ~10GB for Docker images and dataset |

---

## 2. Command Line Parameters Reference

### 2.1 Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--llm-config` | string | **Required**. LLM config name from config.toml (e.g., `llm.kimi-k2`, `llm.gpt4`) |

### 2.2 Dataset Parameters

| Parameter | Type | Default | Description | Example Values |
|-----------|------|---------|-------------|----------------|
| `--dataset` | string | `princeton-nlp/SWE-bench` | HuggingFace dataset name | `princeton-nlp/SWE-bench`, `princeton-nlp/SWE-bench_Lite` |
| `--split` | string | `test` | Dataset split to use | `test`, `dev`, `train` |
| `--repo-filter` | string | `django/django` | Filter issues by repository. **Use empty string `""` for all repos** | `django/django`, `astropy/astropy`, `""` (all repos) |
| `--limit` | int | None | Limit number of issues to process (useful for testing) | `5`, `10`, `50`, `None` (all) |

### 2.3 Agent Parameters

| Parameter | Type | Default | Description | Example Values |
|-----------|------|---------|-------------|----------------|
| `--agent-class` | string | `CodeActAgent` | Agent class to use | `CodeActAgent` |
| `--max-iterations` | int | `100` | Maximum iterations per issue before timeout | `50`, `100`, `150` |

### 2.4 Workflow Learning Parameters

| Parameter | Type | Default | Description | Example Values |
|-----------|------|---------|-------------|----------------|
| `--induction-trigger` | int | `10` | Trigger workflow induction after every N successful experiences | `3`, `5`, `10` |
| `--max-workflows` | int | `50` | Maximum workflows to keep in memory | `20`, `50`, `100` |
| `--truncation` | string/int | None | Token limit for workflow memory injection. See below for details | `true`, `false`, `50000` |

**Truncation parameter values:**
- `--truncation` or `--truncation true` → Use default limit (50,000 tokens)
- `--truncation false` → Disable truncation (inject full workflow memory)
- `--truncation 30000` → Custom token limit

### 2.5 Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output-dir` | string | `evaluation/evaluation_outputs/awm` | Directory for output files |

### 2.6 Runtime Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--resume-from` | string | None | Path to checkpoint file for resuming interrupted runs |
| `--skip-evaluation` | flag | False | Skip evaluation step (for debugging only) |
| `--quiet` | flag | False | Suppress verbose output (disable progress printing) |

---

## 3. Running Commands

### 3.1 Quick Test (Recommended for First Run)

```bash
cd OpenHands

# Test with 3 issues to verify setup
poetry run python -m evaluation.awm.cli \
    --llm-config llm.kimi-k2 \
    --limit 3 \
    --output-dir evaluation/evaluation_outputs/awm_test
```

### 3.2 Run on Django Issues (Default)

```bash
# Process Django issues from SWE-bench (default behavior)
poetry run python -m evaluation.awm.cli \
    --llm-config llm.kimi-k2 \
    --max-iterations 100
```

### 3.3 Run on Full SWE-bench Lite (~300 instances)

```bash
# IMPORTANT: Use --repo-filter "" to process ALL repositories
poetry run python -m evaluation.awm.cli \
    --llm-config llm.kimi-k2 \
    --dataset "princeton-nlp/SWE-bench_Lite" \
    --repo-filter "" \
    --output-dir evaluation/evaluation_outputs/awm_swebench_lite \
    --max-iterations 100 \
    --induction-trigger 10 \
    --max-workflows 50
```

### 3.4 Run with Workflow Memory Truncation

```bash
# Enable truncation with default limit (50,000 tokens)
poetry run python -m evaluation.awm.cli \
    --llm-config llm.kimi-k2 \
    --dataset "princeton-nlp/SWE-bench_Lite" \
    --repo-filter "" \
    --truncation \
    --output-dir evaluation/evaluation_outputs/awm_truncated

# Enable truncation with custom limit
poetry run python -m evaluation.awm.cli \
    --llm-config llm.kimi-k2 \
    --truncation 30000 \
    --output-dir evaluation/evaluation_outputs/awm_truncated_30k
```

### 3.5 Resume from Checkpoint

```bash
# Resume interrupted run from checkpoint
poetry run python -m evaluation.awm.cli \
    --llm-config llm.kimi-k2 \
    --dataset "princeton-nlp/SWE-bench_Lite" \
    --repo-filter "" \
    --output-dir evaluation/evaluation_outputs/awm_swebench_lite \
    --resume-from evaluation/evaluation_outputs/awm_swebench_lite/checkpoint.json
```

### 3.6 Quiet Mode (Minimal Output)

```bash
poetry run python -m evaluation.awm.cli \
    --llm-config llm.kimi-k2 \
    --quiet \
    --output-dir evaluation/evaluation_outputs/awm_quiet
```

---

## 4. Progress Monitoring

### 4.1 Real-time Console Output

When running with verbose mode (default), you'll see rolling progress:

```
============================================================
Processing issue 42/300: astropy__astropy-12907
Success rate so far: 15/41
Workflows learned: 8
✓ PASSED - astropy__astropy-12907

============================================================
Processing issue 43/300: django__django-11099
Success rate so far: 16/42
Workflows learned: 8
✗ FAILED - django__django-11099
```

**Information displayed:**
- Current issue number and total count
- Running success rate (pass/total processed)
- Number of workflows learned so far
- Individual test result (PASS/FAIL)

### 4.2 Real-time File Output

Results are written in real-time to `results.jsonl`:

```bash
# Monitor results in real-time
tail -f evaluation/evaluation_outputs/awm/results.jsonl | jq .
```

---

## 5. Checkpoint and Resume

### 5.1 Automatic Checkpointing

AWM automatically saves checkpoints every 5 samples (configurable in `AWMConfig.checkpoint_interval`).

**Files preserved:**
| File | Content |
|------|---------|
| `checkpoint.json` | Progress counters (total_processed, total_success) |
| `results.jsonl` | All completed results (appended, not overwritten on resume) |
| `memory.json` | Learned workflows |
| `buffer.json` | Experience buffer |

### 5.2 Checkpoint File Structure

```json
{
  "total_processed": 42,
  "total_success": 15,
  "start_time": "2024-11-30T19:31:00.123456",
  "last_checkpoint": "2024-11-30T19:45:30.987654"
}
```

### 5.3 Resume Behavior

When resuming:
1. Progress counters are restored from `checkpoint.json`
2. `results.jsonl` is **appended to** (not cleared)
3. Learned workflows are loaded from `memory.json`
4. Experience buffer is loaded from `buffer.json`
5. Processing continues from where it stopped

---

## 6. Output File Structure

```
evaluation/evaluation_outputs/awm/
├── config.json        # Run configuration
├── checkpoint.json    # Checkpoint for resume
├── buffer.json        # Experience buffer (successful experiences)
├── memory.json        # Learned workflows
├── results.jsonl      # Real-time results (one JSON per line)
└── result.json        # Final summary (after completion)
```

### 6.1 File Descriptions

| File | Purpose | When Written |
|------|---------|--------------|
| `config.json` | Complete run configuration | At start |
| `checkpoint.json` | Progress state for resume | Every 5 samples |
| `buffer.json` | Stores successful experiences by task type | On each success |
| `memory.json` | Workflows inducted from experiences | After induction |
| `results.jsonl` | Each result as JSON line (real-time) | After each issue |
| `result.json` | Final statistics and all results | At completion |

---

## 7. Python API Usage

```python
from evaluation.awm.config import AWMConfig
from evaluation.awm.loop import AWMOnlineLoop, load_django_issues

# Create configuration
config = AWMConfig(
    llm_config_name="llm.kimi-k2",
    dataset_name="princeton-nlp/SWE-bench_Lite",
    repo_filter="",  # All repos
    induction_trigger_count=10,
    max_workflows=50,
    truncation_limit=50000,  # Enable truncation
    output_dir="evaluation/evaluation_outputs/awm_api",
)

# Load dataset
issues = load_django_issues(
    dataset_name=config.dataset_name,
    split=config.split,
    repo_filter=config.repo_filter,
    limit=50,  # Optional limit
)

# Run
loop = AWMOnlineLoop(config)
result = loop.run(issues)

# View results
print(f"Success rate: {result.statistics['success_rate']:.2%}")
print(f"Workflows learned: {len(result.workflows)}")
for wf in result.workflows:
    print(f"  - {wf.name}")
```

---

## 8. AWM Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      AWM Online Learning Loop                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐    ┌────────────┐    ┌────────────┐               │
│   │  Issue  │───▶│ Inference  │───▶│ Evaluation │               │
│   └─────────┘    │ (+ Workflow│    └────────────┘               │
│                  │  Memory)   │           │                      │
│                  └────────────┘           ▼                      │
│                        ▲           ┌────────────┐                │
│                        │           │  PASS/FAIL │                │
│                        │           └────────────┘                │
│                        │                  │                      │
│                        │         ┌────────┴────────┐             │
│                        │         ▼                 ▼             │
│                        │    ┌─────────┐      ┌─────────┐         │
│                        │    │  PASS   │      │  FAIL   │         │
│                        │    └─────────┘      └─────────┘         │
│                        │         │                │              │
│                        │         ▼                │              │
│                        │  ┌────────────┐          │              │
│                        │  │  Buffer    │          │              │
│                        │  │ (Store Exp)│          │              │
│                        │  └────────────┘          │              │
│                        │         │                │              │
│                        │    ┌────┴────┐           │              │
│                        │    │ N hits? │           │              │
│                        │    └────┬────┘           │              │
│                        │         │ Yes            │              │
│                        │         ▼                │              │
│                        │  ┌────────────┐          │              │
│                        │  │ Induction  │          │              │
│                        │  │(Learn WF)  │          │              │
│                        │  └────────────┘          │              │
│                        │         │                │              │
│                        │         ▼                │              │
│                        │  ┌────────────┐          │              │
│                        └──│  Memory    │◀─────────┘              │
│                           │(Inject WF) │                         │
│                           └────────────┘                         │
│                                  │                               │
│                                  ▼                               │
│                           Next Issue                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Docker connection failed | Ensure Docker Desktop is running |
| LLM API error | Check api_key in config.toml |
| Dataset loading failed | Check network connection, may need proxy |
| Out of memory | Reduce `--max-iterations` or use smaller dataset |
| Run interrupted | Use `--resume-from checkpoint.json` to continue |
| Too much console output | Use `--quiet` flag or check Stage 10 truncation |

---

## 10. Cost Considerations

- Each issue may consume **significant tokens** (inference + evaluation)
- Full SWE-bench Lite (~300 instances) can be expensive
- Recommendations:
  - Test with `--limit 5` first
  - Use cheaper models for initial testing
  - Monitor token usage in your LLM provider dashboard

---

## 11. Example: Complete SWE-bench Lite Run

```bash
# Step 1: Test setup with 3 issues
poetry run python -m evaluation.awm.cli \
    --llm-config llm.kimi-k2 \
    --dataset "princeton-nlp/SWE-bench_Lite" \
    --repo-filter "" \
    --limit 3 \
    --output-dir evaluation/evaluation_outputs/awm_test

# Step 2: If test passes, run full dataset
poetry run python -m evaluation.awm.cli \
    --llm-config llm.kimi-k2 \
    --dataset "princeton-nlp/SWE-bench_Lite" \
    --repo-filter "" \
    --max-iterations 100 \
    --induction-trigger 10 \
    --max-workflows 50 \
    --truncation \
    --output-dir evaluation/evaluation_outputs/awm_swebench_lite_full

# Step 3: If interrupted, resume
poetry run python -m evaluation.awm.cli \
    --llm-config llm.kimi-k2 \
    --dataset "princeton-nlp/SWE-bench_Lite" \
    --repo-filter "" \
    --output-dir evaluation/evaluation_outputs/awm_swebench_lite_full \
    --resume-from evaluation/evaluation_outputs/awm_swebench_lite_full/checkpoint.json

# Step 4: Monitor progress
tail -f evaluation/evaluation_outputs/awm_swebench_lite_full/results.jsonl | jq .
```

---

## 12. Related Files

- **Code Location**: `OpenHands/evaluation/awm/`
- **Development Docs**: `OpenHands/stage_dev/dev_per_stage/`
- **Config File**: `OpenHands/config.toml`
- **CLI Entry Point**: `evaluation/awm/cli.py`
- **Main Loop**: `evaluation/awm/loop.py`
- **Configuration**: `evaluation/awm/config.py`

---

*Last Updated: 2024-11-30*
