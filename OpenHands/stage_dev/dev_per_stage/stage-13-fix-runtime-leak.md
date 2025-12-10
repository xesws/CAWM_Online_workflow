# Stage 13: Fix Runtime File Descriptor Leak

## Problem

When running AWM on full SWE-bench Lite (300 samples), the process crashed with:
```
[Errno 24] Too many open files
```

 In openhands/runtime/base.py:164:
  atexit.register(self.close)

  Every runtime instance registers its close() method with atexit. This is a safety fallback.

  The 12+ atexit errors mean:
  1. 12+ runtime instances were created
  2. Their close() was NOT called during normal operation
  3. When program exited, atexit tried to call close() on all of them
  4. But file descriptors were already exhausted, so close() failed too
   
  The connect() failure is a cascade effect, not the root cause:

  1. Root cause: Runtimes not being closed (my fixes address this)
  2. Effect: File descriptors leak
  3. Cascade: After ~10 samples, file descriptors exhausted
  4. Symptom: connect() starts failing, which before my fix caused MORE leaks
  My fixes ensure close() is always called via try-finally blocks, preventing the initial leak.

**Impact:**
- 289 out of 300 samples failed
- Error started at sample 11
- Only 10 samples could run before file descriptors were exhausted

## Root Cause

File descriptor leak in runtime creation/connection flow. Three locations had the pattern:

```python
runtime = create_runtime(config)
call_async_from_sync(runtime.connect)  # If this fails, runtime leaks!
try:
    ...
```

If `runtime.connect()` failed after runtime creation, the runtime object was never closed, leaking file descriptors.

## Solution

### Fix 1: `single_inference.py` (lines 102-108)

```python
# Before
runtime = create_runtime(config)
call_async_from_sync(runtime.connect)
try:
    ...

# After
runtime = create_runtime(config)
try:
    call_async_from_sync(runtime.connect)
except Exception:
    runtime.close()  # Close runtime if connect fails
    raise
try:
    ...
```

### Fix 2: `single_evaluation.py` (lines 145-151)

Same pattern - wrap `runtime.connect()` in try-except to close runtime on failure.

### Fix 3: `pipeline.py` (lines 114-146)

Move `_run_inference()` inside the try block to ensure finally always cleans up:

```python
# Before
inference_output, runtime = self._run_inference(...)  # Outside try
try:
    ...
finally:
    if runtime is not None:
        runtime.close()

# After
runtime = None  # Initialize for finally block safety
try:
    inference_output, runtime = self._run_inference(...)  # Inside try
    ...
finally:
    if runtime is not None:
        runtime.close()
```

## Files Modified

| File | Changes |
|------|---------|
| `evaluation/awm/single_inference.py` | Wrap `runtime.connect()` in try-except |
| `evaluation/awm/single_evaluation.py` | Wrap `runtime.connect()` in try-except |
| `evaluation/awm/pipeline.py` | Move inference call inside try block |

## Expected Result

- No more "Too many open files" errors
- All 300 samples can run without file descriptor exhaustion
- Proper cleanup even when errors occur

## Testing

Run full SWE-bench Lite to verify:
```bash
poetry run python -m evaluation.awm.cli \
    --llm-config llm.kimi-k2 \
    --dataset "princeton-nlp/SWE-bench_Lite" \
    --repo-filter "" \
    --output-dir evaluation/evaluation_outputs/awm_stage13_test
```

Monitor with:
```bash
lsof -p <pid> | wc -l  # Should stay stable, not grow
```


## log at the problem point:
{============================================================
Processing issue 11/300: django__django-11039
Success rate so far: 4/10
Workflows learned: 4
04:05:59 - openhands:INFO: pipeline.py:99 - Processing instance: django__django-11039
04:06:01 - openhands:INFO: pipeline.py:114 - Running inference for django__django-11039...
04:06:01 - openhands:INFO: run_infer.py:94 - Dataset type set to: SWE-bench
04:06:01 - openhands:INFO: shared.py:191 - Using evaluation output directory: evaluation/evaluation_outputs/awm_swebench_lite/awm_single_inference/CodeActAgent/kimi-k2-0905_maxiter_100_N_awm
04:06:01 - openhands:INFO: shared.py:212 - Metadata: {"agent_class":"CodeActAgent","llm_config":{"model":"openrouter/moonshotai/kimi-k2-0905","api_key":"**********","base_url":"https://openrouter.ai/api/v1","api_version":null,"aws_access_key_id":null,"aws_secret_access_key":null,"aws_region_name":null,"openrouter_site_url":"https://docs.all-hands.dev/","openrouter_app_name":"OpenHands","num_retries":5,"retry_multiplier":8.0,"retry_min_wait":8,"retry_max_wait":64,"timeout":null,"max_message_chars":30000,"temperature":0.0,"top_p":1.0,"top_k":null,"custom_llm_provider":null,"max_input_tokens":null,"max_output_tokens":null,"input_cost_per_token":null,"output_cost_per_token":null,"ollama_base_url":null,"drop_params":true,"modify_params":true,"disable_vision":null,"disable_stop_word":false,"caching_prompt":true,"log_completions":false,"log_completions_folder":"/Users/tangyiq/dev/OpenHands/OpenHands/logs/completions","custom_tokenizer":null,"native_tool_calling":null,"reasoning_effort":null,"seed":null,"safety_settings":null,"for_routing":false,"correct_num":5,"completion_kwargs":null},"agent_config":null,"max_iterations":100,"eval_output_dir":"evaluation/evaluation_outputs/awm_swebench_lite/awm_single_inference/CodeActAgent/kimi-k2-0905_maxiter_100_N_awm","start_time":"2025-12-01 04:06:01","git_commit":"38ad33b440370abc1951ae65cfe8df797810cb66","dataset":"awm_single_inference","data_split":null,"details":{"mode":"swe"},"condenser_config":{"type":"noop"},"instruction_template_name":null}
04:06:01 - openhands:INFO: run_infer.py:217 - Using instance container image: docker.io/swebench/sweb.eval.x86_64.django_1776_django-11039:latest. Please make sure this image exists. Submit an issue on https://github.com/OpenHands/OpenHands if you run into any issues.
04:06:01 - openhands:INFO: mapping.py:26 - Resource mapping for awm_single_inference not found.
04:06:01 - openhands:WARNING: utils.py:778 - Model routing config section [model_routing] not found in config.toml
04:06:01 - openhands:INFO: llm_registry.py:87 - [LLM registry e8c693c6-178c-4715-9a54-7c4fe98e208f]: Registering service for agent
04:06:02 - openhands:INFO: runtime_build.py:191 - Building image: ghcr.io/openhands/runtime:oh_v0.62.0_wo3vgyeqtzznmdpy_nwmvmsqqfs4zahb6
================ DOCKER BUILD STARTED ================
04:11:13 - openhands:INFO: docker.py:225 - Image [ghcr.io/openhands/runtime:oh_v0.62.0_wo3vgyeqtzznmdpy_nwmvmsqqfs4zahb6] build finished.
04:11:13 - openhands:INFO: docker.py:230 - Re-tagged image [ghcr.io/openhands/runtime:oh_v0.62.0_wo3vgyeqtzznmdpy_nwmvmsqqfs4zahb6] with more generic tag [oh_v0.62.0_wo3vgyeqtzznmdpy]
04:11:13 - openhands:INFO: docker.py:246 - Image ghcr.io/openhands/runtime with tags [oh_v0.62.0_wo3vgyeqtzznmdpy_nwmvmsqqfs4zahb6, oh_v0.62.0_wo3vgyeqtzznmdpy] built successfully
04:11:15 - openhands:INFO: docker_runtime.py:182 - [runtime 16ffaf38-2e43-4c-39114426476e346] Starting runtime with image: ghcr.io/openhands/runtime:oh_v0.62.0_wo3vgyeqtzznmdpy_nwmvmsqqfs4zahb6
04:11:15 - openhands:INFO: docker_runtime.py:495 - [runtime 16ffaf38-2e43-4c-39114426476e346] Starting server with command: ['/openhands/micromamba/bin/micromamba', 'run', '-n', 'openhands', 'poetry', 'run', 'python', '-u', '-m', 'openhands.runtime.action_execution_server', '31447', '--working-dir', '/workspace', '--plugins', 'agent_skills', 'jupyter', '--username', 'root', '--user-id', '0', '--no-enable-browser']
04:11:37 - openhands:INFO: docker_runtime.py:186 - [runtime 16ffaf38-2e43-4c-39114426476e346] Container started: openhands-runtime-16ffaf38-2e43-4c-39114426476e346. VSCode URL: None
04:11:37 - openhands:INFO: docker_runtime.py:197 - [runtime 16ffaf38-2e43-4c-39114426476e346] Waiting for client to become ready at http://localhost:31447...
04:11:53 - openhands:INFO: docker_runtime.py:203 - [runtime 16ffaf38-2e43-4c-39114426476e346] Runtime is ready.
04:11:56 - openhands:INFO: base.py:1061 - Successfully configured git: name=openhands, email=openhands@all-hands.dev
04:11:56 - openhands:INFO: run_infer.py:280 - ------------------------------
04:11:56 - openhands:INFO: run_infer.py:281 - BEGIN Runtime Initialization Fn
04:11:56 - openhands:INFO: run_infer.py:282 - ------------------------------
04:11:56 - ACTION
**CmdRunAction (source=None, is_input=False)**
COM... [truncated]
04:11:57 - OBSERVATION
**CmdOutputObservation (source=None, exit code=0, ... [truncated]
04:11:57 - ACTION
**CmdRunAction (source=None, is_input=False)**
COM... [truncated]
04:11:58 - OBSERVATION
**CmdOutputObservation (source=None, exit code=0, ... [truncated]
04:11:58 - ACTION
**CmdRunAction (source=None, is_input=False)**
COM... [truncated]
04:11:58 - OBSERVATION
**CmdOutputObservation (source=None, exit code=0, ... [truncated]
04:11:58 - ACTION
**CmdRunAction (source=None, is_input=False)**
COM... [truncated]
04:11:59 - OBSERVATION
**CmdOutputObservation (source=None, exit code=0, ... [truncated]
04:11:59 - ACTION
**CmdRunAction (source=None, is_input=False)**
COM... [truncated]
04:12:00 - OBSERVATION
**CmdOutputObservation (source=None, exit code=0, ... [truncated]
04:12:00 - ACTION
**CmdRunAction (source=None, is_input=False)**
COM... [truncated]
04:12:02 - OBSERVATION
**CmdOutputObservation (source=None, exit code=0, ... [truncated]
04:12:02 - ACTION
**CmdRunAction (source=None, is_input=False)**
COM... [truncated]
04:12:03 - OBSERVATION
**CmdOutputObservation (source=None, exit code=0, ... [truncated]
04:12:03 - ACTION
**CmdRunAction (source=None, is_input=False)**
COM... [truncated]
04:12:03 - OBSERVATION
**CmdOutputObservation (source=None, exit code=0, ... [truncated]
04:12:03 - ACTION
**CmdRunAction (source=None, is_input=False)**
COM... [truncated]
04:12:04 - OBSERVATION
**CmdOutputObservation (source=None, exit code=0, ... [truncated]
04:12:04 - ACTION
**CmdRunAction (source=None, is_input=False)**
COM... [truncated]
04:12:05 - OBSERVATION
**CmdOutputObservation (source=None, exit code=0, ... [truncated]
04:12:05 - openhands:INFO: run_infer.py:435 - ------------------------------
04:12:05 - openhands:INFO: run_infer.py:436 - END Runtime Initialization Fn
04:12:05 - openhands:INFO: run_infer.py:437 - ------------------------------
04:12:05 - openhands:INFO: single_inference.py:213 - Workflow memory injected into instruction
04:12:05 - openhands:INFO: llm_registry.py:87 - [LLM registry ba5b3289-5048-4030-80ae-c43e0c1cb750]: Registering service for agent
04:12:05 - openhands:INFO: base.py:890 - [runtime 16ffaf38-2e43-4c-39114426476e346] Selected repo: None, loading microagents from /workspace/.openhands/microagents (inside runtime)
04:12:05 - openhands:INFO: base.py:637 - [runtime 16ffaf38-2e43-4c-39114426476e346] Attempting to list files in repository microagents directory: /workspace/.openhands/microagents
04:12:05 - openhands:INFO: memory.py:261 - Loading user workspace microagents: []
04:12:05 - openhands:INFO: state_tracker.py:94 - AgentController 16ffaf38-2e43-4c-39114426476e346 - created new state. start_id: 0
04:12:05 - openhands:INFO: agent_controller.py:672 - [Agent Controller 16ffaf38-2e43-4c-39114426476e346] Setting agent(CodeActAgent) state from AgentState.LOADING to AgentState.RUNNING
04:12:05 - openhands:INFO: memory.py:244 - Microagent 'github' triggered by keyword 'github'
04:12:05 - openhands:INFO: memory.py:244 - Microagent 'bitbucket' triggered by keyword 'bitbucket'
04:12:05 - openhands:INFO: memory.py:244 - Microagent 'npm' triggered by keyword 'npm'
04:12:05 - openhands:INFO: memory.py:244 - Microagent 'gitlab' triggered by keyword 'gitlab'
04:12:05 - openhands:INFO: memory.py:244 - Microagent 'security' triggered by keyword 'security'
04:12:05 - openhands:INFO: conversation_stats.py:65 - Saved conversation stats
04:29:34 - openhands:INFO: agent_controller.py:672 - [Agent Controller 16ffaf38-2e43-4c-39114426476e346] Setting agent(CodeActAgent) state from AgentState.RUNNING to AgentState.FINISHED
04:29:34 - openhands:INFO: conversation_stats.py:65 - Saved conversation stats
04:29:35 - openhands:INFO: run_infer.py:450 - ------------------------------
04:29:35 - openhands:INFO: run_infer.py:451 - BEGIN Runtime Completion Fn
04:29:35 - openhands:INFO: run_infer.py:452 - ------------------------------
04:29:35 - ACTION
**CmdRunAction (source=None, is_input=False)**
COM... [truncated]
04:29:35 - OBSERVATION
**CmdOutputObservation (source=None, exit code=0, ... [truncated]
04:29:35 - ACTION
**CmdRunAction (source=None, is_input=False)**
COM... [truncated]
04:29:36 - OBSERVATION
**CmdOutputObservation (source=None, exit code=0, ... [truncated]
04:29:36 - ACTION
**CmdRunAction (source=None, is_input=False)**
COM... [truncated]
04:29:37 - OBSERVATION
**CmdOutputObservation (source=None, exit code=0, ... [truncated]
04:29:37 - ACTION
**CmdRunAction (source=None, is_input=False)**
COM... [truncated]
04:29:38 - OBSERVATION
**CmdOutputObservation (source=None, exit code=0, ... [truncated]
04:29:38 - ACTION
**CmdRunAction (source=None, is_input=False)**
COM... [truncated]
04:29:39 - OBSERVATION
**CmdOutputObservation (source=None, exit code=0, ... [truncated]
04:29:39 - ACTION
**CmdRunAction (source=None, is_input=False)**
COM... [truncated]
04:29:40 - OBSERVATION
**CmdOutputObservation (source=None, exit code=0, ... [truncated]
04:29:40 - ACTION
FileReadAction(path='patch.diff', start=0, end=-1,... [truncated]
04:29:40 - OBSERVATION
[Read from /workspace/django__django__3.0/patch.di... [truncated]
04:29:40 - openhands:INFO: run_infer.py:603 - ------------------------------
04:29:40 - openhands:INFO: run_infer.py:604 - END Runtime Completion Fn
04:29:40 - openhands:INFO: run_infer.py:605 - ------------------------------
04:29:40 - openhands:INFO: pipeline.py:121 - Inference complete. Patch length: 33424
04:29:40 - openhands:INFO: pipeline.py:133 - Running evaluation for django__django-11039 (reusing runtime)...
04:29:40 - openhands:INFO: single_evaluation.py:149 - Reusing runtime from inference phase, resetting testbed...
04:29:42 - openhands:INFO: single_evaluation.py:285 - Testbed reset successful
04:30:17 - openhands:INFO: pipeline.py:139 - Evaluation complete. Resolved: True
04:30:17 - openhands:ERROR: loop.py:203 - Error processing django__django-11039: ('Connection aborted.', OSError(24, 'Too many open files'))

============================================================
Processing issue 12/300: django__django-11049
Success rate so far: 4/11
Workflows learned: 4}