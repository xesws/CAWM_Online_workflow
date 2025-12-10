import os
import sys
import json
from pathlib import Path
from typing import List
import time
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from evaluation.awm.experience import CodingExperience, HistoryStep
from evaluation.awm.log_handler import ExperienceLogHandler, CompressedExperience
from evaluation.awm.induction import WorkflowInductionModule
from evaluation.awm.workflow import Workflow
from openhands.core.config import LLMConfig

# Configuration
OUTPUT_DIR = "evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Lite-test/CodeActAgent/kimi-k2-0905_maxiter_100_N_v0.62.0-no-hint-run_1"
TARGET_INSTANCE = "django__django-12286"
TEST_OUTPUT_DIR = os.path.dirname(__file__)

LLM_CONFIG = LLMConfig(
    model="openrouter/moonshotai/kimi-k2-0905",
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
)

def get_api_key_value(config: LLMConfig) -> str:
    """Helper to get string value of API key"""
    if hasattr(config.api_key, 'get_secret_value'):
        return config.api_key.get_secret_value()
    return str(config.api_key)

def load_experience_from_output(instance_id: str) -> dict:
    """Load a specific instance from output.jsonl"""
    output_path = os.path.join(project_root, OUTPUT_DIR, "output.jsonl")
    
    if not os.path.exists(output_path):
        print(f"Error: Data file not found at {output_path}")
        raise FileNotFoundError(f"Data file not found at {output_path}")

    with open(output_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("instance_id") == instance_id:
                    return data
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Instance {instance_id} not found")

def convert_history(raw_history: list) -> list[HistoryStep]:
    """Convert output.jsonl history format to HistoryStep format"""
    steps = []
    prev_observation = ""

    for i, item in enumerate(raw_history):
        action_type = item.get("action", "unknown")
        message = item.get("message", "")
        if action_type == "system": continue

        thought = message if action_type == "think" else ""
        action_content = message
        if item.get("args"):
            action_content = f"{message} | args: {json.dumps(item['args'])[:200]}"

        step = HistoryStep(
            step_id=i,
            observation=prev_observation[:500],
            thought=thought[:500],
            action=action_content[:500],
            action_type=action_type,
        )
        steps.append(step)
        prev_observation = message
    return steps

def plot_timeline(compressed: CompressedExperience):
    """Draw an ASCII timeline of phases"""
    print("\n" + "=" * 40)
    print(" EXPERIENCE TIMELINE VISUALIZATION")
    print("=" * 40)
    
    total_steps = compressed.original_step_count
    if total_steps == 0:
        print("No steps to visualize.")
        return

    # Calculate scale (e.g., 50 chars wide)
    # Note: Compressed phases don't strictly map to step counts unless we stored start/end indices.
    # But LogHandler merges chunks of `chunk_size`. We can estimate.
    # Each chunk is roughly `chunk_size` steps.
    
    # We will list phases and their "weight" based on assumed chunk size of 10 (default)
    # Since we don't have exact step counts per phase in the CompressedStep struct (we only have lists of files etc),
    # We will just list them sequentially with a fixed visual block, or try to be smarter if we modified CompressedStep.
    # For now, simple list visualization.
    
    print(f"Total Steps: {total_steps} | Compressed Phases: {len(compressed.phases)}")
    print("-" * 40)
    
    for i, phase in enumerate(compressed.phases, 1):
        # Create a visual bar. 
        # Ideally we'd know how many chunks were merged, but CompressedStep doesn't store that yet.
        # We'll just use a fixed representation.
        label = f"{i}. {phase.phase.upper()}"
        content = f"Action: {phase.action_summary[:40]}..."
        print(f"{label:<20} | {content}")
        if phase.files_involved:
             print(f"{ '':<20} | Files: {len(phase.files_involved)}")
        print("-" * 40)

def run_detailed_test():
    print("=" * 60)
    print("Stage 5 Detailed Test: Log Handler & Induction (REAL API CALLS)")
    print("=" * 60)

    # 1. Load Data
    print(f"\n[1/5] Loading experience: {TARGET_INSTANCE}")
    try:
        raw_data = load_experience_from_output(TARGET_INSTANCE)
        history_steps = convert_history(raw_data["history"])
        
        problem = raw_data.get("instance", {}).get("problem_statement", "")
        patch = ""
        if isinstance(raw_data.get("test_result"), dict):
            patch = raw_data["test_result"].get("git_patch", "")

        experience = CodingExperience(
            instance_id=raw_data["instance_id"],
            problem_statement=problem,
            diff_patch=patch,
            history=history_steps,
            test_result="PASS"
        )
        print(f"   - Loaded {len(experience.history)} steps.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

    # 2. Configure Handler & Induction Module
    handler = ExperienceLogHandler(llm_config=LLM_CONFIG, chunk_size=10)
    induction_module = WorkflowInductionModule(llm_config=LLM_CONFIG)

    # 3. Compress
    print("\n[2/5] Compressing experience (Calling LLM)...")
    try:
        compressed = handler.compress(experience)
        print(f"   - Compression ratio: {compressed.original_step_count} steps -> {compressed.compressed_step_count} phases")
    except Exception as e:
        print(f"Error during compression: {e}")
        import traceback; traceback.print_exc()
        return False

    # 4. Visualize
    print("\n[3/5] Visualizing...")
    plot_timeline(compressed)

    # 5. Induce
    print("\n[4/5] Running incremental induction (Calling LLM)...")
    try:
        # We pass an empty list of existing workflows to simulate starting fresh
        updated_workflows = induction_module.induce_from_single(experience, [])
        print(f"   - Induced {len(updated_workflows)} workflows.")
        for wf in updated_workflows:
            print(f"     > {wf.name}")
    except Exception as e:
        print(f"Error during induction: {e}")
        import traceback; traceback.print_exc()
        return False

    # 6. Save Outputs
    print("\n[5/5] Saving outputs to tests/ directory...")
    
    # Save compressed experience
    comp_path = os.path.join(TEST_OUTPUT_DIR, "compressed_experience.json")
    with open(comp_path, "w") as f:
        json.dump(compressed.to_dict(), f, indent=2)
    print(f"   - Saved compressed experience: {comp_path}")
    
    # Save workflows
    wf_path = os.path.join(TEST_OUTPUT_DIR, "inducted_memory.json")
    with open(wf_path, "w") as f:
        data = [wf.to_dict() for wf in updated_workflows]
        json.dump(data, f, indent=2)
    print(f"   - Saved inducted memory: {wf_path}")

    print("\n" + "=" * 60)
    print("Detailed Test Completed Successfully")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_detailed_test()
    sys.exit(0 if success else 1)