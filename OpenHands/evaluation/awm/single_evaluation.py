"""
Single Sample Evaluation Module for AWM

Provides the ability to evaluate a single patch against SWE-bench tests,
independent of the batch evaluation pipeline.
"""

import os
import tempfile
import time
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.runtime.base import Runtime

import pandas as pd

from evaluation.awm.experience import EvaluationResult
from openhands.core.logger import openhands_logger as logger


class SingleEvaluationRunner:
    """
    单样本Evaluation执行器

    通过import复用现有eval_infer.py的逻辑，不修改原文件。
    """

    def __init__(
        self,
        dataset_name: str = "princeton-nlp/SWE-bench",
        split: str = "test",
    ):
        """
        初始化Evaluation Runner

        Args:
            dataset_name: SWE-bench数据集名称
            split: 数据集split
        """
        self.dataset_name = dataset_name
        self.split = split
        self._instance_cache: Dict[str, Any] = {}
        self._test_spec_cache: Dict[str, Any] = {}
        self._load_harness()

    def _load_harness(self):
        """加载SWE-bench harness"""
        try:
            from swebench.harness.grading import get_eval_report
            from swebench.harness.run_evaluation import (
                APPLY_PATCH_FAIL,
                APPLY_PATCH_PASS,
            )
            from swebench.harness.test_spec.test_spec import (
                SWEbenchInstance,
                make_test_spec,
            )
            from swebench.harness.utils import load_swebench_dataset

            self.get_eval_report = get_eval_report
            self.APPLY_PATCH_FAIL = APPLY_PATCH_FAIL
            self.APPLY_PATCH_PASS = APPLY_PATCH_PASS
            self.make_test_spec = make_test_spec
            self.load_swebench_dataset = load_swebench_dataset

            # 预加载数据集
            self._load_dataset()

        except ImportError as e:
            logger.error(f"Failed to import swebench harness: {e}")
            raise

    def _load_dataset(self):
        """加载数据集到缓存"""
        full_dataset = self.load_swebench_dataset(self.dataset_name, self.split)
        for instance in full_dataset:
            self._instance_cache[instance["instance_id"]] = instance

    def run(
        self,
        instance_id: str,
        diff_patch: str,
        timeout: int = 1800,
        runtime: Optional["Runtime"] = None,
    ) -> EvaluationResult:
        """
        对单个instance的patch进行evaluation

        Args:
            instance_id: Instance ID
            diff_patch: 要评估的diff patch
            timeout: 测试超时时间（秒）
            runtime: Optional runtime to reuse. If provided, caller is
                    responsible for closing it. Testbed will be reset
                    before evaluation.

        Returns:
            EvaluationResult: 评估结果
        """
        # 延迟导入
        from evaluation.benchmarks.swe_bench.eval_infer import (
            process_git_patch,
            get_config,
        )
        from evaluation.benchmarks.swe_bench.run_infer import get_instance_docker_image
        from evaluation.utils.shared import get_default_sandbox_config_for_eval
        from openhands.core.main import create_runtime
        from openhands.events.action import CmdRunAction
        from openhands.events.observation import CmdOutputObservation
        from openhands.utils.async_utils import call_async_from_sync

        # 处理patch
        processed_patch = process_git_patch(diff_patch)

        if not processed_patch.strip():
            return EvaluationResult(
                instance_id=instance_id,
                resolved=False,
                test_output="",
                report={"empty_generation": True, "resolved": False},
            )

        # 获取instance信息
        instance = self._get_instance(instance_id)
        test_spec = self._get_test_spec(instance_id, instance)

        # Determine if we're using provided runtime or creating new one
        should_close_runtime = runtime is None

        if runtime is None:
            # 准备Docker环境
            base_container_image = get_instance_docker_image(instance_id)
            sandbox_config = get_default_sandbox_config_for_eval()
            sandbox_config.base_container_image = base_container_image

            # 创建runtime config (minimal)
            from openhands.core.config import OpenHandsConfig, SandboxConfig
            config = OpenHandsConfig(
                sandbox=sandbox_config,
                runtime=os.environ.get("RUNTIME", "docker"),
            )

            # 创建runtime
            runtime = create_runtime(config)
            try:
                call_async_from_sync(runtime.connect)
            except Exception:
                # If connect fails, close the runtime to avoid leaking file descriptors
                runtime.close()
                raise
        else:
            # Reusing provided runtime - reset testbed state first
            logger.info("Reusing runtime from inference phase, resetting testbed...")
            self._reset_testbed(runtime, CmdRunAction, CmdOutputObservation)

        try:
            # 复制patch和eval脚本到容器
            with tempfile.TemporaryDirectory() as temp_dir:
                # 写patch文件
                patch_file = os.path.join(temp_dir, "patch.diff")
                with open(patch_file, "w") as f:
                    f.write(processed_patch)
                runtime.copy_to(patch_file, "/tmp")

                # 写eval脚本
                eval_script = os.path.join(temp_dir, "eval.sh")
                with open(eval_script, "w") as f:
                    f.write(test_spec.eval_script)
                runtime.copy_to(eval_script, "/tmp")

            # 设置执行权限
            action = CmdRunAction(command="chmod +x /tmp/eval.sh")
            action.set_hard_timeout(600)
            obs = runtime.run_action(action)

            # 检查 chmod 是否成功
            if isinstance(obs, CmdOutputObservation) and obs.exit_code != 0:
                logger.warning(f"chmod +x failed: {obs.content}, will use bash to run script")

            # 应用patch
            apply_cmd = (
                "cd /testbed && "
                "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                "(echo 'Trying patch command...' && "
                "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
                "echo 'APPLY_PATCH_FAIL')))"
            )
            action = CmdRunAction(command=apply_cmd)
            action.set_hard_timeout(600)
            obs = runtime.run_action(action)

            apply_output = obs.content if isinstance(obs, CmdOutputObservation) else ""

            if "APPLY_PATCH_FAIL" in apply_output:
                return EvaluationResult(
                    instance_id=instance_id,
                    resolved=False,
                    test_output=apply_output,
                    report={"failed_apply_patch": True, "resolved": False},
                )

            # 运行测试 (使用 bash 执行脚本，避免权限问题)
            log_file = "/tmp/eval_output.log"
            action = CmdRunAction(command=f"bash /tmp/eval.sh > {log_file} 2>&1 & echo $!")
            action.set_hard_timeout(300)
            obs = runtime.run_action(action)

            if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
                pid = obs.content.split()[-1].strip()

                # 等待测试完成
                start_time = time.time()
                while True:
                    if time.time() - start_time > timeout:
                        return EvaluationResult(
                            instance_id=instance_id,
                            resolved=False,
                            test_output="Test timeout",
                            report={"test_timeout": True, "resolved": False},
                        )

                    check_action = CmdRunAction(command=f"ps -p {pid} > /dev/null; echo $?")
                    check_action.set_hard_timeout(300)
                    check_obs = runtime.run_action(check_action)

                    if (isinstance(check_obs, CmdOutputObservation) and
                        check_obs.content.split()[-1].strip() == "1"):
                        break

                    time.sleep(30)

                # 读取测试输出
                cat_action = CmdRunAction(command=f"cat {log_file}")
                cat_action.set_hard_timeout(300)
                cat_obs = runtime.run_action(cat_action)

                if isinstance(cat_obs, CmdOutputObservation) and cat_obs.exit_code == 0:
                    test_output = cat_obs.content

                    # 评估结果
                    resolved = self._grade_result(
                        instance_id, processed_patch, test_output, test_spec
                    )

                    return EvaluationResult(
                        instance_id=instance_id,
                        resolved=resolved,
                        test_output=test_output,
                        report={"resolved": resolved},
                    )

            return EvaluationResult(
                instance_id=instance_id,
                resolved=False,
                test_output="Evaluation failed",
                report={"error_eval": True, "resolved": False},
            )

        finally:
            if should_close_runtime:
                runtime.close()

    def _get_instance(self, instance_id: str) -> Dict[str, Any]:
        """获取instance数据"""
        if instance_id not in self._instance_cache:
            raise ValueError(f"Instance {instance_id} not found in dataset")
        return self._instance_cache[instance_id]

    def _get_test_spec(self, instance_id: str, instance: Dict[str, Any]) -> Any:
        """获取或创建test_spec"""
        if instance_id not in self._test_spec_cache:
            self._test_spec_cache[instance_id] = self.make_test_spec(instance)
        return self._test_spec_cache[instance_id]

    def _reset_testbed(self, runtime, CmdRunAction, CmdOutputObservation) -> None:
        """
        Reset testbed to clean state before evaluation.

        This is called when reusing a runtime from inference phase.
        Cleans up any changes made during inference.
        """
        reset_cmd = "cd /testbed && git reset --hard HEAD && git clean -fd"
        action = CmdRunAction(command=reset_cmd)
        action.set_hard_timeout(300)
        obs = runtime.run_action(action)

        if isinstance(obs, CmdOutputObservation):
            if obs.exit_code == 0:
                logger.info("Testbed reset successful")
            else:
                logger.warning(f"Testbed reset may have issues: {obs.content[:200]}")

    def _grade_result(
        self,
        instance_id: str,
        model_patch: str,
        test_output: str,
        test_spec: Any,
    ) -> bool:
        """评估测试结果"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                log_dir = os.path.join(temp_dir, "logs", instance_id.lower())
                os.makedirs(log_dir, exist_ok=True)

                test_output_path = os.path.join(log_dir, "test_output.txt")
                with open(test_output_path, "w") as f:
                    f.write(test_output)

                report = self.get_eval_report(
                    test_spec=test_spec,
                    prediction={
                        "model_patch": model_patch,
                        "instance_id": instance_id,
                    },
                    include_tests_status=True,
                    test_log_path=test_output_path,
                )

                return report.get(instance_id, {}).get("resolved", False)

        except Exception as e:
            logger.error(f"Error grading result for {instance_id}: {e}")
            return False
