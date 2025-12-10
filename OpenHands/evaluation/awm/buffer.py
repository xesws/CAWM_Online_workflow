"""
Experience Buffer for AWM

Stores successful experiences and provides functionality for:
- Grouping by task type
- Triggering workflow induction
- Persistence and recovery
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Callable

from evaluation.awm.experience import CodingExperience
from evaluation.awm.task_classifier import classify_task, TaskType
from openhands.core.logger import openhands_logger as logger


class ExperienceBuffer:
    """
    存储成功的experiences，支持按任务类型分组

    遵循AWM原论文：按website（这里是task_type）分组存储
    """

    def __init__(
        self,
        induction_trigger_count: int = 10,
        persistence_path: Optional[str] = None,
        auto_persist: bool = True,
    ):
        """
        初始化Experience Buffer

        Args:
            induction_trigger_count: 触发induction的成功experience数量阈值
            persistence_path: 持久化文件路径（可选）
            auto_persist: 是否自动持久化
        """
        # 按任务类型分组存储
        self.buffer: Dict[str, List[CodingExperience]] = defaultdict(list)

        # 触发induction的阈值
        self.induction_trigger_count = induction_trigger_count

        # 已处理的成功experience计数
        self.success_count = 0

        # 持久化设置
        self.persistence_path = persistence_path
        self.auto_persist = auto_persist

        # 如果存在持久化文件，加载
        if persistence_path and os.path.exists(persistence_path):
            self.load()

    def add(self, experience: CodingExperience) -> bool:
        """
        添加experience到buffer

        只接受成功的experience (test_result == "PASS")

        Args:
            experience: 要添加的experience

        Returns:
            bool: 是否应该触发workflow induction
        """
        if experience.test_result != "PASS":
            logger.debug(f"Skipping failed experience: {experience.instance_id}")
            return False

        # 分类任务类型
        if experience.task_type is None:
            experience.task_type = classify_task(experience)

        task_type = experience.task_type

        # 添加到对应分组
        self.buffer[task_type].append(experience)
        self.success_count += 1

        logger.info(
            f"Added experience {experience.instance_id} to buffer "
            f"(type: {task_type}, total: {self.success_count})"
        )

        # 自动持久化
        if self.auto_persist and self.persistence_path:
            self._persist()

        # 判断是否触发induction
        return self.success_count % self.induction_trigger_count == 0

    def get_all_experiences(self) -> List[CodingExperience]:
        """获取所有experience用于induction"""
        all_exp = []
        for task_type, exps in self.buffer.items():
            all_exp.extend(exps)
        return all_exp

    def get_experiences_by_type(self, task_type: str) -> List[CodingExperience]:
        """获取特定类型的experience"""
        return self.buffer.get(task_type, [])

    def get_recent_experiences(self, n: int = 10) -> List[CodingExperience]:
        """获取最近的n个experience"""
        all_exp = self.get_all_experiences()
        # 按时间戳排序
        sorted_exp = sorted(all_exp, key=lambda x: x.timestamp, reverse=True)
        return sorted_exp[:n]

    def get_statistics(self) -> Dict:
        """获取buffer统计信息"""
        type_counts = {t: len(exps) for t, exps in self.buffer.items()}
        return {
            "total_count": self.success_count,
            "type_counts": type_counts,
            "types": list(self.buffer.keys()),
            "induction_trigger_count": self.induction_trigger_count,
            "next_induction_at": (
                (self.success_count // self.induction_trigger_count + 1)
                * self.induction_trigger_count
            ),
        }

    def clear(self):
        """清空buffer"""
        self.buffer = defaultdict(list)
        self.success_count = 0
        if self.auto_persist and self.persistence_path:
            self._persist()

    def _persist(self):
        """持久化buffer到文件"""
        if not self.persistence_path:
            return

        data = {
            "success_count": self.success_count,
            "buffer": {
                task_type: [exp.to_dict() for exp in exps]
                for task_type, exps in self.buffer.items()
            },
            "last_updated": datetime.now().isoformat(),
        }

        # 确保目录存在
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)

        with open(self.persistence_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Buffer persisted to {self.persistence_path}")

    def load(self):
        """从文件加载buffer"""
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return

        try:
            with open(self.persistence_path, "r") as f:
                data = json.load(f)

            self.success_count = data.get("success_count", 0)

            self.buffer = defaultdict(list)
            for task_type, exps in data.get("buffer", {}).items():
                self.buffer[task_type] = [
                    CodingExperience.from_dict(exp) for exp in exps
                ]

            logger.info(
                f"Buffer loaded from {self.persistence_path} "
                f"({self.success_count} experiences)"
            )

        except Exception as e:
            logger.error(f"Failed to load buffer from {self.persistence_path}: {e}")

    def save(self, path: Optional[str] = None):
        """
        手动保存buffer到指定路径

        Args:
            path: 保存路径（如果不指定，使用默认路径）
        """
        original_path = self.persistence_path
        if path:
            self.persistence_path = path
        self._persist()
        self.persistence_path = original_path


class ExperienceBufferWithCallbacks(ExperienceBuffer):
    """
    带回调的Experience Buffer

    支持在特定事件发生时触发回调函数
    """

    def __init__(
        self,
        induction_trigger_count: int = 10,
        persistence_path: Optional[str] = None,
        on_induction_trigger: Optional[Callable[[List[CodingExperience]], None]] = None,
        on_experience_added: Optional[Callable[[CodingExperience], None]] = None,
    ):
        super().__init__(induction_trigger_count, persistence_path)

        self.on_induction_trigger = on_induction_trigger
        self.on_experience_added = on_experience_added

    def add(self, experience: CodingExperience) -> bool:
        """添加experience并触发回调"""
        should_induce = super().add(experience)

        # 触发添加回调
        if experience.test_result == "PASS" and self.on_experience_added:
            self.on_experience_added(experience)

        # 触发induction回调
        if should_induce and self.on_induction_trigger:
            self.on_induction_trigger(self.get_all_experiences())

        return should_induce
