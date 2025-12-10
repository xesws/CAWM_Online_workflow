"""
History Parser for OpenHands Events

Converts OpenHands history format to AWM HistoryStep format.
"""

from typing import List, Dict, Any, Optional
from evaluation.awm.experience import HistoryStep


def parse_openhands_history(history: List[Dict[str, Any]]) -> List[HistoryStep]:
    """
    将OpenHands的history格式转换为HistoryStep格式。

    OpenHands history格式来自 event_to_dict(event)，包含：
    - action类型 (CmdRunAction, FileEditAction, etc.)
    - observation内容
    - thought/reasoning

    Args:
        history: OpenHands事件历史列表

    Returns:
        List[HistoryStep]: 转换后的步骤列表
    """
    steps = []
    step_id = 0

    for i, event in enumerate(history):
        if not isinstance(event, dict):
            continue

        # 检查是否是action事件
        action_type = _extract_action_type(event)
        if action_type is None:
            continue

        step_id += 1

        # 提取各字段
        thought = _extract_thought(event)
        action = _extract_action_content(event)
        observation = _extract_observation(history, i)
        file_path = _extract_file_path(event)
        command = _extract_command(event)

        step = HistoryStep(
            step_id=step_id,
            observation=observation,
            thought=thought,
            action=action,
            action_type=action_type,
            file_path=file_path,
            command=command,
        )
        steps.append(step)

    return steps


def _extract_action_type(event: Dict[str, Any]) -> Optional[str]:
    """提取action类型"""
    # OpenHands事件格式
    if "action" in event:
        action_name = event.get("action", "")

        # 映射到简化的类型名
        type_mapping = {
            "run": "run_command",
            "run_ipython": "run_ipython",
            "read": "read_file",
            "write": "write_file",
            "edit": "edit_file",
            "browse": "browse",
            "finish": "finish",
            "message": "message",
            "think": "think",
        }

        for key, mapped_type in type_mapping.items():
            if key in action_name.lower():
                return mapped_type

        return action_name

    return None


def _extract_thought(event: Dict[str, Any]) -> str:
    """提取思考内容"""
    # 尝试多个可能的字段
    for field in ["thought", "reasoning", "args.thought", "message"]:
        if "." in field:
            parts = field.split(".")
            value = event
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break
            if value:
                return str(value)
        elif field in event:
            return str(event[field])

    return ""


def _extract_action_content(event: Dict[str, Any]) -> str:
    """提取action内容"""
    if "args" in event:
        args = event["args"]
        if isinstance(args, dict):
            # 根据不同action类型提取相关内容
            if "command" in args:
                return args["command"]
            if "code" in args:
                return args["code"]
            if "path" in args:
                return f"path: {args['path']}"
            if "content" in args:
                return args.get("content", "")[:500]  # 截断
        return str(args)[:500]

    return str(event.get("action", ""))


def _extract_observation(history: List[Dict[str, Any]], action_index: int) -> str:
    """提取对应的observation"""
    # 查找action之后的第一个observation
    for i in range(action_index + 1, min(action_index + 3, len(history))):
        event = history[i]
        if isinstance(event, dict) and "observation" in event:
            obs = event.get("content", event.get("observation", ""))
            return str(obs)[:1000]  # 截断

    return ""


def _extract_file_path(event: Dict[str, Any]) -> Optional[str]:
    """提取文件路径"""
    args = event.get("args", {})
    if isinstance(args, dict):
        return args.get("path") or args.get("file_path")
    return None


def _extract_command(event: Dict[str, Any]) -> Optional[str]:
    """提取命令"""
    args = event.get("args", {})
    if isinstance(args, dict):
        return args.get("command") or args.get("code")
    return None


def format_history_for_display(steps: List[HistoryStep], max_steps: int = 20) -> str:
    """
    格式化history用于显示

    Args:
        steps: 步骤列表
        max_steps: 最大显示步骤数

    Returns:
        格式化的字符串
    """
    lines = []
    display_steps = steps[:max_steps]

    for step in display_steps:
        lines.append(f"Step {step.step_id}:")
        if step.thought:
            lines.append(f"  Thought: {step.thought[:200]}...")
        lines.append(f"  Action [{step.action_type}]: {step.action[:100]}...")
        if step.observation:
            lines.append(f"  Observation: {step.observation[:100]}...")
        lines.append("")

    if len(steps) > max_steps:
        lines.append(f"... ({len(steps) - max_steps} more steps)")

    return "\n".join(lines)
