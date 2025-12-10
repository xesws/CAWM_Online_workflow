"""
Token counting utilities for AWM

Provides functions for counting tokens and truncating text to token limits.
Uses tiktoken for accurate token counting.
"""

import tiktoken
from typing import Optional

from openhands.core.logger import openhands_logger as logger

# 默认 truncation 限制
DEFAULT_TRUNCATION_LIMIT = 50000


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    计算文本的 token 数量

    使用 tiktoken，fallback 到 cl100k_base encoding

    Args:
        text: 要计算的文本
        model: 模型名称（用于选择 encoding）

    Returns:
        int: token 数量
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # 如果模型不支持，使用 cl100k_base（GPT-4 使用的 encoding）
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def truncate_to_token_limit(
    text: str,
    max_tokens: int,
    model: str = "gpt-4",
    keep_start: bool = True,
) -> str:
    """
    截断文本到指定 token 数量

    Args:
        text: 要截断的文本
        max_tokens: 最大 token 数量
        model: 模型名称（用于选择 encoding）
        keep_start: True 保留开头（FIFO），False 保留结尾（LIFO）

    Returns:
        截断后的文本
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    if keep_start:
        truncated_tokens = tokens[:max_tokens]
    else:
        truncated_tokens = tokens[-max_tokens:]

    return encoding.decode(truncated_tokens)
