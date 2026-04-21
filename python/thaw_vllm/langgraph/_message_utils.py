"""Message-shape helpers used by the coalescer and ChatThaw.

These utilities stay duck-typed so the coalescer can be unit-tested without
pulling in langchain-core. Real usage passes LangChain BaseMessage subclasses;
tests pass dicts with a ``content`` key.
"""
from __future__ import annotations

from typing import Any, Sequence


def message_key(msg: Any) -> tuple:
    """Stable equality key for a message-like object.

    Two messages produce the same key when they have the same type name and
    content. Handles multi-part content (lists of dict "parts") by flattening
    into a hashable tuple.
    """
    type_name = type(msg).__name__
    content = getattr(msg, "content", msg)
    if isinstance(content, (list, tuple)):
        flat: list = []
        for part in content:
            if isinstance(part, dict):
                flat.append((part.get("type"), part.get("text", part.get("image_url"))))
            else:
                flat.append(part)
        content = tuple(flat)
    return (type_name, content)


def common_prefix_length(message_lists: Sequence[Sequence[Any]]) -> int:
    """Return the length of the longest common prefix across all message lists.

    Messages compare equal via :func:`message_key`. Empty input yields 0.
    """
    if not message_lists:
        return 0
    min_len = min(len(msgs) for msgs in message_lists)
    first = message_lists[0]
    for i in range(min_len):
        key = message_key(first[i])
        if not all(message_key(msgs[i]) == key for msgs in message_lists[1:]):
            return i
    return min_len


def approximate_token_count(messages: Sequence[Any]) -> int:
    """Rough 4-chars-per-token estimate.

    Used when the coalescer has no real tokenizer. Overridden by ChatThaw once
    a vLLM tokenizer is available, for precise threshold decisions.
    """
    total = 0
    for m in messages:
        c = getattr(m, "content", m)
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, (list, tuple)):
            for part in c:
                if isinstance(part, dict):
                    total += len(str(part.get("text", "")))
                else:
                    total += len(str(part))
        else:
            total += len(str(c))
    return total // 4


__all__ = ["message_key", "common_prefix_length", "approximate_token_count"]
