"""Heading-aware chunking adapted from Hamkalam's Persian RAG pipeline.

This is an *approximate* character-budget splitter, not a tokenizer for a specific LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .normalization import normalize_persian

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_SENTENCE = re.compile(r"(?<=[.!?؟؛])\s+|\n+")


@dataclass(frozen=True)
class TextChunk:
    ordinal: int
    content: str
    heading_path: tuple[str, ...]
    estimated_tokens: int


def approximate_tokens(value: str) -> int:
    return max(1, (len(value) + 3) // 4)


def _split_long(sentence: str, budget: int) -> list[str]:
    """Split long unpunctuated text into bounded word sequences."""
    if approximate_tokens(sentence) <= budget:
        return [sentence]
    result: list[str] = []
    current = ""
    for word in sentence.split():
        if approximate_tokens(word) > budget:
            if current:
                result.append(current)
                current = ""
            step = budget * 4
            result.extend(word[i : i + step] for i in range(0, len(word), step))
        elif current and approximate_tokens(current + " " + word) > budget:
            result.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        result.append(current)
    return result


def chunk_text(
    text: str,
    title: str = "",
    *,
    target_tokens: int = 512,
    overlap_ratio: float = 0.12,
) -> list[TextChunk]:
    """Chunk Persian Markdown without carrying overlap across distinct headings.

    `target_tokens` includes the heading prefix in each chunk. For unusual, very
    long headings the prefix itself can exceed the desired approximate budget.
    """
    if target_tokens < 16:
        raise ValueError("target_tokens must be >= 16")
    if not 0 <= overlap_ratio < 0.5:
        raise ValueError("overlap_ratio must be in [0, 0.5)")
    normalized = normalize_persian(text)
    if not normalized:
        return []

    groups: list[tuple[tuple[str, ...], list[str]]] = []
    levels: list[tuple[int, str]] = []
    for line in normalized.splitlines():
        line = line.strip()
        if not line:
            continue
        matched = _HEADING.match(line)
        if matched:
            level = len(matched.group(1))
            levels = [entry for entry in levels if entry[0] < level]
            levels.append((level, matched.group(2).strip()))
            continue
        path = tuple(label for _, label in levels)
        if not groups or groups[-1][0] != path:
            groups.append((path, []))
        groups[-1][1].append(line)

    if not groups:
        return []

    output: list[TextChunk] = []
    for path, paragraphs in groups:
        prefix = " > ".join(part for part in (title, *path) if part)
        budget = max(1, target_tokens - (approximate_tokens(prefix) + 1 if prefix else 0))
        segments: list[str] = []
        for paragraph in paragraphs:
            for sentence in _SENTENCE.split(paragraph):
                if sentence.strip():
                    segments.extend(_split_long(sentence.strip(), budget))
        current: list[str] = []

        def emit(
            items: list[str],
            prefix: str = prefix,
            path: tuple[str, ...] = path,
        ) -> None:
            body = " ".join(items).strip()
            if not body:
                return
            content = f"{prefix}\n{body}" if prefix else body
            output.append(TextChunk(len(output), content, path, approximate_tokens(content)))

        for segment in segments:
            trial = " ".join([*current, segment])
            if current and approximate_tokens(trial) > budget:
                emit(current)
                carry: list[str] = []
                allowed = max(0, int(budget * overlap_ratio))
                for item in reversed(current):
                    proposed = [item, *carry]
                    if approximate_tokens(" ".join(proposed)) > allowed:
                        break
                    carry = proposed
                current = carry
                while current and approximate_tokens(" ".join([*current, segment])) > budget:
                    current.pop(0)
            current.append(segment)
        if current:
            emit(current)
    return output
