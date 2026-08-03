"""Render a config mapping as YAML that carries its explanatory comments.

A plain ``yaml.safe_dump`` loses exactly the information an operator needs to
avoid a silently wrong permeability -- which unit a field is in, what the two
allowed spellings mean, which channel does what. This emitter keeps them.

It handles only the shapes a gasperm config actually contains: nested
mappings, scalars, and flat lists of scalars.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import yaml

__all__ = ["render"]

#: Column that trailing comments are aligned to, when the line is short enough.
_COMMENT_COLUMN = 46


def _scalar(value: Any) -> str:
    """YAML scalar form of ``value`` (``None`` -> ``null``, lists inline)."""
    text = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True).strip()
    # safe_dump appends a document end marker for bare scalars.
    return text.removesuffix("...").strip()


def _is_block(value: Any) -> bool:
    return isinstance(value, Mapping) and len(value) > 0


def _emit(
    data: Mapping[str, Any],
    notes: Mapping[str, str],
    comments: Mapping[str, str],
    prefix: str,
    depth: int,
    out: list[str],
) -> None:
    pad = "  " * depth
    for key, value in data.items():
        path = f"{prefix}{key}"

        note = notes.get(path)
        if note:
            if depth == 0 and out and out[-1] != "":
                out.append("")
            for line in note.strip("\n").split("\n"):
                out.append(f"{pad}# {line}".rstrip())

        if _is_block(value):
            out.append(f"{pad}{key}:")
            _emit(value, notes, comments, f"{path}.", depth + 1, out)
            continue

        if isinstance(value, Mapping):  # empty mapping
            rendered = f"{pad}{key}: {{}}"
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            rendered = f"{pad}{key}: {_scalar(list(value))}"
        else:
            rendered = f"{pad}{key}: {_scalar(value)}"

        comment = comments.get(path)
        if comment:
            padding = max(_COMMENT_COLUMN - len(rendered), 1)
            rendered = f"{rendered}{' ' * padding}# {comment}"
        out.append(rendered)


def render(
    data: Mapping[str, Any],
    *,
    header: str = "",
    notes: Mapping[str, str] | None = None,
    comments: Mapping[str, str] | None = None,
) -> str:
    """Render ``data`` as commented YAML.

    Args:
        data: The mapping to emit, in the order the keys should appear.
        header: Block comment placed at the top of the file.
        notes: Dotted path -> block comment emitted *above* that key. May be
            multi-line.
        comments: Dotted path -> short comment appended after the value.

    Returns:
        YAML text that ``yaml.safe_load`` reads back to ``data``.
    """
    out: list[str] = []
    if header:
        for line in header.strip("\n").split("\n"):
            out.append(f"# {line}".rstrip())
        out.append("")
    _emit(data, notes or {}, comments or {}, "", 0, out)
    return "\n".join(out).strip("\n") + "\n"
