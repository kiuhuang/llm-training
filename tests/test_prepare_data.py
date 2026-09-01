#!/usr/bin/env python
"""Offline unit tests for data_prep/prepare_data.normalize (stdlib only).

Run:  python tests/test_prepare_data.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_prep"))
from prepare_data import normalize  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def test_full_row_keeps_system():
    msgs, reason = normalize({"system": "You are an analyst.", "user": "Q?", "assistant": "A."})
    check(msgs is not None and reason == "", "complete row is kept")
    check([m["role"] for m in msgs] == ["system", "user", "assistant"], "roles preserved")


def test_newline_system_dropped():
    msgs, reason = normalize({"system": "\n", "user": "Q?", "assistant": "A."})
    check(msgs is not None, "newline-only system row is kept")
    check([m["role"] for m in msgs] == ["user", "assistant"],
          f"system message dropped (got {[m['role'] for m in msgs]})")


def test_empty_fields_dropped():
    for field, expected in [("user", "empty_user"), ("assistant", "empty_assistant")]:
        row = {"system": "s", "user": "Q?", "assistant": "A."}
        row[field] = ""
        msgs, reason = normalize(row)
        check(msgs is None and reason == expected, f"empty {field} -> {expected}")


def test_whitespace_stripped():
    msgs, _ = normalize({"system": "  S\n ", "user": " Q ", "assistant": " A \n"})
    check(msgs[0]["content"] == "S" and msgs[1]["content"] == "Q" and msgs[2]["content"] == "A",
          "surrounding whitespace stripped from all fields")


def test_absurd_length_dropped():
    row = {"system": "", "user": "x" * 5_000_000, "assistant": "y" * 5_000_001}
    msgs, reason = normalize(row)
    check(msgs is None and reason == "absurd_length", ">10M chars row dropped")


def test_rag_style_row_kept():
    context = "10-K excerpt: " + "num " * 1000  # ~5k chars of RAG context in user
    msgs, reason = normalize({"system": "", "user": context + "\nQuestion?", "assistant": "A"})
    check(msgs is not None and reason == "", "long-but-reasonable RAG row kept")


if __name__ == "__main__":
    test_full_row_keeps_system()
    test_newline_system_dropped()
    test_empty_fields_dropped()
    test_whitespace_stripped()
    test_absurd_length_dropped()
    test_rag_style_row_kept()
    print("\nALL DATA-PREP TESTS PASSED")
