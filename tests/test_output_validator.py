"""tests/test_output_validator.py — Output-leak detection cases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from security.output_validator import output_validator


def test_normal_answer_is_allowed():
    decision = output_validator.check("The mileage reimbursement rate is $0.67 per mile [Source: policy, Chunk 3].")
    assert decision.action == "ALLOW"


def test_groq_key_shaped_string_is_redacted():
    leaked = "Sure, here it is: gsk_abcdefghijklmnopqrstuvwxyz0123456789ABCD"
    decision = output_validator.check(leaked)
    assert decision.action == "BLOCK"
    assert "gsk_" not in decision.redacted_text
    assert "[REDACTED]" in decision.redacted_text


def test_anthropic_key_shaped_string_is_redacted():
    leaked = "The key is sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"
    decision = output_validator.check(leaked)
    assert decision.action == "BLOCK"
    assert "sk-ant" not in decision.redacted_text


def test_aws_key_shaped_string_is_redacted():
    leaked = "Found this in the doc: AKIAABCDEFGHIJKLMNOP"
    decision = output_validator.check(leaked)
    assert decision.action == "BLOCK"
    assert "AKIA" not in decision.redacted_text


def test_system_prompt_echo_is_blocked():
    leaked = "My instructions say: you are a precise document analyst."
    decision = output_validator.check(leaked)
    assert decision.action == "BLOCK"


def test_empty_answer_is_allowed():
    decision = output_validator.check("")
    assert decision.action == "ALLOW"
