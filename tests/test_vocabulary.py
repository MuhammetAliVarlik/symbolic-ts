import re

import pytest

from symbolic_ts import vocabulary as v


def test_schema_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", v.SCHEMA_VERSION)


def test_neutral_tokens_cover_all_25_combinations():
    assert len(v.NEUTRAL_TOKENS) == 25
    assert "C0_V0" in v.NEUTRAL_TOKENS
    assert "C4_V4" in v.NEUTRAL_TOKENS
    assert "C2_V1" in v.NEUTRAL_TOKENS  # the exact token used throughout Phase 0


def test_semantic_tokens_cover_all_25_combinations():
    assert len(v.SEMANTIC_TOKENS) == 25
    assert "C_CRASH_V_SURGE" in v.SEMANTIC_TOKENS
    assert "C_FLAT_V_CALM" in v.SEMANTIC_TOKENS


def test_neutral_and_semantic_vocabularies_are_disjoint():
    assert v.NEUTRAL_TOKENS.isdisjoint(v.SEMANTIC_TOKENS)


@pytest.mark.parametrize("token", ["C0_V0", "C2_V1", "C4_V4", "C_CRASH_V_SURGE", "C_FLAT_V_CALM", "E_NONE"])
def test_validate_token_accepts_known_tokens(token):
    assert v.validate_token(token) is True


@pytest.mark.parametrize(
    "token",
    [
        "",
        "garbage",
        "C5_V0",  # out-of-range bucket digit
        "C0_V5",
        "c0_v0",  # wrong case
        "C0V0",  # missing underscore
        "C0_V0_extra",  # malformed
        "C_CRASH",  # semantic change without volatility half
        "E_WEEKEND",  # not registered yet
    ],
)
def test_validate_token_rejects_unknown_tokens(token):
    assert v.validate_token(token) is False


def test_validate_token_respects_allowed_subset():
    assert v.validate_token("C0_V0", allowed=v.NEUTRAL_TOKENS) is True
    assert v.validate_token("C_CRASH_V_SURGE", allowed=v.NEUTRAL_TOKENS) is False


def test_event_none_is_reserved_by_default():
    assert v.EVENT_NONE in v.registered_event_tokens()
    assert v.validate_token(v.EVENT_NONE) is True


def test_register_event_tokens_extends_vocabulary():
    v.register_event_tokens(["E_WEEKEND", "E_HOUR_PEAK"])
    try:
        assert v.validate_token("E_WEEKEND") is True
        assert v.validate_token("E_HOUR_PEAK") is True
        assert "E_WEEKEND" in v.registered_event_tokens()
    finally:
        v._registered_event_tokens.discard("E_WEEKEND")
        v._registered_event_tokens.discard("E_HOUR_PEAK")


def test_register_event_tokens_rejects_wrong_prefix():
    with pytest.raises(ValueError):
        v.register_event_tokens(["WEEKEND"])


def test_validate_sequence_all_valid():
    report = v.validate_sequence(["C0_V0", "C2_V1", "E_NONE"])
    assert report.is_valid
    assert report.invalid_count == 0
    assert report.conformance_rate == 1.0


def test_validate_sequence_reports_invalid_positions():
    report = v.validate_sequence(["C0_V0", "garbage", "C2_V1", "also_bad"])
    assert not report.is_valid
    assert report.invalid_count == 2
    assert report.invalid_tokens == [(1, "garbage"), (3, "also_bad")]
    assert report.conformance_rate == 0.5


def test_validate_sequence_empty_sequence_is_valid():
    report = v.validate_sequence([])
    assert report.is_valid
    assert report.conformance_rate == 1.0


def test_validate_sequence_respects_allowed_subset():
    report = v.validate_sequence(["C0_V0", "C_CRASH_V_SURGE"], allowed=v.NEUTRAL_TOKENS)
    assert not report.is_valid
    assert report.invalid_tokens == [(1, "C_CRASH_V_SURGE")]


def test_known_tokens_includes_all_three_namespaces():
    known = v.known_tokens()
    assert v.NEUTRAL_TOKENS <= known
    assert v.SEMANTIC_TOKENS <= known
    assert v.EVENT_NONE in known
