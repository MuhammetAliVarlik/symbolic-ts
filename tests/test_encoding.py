import pytest

from symbolic_ts import encoding as enc

SAMPLE_TOKENS = ["C2_V1", "E_NONE", "C0_V4", "C4_V0", "E_WEEKEND", "C2_V2"]


class WhitespaceTokenizer:
    """Minimal stand-in satisfying the Tokenizer protocol -- one 'token' per
    whitespace-separated symbol. Good enough to exercise token_count()
    without pulling in a real tokeniser package as a test dependency."""

    def encode(self, text: str) -> list[str]:
        return text.split(" ") if text else []


class CharTokenizer:
    """A second stand-in tokenizer where every character costs one token --
    makes the single-char encoder's efficiency advantage checkable directly."""

    def encode(self, text: str) -> list[str]:
        return list(text)


@pytest.mark.parametrize("encoder", enc.ALL_ENCODERS)
def test_encoder_satisfies_protocol(encoder):
    assert isinstance(encoder, enc.SequenceEncoder)


def test_neutral_encoder_round_trips():
    result = enc.NEUTRAL_ENCODER.encode(SAMPLE_TOKENS)
    assert enc.NEUTRAL_ENCODER.decode(result.text) == SAMPLE_TOKENS


def test_semantic_encoder_round_trips_to_canonical_neutral_form():
    encoded = enc.SEMANTIC_ENCODER.encode(SAMPLE_TOKENS)
    assert enc.SEMANTIC_ENCODER.decode(encoded.text) == SAMPLE_TOKENS


def test_single_char_encoder_round_trips_to_canonical_neutral_form():
    encoded = enc.SINGLE_CHAR_ENCODER.encode(SAMPLE_TOKENS)
    assert enc.SINGLE_CHAR_ENCODER.decode(encoded.text) == SAMPLE_TOKENS


def test_semantic_encoding_specific_tokens():
    encoded = enc.SEMANTIC_ENCODER.encode(["C0_V0", "C4_V4", "C2_V1"])
    assert encoded.text == "C_CRASH_V_CALM C_SURGE_V_SURGE C_FLAT_V_LOW"


def test_single_char_encoding_is_one_char_per_symbol():
    encoded = enc.SINGLE_CHAR_ENCODER.encode(["C0_V0", "C4_V4", "C2_V1"])
    symbols = encoded.text.split(" ")
    assert all(len(s) == 1 for s in symbols)
    assert len(set(symbols)) == 3  # distinct buckets map to distinct characters


def test_single_char_encoding_is_deterministic_and_reversible_bijection():
    from symbolic_ts.vocabulary import NEUTRAL_TOKENS

    seen_chars = set()
    for token in sorted(NEUTRAL_TOKENS):
        char = enc.SINGLE_CHAR_ENCODER.encode([token]).text
        assert char not in seen_chars, f"collision on {token!r} -> {char!r}"
        seen_chars.add(char)
        assert enc.SINGLE_CHAR_ENCODER.decode(char) == [token]


def test_event_tokens_pass_through_unchanged_in_every_encoder():
    for encoder in enc.ALL_ENCODERS:
        encoded = encoder.encode(["E_NONE", "E_WEEKEND"])
        assert encoded.text == "E_NONE E_WEEKEND"


def test_unrecognized_compound_token_raises():
    with pytest.raises(ValueError):
        enc.NEUTRAL_ENCODER.encode(["NOT_A_TOKEN"])


def test_single_char_decode_rejects_unknown_symbol():
    with pytest.raises(ValueError):
        enc.SINGLE_CHAR_ENCODER.decode("Z")  # 26th letter, outside the 25-symbol alphabet


def test_token_count_under_given_tokenizer():
    encoded = enc.NEUTRAL_ENCODER.encode(SAMPLE_TOKENS)
    assert encoded.token_count(WhitespaceTokenizer()) == len(SAMPLE_TOKENS)


def test_metadata_records_encoder_identity_for_mlflow():
    encoded = enc.SEMANTIC_ENCODER.encode(["C0_V0"])
    assert encoded.metadata == {"encoding": "semantic", "schema_version": encoded.schema_version}


def test_compare_encoders_reports_same_input_across_all_encoders():
    reports = enc.compare_encoders(SAMPLE_TOKENS, WhitespaceTokenizer())
    assert [r.encoder_name for r in reports] == ["neutral", "semantic", "single_char"]
    for report in reports:
        assert report.token_count > 0
        assert report.char_count == len(report.text)


def test_single_char_encoding_uses_fewer_tokens_than_semantic_under_char_tokenizer():
    tokenizer = CharTokenizer()
    compound_only = ["C0_V0", "C1_V2", "C4_V4", "C2_V1"]
    reports = {r.encoder_name: r for r in enc.compare_encoders(compound_only, tokenizer)}
    assert reports["single_char"].token_count < reports["semantic"].token_count
    assert reports["single_char"].token_count < reports["neutral"].token_count
