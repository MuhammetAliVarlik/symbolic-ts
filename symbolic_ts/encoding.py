"""Sequence encoders: turn a token sequence (e.g. ["C2_V1", "E_NONE", ...]) into
a single model-ready prompt string, and back.

Three encodings exist because comparing them IS the E2-E4 representation
ablation (WORKING_PLAN.md), not a preference for one over the others:

- `NeutralEncoder`  (E3, default) -- neutral compound tokens, space-joined.
- `SemanticEncoder` (E2)          -- the same buckets spelled out as words.
- `SingleCharEncoder` (E4)        -- the same buckets as one character each.
  This one exists purely for token-budget efficiency: at semantic labelling
  a compound token costs ~5-7 subword tokens under a typical tokeniser
  (symbolic-ts-research's F0-09 notes, ~6.7 tokens/symbol measured), so a
  50-symbol context alone can consume ~335 tokens of a 0.5B model's scarce
  budget. A single character is ~1 subword token regardless of scheme.

All three accept the SAME input token sequence and re-render it -- they do
not require re-fitting a binner with a different `labels=` scheme just to
compare encodings, since a compound token's bucket identity (which change
bucket, which volatility bucket) is scheme-independent; only its spelling
changes. Event tokens (`E_` prefix) are passed through unchanged by every
encoder -- compressing those is a separate, later concern (WORKING_PLAN's
B4, event token integration), out of scope here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from symbolic_ts.vocabulary import (
    EVENT_PREFIX,
    N_BUCKETS,
    NEUTRAL_CHANGE_LABELS,
    NEUTRAL_VOLATILITY_LABELS,
    SCHEMA_VERSION,
    SEMANTIC_CHANGE_LABELS,
    SEMANTIC_VOLATILITY_LABELS,
)

DEFAULT_SEPARATOR = " "

# Every compound token spelling (neutral AND semantic) maps to the same
# (change_bucket_index, volatility_bucket_index) pair -- computed once here
# so every encoder parses input the same way regardless of which scheme the
# input happens to already be spelled in.
_COMPOUND_TOKEN_TO_INDICES: dict[str, tuple[int, int]] = {}
for _c_idx, (_nc, _sc) in enumerate(zip(NEUTRAL_CHANGE_LABELS, SEMANTIC_CHANGE_LABELS)):
    for _v_idx, (_nv, _sv) in enumerate(zip(NEUTRAL_VOLATILITY_LABELS, SEMANTIC_VOLATILITY_LABELS)):
        _COMPOUND_TOKEN_TO_INDICES[f"{_nc}_{_nv}"] = (_c_idx, _v_idx)
        _COMPOUND_TOKEN_TO_INDICES[f"{_sc}_{_sv}"] = (_c_idx, _v_idx)

# Single-character alphabet: chr('A' + change_idx * N_BUCKETS + volatility_idx).
# Assumes N_BUCKETS**2 <= 26 (currently 25) -- true for D1/D4's fixed 5-bucket
# design; revisit this scheme, not just its constant, if that ever changes.
assert N_BUCKETS * N_BUCKETS <= 26, "single-character alphabet needs N_BUCKETS**2 <= 26"
_CHAR_TO_INDICES: dict[str, tuple[int, int]] = {
    chr(ord("A") + i): divmod(i, N_BUCKETS) for i in range(N_BUCKETS * N_BUCKETS)
}


def _indices_of(token: str) -> tuple[int, int]:
    try:
        return _COMPOUND_TOKEN_TO_INDICES[token]
    except KeyError:
        raise ValueError(f"unrecognized compound token {token!r}") from None


@dataclass(frozen=True)
class EncodedSequence:
    """The result of encoding a token sequence: the prompt text plus enough
    identity to log and to measure."""

    text: str
    encoder_name: str
    schema_version: str = SCHEMA_VERSION

    def token_count(self, tokenizer: "Tokenizer") -> int:
        return len(tokenizer.encode(self.text))

    @property
    def metadata(self) -> dict[str, str]:
        """Encoder identity for MLflow logging -- `encoding` and
        `schema_version` specifically; `domain`/`is_fallback` (also listed in
        WORKING_PLAN's definition of done) are populated by the caller, not
        this module, since an encoder has no notion of either."""
        return {"encoding": self.encoder_name, "schema_version": self.schema_version}


@runtime_checkable
class Tokenizer(Protocol):
    """The minimal shape this module needs from a tokeniser -- satisfied by
    e.g. a Hugging Face `PreTrainedTokenizerBase` -- so this library never
    takes a hard dependency on any specific tokeniser package. The caller
    supplies whichever tokeniser the target model actually uses."""

    def encode(self, text: str) -> Sequence[object]:
        ...


@runtime_checkable
class SequenceEncoder(Protocol):
    name: str

    def encode(self, tokens: Sequence[str]) -> EncodedSequence:
        ...

    def decode(self, text: str) -> list[str]:
        ...


@dataclass(frozen=True)
class NeutralEncoder:
    """E3 (default): compound tokens spelled as neutral C{0-4}_V{0-4}
    labels, space-joined. `decode()` always reconstructs the canonical
    neutral spelling regardless of what the original input was spelled as."""

    name: str = "neutral"
    separator: str = DEFAULT_SEPARATOR

    def encode(self, tokens: Sequence[str]) -> EncodedSequence:
        rendered = []
        for token in tokens:
            if token.startswith(EVENT_PREFIX):
                rendered.append(token)
                continue
            c, v = _indices_of(token)
            rendered.append(f"{NEUTRAL_CHANGE_LABELS[c]}_{NEUTRAL_VOLATILITY_LABELS[v]}")
        return EncodedSequence(text=self.separator.join(rendered), encoder_name=self.name)

    def decode(self, text: str) -> list[str]:
        return text.split(self.separator) if text else []


@dataclass(frozen=True)
class SemanticEncoder:
    """E2: the same 25 buckets spelled out as words, e.g. `C_FLAT_V_LOW`."""

    name: str = "semantic"
    separator: str = DEFAULT_SEPARATOR

    def encode(self, tokens: Sequence[str]) -> EncodedSequence:
        rendered = []
        for token in tokens:
            if token.startswith(EVENT_PREFIX):
                rendered.append(token)
                continue
            c, v = _indices_of(token)
            rendered.append(f"{SEMANTIC_CHANGE_LABELS[c]}_{SEMANTIC_VOLATILITY_LABELS[v]}")
        return EncodedSequence(text=self.separator.join(rendered), encoder_name=self.name)

    def decode(self, text: str) -> list[str]:
        # Decoding reconstructs the canonical neutral spelling (see
        # NeutralEncoder), not the semantic spelling that was encoded --
        # every encoder's decode() output is comparable this way.
        symbols = text.split(self.separator) if text else []
        rebuilt = []
        for symbol in symbols:
            if symbol.startswith(EVENT_PREFIX):
                rebuilt.append(symbol)
                continue
            c, v = _indices_of(symbol)
            rebuilt.append(f"{NEUTRAL_CHANGE_LABELS[c]}_{NEUTRAL_VOLATILITY_LABELS[v]}")
        return rebuilt


@dataclass(frozen=True)
class SingleCharEncoder:
    """E4: each compound token becomes exactly one character. Event tokens
    are passed through unchanged (see module docstring) -- the token-budget
    win this encoder exists for is specifically the compound C/V token."""

    name: str = "single_char"
    separator: str = DEFAULT_SEPARATOR

    def encode(self, tokens: Sequence[str]) -> EncodedSequence:
        rendered = []
        for token in tokens:
            if token.startswith(EVENT_PREFIX):
                rendered.append(token)
                continue
            c, v = _indices_of(token)
            rendered.append(chr(ord("A") + c * N_BUCKETS + v))
        return EncodedSequence(text=self.separator.join(rendered), encoder_name=self.name)

    def decode(self, text: str) -> list[str]:
        symbols = text.split(self.separator) if text else []
        rebuilt = []
        for symbol in symbols:
            if symbol.startswith(EVENT_PREFIX):
                rebuilt.append(symbol)
                continue
            if symbol not in _CHAR_TO_INDICES:
                raise ValueError(f"unrecognized single-character symbol {symbol!r}")
            c, v = _CHAR_TO_INDICES[symbol]
            rebuilt.append(f"{NEUTRAL_CHANGE_LABELS[c]}_{NEUTRAL_VOLATILITY_LABELS[v]}")
        return rebuilt


NEUTRAL_ENCODER = NeutralEncoder()
SEMANTIC_ENCODER = SemanticEncoder()
SINGLE_CHAR_ENCODER = SingleCharEncoder()
ALL_ENCODERS: tuple[SequenceEncoder, ...] = (NEUTRAL_ENCODER, SEMANTIC_ENCODER, SINGLE_CHAR_ENCODER)


@dataclass(frozen=True)
class EncodingReport:
    """One row of the encoder comparison table."""

    encoder_name: str
    text: str
    char_count: int
    token_count: int


def compare_encoders(
    tokens: Sequence[str], tokenizer: Tokenizer, encoders: Sequence[SequenceEncoder] = ALL_ENCODERS
) -> list[EncodingReport]:
    """Encode the SAME input sequence with every given encoder and report
    each one's resulting token count under `tokenizer` -- the comparison the
    E2-E4 ablation is actually about."""
    reports = []
    for encoder in encoders:
        encoded = encoder.encode(tokens)
        reports.append(
            EncodingReport(
                encoder_name=encoded.encoder_name,
                text=encoded.text,
                char_count=len(encoded.text),
                token_count=encoded.token_count(tokenizer),
            )
        )
    return reports
