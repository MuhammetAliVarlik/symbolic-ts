"""Token vocabulary: the single source of truth for symbolic-ts's token schema.

Every other module in this package, and every downstream consumer
(symbolic-ts-research, FinwiseBackend), imports the vocabulary from here rather
than defining its own token strings -- the inconsistent token schemes
duplicated across FinwiseBackend were the largest structural defect this
library exists to fix.

No bin edges, window sizes, or other numeric binning parameters live in this
module -- see `binning.py` for those. This module only defines which token
strings are valid and what they mean, not how raw data gets mapped to them.

Design: two channels (change, volatility), five buckets each (0 = lowest/
most-negative .. 4 = highest/most-positive), combined into one compound token
per timestep, e.g. "C2_V1". This is a 2-channel design, not the originally
planned 3 (no `level`) -- see symbolic-ts-research's Phase 0 notes
(F0-04 through F0-11) for why: a raw level is non-stationary and doesn't fit
fixed training-derived thresholds, so it was dropped rather than tokenised.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

SCHEMA_VERSION = "1.0.0"

N_BUCKETS = 5
BUCKET_INDICES: tuple[int, ...] = tuple(range(N_BUCKETS))

CHANGE_PREFIX = "C"
VOLATILITY_PREFIX = "V"

# Neutral labels (D4 default).
NEUTRAL_CHANGE_LABELS: tuple[str, ...] = tuple(f"{CHANGE_PREFIX}{i}" for i in BUCKET_INDICES)
NEUTRAL_VOLATILITY_LABELS: tuple[str, ...] = tuple(f"{VOLATILITY_PREFIX}{i}" for i in BUCKET_INDICES)

# Semantic labels (D4 ablation). Bucket 0 = most extreme low, bucket 4 = most
# extreme high -- matches the existing "P_CRASH_V_SURGE" convention already
# cited elsewhere in the project (an extreme-negative change alongside
# extreme-high volatility).
SEMANTIC_CHANGE_WORDS: tuple[str, ...] = ("CRASH", "DROP", "FLAT", "RISE", "SURGE")
SEMANTIC_VOLATILITY_WORDS: tuple[str, ...] = ("CALM", "LOW", "MODERATE", "HIGH", "SURGE")
SEMANTIC_CHANGE_LABELS: tuple[str, ...] = tuple(f"{CHANGE_PREFIX}_{w}" for w in SEMANTIC_CHANGE_WORDS)
SEMANTIC_VOLATILITY_LABELS: tuple[str, ...] = tuple(f"{VOLATILITY_PREFIX}_{w}" for w in SEMANTIC_VOLATILITY_WORDS)


def _compound_tokens(change_labels: tuple[str, ...], volatility_labels: tuple[str, ...]) -> frozenset[str]:
    return frozenset(f"{c}_{v}" for c in change_labels for v in volatility_labels)


NEUTRAL_TOKENS: frozenset[str] = _compound_tokens(NEUTRAL_CHANGE_LABELS, NEUTRAL_VOLATILITY_LABELS)
SEMANTIC_TOKENS: frozenset[str] = _compound_tokens(SEMANTIC_CHANGE_LABELS, SEMANTIC_VOLATILITY_LABELS)

# Event token namespace (D3) -- reserved here, populated by domain event
# adapters in events.py (F1-04). E_NONE is the one guaranteed member: "no
# event", emitted explicitly rather than omitted, so sequence length stays
# stable regardless of whether an event fired for a given timestep.
EVENT_PREFIX = "E_"
EVENT_NONE = f"{EVENT_PREFIX}NONE"

_registered_event_tokens: set[str] = {EVENT_NONE}


def register_event_tokens(tokens: Iterable[str]) -> None:
    """Extension point for domain event adapters (F1-04) to add their own
    event tokens (e.g. E_WEEKEND, E_HOUR_PEAK) to the recognised vocabulary.
    Every registered token must start with EVENT_PREFIX."""
    for token in tokens:
        if not token.startswith(EVENT_PREFIX):
            raise ValueError(f"event token {token!r} must start with {EVENT_PREFIX!r}")
        _registered_event_tokens.add(token)


def registered_event_tokens() -> frozenset[str]:
    return frozenset(_registered_event_tokens)


def known_tokens() -> frozenset[str]:
    """All currently-recognised tokens: neutral, semantic, and registered event."""
    return NEUTRAL_TOKENS | SEMANTIC_TOKENS | _registered_event_tokens


def validate_token(token: str, *, allowed: frozenset[str] | None = None) -> bool:
    """True if `token` is a recognised vocabulary member. By default any
    known token (neutral, semantic, or a registered event token) is accepted;
    pass `allowed` (e.g. NEUTRAL_TOKENS) to check conformance to one specific
    label scheme instead -- e.g. verifying a model trained on neutral labels
    never emits a semantic one."""
    return token in (allowed if allowed is not None else known_tokens())


@dataclass
class ValidationReport:
    total: int
    valid_count: int
    invalid_tokens: list[tuple[int, str]] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.invalid_tokens

    @property
    def invalid_count(self) -> int:
        return len(self.invalid_tokens)

    @property
    def conformance_rate(self) -> float:
        return self.valid_count / self.total if self.total else 1.0


def validate_sequence(sequence: list[str], *, allowed: frozenset[str] | None = None) -> ValidationReport:
    invalid = [(i, token) for i, token in enumerate(sequence) if not validate_token(token, allowed=allowed)]
    return ValidationReport(total=len(sequence), valid_count=len(sequence) - len(invalid), invalid_tokens=invalid)
