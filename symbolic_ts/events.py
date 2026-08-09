"""Event tokens: calendar/seasonality effects (D3), layered onto the
change/volatility token stream as a separate additive namespace rather than a
third numeric channel -- see vocabulary.py's EVENT_PREFIX/EVENT_NONE.

A domain adapter maps one timestamp (plus optional per-call metadata) to zero
or more event tokens. `EVENT_NONE` is returned explicitly whenever nothing
else fires, so a token sequence's length never depends on how many events
happened to occur -- the caller gets exactly one events_for() result per
timestep, never a variable-length list to reconcile against the numeric
token stream.

New domains don't require a library change: `EventAdapter` is a structural
Protocol, not a base class to inherit from -- any object with a matching
events_for() method satisfies it -- and adapters are looked up by name via
register_event_adapter()/get_event_adapter() rather than an
`if domain == ...` branch anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol, runtime_checkable

from symbolic_ts.vocabulary import EVENT_NONE, register_event_tokens


@runtime_checkable
class EventAdapter(Protocol):
    def events_for(self, timestamp: datetime, context: Any = None) -> list[str]:
        ...


_registered_adapters: dict[str, "EventAdapter"] = {}


def register_event_adapter(name: str, adapter: EventAdapter) -> None:
    """Extension point: register an adapter instance under a domain name
    instead of adding a branch to this module. A third domain (neither
    finance nor ETT) needs only this call, not a library change."""
    _registered_adapters[name] = adapter


def get_event_adapter(name: str) -> EventAdapter:
    try:
        return _registered_adapters[name]
    except KeyError:
        raise KeyError(
            f"no event adapter registered under {name!r}; call register_event_adapter() first"
        ) from None


def registered_event_adapter_names() -> frozenset[str]:
    return frozenset(_registered_adapters)


def _as_date(timestamp: datetime) -> date:
    return timestamp.date() if isinstance(timestamp, datetime) else timestamp


@dataclass
class FinanceEventAdapter:
    """Finance calendar events: earnings, dividends, FOMC days. Each is a
    caller-supplied set of dates -- this adapter has no built-in knowledge of
    any specific ticker's or year's calendar, matching ProjectionConfig's
    stance (projection.py) that a domain is configuration, not a hardcoded
    branch."""

    earnings_dates: frozenset[date] = field(default_factory=frozenset)
    dividend_dates: frozenset[date] = field(default_factory=frozenset)
    fomc_dates: frozenset[date] = field(default_factory=frozenset)

    EVENT_TOKENS = ("E_EARNINGS", "E_DIVIDEND", "E_FOMC")

    def __post_init__(self) -> None:
        register_event_tokens(self.EVENT_TOKENS)

    def events_for(self, timestamp: datetime, context: Any = None) -> list[str]:
        d = _as_date(timestamp)
        events = []
        if d in self.earnings_dates:
            events.append("E_EARNINGS")
        if d in self.dividend_dates:
            events.append("E_DIVIDEND")
        if d in self.fomc_dates:
            events.append("E_FOMC")
        return events or [EVENT_NONE]


# Meteorological (not astronomical) Northern Hemisphere season boundaries --
# a modelling choice, not a Phase 0 finding. F0-09 established that ETT has a
# genuine, exact 24-hour seasonal cycle worth flagging
# (symbolic-ts-research/experiments/notes/F0-09.md) but did not pin a
# specific season-boundary convention or a "peak" clock hour; inventing
# either as a silent default would be a number this library can't back with
# evidence, so `peak_hours`/`holiday_dates` below have no built-in default.
_SEASON_START_MONTHS: dict[int, str] = {12: "WINTER", 3: "SPRING", 6: "SUMMER", 9: "AUTUMN"}


@dataclass
class ETTEventAdapter:
    """ETT calendar/seasonality events: weekend, holiday, peak hour, and
    season change.

    `holiday_dates` and `peak_hours` are caller-supplied and empty by
    default (see the module-level note above) -- with both left empty, only
    `E_WEEKEND` and `E_SEASON_CHANGE` can ever fire.

    "Season change" fires on the first calendar day of each meteorological
    season rather than naming which season it is: `events_for()` is called
    per-timestamp with no memory of the previous call, so detecting an
    edge (this is a NEW season) has to be a property of the timestamp alone,
    not a comparison against a prior one.
    """

    holiday_dates: frozenset[date] = field(default_factory=frozenset)
    peak_hours: frozenset[int] = field(default_factory=frozenset)

    EVENT_TOKENS = ("E_WEEKEND", "E_HOLIDAY", "E_HOUR_PEAK", "E_SEASON_CHANGE")

    def __post_init__(self) -> None:
        register_event_tokens(self.EVENT_TOKENS)

    def events_for(self, timestamp: datetime, context: Any = None) -> list[str]:
        events = []
        if timestamp.weekday() >= 5:
            events.append("E_WEEKEND")
        if _as_date(timestamp) in self.holiday_dates:
            events.append("E_HOLIDAY")
        if timestamp.hour in self.peak_hours:
            events.append("E_HOUR_PEAK")
        if timestamp.day == 1 and timestamp.month in _SEASON_START_MONTHS:
            events.append("E_SEASON_CHANGE")
        return events or [EVENT_NONE]
