from datetime import date, datetime

import pytest

from symbolic_ts import events as ev
from symbolic_ts import vocabulary as v


@pytest.fixture(autouse=True)
def clean_registries():
    token_snapshot = set(v._registered_event_tokens)
    ev._registered_adapters.clear()
    yield
    ev._registered_adapters.clear()
    v._registered_event_tokens.clear()
    v._registered_event_tokens.update(token_snapshot)


def test_finance_event_adapter_satisfies_protocol():
    assert isinstance(ev.FinanceEventAdapter(), ev.EventAdapter)


def test_ett_event_adapter_satisfies_protocol():
    assert isinstance(ev.ETTEventAdapter(), ev.EventAdapter)


def test_finance_adapter_fires_on_configured_dates():
    adapter = ev.FinanceEventAdapter(
        earnings_dates=frozenset({date(2024, 1, 25)}),
        dividend_dates=frozenset({date(2024, 2, 9)}),
        fomc_dates=frozenset({date(2024, 3, 20)}),
    )
    assert adapter.events_for(datetime(2024, 1, 25, 9, 30)) == ["E_EARNINGS"]
    assert adapter.events_for(datetime(2024, 2, 9, 9, 30)) == ["E_DIVIDEND"]
    assert adapter.events_for(datetime(2024, 3, 20, 9, 30)) == ["E_FOMC"]


def test_finance_adapter_fires_multiple_events_same_day():
    same_day = date(2024, 1, 25)
    adapter = ev.FinanceEventAdapter(
        earnings_dates=frozenset({same_day}), dividend_dates=frozenset({same_day})
    )
    result = adapter.events_for(datetime(2024, 1, 25))
    assert set(result) == {"E_EARNINGS", "E_DIVIDEND"}


def test_finance_adapter_emits_event_none_explicitly_when_nothing_fires():
    adapter = ev.FinanceEventAdapter()
    assert adapter.events_for(datetime(2024, 1, 25)) == [v.EVENT_NONE]


def test_finance_adapter_tokens_validate_against_vocabulary():
    adapter = ev.FinanceEventAdapter(
        earnings_dates=frozenset({date(2024, 1, 25)}),
        dividend_dates=frozenset({date(2024, 1, 25)}),
        fomc_dates=frozenset({date(2024, 1, 25)}),
    )
    for token in adapter.events_for(datetime(2024, 1, 25)):
        assert v.validate_token(token)
    assert v.validate_token(v.EVENT_NONE)


def test_ett_adapter_fires_on_weekend():
    adapter = ev.ETTEventAdapter()
    saturday = datetime(2024, 1, 6, 12)
    monday = datetime(2024, 1, 8, 12)
    assert "E_WEEKEND" in adapter.events_for(saturday)
    assert "E_WEEKEND" not in adapter.events_for(monday)


def test_ett_adapter_fires_on_configured_holiday():
    adapter = ev.ETTEventAdapter(holiday_dates=frozenset({date(2024, 1, 1)}))
    assert "E_HOLIDAY" in adapter.events_for(datetime(2024, 1, 1, 12))
    assert "E_HOLIDAY" not in adapter.events_for(datetime(2024, 1, 2, 12))


def test_ett_adapter_fires_on_configured_peak_hour():
    adapter = ev.ETTEventAdapter(peak_hours=frozenset({18, 19, 20}))
    assert "E_HOUR_PEAK" in adapter.events_for(datetime(2024, 1, 8, 19))
    assert "E_HOUR_PEAK" not in adapter.events_for(datetime(2024, 1, 8, 3))


def test_ett_adapter_defaults_never_fire_holiday_or_peak_hour():
    adapter = ev.ETTEventAdapter()
    result = adapter.events_for(datetime(2024, 1, 8, 19))  # Monday, would-be peak hour
    assert "E_HOLIDAY" not in result
    assert "E_HOUR_PEAK" not in result


@pytest.mark.parametrize(
    "timestamp",
    [datetime(2024, 3, 1), datetime(2024, 6, 1), datetime(2024, 9, 1), datetime(2024, 12, 1)],
)
def test_ett_adapter_fires_season_change_on_first_day_of_season(timestamp):
    adapter = ev.ETTEventAdapter()
    assert "E_SEASON_CHANGE" in adapter.events_for(timestamp)


def test_ett_adapter_does_not_fire_season_change_mid_season():
    adapter = ev.ETTEventAdapter()
    assert "E_SEASON_CHANGE" not in adapter.events_for(datetime(2024, 3, 15))


def test_ett_adapter_emits_event_none_explicitly_when_nothing_fires():
    adapter = ev.ETTEventAdapter()
    # Monday, mid-season, no configured holiday/peak hour.
    assert adapter.events_for(datetime(2024, 1, 8, 12)) == [v.EVENT_NONE]


def test_ett_adapter_tokens_validate_against_vocabulary():
    adapter = ev.ETTEventAdapter(
        holiday_dates=frozenset({date(2024, 1, 6)}), peak_hours=frozenset({12})
    )
    for token in adapter.events_for(datetime(2024, 1, 6, 12)):
        assert v.validate_token(token)


def test_register_and_get_event_adapter_roundtrip():
    adapter = ev.FinanceEventAdapter()
    ev.register_event_adapter("finance", adapter)
    assert ev.get_event_adapter("finance") is adapter
    assert "finance" in ev.registered_event_adapter_names()


def test_get_event_adapter_raises_for_unregistered_name():
    with pytest.raises(KeyError):
        ev.get_event_adapter("no_such_domain")


def test_a_third_domain_needs_no_library_change():
    class CustomAdapter:
        def events_for(self, timestamp, context=None):
            return [v.EVENT_NONE]

    adapter = CustomAdapter()
    assert isinstance(adapter, ev.EventAdapter)
    ev.register_event_adapter("custom", adapter)
    assert ev.get_event_adapter("custom").events_for(datetime(2024, 1, 1)) == [v.EVENT_NONE]
