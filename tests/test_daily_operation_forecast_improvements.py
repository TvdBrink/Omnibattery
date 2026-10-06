"""Tests for Daily Operation tomorrow solar forecast and planned discharge projection."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from custom_components.omnibattery import ChargeDischargeController
from custom_components.omnibattery.pricing import PriceSlot
from custom_components.omnibattery.pricing.curtailment import PreDischargeSlot
from custom_components.omnibattery.pricing.daily_timeline import (
    ACTION_DISCHARGE,
    BatteryProjectionInput,
    ProjectionIntervalInput,
    simulate_battery_projection,
)
from custom_components.omnibattery.pricing.high_price_discharge import (
    HighPriceDischargePlan,
    TriggerAllocation,
)
from custom_components.omnibattery.solar_forecast import (
    SolarForecastPeriod,
    read_solar_forecast_kwh,
    solar_periods_from_wh_hours,
)
from custom_components.omnibattery.tracking.daily_projection import (
    DailyOperationProjectionRequest,
    build_daily_operation_projection,
)


MADRID = ZoneInfo("Europe/Madrid")


def test_solar_periods_from_wh_hours_conversion():
    """Verify wh_hours dictionary is accurately converted to SolarForecastPeriod items."""
    wh_hours = {
        "2026-08-24T06:00:00+02:00": 150.0,
        "2026-08-24T07:00:00+02:00": 600.0,
        "2026-08-24T08:00:00+02:00": 1200.0,
        "2026-08-25T06:00:00+02:00": 200.0,
        "2026-08-25T07:00:00+02:00": 800.0,
    }
    periods = solar_periods_from_wh_hours(wh_hours, default_timezone=MADRID)
    assert len(periods) == 5

    # Check first period
    assert periods[0].start == datetime(2026, 8, 24, 6, 0, tzinfo=MADRID)
    assert periods[0].end == datetime(2026, 8, 24, 7, 0, tzinfo=MADRID)
    assert periods[0].energy_kwh == pytest.approx(0.15)

    # Check tomorrow period
    assert periods[3].start == datetime(2026, 8, 25, 6, 0, tzinfo=MADRID)
    assert periods[3].end == datetime(2026, 8, 25, 7, 0, tzinfo=MADRID)
    assert periods[3].energy_kwh == pytest.approx(0.20)


def test_solar_periods_from_wh_hours_handles_empty_and_invalid():
    """Empty or invalid dictionaries return an empty tuple."""
    assert solar_periods_from_wh_hours(None) == ()
    assert solar_periods_from_wh_hours({}) == ()
    assert solar_periods_from_wh_hours({"invalid-timestamp": 500}) == ()
    assert solar_periods_from_wh_hours({"2026-08-24T06:00:00+02:00": -100}) == ()


def test_read_solar_forecast_uses_cached_controller_periods():
    """If the entity state lacks period attributes, cached periods on controller are used."""
    cached_period = SolarForecastPeriod(
        datetime(2026, 8, 24, 6, 0, tzinfo=MADRID),
        datetime(2026, 8, 24, 7, 0, tzinfo=MADRID),
        0.5,
    )
    controller = SimpleNamespace(
        solar_forecast_sensor="sensor.energy_production_today",
        solar_forecast_remaining_sensor=None,
        _energy_solar_periods=(cached_period,),
    )
    state = SimpleNamespace(state="5.0", attributes={})
    hass = SimpleNamespace(states=SimpleNamespace(get=lambda _id: state))

    forecast = read_solar_forecast_kwh(hass, controller, update_controller=True)
    assert forecast is not None
    assert forecast.kwh == 5.0
    assert forecast.periods == (cached_period,)
    assert controller.solar_forecast_periods == (cached_period,)


def test_simulate_battery_projection_with_planned_discharge_allocation():
    """Planned export discharge drops stored energy, sets battery_to_grid, and sets ACTION_DISCHARGE."""
    start = datetime(2026, 8, 24, 18, 0, tzinfo=MADRID)
    intervals = [
        ProjectionIntervalInput(
            start=start + timedelta(minutes=15 * i),
            end=start + timedelta(minutes=15 * (i + 1)),
            solar_kwh=0.0,
            consumption_kwh=0.1,  # 0.1 kWh consumption per interval
        )
        for i in range(4)  # 1 hour total
    ]

    battery = BatteryProjectionInput(
        key="batt1",
        stored_kwh=5.0,
        capacity_kwh=10.0,
        min_soc_pct=10.0,
        max_soc_pct=100.0,
        charge_power_w=3000.0,
        discharge_power_w=3000.0,
        can_discharge=True,
    )

    # Case 1: No planned export discharge (only home deficit 0.1 kWh/interval)
    base_result = simulate_battery_projection(intervals, [battery])
    base_flow = base_result.intervals[0]
    assert base_flow.battery_to_home_kwh == pytest.approx(0.1)
    assert base_flow.battery_to_grid_kwh == pytest.approx(0.0)
    assert base_flow.stored_energy_end_kwh == pytest.approx(4.9)

    # Case 2: With planned export discharge allocation of 0.8 kWh over the hour (0.2 kWh per interval)
    discharge_alloc = TriggerAllocation(
        start=start,
        end=start + timedelta(hours=1),
        export_price=0.35,
        threshold=0.20,
        energy_kwh=0.8,
        power_w=800.0,
    )
    export_result = simulate_battery_projection(
        intervals, [battery], discharge_allocations=[discharge_alloc]
    )

    export_flow = export_result.intervals[0]
    # Covers home deficit (0.1 kWh) AND export quota (0.2 kWh)
    assert export_flow.battery_to_home_kwh == pytest.approx(0.1)
    assert export_flow.battery_to_grid_kwh == pytest.approx(0.2)
    assert export_flow.stored_energy_end_kwh == pytest.approx(4.7)
    assert export_flow.action_mask & ACTION_DISCHARGE
    # Total discharged over 4 intervals: 4 * (0.1 + 0.2) = 1.2 kWh
    assert export_result.intervals[-1].stored_energy_end_kwh == pytest.approx(3.8)


def test_simulate_battery_projection_respects_minimum_soc_floor_during_export():
    """Planned export discharge is capped and never breaches the battery minimum stored energy."""
    start = datetime(2026, 8, 24, 18, 0, tzinfo=MADRID)
    interval = ProjectionIntervalInput(
        start=start,
        end=start + timedelta(minutes=15),
        solar_kwh=0.0,
        consumption_kwh=0.0,
    )

    battery = BatteryProjectionInput(
        key="batt1",
        stored_kwh=1.2,
        capacity_kwh=10.0,
        min_soc_pct=10.0,  # 1.0 kWh minimum stored
        max_soc_pct=100.0,
        charge_power_w=3000.0,
        discharge_power_w=3000.0,
        can_discharge=True,
    )

    # Allocation requests 1.0 kWh export, but only 0.2 kWh is available above minimum
    discharge_alloc = TriggerAllocation(
        start=start,
        end=start + timedelta(minutes=15),
        export_price=0.35,
        threshold=0.20,
        energy_kwh=1.0,
        power_w=4000.0,
    )

    result = simulate_battery_projection(
        [interval], [battery], discharge_allocations=[discharge_alloc]
    )
    flow = result.intervals[0]
    assert flow.battery_to_grid_kwh == pytest.approx(0.2)
    assert flow.stored_energy_end_kwh == pytest.approx(1.0)  # Exactly at minimum floor


def test_build_daily_operation_projection_includes_discharge_allocations():
    """DailyOperationProjectionRequest carries discharge allocations and reflects them in soc_end_pct."""
    now = datetime(2026, 8, 24, 12, 0, tzinfo=MADRID)
    interval = ProjectionIntervalInput(
        start=datetime(2026, 8, 24, 18, 0, tzinfo=MADRID),
        end=datetime(2026, 8, 24, 18, 15, tzinfo=MADRID),
        solar_kwh=0.0,
        consumption_kwh=0.1,
    )

    battery = BatteryProjectionInput(
        key="batt1",
        stored_kwh=8.0,
        capacity_kwh=10.0,
        min_soc_pct=10.0,
        max_soc_pct=100.0,
        charge_power_w=3000.0,
        discharge_power_w=3000.0,
        can_discharge=True,
    )

    alloc = TriggerAllocation(
        start=datetime(2026, 8, 24, 18, 0, tzinfo=MADRID),
        end=datetime(2026, 8, 24, 18, 15, tzinfo=MADRID),
        export_price=0.40,
        threshold=0.20,
        energy_kwh=0.5,
        power_w=2000.0,
    )

    req = DailyOperationProjectionRequest(
        now=now,
        plan_intervals=(interval,),
        allocations=(),
        battery_inputs=(battery,),
        mode="dynamic_pricing",
        decision_data={},
        discharge_allocations=(alloc,),
    )

    projection = build_daily_operation_projection(req)
    assert projection is not None
    assert len(projection["intervals"]) == 1
    item = projection["intervals"][0]

    # Home deficit 0.1 kWh + export 0.5 kWh = 0.6 kWh total discharge
    assert item["battery_to_home_kwh"] == pytest.approx(0.1)
    assert item["battery_to_grid_kwh"] == pytest.approx(0.5)
    assert item["discharge_from_battery_kwh"] == pytest.approx(0.6)
    assert item["stored_energy_end_kwh"] == pytest.approx(7.4)
    assert item["soc_end_pct"] == pytest.approx(74.0)  # 7.4 / 10.0 * 100%
    assert item["action_mask"] & ACTION_DISCHARGE


def test_build_daily_operation_projection_includes_curtailment_predischarge():
    """Curtailment PreDischargeSlot with planned_energy_kwh is properly modeled in projection."""
    now = datetime(2026, 8, 24, 10, 0, tzinfo=MADRID)
    interval = ProjectionIntervalInput(
        start=datetime(2026, 8, 24, 11, 0, tzinfo=MADRID),
        end=datetime(2026, 8, 24, 11, 15, tzinfo=MADRID),
        solar_kwh=0.0,
        consumption_kwh=0.0,
    )
    battery = BatteryProjectionInput(
        key="batt1",
        stored_kwh=9.0,
        capacity_kwh=10.0,
        min_soc_pct=10.0,
        max_soc_pct=100.0,
        charge_power_w=3000.0,
        discharge_power_w=3000.0,
        can_discharge=True,
    )
    pre_slot = PreDischargeSlot(
        start=datetime(2026, 8, 24, 11, 0, tzinfo=MADRID),
        end=datetime(2026, 8, 24, 11, 15, tzinfo=MADRID),
        price=0.05,
        planned_energy_kwh=0.75,
        power_w=3000.0,
        export_target_w=3000.0,
    )

    req = DailyOperationProjectionRequest(
        now=now,
        plan_intervals=(interval,),
        allocations=(),
        battery_inputs=(battery,),
        mode="dynamic_pricing",
        decision_data={},
        discharge_allocations=(pre_slot,),
    )

    projection = build_daily_operation_projection(req)
    assert projection is not None
    item = projection["intervals"][0]
    assert item["battery_to_grid_kwh"] == pytest.approx(0.75)
    assert item["discharge_from_battery_kwh"] == pytest.approx(0.75)
    assert item["stored_energy_end_kwh"] == pytest.approx(8.25)
    assert item["soc_end_pct"] == pytest.approx(82.5)
    assert item["action_mask"] & ACTION_DISCHARGE


def test_extended_projection_with_periods_renders_tomorrow_solar():
    """Tomorrow morning periods appear in the extended projection intervals."""
    now = datetime(2026, 8, 24, 20, 0, tzinfo=MADRID)
    local_midnight = datetime(2026, 8, 25, 0, 0, tzinfo=MADRID)

    # 48 intervals from midnight to noon tomorrow
    intervals = []
    for i in range(48):
        start = local_midnight + timedelta(minutes=15 * i)
        end = start + timedelta(minutes=15)
        # Solar period active from 06:00 to 12:00
        solar_val = 0.1 if start >= datetime(2026, 8, 25, 6, 0, tzinfo=MADRID) else 0.0
        intervals.append(
            ProjectionIntervalInput(
                start=start,
                end=end,
                solar_kwh=solar_val,
                consumption_kwh=0.05,
            )
        )

    battery = BatteryProjectionInput(
        key="batt1",
        stored_kwh=5.0,
        capacity_kwh=10.0,
        min_soc_pct=10.0,
        max_soc_pct=100.0,
        charge_power_w=3000.0,
        discharge_power_w=3000.0,
        can_discharge=True,
    )

    req = DailyOperationProjectionRequest(
        now=now,
        plan_intervals=tuple(intervals),
        allocations=(),
        battery_inputs=(battery,),
        mode="dynamic_pricing",
        decision_data={},
        extension_hours=12,
    )

    projection = build_daily_operation_projection(req)
    assert projection is not None
    assert len(projection["extended_intervals"]) == 48

    # Verify that daytime intervals tomorrow have non-zero solar forecast
    morning_interval = [
        item for item in projection["extended_intervals"]
        if item["index"] == 28  # 07:00 tomorrow
    ][0]
    assert morning_interval["solar_kwh"] == pytest.approx(0.1)
    assert morning_interval["solar_forecast_kwh"] == pytest.approx(0.1)


def test_extract_forecast_periods_solcast_detailed_forecast():
    """Verify Solcast detailedForecast half-hourly attributes are parsed into SolarForecastPeriod instances."""
    from custom_components.omnibattery.solar_forecast import _extract_forecast_periods

    state = SimpleNamespace(
        state="14.2",
        attributes={
            "detailedForecast": [
                {
                    "period_start": "2026-10-07T07:00:00+02:00",
                    "period_end": "2026-10-07T07:30:00+02:00",
                    "pv_estimate": 2.4,  # kW -> 2.4 * 0.5h = 1.2 kWh
                },
                {
                    "period_start": "2026-10-07T07:30:00+02:00",
                    "period_end": "2026-10-07T08:00:00+02:00",
                    "pv_estimate": 3.6,  # kW -> 3.6 * 0.5h = 1.8 kWh
                },
            ]
        },
    )
    periods = _extract_forecast_periods(state)
    assert len(periods) == 2
    assert periods[0].energy_kwh == pytest.approx(1.2)
    assert periods[1].energy_kwh == pytest.approx(1.8)


def test_build_solar_timeline_provider_weights_when_mode_is_off():
    """External provider forecast must not be blocked when learned profile mode is off."""
    from custom_components.omnibattery.pricing.solar_timeline import (
        build_boundaries,
        build_solar_timeline,
    )

    now = datetime(2026, 10, 6, 21, 30, tzinfo=MADRID)
    horizon_end = datetime(2026, 10, 7, 12, 0, tzinfo=MADRID)
    boundaries = build_boundaries(now, horizon_end)

    # Synthetic temporal shape with sun tomorrow morning
    temporal_shape = [0.0] * len(boundaries)
    temporal_shape[-10] = 0.5
    temporal_shape[-9] = 1.0

    timeline = build_solar_timeline(
        boundaries,
        1.5,
        safety_margin_kwh=0.0,
        provider_periods=None,
        temporal_shape=temporal_shape,
        mode="off",  # learned profile off
    )
    assert timeline.source == "provider"
    assert sum(timeline.intervals_kwh) == pytest.approx(1.5)


def test_helios_forecast_in_memory_coordinator_populates_tomorrow_solar():
    """Verify Helios Forecast data in hass.data is extracted and populates tomorrow's projection."""
    from custom_components.omnibattery.pricing.engine import PricingManager
    from custom_components.omnibattery.tracking.consumption_profile import ConsumptionForecast

    wh_hours = {
        "2026-10-07T04:00:00+00:00": 150.0,
        "2026-10-07T05:00:00+00:00": 600.0,
        "2026-10-08T05:00:00+00:00": 400.0,
        "2026-10-08T06:00:00+00:00": 1000.0,
        "2026-10-08T07:00:00+00:00": 1800.0,
        "2026-10-08T08:00:00+00:00": 2500.0,
    }
    summary = SimpleNamespace(wh_hours=wh_hours)
    coordinator = SimpleNamespace(data=SimpleNamespace(summary=summary))

    now = datetime(2026, 10, 7, 21, 30, tzinfo=MADRID)
    day_end = datetime(2026, 10, 8, 0, 0, tzinfo=MADRID)
    horizon_end = datetime(2026, 10, 8, 12, 0, tzinfo=MADRID)

    sensor_state = SimpleNamespace(state="0.0", attributes={})
    hass = SimpleNamespace(
        data={"helios_forecast": {"entry_1": coordinator}},
        states=SimpleNamespace(
            get=lambda eid: sensor_state
            if eid == "sensor.helios_forecast_energy_today_remaining"
            else None
        ),
        config=SimpleNamespace(time_zone="Europe/Madrid"),
    )

    forecast = ConsumptionForecast(10.0, [0.1] * 96, "legacy_daily", True)
    tracker = SimpleNamespace(
        consumption_profile=SimpleNamespace(),
        forecast_consumption_between=lambda *_args, **_kwargs: forecast,
    )

    controller = SimpleNamespace(
        solar_forecast_remaining_sensor="sensor.helios_forecast_energy_today_remaining",
        solar_forecast_sensor=None,
        _energy_solar_periods=(),
        solar_profile_mode="off",
        _consumption_tracker=tracker,
        _raw_consumption_tracker=tracker,
        capacity_protection_limit=0.0,
        charge_delay_enabled=False,
        predictive_charging_enabled=False,
        time_slot_enabled=False,
        dynamic_pricing_enabled=True,
        nordpool_enabled=False,
        price_discharge_control_enabled=False,
        _predictive_safety_margin_kwh=0.0,
        coordinators=[],
        _is_battery_manual_owned=lambda _coordinator: False,
        max_contracted_power=0.0,
        max_charge_capacity=0.0,
        _last_chronological_diagnostics=None,
    )

    pricing = PricingManager(hass, controller)
    slots = [PriceSlot(now, horizon_end, 0.15)]
    proj = pricing.build_extended_chronological_projection(
        now=now,
        slots=slots,
        base_decision_data={},
        price_ceiling=None,
        horizon_end=horizon_end,
    )
    assert proj.plan is not None
    tomorrow_intervals = [i for i in proj.plan.intervals if i.start >= day_end]
    solar_tomorrow = [
        (i.start.strftime("%H:%M"), i.solar_kwh)
        for i in tomorrow_intervals
        if i.solar_kwh > 0
    ]
    assert len(solar_tomorrow) > 0



