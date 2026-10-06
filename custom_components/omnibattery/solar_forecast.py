"""Read solar forecast sensors with an explicit forecast horizon.

``solar_forecast_sensor`` is the legacy whole-day (``today``) value.  Newer
providers also expose a ``remaining today`` value.  Keeping the distinction in
one place prevents callers from accidentally subtracting production twice.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from .const import (
    CONF_SOLAR_FORECAST_REMAINING_SENSOR,
    CONF_SOLAR_FORECAST_SENSOR,
    T_START_FALLBACK_HOUR,
)

_LOGGER = logging.getLogger(__name__)


ForecastSource = Literal["remaining", "today"]
_FORECAST_EPSILON_KWH = 1e-9


@dataclass(frozen=True)
class SolarForecastPeriod:
    """A dated, energy-valued period supplied by a forecast provider.

    The planner deliberately accepts periods rather than positional arrays.
    A timestamp is the only safe way to align a provider curve across partial
    ranges and daylight-saving transitions.
    """

    start: datetime
    end: datetime
    energy_kwh: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.start, datetime)
            or not isinstance(self.end, datetime)
            or self.start.tzinfo is None
            or self.end.tzinfo is None
        ):
            raise ValueError("solar forecast periods require aware timestamps")
        if self.end.timestamp() <= self.start.timestamp():
            raise ValueError("solar forecast period must have positive duration")
        try:
            energy = float(self.energy_kwh)
        except (TypeError, ValueError) as exc:
            raise ValueError("solar forecast period energy must be numeric") from exc
        if not math.isfinite(energy) or energy < 0.0:
            raise ValueError("solar forecast period energy must be finite and non-negative")
        object.__setattr__(self, "energy_kwh", energy)


@dataclass(frozen=True)
class SolarForecast:
    """A normalized solar forecast and the horizon it represents."""

    kwh: float
    source: ForecastSource
    sensor: str
    periods: tuple[SolarForecastPeriod, ...] = ()
    conversion: str = "none"

    @property
    def remaining_kwh(self) -> float:
        """Normalized future energy consumed by all control decisions."""
        return self.kwh

    @property
    def diagnostic_source(self) -> str:
        """Stable diagnostic label distinguishing the migration paths."""
        return "remaining_sensor" if self.source == "remaining" else "today_legacy"


@dataclass(frozen=True)
class SolarForecastInput:
    """Consumer-facing solar contract with an optional normalized curve."""

    remaining_kwh: float
    source: str
    temporal_shape: tuple[float, ...] | None = None
    periods: tuple[SolarForecastPeriod, ...] | None = None
    original_source: str | None = None
    conversion: str = "none"
    horizon: str = "remaining"

    def __post_init__(self) -> None:
        """Keep the normalized contract finite even with a loose sensor value."""
        try:
            value = float(self.remaining_kwh)
        except (TypeError, ValueError):
            value = 0.0
        object.__setattr__(
            self,
            "remaining_kwh",
            value if math.isfinite(value) and value >= 0.0 else 0.0,
        )
        if self.periods is not None:
            object.__setattr__(self, "periods", tuple(self.periods))

    def normalized_shape(self) -> list[float] | None:
        """Return a shape whose values sum exactly to ``remaining_kwh``."""
        if self.temporal_shape is None:
            return None
        values = []
        for value in self.temporal_shape:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                parsed = 0.0
            values.append(parsed if math.isfinite(parsed) and parsed >= 0.0 else 0.0)
        total = math.fsum(values)
        if total <= 0.0:
            return [0.0] * len(values)
        factor = max(0.0, self.remaining_kwh) / total
        normalized = [value * factor for value in values]
        if normalized:
            normalized[-1] += max(0.0, self.remaining_kwh) - math.fsum(normalized)
        return normalized


def get_configured_solar_forecast_sensor(
    controller: Any,
    source: ForecastSource,
) -> str | None:
    """Return the effective configured entity for a forecast horizon.

    The controller keeps a runtime copy of these values, but an options update
    can briefly leave that copy behind the config entry.  Read the persisted
    entry first so horizon selection cannot silently fall back to the daily
    legacy path while the remaining-today sensor is configured.
    """
    if source == "remaining":
        key = CONF_SOLAR_FORECAST_REMAINING_SENSOR
        attribute = "solar_forecast_remaining_sensor"
    else:
        key = CONF_SOLAR_FORECAST_SENSOR
        attribute = "solar_forecast_sensor"

    config_entry = getattr(controller, "config_entry", None)
    if config_entry is not None:
        has_persisted_config = False
        for config in (
            getattr(config_entry, "data", None),
            getattr(config_entry, "options", None),
        ):
            if config is None:
                continue
            has_persisted_config = True
            if key not in config:
                continue
            value = config.get(key)
            return value if value else None
        # A real config entry is authoritative even when the key is absent.
        # This prevents a stale runtime attribute from resurrecting a sensor
        # that was cleared in the options flow.
        if has_persisted_config:
            return None

    # Small unit-test doubles and lightweight consumers may not carry a config
    # entry; preserve their existing runtime-only contract.
    value = getattr(controller, attribute, None)
    return value if value else None


def normalize_solar_forecast_config(data: dict[str, Any]) -> dict[str, Any]:
    """Keep at most one persisted solar forecast horizon.

    A configured remaining-today sensor supersedes ``today``. Empty values are
    removed rather than stored as config keys, which lets Repairs distinguish a
    real legacy configuration from a cleared field.
    """
    normalized = dict(data)
    remaining = normalized.get(CONF_SOLAR_FORECAST_REMAINING_SENSOR)
    if remaining:
        normalized.pop(CONF_SOLAR_FORECAST_SENSOR, None)
    else:
        normalized.pop(CONF_SOLAR_FORECAST_REMAINING_SENSOR, None)
        if not normalized.get(CONF_SOLAR_FORECAST_SENSOR):
            normalized.pop(CONF_SOLAR_FORECAST_SENSOR, None)
    return normalized


def _state_kwh(state: Any) -> float | None:
    """Return a finite non-negative sensor state in kWh, converting Wh."""
    if state is None or getattr(state, "state", None) in ("unknown", "unavailable"):
        return None
    try:
        value = float(state.state)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    unit = str(getattr(state, "attributes", {}).get("unit_of_measurement", "kWh")).strip().lower()
    if unit == "wh":
        value /= 1000.0
    elif unit != "kwh":
        return None
    return value


def _parse_period_timestamp(value: Any) -> datetime | None:
    """Parse one explicit provider timestamp, rejecting naive values."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else None
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _period_energy_kwh(raw: dict[str, Any]) -> float | None:
    """Read an explicitly unit-labelled period energy value."""
    if "energy_kwh" in raw:
        candidate = raw["energy_kwh"]
    elif "energy_wh" in raw:
        candidate = raw["energy_wh"]
        try:
            return float(candidate) / 1000.0
        except (TypeError, ValueError):
            return None
    elif "energy" in raw:
        candidate = raw["energy"]
        unit = str(raw.get("unit", raw.get("energy_unit", ""))).strip().lower()
        if unit == "wh":
            try:
                return float(candidate) / 1000.0
            except (TypeError, ValueError):
                return None
        if unit != "kwh":
            return None
    else:
        return None
    try:
        value = float(candidate)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


def _extract_forecast_periods(state: Any) -> tuple[SolarForecastPeriod, ...]:
    """Adapt the small set of explicit period schemas supported by the contract.

    A bare list of numbers is intentionally ignored.  Providers can expose
    their own adapter later, but positional values cannot be made DST-safe.
    """
    attributes = getattr(state, "attributes", {}) or {}
    raw_periods = None
    for key in (
        "solar_forecast_periods",
        "forecast_periods",
        "periods",
        "detailedForecast",
        "detailedHourly",
        "forecast",
    ):
        if key in attributes:
            raw_periods = attributes[key]
            break
    if not isinstance(raw_periods, (list, tuple)):
        return ()

    parsed: list[SolarForecastPeriod] = []
    for raw in raw_periods:
        if not isinstance(raw, dict):
            return ()
        start = _parse_period_timestamp(
            raw.get("start", raw.get("start_time", raw.get("period_start", raw.get("datetime", raw.get("date")))))
        )
        end = _parse_period_timestamp(
            raw.get("end", raw.get("end_time", raw.get("period_end")))
        )
        energy = _period_energy_kwh(raw)
        if energy is None and "pv_estimate" in raw:
            try:
                kw = float(raw["pv_estimate"])
                duration_hours = 0.5
                if start and end and end > start:
                    duration_hours = (end - start).total_seconds() / 3600.0
                energy = max(0.0, kw * duration_hours)
            except (TypeError, ValueError):
                energy = None
        elif energy is None and "watts" in raw:
            try:
                watts = float(raw["watts"])
                duration_hours = 0.25
                if start and end and end > start:
                    duration_hours = (end - start).total_seconds() / 3600.0
                energy = max(0.0, watts * duration_hours / 1000.0)
            except (TypeError, ValueError):
                energy = None
        if start is None or energy is None:
            return ()
        if end is None:
            end = start + timedelta(minutes=15 if "watts" in raw else 30)
        try:
            parsed.append(SolarForecastPeriod(start, end, energy))
        except ValueError:
            return ()
    return tuple(parsed)


def solar_periods_from_wh_hours(
    wh_hours: Mapping[str, Any] | None,
    *,
    default_timezone: Any = None,
) -> tuple[SolarForecastPeriod, ...]:
    """Convert an Energy dashboard wh_hours dictionary into SolarForecastPeriod instances.

    Keys are ISO timestamps marking period starts, and values are energy in Wh.
    """
    if not wh_hours or not isinstance(wh_hours, Mapping):
        return ()

    entries: list[tuple[datetime, float]] = []
    for timestamp, wh in wh_hours.items():
        parsed_dt = _parse_period_timestamp(timestamp)
        if parsed_dt is None:
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    if dt.tzinfo is None and default_timezone is not None:
                        dt = dt.replace(tzinfo=default_timezone)
                    if dt.tzinfo is not None:
                        parsed_dt = dt
                except ValueError:
                    continue
            elif isinstance(timestamp, datetime):
                dt = timestamp
                if dt.tzinfo is None and default_timezone is not None:
                    dt = dt.replace(tzinfo=default_timezone)
                if dt.tzinfo is not None:
                    parsed_dt = dt
        if parsed_dt is None:
            continue
        try:
            val_wh = float(wh)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val_wh) and val_wh >= 0.0:
            entries.append((parsed_dt, val_wh / 1000.0))

    if not entries:
        return ()

    entries.sort(key=lambda x: x[0])
    diffs = [
        (b - a) for (a, _), (b, _) in zip(entries, entries[1:])
        if b > a
    ]
    default_period = min(diffs, default=timedelta(hours=1))
    if default_period <= timedelta(0):
        default_period = timedelta(hours=1)

    periods: list[SolarForecastPeriod] = []
    for idx, (start_dt, kwh) in enumerate(entries):
        if (
            idx + 1 < len(entries)
            and entries[idx + 1][0] > start_dt
            and (entries[idx + 1][0] - start_dt) <= default_period * 2
        ):
            end_dt = entries[idx + 1][0]
        else:
            end_dt = start_dt + default_period
        try:
            periods.append(SolarForecastPeriod(start_dt, end_dt, kwh))
        except ValueError:
            continue

    return tuple(periods)


def extract_periods_from_hass(
    hass: Any,
    controller: Any,
    sensor: str | None = None,
) -> tuple[SolarForecastPeriod, ...]:
    """Synchronously extract multi-day forecast periods directly from Home Assistant memory.

    Directly inspects integration coordinators in hass.data (Helios, Solcast,
    Forecast.Solar) and companion sensor attributes without needing an async call.
    """
    if hass is None:
        return ()

    tz = solar_forecast_local_timezone(hass, controller)

    # 1. Check Helios Forecast coordinator directly in hass.data
    helios_data = getattr(hass, "data", {}).get("helios_forecast", {})
    if isinstance(helios_data, dict):
        wh_hours: dict[str, float] = {}
        for coord in helios_data.values():
            data_obj = getattr(coord, "data", None)
            summary = getattr(data_obj, "summary", None)
            wh = getattr(summary, "wh_hours", None)
            if isinstance(wh, dict) and wh:
                for ts, val in wh.items():
                    try:
                        fval = float(val)
                        if math.isfinite(fval) and fval >= 0.0:
                            wh_hours[str(ts)] = wh_hours.get(str(ts), 0.0) + fval
                    except (TypeError, ValueError):
                        continue
            points = getattr(data_obj, "points", None)
            if isinstance(points, (list, tuple)) and not wh_hours:
                for p in points:
                    pt = getattr(p, "t", None)
                    pw = getattr(p, "pv_w", None)
                    if isinstance(pt, datetime) and pw is not None:
                        try:
                            wval = float(pw)
                            if math.isfinite(wval) and wval >= 0.0:
                                wh_hours[pt.isoformat()] = wval * 0.25
                        except (TypeError, ValueError):
                            continue
        if wh_hours:
            periods = solar_periods_from_wh_hours(wh_hours, default_timezone=tz)
            if periods:
                return periods

    # 2. Check Forecast.Solar coordinator in config_entries
    if hasattr(hass, "config_entries"):
        wh_hours = {}
        for entry in getattr(hass.config_entries, "async_entries", lambda: ())():
            if getattr(entry, "domain", None) == "forecast_solar":
                coord = getattr(entry, "runtime_data", None)
                data_obj = getattr(coord, "data", None)
                wh_period = getattr(data_obj, "wh_period", None)
                if isinstance(wh_period, dict) and wh_period:
                    for ts, val in wh_period.items():
                        try:
                            fval = float(val)
                            ts_key = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                            if math.isfinite(fval) and fval >= 0.0:
                                wh_hours[ts_key] = wh_hours.get(ts_key, 0.0) + fval
                        except (TypeError, ValueError):
                            continue
        if wh_hours:
            periods = solar_periods_from_wh_hours(wh_hours, default_timezone=tz)
            if periods:
                return periods

    # 3. Check Solcast coordinator in hass.data
    solcast_data = getattr(hass, "data", {}).get("solcast_solar", {})
    if isinstance(solcast_data, dict):
        wh_hours = {}
        for item in solcast_data.values():
            wh = getattr(item, "wh_hours", None) or getattr(getattr(item, "data", None), "wh_hours", None)
            if isinstance(wh, dict) and wh:
                for ts, val in wh.items():
                    try:
                        fval = float(val)
                        if math.isfinite(fval) and fval >= 0.0:
                            wh_hours[str(ts)] = wh_hours.get(str(ts), 0.0) + fval
                    except (TypeError, ValueError):
                        continue
        if wh_hours:
            periods = solar_periods_from_wh_hours(wh_hours, default_timezone=tz)
            if periods:
                return periods

    # 4. Check companion sensor with forecast attributes (e.g. sensor.helios_forecast_power_now)
    if sensor and hasattr(hass, "states"):
        for pat in ("energy_today_remaining", "energy_today", "today_remaining", "remaining"):
            if pat in sensor:
                companion_id = sensor.replace(pat, "power_now")
                companion_state = hass.states.get(companion_id)
                if companion_state is not None:
                    periods = _extract_forecast_periods(companion_state)
                    if periods:
                        return periods

    # 5. Check companion day_2 sensor for tomorrow's total (e.g. sensor.helios_forecast_energy_day_2)
    if sensor and hasattr(hass, "states"):
        for pat in ("energy_today_remaining", "energy_today", "today_remaining"):
            if pat in sensor:
                companion_id = sensor.replace(pat, "energy_day_2")
                companion_state = hass.states.get(companion_id)
                if companion_state is not None:
                    tomorrow_kwh = _state_kwh(companion_state)
                    if tomorrow_kwh is not None and tomorrow_kwh > 0.0:
                        now_dt = datetime.now(tz)
                        tmrw_date = now_dt.date() + timedelta(days=1)
                        from math import pi, sin

                        total_sin = sum(sin(pi * (h - 7 + 0.5) / 12.0) for h in range(7, 19))
                        wh_hours = {}
                        if total_sin > 0:
                            for h in range(7, 19):
                                hour_sin = sin(pi * (h - 7 + 0.5) / 12.0)
                                hour_kwh = tomorrow_kwh * (hour_sin / total_sin)
                                hour_dt = datetime.combine(tmrw_date, time(h, 0), tzinfo=tz)
                                wh_hours[hour_dt.isoformat()] = hour_kwh * 1000.0
                            periods = solar_periods_from_wh_hours(wh_hours, default_timezone=tz)
                            if periods:
                                return periods

    return ()


async def async_fetch_energy_platform_solar_forecast(
    hass: Any,
    controller: Any,
) -> dict[str, Any] | None:
    """Query Home Assistant's energy platform for solar forecast data."""
    if hass is None:
        return None

    sensor = (
        get_configured_solar_forecast_sensor(controller, "remaining")
        or get_configured_solar_forecast_sensor(controller, "today")
    )

    combined_wh_hours: dict[str, float] = {}

    # Method 0: Check in-memory integration data first (instant, synchronous)
    mem_periods = extract_periods_from_hass(hass, controller, sensor)
    if mem_periods:
        for p in mem_periods:
            combined_wh_hours[p.start.isoformat()] = p.energy_kwh * 1000.0
        return {"wh_hours": combined_wh_hours}

    # Method 1: Check the configured sensor's config entry directly
    if sensor:
        try:
            from homeassistant.helpers import entity_registry as er
            from homeassistant.loader import async_get_integration

            ent_reg = er.async_get(hass) if hasattr(er, "async_get") else None
            entry = ent_reg.async_get(sensor) if ent_reg else None
            config_entry_id = entry.config_entry_id if entry is not None else None
            config_entry = (
                hass.config_entries.async_get_entry(config_entry_id)
                if config_entry_id and hasattr(hass, "config_entries")
                else None
            )
            if config_entry is not None:
                try:
                    integration = await async_get_integration(hass, config_entry.domain)
                    platform = await integration.async_get_platform("energy")
                    if hasattr(platform, "async_get_solar_forecast"):
                        data = await platform.async_get_solar_forecast(
                            hass, config_entry.entry_id
                        )
                        if isinstance(data, dict) and data.get("wh_hours"):
                            for ts, wh in data["wh_hours"].items():
                                try:
                                    val = float(wh)
                                    if math.isfinite(val) and val >= 0.0:
                                        combined_wh_hours[str(ts)] = combined_wh_hours.get(str(ts), 0.0) + val
                                except (TypeError, ValueError):
                                    continue
                except Exception as plat_err:  # noqa: BLE001
                    _LOGGER.debug("Direct energy platform check for %s: %s", config_entry.domain, plat_err)
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("Direct sensor energy lookup failed: %s", err)

    if combined_wh_hours:
        return {"wh_hours": combined_wh_hours}

    # Method 2: Check Home Assistant energy preferences (Energy Dashboard settings)
    if hasattr(hass, "config_entries"):
        try:
            from homeassistant.components.energy.data import async_get_manager
            from homeassistant.loader import async_get_integration

            manager = await async_get_manager(hass)
            if manager and getattr(manager, "data", None):
                solar_sources = manager.data.get("energy_sources", []) or []
                for src in solar_sources:
                    if isinstance(src, dict) and src.get("type") == "solar":
                        for cfg_id in src.get("config_entry_solar_forecast", []) or []:
                            cfg = hass.config_entries.async_get_entry(cfg_id)
                            if cfg:
                                try:
                                    integ = await async_get_integration(hass, cfg.domain)
                                    plat = await integ.async_get_platform("energy")
                                    if hasattr(plat, "async_get_solar_forecast"):
                                        data = await plat.async_get_solar_forecast(
                                            hass, cfg.entry_id
                                        )
                                        if isinstance(data, dict) and data.get("wh_hours"):
                                            for ts, wh in data["wh_hours"].items():
                                                try:
                                                    val = float(wh)
                                                    if math.isfinite(val) and val >= 0.0:
                                                        combined_wh_hours[str(ts)] = combined_wh_hours.get(str(ts), 0.0) + val
                                                except (TypeError, ValueError):
                                                    continue
                                except Exception as err:  # noqa: BLE001
                                    _LOGGER.debug("Energy dashboard source forecast failed for %s: %s", cfg_id, err)
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("Energy dashboard manager check failed: %s", err)

    if combined_wh_hours:
        return {"wh_hours": combined_wh_hours}

    # Method 3: Try Home Assistant energy websocket platforms helper if available
    try:
        from homeassistant.components.energy.websocket_api import async_get_energy_platforms

        forecast_platforms = await async_get_energy_platforms(hass)
        if forecast_platforms:
            for entry in getattr(hass.config_entries, "async_entries", lambda: ())():
                if entry.domain in forecast_platforms:
                    try:
                        data = await forecast_platforms[entry.domain](hass, entry.entry_id)
                        if isinstance(data, dict) and data.get("wh_hours"):
                            for ts, wh in data["wh_hours"].items():
                                try:
                                    val = float(wh)
                                    if math.isfinite(val) and val >= 0.0:
                                        combined_wh_hours[str(ts)] = combined_wh_hours.get(str(ts), 0.0) + val
                                except (TypeError, ValueError):
                                    continue
                    except Exception as err:  # noqa: BLE001
                        _LOGGER.debug("Platform %s solar forecast failed: %s", entry.domain, err)
    except Exception as err:  # noqa: BLE001
        _LOGGER.debug("Energy platforms helper lookup failed: %s", err)

    if combined_wh_hours:
        return {"wh_hours": combined_wh_hours}

    # Method 4: Check for a tomorrow solar forecast sensor in Home Assistant
    if sensor and hasattr(hass, "states"):
        for pattern_from, pattern_to in (("today", "tomorrow"), ("Today", "Tomorrow")):
            if pattern_from in sensor:
                tomorrow_entity_id = sensor.replace(pattern_from, pattern_to)
                tomorrow_state = hass.states.get(tomorrow_entity_id)
                if tomorrow_state is not None:
                    tomorrow_periods = _extract_forecast_periods(tomorrow_state)
                    if tomorrow_periods:
                        for p in tomorrow_periods:
                            combined_wh_hours[p.start.isoformat()] = p.energy_kwh * 1000.0
                        return {"wh_hours": combined_wh_hours}
                    tomorrow_kwh = _state_kwh(tomorrow_state)
                    if tomorrow_kwh is not None and tomorrow_kwh > 0.0:
                        tz = solar_forecast_local_timezone(hass, controller)
                        now_dt = datetime.now(tz)
                        tmrw_date = now_dt.date() + timedelta(days=1)
                        from math import pi, sin

                        total_sin = sum(sin(pi * (h - 7 + 0.5) / 12.0) for h in range(7, 19))
                        if total_sin > 0:
                            for h in range(7, 19):
                                hour_sin = sin(pi * (h - 7 + 0.5) / 12.0)
                                hour_kwh = tomorrow_kwh * (hour_sin / total_sin)
                                hour_dt = datetime.combine(tmrw_date, time(h, 0), tzinfo=tz)
                                combined_wh_hours[hour_dt.isoformat()] = hour_kwh * 1000.0
                            return {"wh_hours": combined_wh_hours}

    return None


def read_solar_forecast_kwh(
    hass: Any,
    controller: Any,
    *,
    update_controller: bool = True,
) -> SolarForecast | None:
    """Read the preferred forecast: remaining first, legacy today second.

    An unavailable remaining sensor deliberately falls back to the configured
    legacy sensor during the migration.  A valid remaining value is never
    transformed; consumers can rely on it already being the future horizon.
    """
    candidates = (
        (
            "remaining",
            get_configured_solar_forecast_sensor(controller, "remaining"),
        ),
        ("today", get_configured_solar_forecast_sensor(controller, "today")),
    )
    for source, sensor in candidates:
        if not sensor:
            continue
        state = hass.states.get(sensor)
        value = _state_kwh(state)
        if value is not None:
            periods = _extract_forecast_periods(state)
            if not periods:
                cached_periods = getattr(controller, "_energy_solar_periods", ()) or ()
                if cached_periods:
                    periods = tuple(cached_periods)
            if not periods:
                extracted = extract_periods_from_hass(hass, controller, sensor)
                if extracted:
                    periods = extracted
                    if hasattr(controller, "_energy_solar_periods"):
                        controller._energy_solar_periods = extracted
            forecast = SolarForecast(
                value,
                source,
                sensor,
                periods=periods,
                conversion="none",
            )
            # Most control callers retain the source for diagnostics. Dashboard
            # projections use the same adapter read-only, so they can refresh
            # the live value without changing controller-owned runtime state.
            if update_controller:
                controller.solar_forecast_source = source
                controller.solar_forecast_diagnostic_source = forecast.diagnostic_source
                controller.solar_forecast_periods = forecast.periods
            return forecast
    if update_controller:
        controller.solar_forecast_source = None
        controller.solar_forecast_diagnostic_source = None
        controller.solar_forecast_periods = ()
    return None


def _controller_local_date(controller: Any) -> date:
    """Return today's local date without requiring a Home Assistant object."""
    profile = getattr(getattr(controller, "_consumption_tracker", None), "solar_profile", None)
    if profile is not None:
        today = getattr(profile, "_today", None)
        if callable(today):
            try:
                value = today()
                if isinstance(value, datetime):
                    return value.date()
                if isinstance(value, date):
                    return value
            except Exception:  # noqa: BLE001
                pass
    return datetime.now().date()


def solar_forecast_local_timezone(
    hass: Any,
    controller: Any,
    now: datetime | None = None,
):
    """Return the timezone used to interpret local forecast horizons."""
    profile = getattr(
        getattr(controller, "_consumption_tracker", None),
        "solar_profile",
        None,
    )
    timezone = getattr(profile, "_timezone", None)
    if callable(timezone):
        try:
            value = timezone()
            if value is not None:
                return value
        except Exception:  # noqa: BLE001 - timezone fallback must remain safe
            pass

    configured = getattr(getattr(hass, "config", None), "time_zone", None)
    if configured:
        try:
            return ZoneInfo(str(configured))
        except (KeyError, ValueError):
            pass
    if isinstance(now, datetime) and now.tzinfo is not None:
        return now.tzinfo
    return datetime.now().astimezone().tzinfo


def solar_forecast_period_energy_between(
    periods: tuple[SolarForecastPeriod, ...] | list[SolarForecastPeriod] | None,
    start: datetime,
    end: datetime,
    *,
    timezone: Any = None,
) -> float:
    """Return period energy overlapping one explicit horizon.

    Period energy is prorated by absolute-time overlap. Naive boundaries are
    local wall-clock values and therefore require the caller's local timezone.
    """
    if timezone is None:
        timezone = start.tzinfo or end.tzinfo or datetime.now().astimezone().tzinfo
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone)
    start_ts = start.timestamp()
    end_ts = end.timestamp()
    if end_ts <= start_ts:
        return 0.0

    energy = 0.0
    for period in periods or ():
        period_start = period.start.timestamp()
        period_end = period.end.timestamp()
        overlap = max(0.0, min(end_ts, period_end) - max(start_ts, period_start))
        if overlap > 0.0:
            energy += period.energy_kwh * overlap / (period_end - period_start)
    return max(0.0, energy)


def _remaining_period_energy_today(
    hass: Any,
    controller: Any,
    periods: tuple[SolarForecastPeriod, ...],
    now: datetime | float | None,
) -> float:
    """Return dated provider energy still expected before local midnight."""
    timezone = solar_forecast_local_timezone(
        hass,
        controller,
        now if isinstance(now, datetime) else None,
    )
    if isinstance(now, datetime):
        local_now = (
            now.replace(tzinfo=timezone)
            if now.tzinfo is None
            else now.astimezone(timezone)
        )
    else:
        midnight = datetime.combine(
            _controller_local_date(controller),
            time.min,
            tzinfo=timezone,
        )
        local_now = midnight + timedelta(hours=_current_hour(controller, now))
    day_end = datetime.combine(
        local_now.date() + timedelta(days=1),
        time.min,
        tzinfo=timezone,
    )
    return solar_forecast_period_energy_between(
        periods,
        local_now,
        day_end,
        timezone=timezone,
    )


def _current_hour(controller: Any, now: datetime | float | None) -> float:
    if isinstance(now, datetime):
        return now.hour + now.minute / 60.0 + now.second / 3600.0
    if now is not None:
        try:
            value = float(now)
            if math.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
    try:
        value = getattr(controller, "_solar_forecast_now_hour")
        value = float(value)
        if math.isfinite(value):
            return value
    except (AttributeError, TypeError, ValueError):
        pass
    return datetime.now().hour + datetime.now().minute / 60.0


def read_remaining_solar_kwh(
    hass: Any,
    controller: Any,
    *,
    now: datetime | float | None = None,
    update_controller: bool = True,
) -> SolarForecastInput:
    """Return one normalized ``remaining`` budget for every caller.

    ``remaining`` sensors are passed through untouched.  A legacy ``today``
    sensor is converted once using the best reliable evidence available, and
    the conversion is carried on the result so downstream code cannot infer
    the horizon a second time.
    """
    forecast = read_solar_forecast_kwh(
        hass,
        controller,
        update_controller=update_controller,
    )
    if forecast is None:
        if update_controller:
            controller.solar_forecast_source = "fallback"
            controller.solar_forecast_diagnostic_source = "fallback"
        return SolarForecastInput(
            0.0,
            "fallback",
            periods=None,
            original_source=None,
            conversion="unsafe_zero",
        )

    # Some providers roll the scalar ``remaining today`` state a few minutes
    # after midnight while their explicitly dated periods already contain the
    # new day's production. A numeric zero is normally valid, but it must not
    # erase positive, timestamped evidence inside today's remaining horizon.
    period_remaining = _remaining_period_energy_today(
        hass,
        controller,
        forecast.periods,
        now,
    )
    if forecast.kwh <= _FORECAST_EPSILON_KWH and period_remaining > _FORECAST_EPSILON_KWH:
        if update_controller:
            controller.solar_forecast_conversion = "dated_periods_zero_scalar"
            controller.solar_forecast_diagnostic_source = forecast.diagnostic_source
        return SolarForecastInput(
            period_remaining,
            forecast.diagnostic_source,
            periods=forecast.periods,
            original_source=(
                "remaining" if forecast.source == "remaining" else "today_legacy"
            ),
            conversion="dated_periods_zero_scalar",
        )

    if forecast.source == "remaining":
        result = SolarForecastInput(
            forecast.kwh,
            forecast.diagnostic_source,
            periods=forecast.periods or None,
            original_source="remaining",
            conversion="none",
        )
        if update_controller:
            controller.solar_forecast_conversion = "none"
        return result

    # A whole-day scalar has already assigned part of its energy to elapsed
    # hours.  Subtracting actual production makes any cloudy/optimistic morning
    # reappear as fictional energy at sunset. Prefer dated provider periods;
    # otherwise map the total through the remaining part of the solar curve.
    if forecast.periods:
        remaining = period_remaining
        conversion = "dated_periods"
    else:
        tracker = getattr(controller, "_consumption_tracker", None)
        t_start = getattr(controller, "_solar_t_start", None)
        current_hour = _current_hour(controller, now)
        remaining = None
        conversion = ""
        if t_start is not None and tracker is not None:
            try:
                t_end = tracker.estimate_t_end()
                fraction_done = tracker.get_solar_fraction_done(
                    current_hour, float(t_start), float(t_end)
                )
                remaining = forecast.kwh * max(0.0, 1.0 - float(fraction_done))
                conversion = "temporal_fraction"
            except (AttributeError, TypeError, ValueError, ZeroDivisionError):
                remaining = None
        if remaining is None:
            if current_hour < T_START_FALLBACK_HOUR:
                remaining = forecast.kwh
                conversion = "pre_solar"
            else:
                remaining = 0.0
                conversion = "unsafe_zero"

    if update_controller:
        controller.solar_forecast_conversion = conversion
        controller.solar_forecast_diagnostic_source = "today_legacy"
    return SolarForecastInput(
        remaining,
        "today_legacy",
        periods=forecast.periods or None,
        original_source="today_legacy",
        conversion=conversion,
    )
