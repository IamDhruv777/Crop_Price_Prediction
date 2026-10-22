"""
Repository layer for the Crop Price Prediction system.

All public functions accept a SQLAlchemy Session as their first argument
so they compose cleanly with the `get_session()` context manager.

Upsert functions use PostgreSQL ``INSERT … ON CONFLICT DO UPDATE`` via
SQLAlchemy's ``insert().on_conflict_do_update()`` dialect helper, with a
graceful fallback to a select-then-insert/update approach for other backends
(e.g. SQLite used in tests).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import (
    Alert,
    DataQuality,
    FeatureSet,
    MandiPrice,
    Market,
    MarketArrival,
    ModelRun,
    Prediction,
    WeatherObservation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_postgres(session: Session) -> bool:
    """Return True if the session is bound to a PostgreSQL engine."""
    url = str(session.bind.url) if session.bind else ""  # type: ignore[union-attr]
    return "postgresql" in url or "psycopg" in url


def _pg_upsert(session: Session, model_cls, conflict_columns: list[str], data: dict) -> object:
    """
    PostgreSQL-specific upsert using INSERT … ON CONFLICT DO UPDATE.

    Returns the persisted ORM instance (refreshed from the DB).
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    table = model_cls.__table__
    stmt = (
        pg_insert(table)
        .values(**data)
        .on_conflict_do_update(
            index_elements=conflict_columns,
            set_={k: v for k, v in data.items() if k not in conflict_columns},
        )
        .returning(table.c.id)
    )
    result = session.execute(stmt)
    row_id = result.scalar_one()
    session.flush()
    return session.get(model_cls, row_id)


def _generic_upsert(session: Session, model_cls, conflict_fields: dict, data: dict) -> object:
    """
    Database-agnostic upsert: SELECT → UPDATE or INSERT.

    ``conflict_fields`` is a dict of {column_name: value} used to find
    an existing row.
    """
    stmt = select(model_cls)
    for field, value in conflict_fields.items():
        stmt = stmt.where(getattr(model_cls, field) == value)
    existing = session.scalars(stmt).first()
    if existing is not None:
        for k, v in data.items():
            if hasattr(existing, k):
                setattr(existing, k, v)
        session.flush()
        return existing
    new_obj = model_cls(**data)
    session.add(new_obj)
    try:
        session.flush()
    except IntegrityError:
        session.rollback()
        # Race condition: another writer inserted concurrently
        existing = session.scalars(stmt).first()
        if existing is None:
            raise
        return existing
    return new_obj


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------

def upsert_market(session: Session, market_data: dict) -> Market:
    """
    Insert or update a Market record keyed on ``market_id``.

    Args:
        session: Active SQLAlchemy session.
        market_data: Dict with Market column values. ``market_id`` is required.

    Returns:
        Persisted Market ORM instance.
    """
    if "market_id" not in market_data:
        raise ValueError("market_data must contain 'market_id'.")

    conflict_key = {"market_id": market_data["market_id"]}

    if _is_postgres(session):
        return _pg_upsert(session, Market, ["market_id"], market_data)  # type: ignore[return-value]
    return _generic_upsert(session, Market, conflict_key, market_data)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# MandiPrice
# ---------------------------------------------------------------------------

def upsert_mandi_price(session: Session, price_data: dict) -> MandiPrice:
    """
    Insert or update a MandiPrice record.

    Conflict key: (market_id, commodity, price_date).

    Args:
        session: Active SQLAlchemy session.
        price_data: Dict with MandiPrice column values.

    Returns:
        Persisted MandiPrice ORM instance.
    """
    required = {"market_id", "commodity", "price_date"}
    _check_required(price_data, required, "price_data")

    conflict_cols = ["market_id", "commodity", "price_date"]
    conflict_key = {c: price_data[c] for c in conflict_cols}

    if _is_postgres(session):
        return _pg_upsert(session, MandiPrice, conflict_cols, price_data)  # type: ignore[return-value]
    return _generic_upsert(session, MandiPrice, conflict_key, price_data)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# WeatherObservation
# ---------------------------------------------------------------------------

def upsert_weather(session: Session, weather_data: dict) -> WeatherObservation:
    """
    Insert or update a WeatherObservation record.

    Conflict key: (market_id, obs_date).

    Args:
        session: Active SQLAlchemy session.
        weather_data: Dict with WeatherObservation column values.

    Returns:
        Persisted WeatherObservation ORM instance.
    """
    required = {"market_id", "obs_date"}
    _check_required(weather_data, required, "weather_data")

    conflict_cols = ["market_id", "obs_date"]
    conflict_key = {c: weather_data[c] for c in conflict_cols}

    if _is_postgres(session):
        return _pg_upsert(session, WeatherObservation, conflict_cols, weather_data)  # type: ignore[return-value]
    return _generic_upsert(session, WeatherObservation, conflict_key, weather_data)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# MarketArrival
# ---------------------------------------------------------------------------

def upsert_arrival(session: Session, arrival_data: dict) -> MarketArrival:
    """
    Insert or update a MarketArrival record.

    Conflict key: (market_id, commodity, arrival_date).

    Args:
        session: Active SQLAlchemy session.
        arrival_data: Dict with MarketArrival column values.

    Returns:
        Persisted MarketArrival ORM instance.
    """
    required = {"market_id", "commodity", "arrival_date"}
    _check_required(arrival_data, required, "arrival_data")

    conflict_cols = ["market_id", "commodity", "arrival_date"]
    conflict_key = {c: arrival_data[c] for c in conflict_cols}

    if _is_postgres(session):
        return _pg_upsert(session, MarketArrival, conflict_cols, arrival_data)  # type: ignore[return-value]
    return _generic_upsert(session, MarketArrival, conflict_key, arrival_data)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Query helpers → DataFrames
# ---------------------------------------------------------------------------

def get_mandi_prices(
    session: Session,
    market_id: int,
    commodity: str,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
) -> pd.DataFrame:
    """
    Retrieve MandiPrice rows for a market/commodity window as a DataFrame.

    Args:
        session: Active SQLAlchemy session.
        market_id: Primary key of the Market row.
        commodity: Commodity name (case-sensitive).
        start_date: Inclusive start of the date range.
        end_date: Inclusive end of the date range.

    Returns:
        DataFrame with columns matching MandiPrice table columns, ordered by
        ``price_date`` ascending. Empty DataFrame if no rows match.
    """
    stmt = (
        select(MandiPrice)
        .where(MandiPrice.market_id == market_id)
        .where(MandiPrice.commodity == commodity)
        .where(MandiPrice.price_date >= _to_dt(start_date))
        .where(MandiPrice.price_date <= _to_dt(end_date))
        .order_by(MandiPrice.price_date.asc())
    )
    rows = session.scalars(stmt).all()
    return _orm_rows_to_df(rows)


def get_weather(
    session: Session,
    market_id: int,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
) -> pd.DataFrame:
    """
    Retrieve WeatherObservation rows for a market/date window as a DataFrame.

    Args:
        session: Active SQLAlchemy session.
        market_id: Primary key of the Market row.
        start_date: Inclusive start of the observation date range.
        end_date: Inclusive end of the observation date range.

    Returns:
        DataFrame ordered by ``obs_date`` ascending.
    """
    stmt = (
        select(WeatherObservation)
        .where(WeatherObservation.market_id == market_id)
        .where(WeatherObservation.obs_date >= _to_dt(start_date))
        .where(WeatherObservation.obs_date <= _to_dt(end_date))
        .order_by(WeatherObservation.obs_date.asc())
    )
    rows = session.scalars(stmt).all()
    return _orm_rows_to_df(rows)


def get_arrivals(
    session: Session,
    market_id: int,
    commodity: str,
    start_date: date | datetime | str,
    end_date: date | datetime | str,
) -> pd.DataFrame:
    """
    Retrieve MarketArrival rows for a market/commodity window as a DataFrame.

    Args:
        session: Active SQLAlchemy session.
        market_id: Primary key of the Market row.
        commodity: Commodity name.
        start_date: Inclusive start of the arrival date range.
        end_date: Inclusive end of the arrival date range.

    Returns:
        DataFrame ordered by ``arrival_date`` ascending.
    """
    stmt = (
        select(MarketArrival)
        .where(MarketArrival.market_id == market_id)
        .where(MarketArrival.commodity == commodity)
        .where(MarketArrival.arrival_date >= _to_dt(start_date))
        .where(MarketArrival.arrival_date <= _to_dt(end_date))
        .order_by(MarketArrival.arrival_date.asc())
    )
    rows = session.scalars(stmt).all()
    return _orm_rows_to_df(rows)


# ---------------------------------------------------------------------------
# Market lookups
# ---------------------------------------------------------------------------

def get_market_by_name(
    session: Session,
    market_name: str,
    state: str,
) -> Optional[Market]:
    """
    Look up a Market by its human-readable name and state.

    The match is case-insensitive.

    Args:
        session: Active SQLAlchemy session.
        market_name: Human-readable market/mandi name.
        state: State name.

    Returns:
        Matching Market instance, or ``None`` if not found.
    """
    stmt = (
        select(Market)
        .where(Market.market_name.ilike(market_name))
        .where(Market.state.ilike(state))
        .limit(1)
    )
    return session.scalars(stmt).first()


# ---------------------------------------------------------------------------
# ModelRun
# ---------------------------------------------------------------------------

def save_model_run(session: Session, run_data: dict) -> ModelRun:
    """
    Persist a new ModelRun record.

    This always creates a new row (a run is immutable once saved). If
    ``is_active`` is True, all prior active runs for the same
    (model_name, crop, horizon) are deactivated first.

    Args:
        session: Active SQLAlchemy session.
        run_data: Dict with ModelRun column values.

    Returns:
        Newly created ModelRun ORM instance.
    """
    if run_data.get("is_active"):
        _deactivate_prior_runs(
            session,
            model_name=run_data.get("model_name"),
            crop=run_data.get("crop"),
            horizon=run_data.get("horizon"),
        )

    model_run = ModelRun(**run_data)
    session.add(model_run)
    session.flush()
    logger.info("Saved ModelRun id=%d model=%s crop=%s", model_run.id, model_run.model_name, model_run.crop)
    return model_run


def get_best_model(
    session: Session,
    crop: str,
    market_id: int,
    horizon: int,
) -> Optional[ModelRun]:
    """
    Retrieve the best active ModelRun for a given crop, market, and horizon.

    "Best" is defined as: is_active=True, matching crop + horizon, with the
    lowest MAE. When ``market_id`` does not uniquely narrow results the
    global best active model is returned.

    Args:
        session: Active SQLAlchemy session.
        crop: Crop name.
        market_id: Market primary key (used to filter by market_scope).
        horizon: Forecast horizon in days.

    Returns:
        Best ModelRun or ``None``.
    """
    stmt = (
        select(ModelRun)
        .where(ModelRun.crop == crop)
        .where(ModelRun.horizon == horizon)
        .where(ModelRun.is_active.is_(True))
        .order_by(ModelRun.mae.asc().nullslast())
        .limit(1)
    )
    return session.scalars(stmt).first()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def save_prediction(session: Session, pred_data: dict) -> Prediction:
    """
    Persist a Prediction record (upsert on unique constraint).

    Conflict key: (market_id, commodity, prediction_date, horizon_days, model_run_id).

    Args:
        session: Active SQLAlchemy session.
        pred_data: Dict with Prediction column values.

    Returns:
        Persisted Prediction ORM instance.
    """
    required = {"market_id", "commodity", "prediction_date", "horizon_days", "predicted_price"}
    _check_required(pred_data, required, "pred_data")

    conflict_cols = ["market_id", "commodity", "prediction_date", "horizon_days", "model_run_id"]

    if _is_postgres(session):
        return _pg_upsert(session, Prediction, conflict_cols, pred_data)  # type: ignore[return-value]

    conflict_key = {c: pred_data.get(c) for c in conflict_cols}
    return _generic_upsert(session, Prediction, conflict_key, pred_data)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------

def save_alert(session: Session, alert_data: dict) -> Alert:
    """
    Persist a new Alert record.

    Alerts are treated as append-only events; no deduplication is applied.

    Args:
        session: Active SQLAlchemy session.
        alert_data: Dict with Alert column values.

    Returns:
        Newly created Alert ORM instance.
    """
    required = {"market_id", "commodity", "alert_date", "alert_type", "description"}
    _check_required(alert_data, required, "alert_data")

    alert = Alert(**alert_data)
    session.add(alert)
    session.flush()
    logger.info(
        "Saved Alert id=%d market_id=%d commodity=%s type=%s",
        alert.id,
        alert.market_id,
        alert.commodity,
        alert.alert_type,
    )
    return alert


# ---------------------------------------------------------------------------
# DataQuality
# ---------------------------------------------------------------------------

def get_data_quality_score(
    session: Session,
    table_name: str,
    market_id: Optional[int],
    commodity: Optional[str],
) -> float:
    """
    Return the most recent quality score for a table/market/commodity triple.

    Args:
        session: Active SQLAlchemy session.
        table_name: Target table (e.g. ``"mandi_prices"``).
        market_id: Optional market primary key filter.
        commodity: Optional commodity name filter.

    Returns:
        Quality score in [0.0, 1.0], or ``1.0`` if no record is found
        (assume perfect quality when no audit has run yet).
    """
    stmt = (
        select(DataQuality.quality_score)
        .where(DataQuality.table_name == table_name)
        .order_by(DataQuality.check_date.desc())
    )
    if market_id is not None:
        stmt = stmt.where(DataQuality.market_id == market_id)
    if commodity is not None:
        stmt = stmt.where(DataQuality.commodity == commodity)

    score = session.scalars(stmt.limit(1)).first()
    return float(score) if score is not None else 1.0


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _check_required(data: dict, required: set[str], data_name: str) -> None:
    missing = required - data.keys()
    if missing:
        raise ValueError(f"{data_name} is missing required fields: {missing}")


def _to_dt(value: date | datetime | str) -> datetime:
    """Coerce a date / date-string to a datetime for ORM comparisons."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    # string
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date string: {value!r}")


def _orm_rows_to_df(rows: list) -> pd.DataFrame:
    """Convert a list of ORM instances to a pandas DataFrame."""
    if not rows:
        return pd.DataFrame()
    records = [
        {c.name: getattr(row, c.name) for c in row.__table__.columns}
        for row in rows
    ]
    return pd.DataFrame(records)


def _deactivate_prior_runs(
    session: Session,
    model_name: Optional[str],
    crop: Optional[str],
    horizon: Optional[int],
) -> None:
    """Set is_active=False on all prior active runs matching the signature."""
    stmt = (
        select(ModelRun)
        .where(ModelRun.is_active.is_(True))
    )
    if model_name:
        stmt = stmt.where(ModelRun.model_name == model_name)
    if crop:
        stmt = stmt.where(ModelRun.crop == crop)
    if horizon is not None:
        stmt = stmt.where(ModelRun.horizon == horizon)

    runs = session.scalars(stmt).all()
    for run in runs:
        run.is_active = False
    if runs:
        logger.info("Deactivated %d prior ModelRun(s).", len(runs))
    session.flush()
