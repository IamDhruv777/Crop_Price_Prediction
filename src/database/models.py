"""
SQLAlchemy ORM models for the Crop Price Prediction system.

All tables use SQLAlchemy 2.x declarative_base style with proper indexes,
composite unique constraints, and __repr__ methods.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Common declarative base for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------

class Market(Base):
    """Physical agricultural market (mandi)."""

    __tablename__ = "markets"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    market_id: str = Column(String(64), nullable=False, unique=True, index=True)
    market_name: str = Column(String(256), nullable=False)
    district: str = Column(String(128), nullable=False)
    state: str = Column(String(128), nullable=False)
    latitude: Optional[float] = Column(Float, nullable=True)
    longitude: Optional[float] = Column(Float, nullable=True)
    region_id: Optional[str] = Column(String(64), nullable=True)
    is_active: bool = Column(Boolean, nullable=False, default=True)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: datetime = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    # Relationships
    prices = relationship("MandiPrice", back_populates="market", lazy="dynamic")
    arrivals = relationship("MarketArrival", back_populates="market", lazy="dynamic")
    weather_obs = relationship("WeatherObservation", back_populates="market", lazy="dynamic")
    features = relationship("FeatureSet", back_populates="market", lazy="dynamic")
    predictions = relationship("Prediction", back_populates="market", lazy="dynamic")
    alerts = relationship("Alert", back_populates="market", lazy="dynamic")
    data_quality = relationship("DataQuality", back_populates="market", lazy="dynamic")

    __table_args__ = (
        Index("ix_markets_state", "state"),
        Index("ix_markets_district", "district"),
        Index("ix_markets_state_district", "state", "district"),
    )

    def __repr__(self) -> str:
        return (
            f"<Market id={self.id} market_id={self.market_id!r} "
            f"name={self.market_name!r} district={self.district!r} state={self.state!r}>"
        )


# ---------------------------------------------------------------------------
# MandiPrice
# ---------------------------------------------------------------------------

class MandiPrice(Base):
    """Daily commodity price records from mandi markets."""

    __tablename__ = "mandi_prices"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    market_id: int = Column(Integer, ForeignKey("markets.id", ondelete="CASCADE"), nullable=False)
    commodity: str = Column(String(128), nullable=False)
    variety: Optional[str] = Column(String(128), nullable=True)
    grade: Optional[str] = Column(String(64), nullable=True)
    price_date: datetime = Column(DateTime(timezone=False), nullable=False)
    min_price: Numeric = Column(Numeric(12, 2), nullable=False)
    max_price: Numeric = Column(Numeric(12, 2), nullable=False)
    modal_price: Numeric = Column(Numeric(12, 2), nullable=False)
    arrival_quantity: Optional[float] = Column(Float, nullable=True)
    quality_flag: str = Column(String(16), nullable=False, default="OK")
    price_anomaly: bool = Column(Boolean, nullable=False, default=False)
    validation_status: str = Column(String(32), nullable=False, default="pending")
    data_source: str = Column(String(64), nullable=False)
    download_timestamp: datetime = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_at: datetime = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    market = relationship("Market", back_populates="prices")

    __table_args__ = (
        UniqueConstraint("market_id", "commodity", "price_date", name="uq_mandi_prices_market_commodity_date"),
        Index("ix_mandi_prices_market_id", "market_id"),
        Index("ix_mandi_prices_commodity", "commodity"),
        Index("ix_mandi_prices_price_date", "price_date"),
        Index("ix_mandi_prices_market_commodity", "market_id", "commodity"),
        Index("ix_mandi_prices_validation", "validation_status"),
    )

    def __repr__(self) -> str:
        return (
            f"<MandiPrice id={self.id} market_id={self.market_id} "
            f"commodity={self.commodity!r} date={self.price_date} modal={self.modal_price}>"
        )


# ---------------------------------------------------------------------------
# MarketArrival
# ---------------------------------------------------------------------------

class MarketArrival(Base):
    """Daily commodity arrival quantities at mandi markets."""

    __tablename__ = "market_arrivals"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    market_id: int = Column(Integer, ForeignKey("markets.id", ondelete="CASCADE"), nullable=False)
    commodity: str = Column(String(128), nullable=False)
    arrival_date: datetime = Column(DateTime(timezone=False), nullable=False)
    arrival_quantity: float = Column(Float, nullable=False)
    unit: str = Column(String(32), nullable=False, default="tonnes")
    data_source: str = Column(String(64), nullable=False)
    is_imputed: bool = Column(Boolean, nullable=False, default=False)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    market = relationship("Market", back_populates="arrivals")

    __table_args__ = (
        UniqueConstraint("market_id", "commodity", "arrival_date", name="uq_market_arrivals_market_commodity_date"),
        Index("ix_market_arrivals_market_id", "market_id"),
        Index("ix_market_arrivals_commodity", "commodity"),
        Index("ix_market_arrivals_arrival_date", "arrival_date"),
        Index("ix_market_arrivals_market_commodity", "market_id", "commodity"),
    )

    def __repr__(self) -> str:
        return (
            f"<MarketArrival id={self.id} market_id={self.market_id} "
            f"commodity={self.commodity!r} date={self.arrival_date} qty={self.arrival_quantity} {self.unit}>"
        )


# ---------------------------------------------------------------------------
# WeatherObservation
# ---------------------------------------------------------------------------

class WeatherObservation(Base):
    """Daily weather observations associated with a market location."""

    __tablename__ = "weather"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    market_id: int = Column(Integer, ForeignKey("markets.id", ondelete="CASCADE"), nullable=False)
    obs_date: datetime = Column(DateTime(timezone=False), nullable=False)
    latitude: float = Column(Float, nullable=False)
    longitude: float = Column(Float, nullable=False)
    temperature: Optional[float] = Column(Float, nullable=True)        # mean °C
    temperature_max: Optional[float] = Column(Float, nullable=True)    # max °C
    temperature_min: Optional[float] = Column(Float, nullable=True)    # min °C
    rainfall: Optional[float] = Column(Float, nullable=True)           # mm
    humidity: Optional[float] = Column(Float, nullable=True)           # %
    wind_speed: Optional[float] = Column(Float, nullable=True)         # m/s
    data_source: str = Column(String(64), nullable=False)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    market = relationship("Market", back_populates="weather_obs")

    __table_args__ = (
        UniqueConstraint("market_id", "obs_date", name="uq_weather_market_date"),
        Index("ix_weather_market_id", "market_id"),
        Index("ix_weather_obs_date", "obs_date"),
        Index("ix_weather_market_date", "market_id", "obs_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<WeatherObservation id={self.id} market_id={self.market_id} "
            f"date={self.obs_date} temp={self.temperature} rain={self.rainfall}>"
        )


# ---------------------------------------------------------------------------
# CropProduction
# ---------------------------------------------------------------------------

class CropProduction(Base):
    """Annual crop production statistics at state/district level."""

    __tablename__ = "crop_production"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    state: str = Column(String(128), nullable=False)
    district: str = Column(String(128), nullable=False)
    crop: str = Column(String(128), nullable=False)
    season: str = Column(String(64), nullable=False)   # kharif / rabi / zaid
    year: int = Column(Integer, nullable=False)
    area_ha: Optional[float] = Column(Float, nullable=True)
    production_tonnes: Optional[float] = Column(Float, nullable=True)
    yield_kg_per_ha: Optional[float] = Column(Float, nullable=True)
    data_source: str = Column(String(64), nullable=False)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("state", "district", "crop", "season", "year", name="uq_crop_production"),
        Index("ix_crop_production_state", "state"),
        Index("ix_crop_production_district", "district"),
        Index("ix_crop_production_crop", "crop"),
        Index("ix_crop_production_year", "year"),
        Index("ix_crop_production_state_crop_year", "state", "crop", "year"),
    )

    def __repr__(self) -> str:
        return (
            f"<CropProduction id={self.id} state={self.state!r} district={self.district!r} "
            f"crop={self.crop!r} season={self.season!r} year={self.year} "
            f"prod={self.production_tonnes} t>"
        )


# ---------------------------------------------------------------------------
# FeatureSet
# ---------------------------------------------------------------------------

class FeatureSet(Base):
    """Computed ML feature vectors stored as JSON, keyed by market+commodity+date."""

    __tablename__ = "features"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    market_id: int = Column(Integer, ForeignKey("markets.id", ondelete="CASCADE"), nullable=False)
    commodity: str = Column(String(128), nullable=False)
    feature_date: datetime = Column(DateTime(timezone=False), nullable=False)
    features_json: dict = Column(JSON, nullable=False)
    feature_version: str = Column(String(32), nullable=False, default="v1")
    created_at: datetime = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    market = relationship("Market", back_populates="features")

    __table_args__ = (
        UniqueConstraint(
            "market_id", "commodity", "feature_date", "feature_version",
            name="uq_features_market_commodity_date_version",
        ),
        Index("ix_features_market_id", "market_id"),
        Index("ix_features_commodity", "commodity"),
        Index("ix_features_feature_date", "feature_date"),
        Index("ix_features_market_commodity", "market_id", "commodity"),
    )

    def __repr__(self) -> str:
        return (
            f"<FeatureSet id={self.id} market_id={self.market_id} "
            f"commodity={self.commodity!r} date={self.feature_date} version={self.feature_version!r}>"
        )


# ---------------------------------------------------------------------------
# MarketGraphEdge
# ---------------------------------------------------------------------------

class MarketGraphEdge(Base):
    """Directed weighted edge between two markets in the price-influence graph."""

    __tablename__ = "market_graph_edges"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    source_market_id: str = Column(String(64), nullable=False, index=True)
    target_market_id: str = Column(String(64), nullable=False, index=True)
    distance_km: Optional[float] = Column(Float, nullable=True)
    edge_weight: float = Column(Float, nullable=False, default=1.0)
    commodity: Optional[str] = Column(String(128), nullable=True, index=True)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "source_market_id", "target_market_id", "commodity",
            name="uq_market_graph_edge",
        ),
        Index("ix_market_graph_edges_source", "source_market_id"),
        Index("ix_market_graph_edges_target", "target_market_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<MarketGraphEdge id={self.id} src={self.source_market_id!r} "
            f"tgt={self.target_market_id!r} commodity={self.commodity!r} "
            f"weight={self.edge_weight:.4f}>"
        )


# ---------------------------------------------------------------------------
# ModelRun
# ---------------------------------------------------------------------------

class ModelRun(Base):
    """Training run metadata and evaluation metrics for a prediction model."""

    __tablename__ = "model_runs"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    model_name: str = Column(String(128), nullable=False)
    model_version: str = Column(String(64), nullable=False)
    crop: str = Column(String(128), nullable=False)
    state: Optional[str] = Column(String(128), nullable=True)
    district: Optional[str] = Column(String(128), nullable=True)
    market_scope: Optional[str] = Column(String(64), nullable=True)  # e.g. "all", "regional"

    # Date ranges
    train_start: Optional[datetime] = Column(DateTime(timezone=False), nullable=True)
    train_end: Optional[datetime] = Column(DateTime(timezone=False), nullable=True)
    val_start: Optional[datetime] = Column(DateTime(timezone=False), nullable=True)
    val_end: Optional[datetime] = Column(DateTime(timezone=False), nullable=True)
    test_start: Optional[datetime] = Column(DateTime(timezone=False), nullable=True)
    test_end: Optional[datetime] = Column(DateTime(timezone=False), nullable=True)

    # Config
    features: Optional[dict] = Column(JSON, nullable=True)
    hyperparameters: Optional[dict] = Column(JSON, nullable=True)

    # Metrics
    mae: Optional[float] = Column(Float, nullable=True)
    rmse: Optional[float] = Column(Float, nullable=True)
    mape: Optional[float] = Column(Float, nullable=True)
    smape: Optional[float] = Column(Float, nullable=True)
    wape: Optional[float] = Column(Float, nullable=True)
    r2: Optional[float] = Column(Float, nullable=True)

    # Forecast horizon in days
    horizon: int = Column(Integer, nullable=False, default=7)

    # Artifact paths
    training_timestamp: datetime = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    model_path: Optional[str] = Column(Text, nullable=True)
    scaler_path: Optional[str] = Column(Text, nullable=True)
    is_active: bool = Column(Boolean, nullable=False, default=False)

    predictions = relationship("Prediction", back_populates="model_run", lazy="dynamic")

    __table_args__ = (
        Index("ix_model_runs_crop", "crop"),
        Index("ix_model_runs_model_name", "model_name"),
        Index("ix_model_runs_is_active", "is_active"),
        Index("ix_model_runs_crop_horizon", "crop", "horizon"),
        Index("ix_model_runs_training_timestamp", "training_timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelRun id={self.id} model={self.model_name!r} v={self.model_version!r} "
            f"crop={self.crop!r} horizon={self.horizon}d mae={self.mae} active={self.is_active}>"
        )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class Prediction(Base):
    """Forward price predictions generated by a ModelRun."""

    __tablename__ = "predictions"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    market_id: int = Column(Integer, ForeignKey("markets.id", ondelete="CASCADE"), nullable=False)
    commodity: str = Column(String(128), nullable=False)
    prediction_date: datetime = Column(DateTime(timezone=False), nullable=False)
    horizon_days: int = Column(Integer, nullable=False)
    predicted_price: Numeric = Column(Numeric(12, 2), nullable=False)
    lower_bound: Optional[Numeric] = Column(Numeric(12, 2), nullable=True)
    upper_bound: Optional[Numeric] = Column(Numeric(12, 2), nullable=True)
    confidence_level: Optional[float] = Column(Float, nullable=True)  # e.g. 0.95
    model_run_id: int = Column(Integer, ForeignKey("model_runs.id", ondelete="SET NULL"), nullable=True)
    actual_price: Optional[Numeric] = Column(Numeric(12, 2), nullable=True)
    error: Optional[float] = Column(Float, nullable=True)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    market = relationship("Market", back_populates="predictions")
    model_run = relationship("ModelRun", back_populates="predictions")

    __table_args__ = (
        UniqueConstraint(
            "market_id", "commodity", "prediction_date", "horizon_days", "model_run_id",
            name="uq_prediction_market_commodity_date_horizon_run",
        ),
        Index("ix_predictions_market_id", "market_id"),
        Index("ix_predictions_commodity", "commodity"),
        Index("ix_predictions_prediction_date", "prediction_date"),
        Index("ix_predictions_model_run_id", "model_run_id"),
        Index("ix_predictions_market_commodity", "market_id", "commodity"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} market_id={self.market_id} "
            f"commodity={self.commodity!r} date={self.prediction_date} "
            f"horizon={self.horizon_days}d price={self.predicted_price}>"
        )


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------

class Alert(Base):
    """Price anomaly and market alerts."""

    __tablename__ = "alerts"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    market_id: int = Column(Integer, ForeignKey("markets.id", ondelete="CASCADE"), nullable=False)
    commodity: str = Column(String(128), nullable=False)
    alert_date: datetime = Column(DateTime(timezone=False), nullable=False)
    alert_type: str = Column(String(64), nullable=False)   # e.g. "price_spike", "supply_drop"
    severity: str = Column(String(16), nullable=False, default="medium")  # low/medium/high/critical
    description: str = Column(Text, nullable=False)
    threshold_value: Optional[float] = Column(Float, nullable=True)
    actual_value: Optional[float] = Column(Float, nullable=True)
    is_resolved: bool = Column(Boolean, nullable=False, default=False)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    market = relationship("Market", back_populates="alerts")

    __table_args__ = (
        Index("ix_alerts_market_id", "market_id"),
        Index("ix_alerts_commodity", "commodity"),
        Index("ix_alerts_alert_date", "alert_date"),
        Index("ix_alerts_alert_type", "alert_type"),
        Index("ix_alerts_severity", "severity"),
        Index("ix_alerts_is_resolved", "is_resolved"),
        Index("ix_alerts_market_commodity", "market_id", "commodity"),
    )

    def __repr__(self) -> str:
        return (
            f"<Alert id={self.id} market_id={self.market_id} commodity={self.commodity!r} "
            f"type={self.alert_type!r} severity={self.severity!r} resolved={self.is_resolved}>"
        )


# ---------------------------------------------------------------------------
# DataQuality
# ---------------------------------------------------------------------------

class DataQuality(Base):
    """Data quality audit records for any table/market/commodity combination."""

    __tablename__ = "data_quality"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    table_name: str = Column(String(128), nullable=False)
    market_id: Optional[int] = Column(Integer, ForeignKey("markets.id", ondelete="SET NULL"), nullable=True)
    commodity: Optional[str] = Column(String(128), nullable=True)
    check_date: datetime = Column(DateTime(timezone=False), nullable=False)
    total_records: int = Column(Integer, nullable=False, default=0)
    missing_records: int = Column(Integer, nullable=False, default=0)
    duplicate_records: int = Column(Integer, nullable=False, default=0)
    invalid_records: int = Column(Integer, nullable=False, default=0)
    quality_score: float = Column(Float, nullable=False, default=1.0)  # 0.0 – 1.0
    details: Optional[dict] = Column(JSON, nullable=True)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    market = relationship("Market", back_populates="data_quality")

    __table_args__ = (
        Index("ix_data_quality_table_name", "table_name"),
        Index("ix_data_quality_market_id", "market_id"),
        Index("ix_data_quality_commodity", "commodity"),
        Index("ix_data_quality_check_date", "check_date"),
        Index("ix_data_quality_table_market", "table_name", "market_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<DataQuality id={self.id} table={self.table_name!r} market_id={self.market_id} "
            f"commodity={self.commodity!r} date={self.check_date} score={self.quality_score:.3f}>"
        )
