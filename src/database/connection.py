"""
Database connection management for the Crop Price Prediction system.

Provides engine creation, session factory, context-managed sessions,
and table initialisation with graceful PostGIS fallback.
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """
    Load database configuration.

    Tries to import from the project's config module first; falls back to
    sensible environment-variable-based defaults so the module is importable
    in isolation.
    """
    try:
        from config import settings  # type: ignore[import]
        return {
            "db_url": settings.DATABASE_URL,
            "pool_size": getattr(settings, "DB_POOL_SIZE", 10),
            "max_overflow": getattr(settings, "DB_MAX_OVERFLOW", 20),
            "pool_timeout": getattr(settings, "DB_POOL_TIMEOUT", 30),
            "pool_recycle": getattr(settings, "DB_POOL_RECYCLE", 1800),
            "pool_pre_ping": getattr(settings, "DB_POOL_PRE_PING", True),
            "echo": getattr(settings, "DB_ECHO", False),
        }
    except (ImportError, AttributeError):
        import os
        return {
            "db_url": os.environ.get(
                "DATABASE_URL",
                "postgresql+psycopg2://postgres:postgres@localhost:5432/crop_price_db",
            ),
            "pool_size": int(os.environ.get("DB_POOL_SIZE", "10")),
            "max_overflow": int(os.environ.get("DB_MAX_OVERFLOW", "20")),
            "pool_timeout": int(os.environ.get("DB_POOL_TIMEOUT", "30")),
            "pool_recycle": int(os.environ.get("DB_POOL_RECYCLE", "1800")),
            "pool_pre_ping": os.environ.get("DB_POOL_PRE_PING", "true").lower() == "true",
            "echo": os.environ.get("DB_ECHO", "false").lower() == "true",
        }


# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialised)
# ---------------------------------------------------------------------------

_engine: Engine | None = None
_session_factory: sessionmaker | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_engine() -> Engine:
    """
    Return (or create) the shared SQLAlchemy engine.

    The engine is created once and reused for the lifetime of the process.
    Connection pool parameters are loaded from config.
    """
    global _engine
    if _engine is not None:
        return _engine

    cfg = _load_config()
    db_url: str = cfg["db_url"]

    logger.info("Creating SQLAlchemy engine for %s", db_url.split("@")[-1])  # hide credentials

    # SQLite does not support pool_size / max_overflow
    is_sqlite = db_url.startswith("sqlite")

    connect_args: dict = {}
    if is_sqlite:
        connect_args["check_same_thread"] = False

    engine_kwargs: dict = {
        "echo": cfg["echo"],
        "pool_pre_ping": cfg["pool_pre_ping"],
        "connect_args": connect_args,
    }

    if not is_sqlite:
        engine_kwargs.update(
            {
                "pool_size": cfg["pool_size"],
                "max_overflow": cfg["max_overflow"],
                "pool_timeout": cfg["pool_timeout"],
                "pool_recycle": cfg["pool_recycle"],
            }
        )

    _engine = create_engine(db_url, **engine_kwargs)

    # Enable WAL mode for SQLite to improve concurrent read performance
    if is_sqlite:
        @event.listens_for(_engine, "connect")
        def _set_wal(dbapi_conn, _connection_record):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    logger.info("Engine created successfully.")
    return _engine


def get_session_factory() -> sessionmaker:
    """
    Return (or create) the shared session factory.

    The factory is bound to the shared engine and produces sessions that
    do *not* auto-commit or auto-flush to give callers explicit control.
    """
    global _session_factory
    if _session_factory is not None:
        return _session_factory

    engine = get_engine()
    _session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
    logger.debug("Session factory created.")
    return _session_factory


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager that yields a database session.

    Automatically commits on success and rolls back on any exception.
    The session is always closed after use.

    Usage::

        with get_session() as session:
            session.add(some_obj)
    """
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
        logger.debug("Session committed successfully.")
    except Exception:
        session.rollback()
        logger.exception("Session rolled back due to an error.")
        raise
    finally:
        session.close()
        logger.debug("Session closed.")


def init_db(engine: Engine | None = None) -> None:
    """
    Create all database tables defined in the ORM models.

    If a PostgreSQL backend is detected, attempts to enable the PostGIS
    extension first and gracefully falls back if it is not available.

    Args:
        engine: Optional engine to use. Defaults to the shared engine.
    """
    if engine is None:
        engine = get_engine()

    db_url = str(engine.url)
    is_postgres = "postgresql" in db_url or "psycopg" in db_url

    if is_postgres:
        _try_enable_postgis(engine)

    logger.info("Creating database tables (if not exist)…")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("All tables created / verified successfully.")
    except Exception:
        logger.exception("Failed to create database tables.")
        raise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_enable_postgis(engine: Engine) -> None:
    """Attempt to enable the PostGIS extension; log a warning if unavailable."""
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        logger.info("PostGIS extension enabled (or already present).")
    except (OperationalError, ProgrammingError) as exc:
        logger.warning(
            "PostGIS is not available on this PostgreSQL instance — spatial "
            "features will be disabled. Details: %s",
            exc,
        )
    except Exception:
        logger.warning(
            "Unexpected error while enabling PostGIS; continuing without it.",
            exc_info=True,
        )
