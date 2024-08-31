from sqlalchemy import (
    Column,
    BigInteger,
    DateTime,
    Date,
    Numeric,
    Integer,
    SmallInteger,
    String,
    func,
    Time,
    UniqueConstraint,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


# ====================================
# FNO Expiry Dates Table
# ====================================
class FNOExpiryDates(Base):
    __tablename__ = "fno_expiry_dates"
    __table_args__ = (
        Index("idx_exchange", "exchange"),
        Index("idx_segment", "segment"),
        Index("idx_name", "name"),
        Index("idx_instrument_type", "instrument_type"),
        Index("idx_expiry_date", "expiry_date"),
        UniqueConstraint(
            "exchange", "segment", "name", "instrument_type", "expiry_date"
        ),
        {"mysql_engine": "InnoDB"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    exchange = Column(String(10), nullable=False)
    segment = Column(String(10), nullable=False)
    name = Column(String(255), nullable=False)
    instrument_type = Column(String(10), nullable=False)
    expiry_date = Column(Date, nullable=False)
    expiry_weekday = Column(String(10), nullable=False)
    status = Column(SmallInteger, nullable=False, server_default="1")
    updated_at = Column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.current_timestamp(),
    )
    created_at = Column(DateTime, nullable=False, server_default=func.now())


# =============================
# Historical Updates Table
# =============================
class HistoricalDataSyncDetails(Base):
    __tablename__ = "historical_data_sync_details"
    __table_args__ = (
        Index("idx_exchange", "exchange"),
        Index("idx_segment", "segment"),
        Index("idx_name", "name"),
        Index("idx_instrument_type", "instrument_type"),
        Index("idx_expiry", "expiry"),
        Index("idx_interval", "interval"),
        UniqueConstraint(
            "exchange", "segment", "name", "instrument_type", "expiry", "interval"
        ),
        {"mysql_engine": "InnoDB"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    exchange = Column(String(10), nullable=False)
    segment = Column(String(10), nullable=False)
    name = Column(String(255), nullable=False)
    instrument_type = Column(String(10), nullable=False)
    expiry = Column(String(10), nullable=False)  # `NONE`, `CURRENT`, `MID` OR `FAR`
    interval = Column(String(8), nullable=False)
    from_date = Column(Date, nullable=False)
    to_date = Column(Date, nullable=False)
    status = Column(SmallInteger, nullable=False, server_default="1")
    updated_at = Column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.current_timestamp(),
    )
    created_at = Column(DateTime, nullable=False, server_default=func.now())


# =============================
# Historical Data Table
# =============================
class HistoricalData(Base):
    __tablename__ = "historical_data"
    __table_args__ = (
        Index("idx_exchange", "exchange"),
        Index("idx_segment", "segment"),
        Index("idx_name", "name"),
        Index("idx_instrument_type", "instrument_type"),
        Index("idx_expiry", "expiry"),
        Index("idx_interval", "interval"),
        Index("idx_record_date", "record_date"),
        UniqueConstraint(
            "record_datetime",
            "exchange",
            "segment",
            "name",
            "instrument_type",
            "expiry",
            "interval",
        ),
        {"mysql_engine": "InnoDB"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    exchange = Column(String(10), nullable=False)
    segment = Column(String(10), nullable=False)
    name = Column(String(255), nullable=False)
    instrument_type = Column(String(10), nullable=False)
    expiry = Column(String(10), nullable=False)  # `NONE`, `CURRENT`, `MID` OR `FAR`
    record_datetime = Column(DateTime, nullable=False)
    record_date = Column(Date, nullable=False)
    record_time = Column(Time, nullable=False)
    interval = Column(String(8), nullable=False)
    open_price = Column(Numeric(10, 2), nullable=False)
    high_price = Column(Numeric(10, 2), nullable=False)
    low_price = Column(Numeric(10, 2), nullable=False)
    close_price = Column(Numeric(10, 2), nullable=False)
    continuous = Column(SmallInteger, nullable=False)
    oi = Column(BigInteger, nullable=False)
    volume = Column(BigInteger, nullable=False)
    status = Column(SmallInteger, nullable=False, server_default="1")
    updated_at = Column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.current_timestamp(),
    )
    created_at = Column(DateTime, nullable=False, server_default=func.now())


# =============================
# Instruments Table
# =============================
class Instruments(Base):
    __tablename__ = "instruments"
    __table_args__ = (
        Index("idx_tradingsymbol", "tradingsymbol"),
        Index("idx_segment", "segment"),
        Index("idx_exchange", "exchange"),
        Index("idx_name", "name"),
        Index("idx_instrument_type", "instrument_type"),
        UniqueConstraint("instrument_token"),
        {"mysql_engine": "InnoDB"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tradingsymbol = Column(String(255), nullable=False)
    segment = Column(String(10), nullable=False)
    exchange = Column(String(10), nullable=False)
    instrument_token = Column(Integer, nullable=False)
    lot_size = Column(Integer, nullable=False)
    name = Column(String(255), nullable=True)
    instrument_type = Column(String(10), nullable=False)
    strike = Column(Numeric(10, 2), nullable=False)
    last_price = Column(Numeric(10, 2), nullable=False)
    exchange_token = Column(Integer, nullable=False)
    expiry = Column(Date, nullable=True)
    tick_size = Column(Numeric(10, 2), nullable=False)
    status = Column(SmallInteger, nullable=False, server_default="1")
    updated_at = Column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.current_timestamp(),
    )
    created_at = Column(DateTime, nullable=False, server_default=func.now())
