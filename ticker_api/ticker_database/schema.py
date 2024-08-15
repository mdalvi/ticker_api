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


# =============================
# Historical Data Table
# =============================
class HistoricalData(Base):
    __tablename__ = "historical_data"
    __table_args__ = (
        Index("idx_instrument_token", "instrument_token"),
        Index("idx_record_datetime", "record_datetime"),
        Index("idx_interval", "interval"),
        Index("idx_record_date", "record_date"),
        Index("idx_record_time", "record_time"),
        UniqueConstraint("instrument_token", "record_datetime", "interval"),
        {"mysql_engine": "MyISAM"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tradingsymbol = Column(String(255), nullable=False)
    instrument_token = Column(Integer, nullable=False)
    record_datetime = Column(DateTime, nullable=False)
    record_date = Column(Date, nullable=False)
    record_time = Column(Time, nullable=False)
    interval = Column(String(8), nullable=False)
    open_price = Column(Numeric(10, 2), nullable=False)
    high_price = Column(Numeric(10, 2), nullable=False)
    low_price = Column(Numeric(10, 2), nullable=False)
    close_price = Column(Numeric(10, 2), nullable=False)
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
    exchange_token = Column(Numeric(10, 2), nullable=False)
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
