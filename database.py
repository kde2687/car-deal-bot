from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean,
    DateTime, Text, JSON, Index, ForeignKey, event
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

import config

_connect_args = {"check_same_thread": False, "timeout": 30} if "sqlite" in config.DATABASE_URL else {}
engine = create_engine(config.DATABASE_URL, connect_args=_connect_args)

# Enable WAL mode for SQLite — allows concurrent reads during writes
if "sqlite" in config.DATABASE_URL:
    @event.listens_for(engine, "connect")
    def _set_wal_mode(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA journal_mode=WAL")
        dbapi_conn.execute("PRAGMA synchronous=NORMAL")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Listing(Base):
    __tablename__ = "listings"

    id = Column(String, primary_key=True)
    source = Column(String, nullable=False)
    title = Column(String)
    brand = Column(String)
    model = Column(String)
    year = Column(Integer)
    km = Column(Integer)
    price_ars = Column(Float)
    price_usd = Column(Float, nullable=True)
    fuel = Column(String)
    transmission = Column(String)
    condition = Column(String)
    url = Column(String)
    thumbnail = Column(String)
    score = Column(Float, default=0.0)
    discount_pct = Column(Float, default=0.0)
    is_deal = Column(Boolean, default=False)
    alerted = Column(Boolean, default=False)
    deal_reason = Column(Text)
    is_agency = Column(Boolean, default=False)   # flagged as dealer post-scan
    hidden = Column(Boolean, default=False)       # manually hidden by user
    market_price_ars = Column(Float, nullable=True)  # market median used for scoring
    price_usd_equiv  = Column(Float, nullable=True)  # price converted to USD at MEP rate
    percentile_rank  = Column(Integer, nullable=True) # price percentile within comparable set (lower = cheaper)
    ref_type         = Column(String, nullable=True)  # reference type: exact/broad/curve/etc.
    confidence_index = Column(Integer, nullable=True) # 0–100: type_weight × √(n/30) × exp(-3×QCD)
    # Panel tracking
    status              = Column(String, default="active")   # active / sold / expired
    sold_at             = Column(DateTime, nullable=True)    # approx time of sale (last_seen when flagged)
    price_changes_count = Column(Integer, default=0)         # how many times price was revised
    last_price_change   = Column(DateTime, nullable=True)    # when price last changed

    price_history = relationship("PriceHistory", back_populates="listing",
                                  order_by="PriceHistory.recorded_at")
    seller_city = Column(String)
    seller_lat = Column(Float, nullable=True)
    seller_lon = Column(Float, nullable=True)
    distance_km = Column(Float, nullable=True)
    first_seen = Column(DateTime, default=datetime.utcnow, index=True)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    raw_data = Column(JSON)

    __table_args__ = (
        Index("idx_listing_brand_year",   "brand", "year"),
        Index("idx_listing_deal_hidden",  "is_deal", "hidden", "is_agency"),
        Index("idx_listing_score",        "score"),
    )


class PriceHistory(Base):
    """
    Append-only log of price changes per listing.
    event_type: 'initial' | 'price_change' | 'sold'
    Used for panel data analysis and survival modelling (time-to-sale).
    """
    __tablename__ = "price_history"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    listing_id    = Column(String, ForeignKey("listings.id"), nullable=False, index=True)
    price_ars     = Column(Float)
    price_usd_equiv = Column(Float, nullable=True)
    recorded_at   = Column(DateTime, default=datetime.utcnow, index=True)
    days_on_market = Column(Integer)   # days since first_seen at time of event
    event_type    = Column(String)     # 'initial' | 'price_change' | 'sold'

    listing = relationship("Listing", back_populates="price_history")


class MarketReference(Base):
    __tablename__ = "market_references"

    id = Column(Integer, primary_key=True, autoincrement=True)
    brand = Column(String)
    model = Column(String)
    year = Column(Integer)
    avg_price_ars = Column(Float)
    median_price_ars = Column(Float)
    median_usd = Column(Float, nullable=True)
    usd_rate_used = Column(Float, nullable=True)
    sample_count = Column(Integer)
    updated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_mktref_bmh", "brand", "model", "year", unique=True),
    )


def init_db():
    Base.metadata.create_all(engine)
