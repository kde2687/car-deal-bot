from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean,
    DateTime, Text, JSON, LargeBinary, Index, ForeignKey, event, CheckConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

import config

if "sqlite" in config.DATABASE_URL:
    _connect_args = {"check_same_thread": False, "timeout": 30}
else:
    _connect_args = {"connect_timeout": 30}  # PostgreSQL connection timeout
engine = create_engine(config.DATABASE_URL, connect_args=_connect_args)

# Enable WAL mode for SQLite — allows concurrent reads during writes
if "sqlite" in config.DATABASE_URL:
    @event.listens_for(engine, "connect")
    def _set_wal_mode(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA journal_mode=WAL")
        dbapi_conn.execute("PRAGMA synchronous=NORMAL")
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

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
    percentile_rank  = Column(Float, nullable=True)   # price percentile within comparable set (lower = cheaper) — Float for sub-integer precision
    ref_type         = Column(String, nullable=True)  # reference type: exact/broad/curve/etc.
    confidence_index = Column(Integer, nullable=True) # 0–100: type_weight × √(n/30) × exp(-3×QCD)
    # Preserved original price — set once at insert, never updated.
    # Allows tracking total price movement over the lifetime of the listing.
    initial_price_ars   = Column(Float, nullable=True)
    initial_price_usd   = Column(Float, nullable=True)
    # Denormalized seller name (from raw_data) for fast blocklist matching and display.
    # Stored normalized (lowercase, collapsed whitespace) to match BlockedSeller.seller_name.
    seller_name         = Column(String, nullable=True, index=True)
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
        Index("idx_listing_brand_year",          "brand", "year"),
        Index("idx_listing_brand_model_year",    "brand", "model", "year"),  # speeds up market reference queries
        # Composite index for comparables queries: (brand, model, year, source) — avoids full scan
        # when score_listing filters active listings by segment + source in the same query.
        Index("idx_listing_bmys",                "brand", "model", "year", "source"),
        Index("idx_listing_deal_hidden",         "is_deal", "hidden", "is_agency"),
        Index("idx_listing_score",               "score"),
        Index("idx_listing_status_last_seen",    "status", "last_seen"),
        Index("idx_listing_source",              "source"),
        Index("idx_listing_status",              "status"),  # standalone status index for active-only queries
        # Partial-style substitute: index price columns used in _mark_sold_listings / update_market_references
        Index("idx_listing_price_ars",           "price_ars"),
        Index("idx_listing_price_usd_equiv",     "price_usd_equiv"),
        # Enforce valid status values at the DB level
        CheckConstraint("status IN ('active', 'sold', 'expired')", name="ck_listing_status"),
    )


class PriceHistory(Base):
    """
    Append-only log of price changes per listing.
    event_type:
      'initial'      — first time a listing is seen (inserted)
      'price_change' — price changed by >0.5% vs previous value
      'sold'         — listing disappeared for >24h and was marked sold
      'resurrected'  — listing reappeared after being marked sold
                       (scorer logs this but does not yet write this event;
                        add _record_price_event call in the resurrection branch)
    Used for panel data analysis and survival modelling (time-to-sale).
    """
    __tablename__ = "price_history"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    listing_id    = Column(String, ForeignKey("listings.id"), nullable=False, index=True)
    price_ars     = Column(Float)
    price_usd_equiv = Column(Float, nullable=True)
    recorded_at   = Column(DateTime, default=datetime.utcnow, index=True)
    days_on_market = Column(Integer)   # days since first_seen at time of event
    event_type    = Column(String)     # see class docstring for valid values

    listing = relationship("Listing", back_populates="price_history")

    __table_args__ = (
        CheckConstraint(
            "event_type IN ('initial', 'price_change', 'sold', 'resurrected')",
            name="ck_price_history_event_type",
        ),
        # Index for filtering events by type (e.g. querying all 'sold' events for survival analysis)
        Index("idx_price_history_event_type", "event_type"),
    )


class MarketReference(Base):
    """
    Cached median market price per brand/model/year segment.

    Staleness risk: rows are only refreshed when update_market_references() runs
    (called after each full scan).  If a segment disappears from the market (no
    active listings), its row is never deleted — callers should treat rows older
    than `expires_at` (or older than a configurable threshold) as unreliable.

    TTL convention: `expires_at` is set to updated_at + 7 days by the writer
    (_save_market_ref in scorer.py).  Queries that care about freshness should
    add a filter: MarketReference.expires_at > datetime.utcnow().
    NOTE: scorer.py must be updated to populate expires_at.
    """
    __tablename__ = "market_references"

    id = Column(Integer, primary_key=True, autoincrement=True)
    brand = Column(String)
    model = Column(String)
    year = Column(Integer)
    avg_price_ars = Column(Float)
    median_price_ars = Column(Float)
    median_usd = Column(Float, nullable=True)
    usd_rate_used = Column(Float, nullable=True)
    sample_count = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow)
    # Expiry timestamp — set by writer to updated_at + 7 days.
    # Allows callers to skip stale references without relying on wall-clock math.
    expires_at = Column(DateTime, nullable=True, index=True)

    __table_args__ = (
        Index("idx_mktref_bmh", "brand", "model", "year", unique=True),
        CheckConstraint("sample_count >= 0", name="ck_mktref_sample_count"),
    )


class SegmentVelocity(Base):
    """
    Average days-to-sale per brand/model/year — computed from sold listings.
    Used to flag segments with fast/slow turnover and to calibrate time-on-market signals.

    Cold-start note: this table will be empty on fresh deployments until enough
    listings have been marked sold (requires >= 2 confirmed sold samples per segment
    in update_segment_velocity()).  Callers must handle the None/missing-row case.

    Staleness: like MarketReference, rows are never deleted.  `expires_at` follows
    the same 7-day TTL convention; scorer.py must be updated to populate it.
    """
    __tablename__ = "segment_velocity"

    id = Column(Integer, primary_key=True, autoincrement=True)
    brand = Column(String)
    model = Column(String)
    year = Column(Integer)
    avg_days_to_sale = Column(Float)
    median_days_to_sale = Column(Float)
    sample_count = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow)
    # Expiry timestamp — same TTL convention as MarketReference.
    expires_at = Column(DateTime, nullable=True, index=True)

    __table_args__ = (
        Index("idx_segvel_bmy", "brand", "model", "year", unique=True),
        CheckConstraint("sample_count >= 2", name="ck_segvel_min_sample_count"),
    )


class ModelArtifact(Base):
    """
    Stores serialized ML model blobs in PostgreSQL so they survive Railway deploys.
    Each model_name has one row (upserted on save).
    """
    __tablename__ = "model_artifacts"

    model_name  = Column(String, primary_key=True)   # e.g. "pricing_v2"
    version     = Column(Integer, nullable=False)
    artifact    = Column(LargeBinary, nullable=False)  # pickle bytes
    trained_at  = Column(DateTime, nullable=True)
    sample_count = Column(Integer, nullable=True)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BlockedSeller(Base):
    """
    Seller names/nicknames manually flagged as agencies via the dashboard.
    Loaded at scan start and used to auto-flag new listings from the same seller.
    """
    __tablename__ = "blocked_sellers"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    seller_name = Column(String, nullable=False, unique=True)   # normalised lowercase
    added_at    = Column(DateTime, default=datetime.utcnow)
    notes       = Column(String, nullable=True)   # e.g. listing ID that triggered the block


def _migrate_existing_db():
    """
    Apply additive schema migrations (ALTER TABLE ADD COLUMN) for columns that
    were added after the initial schema was deployed.  SQLAlchemy's create_all()
    only creates *missing tables* — it does NOT add columns to existing tables.

    Pattern: attempt the ALTER TABLE; silently ignore "duplicate column" errors
    (both SQLite and PostgreSQL raise an error when the column already exists).
    All columns added here must be nullable or have a DEFAULT so existing rows
    are valid after the migration.

    IMPORTANT: When adding a new column to a model, add a corresponding entry
    here.  Keep entries in chronological order.
    """
    from sqlalchemy import text as _sql_text

    migrations = [
        # 2026-04 — initial_price_ars / initial_price_usd: preserved first price
        "ALTER TABLE listings ADD COLUMN initial_price_ars REAL",
        "ALTER TABLE listings ADD COLUMN initial_price_usd REAL",
        # 2026-04 — seller_name: denormalized for fast blocklist matching
        "ALTER TABLE listings ADD COLUMN seller_name VARCHAR",
        # 2026-04 — expires_at on MarketReference and SegmentVelocity for TTL support
        "ALTER TABLE market_references ADD COLUMN expires_at TIMESTAMP",
        "ALTER TABLE segment_velocity ADD COLUMN expires_at TIMESTAMP",
    ]

    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(_sql_text(sql))
                conn.commit()
            except Exception:
                # Column already exists — safe to ignore
                try:
                    conn.rollback()
                except Exception:
                    pass


def init_db():
    # Create any tables that don't exist yet (idempotent)
    Base.metadata.create_all(engine)
    # Apply additive column migrations for existing tables
    _migrate_existing_db()
