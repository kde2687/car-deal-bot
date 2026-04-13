import asyncio
import atexit
import logging
import logging.handlers
import os
import signal
import sys
import time
import threading
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    import config as cfg

    log_level = getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO)

    rotating_handler = logging.handlers.RotatingFileHandler(
        "logs/bot.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    rotating_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(rotating_handler)
    root_logger.addHandler(stream_handler)
    atexit.register(logging.shutdown)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


async def run_scan(telegram_alerter=None):
    import config as cfg
    from scorer import process_listings
    from database import SessionLocal

    logger.info("Starting scan...")
    t_start = time.time()

    from scrapers.mercadolibre import MercadoLibreScraper
    from scrapers.autocosmos import AutocosmosScraper
    from scrapers.kavak import KavakScraper

    ml_scraper = MercadoLibreScraper(cfg.BRANDS_LIST, cfg.MIN_YEAR, cfg.MAX_KM)
    ac_scraper = AutocosmosScraper(cfg.BRANDS_LIST, cfg.MIN_YEAR, cfg.MAX_KM)
    kv_scraper = KavakScraper(cfg.BRANDS_LIST, cfg.MIN_YEAR, cfg.MAX_KM)

    async def safe_scrape(name, scraper):
        t = time.time()
        try:
            # Per-scraper timeout: one slow source won't block the others
            result = await asyncio.wait_for(scraper.fetch_listings(), timeout=380.0)
            return result or [], time.time() - t
        except asyncio.TimeoutError:
            logger.error(f"{name} scraper timed out after 380s — returning empty")
            return [], time.time() - t
        except Exception as e:
            logger.error(f"{name} scraper failed: {e}")
            return [], time.time() - t

    try:
        (ml_listings, t_ml), (ac_listings, t_ac), (kavak_listings, t_kavak) = await asyncio.wait_for(
            asyncio.gather(
                safe_scrape("MercadoLibre", ml_scraper),
                safe_scrape("Autocosmos", ac_scraper),
                safe_scrape("Kavak", kv_scraper),
            ),
            timeout=420.0,
        )
    except asyncio.TimeoutError:
        # Global timeout: all scrapers hung simultaneously.
        # Do NOT process empty results — process_listings([]) with empty scraped_sources
        # would mark ALL active listings as sold (cascading data loss).
        logger.error("Scrape phase exceeded 420s global timeout — skipping scan to preserve DB state")
        return

    all_listings = ml_listings + ac_listings + kavak_listings
    logger.info(f"Fetched {len(ml_listings)} ML + {len(ac_listings)} Autocosmos + {len(kavak_listings)} Kavak = {len(all_listings)} total")

    new_count, deal_count, updated_count = (0, 0, 0)
    new_ml_ids = []
    try:
        loop = asyncio.get_running_loop()
        new_count, deal_count, updated_count, new_ml_ids = await loop.run_in_executor(
            None, process_listings, all_listings
        )
    except Exception as e:
        logger.error(f"process_listings failed: {e}")

    # Background enrichment: check new ML listings for dealers that slipped through
    if new_ml_ids:
        try:
            from scrapers.ml_enrich import enrich_ml_new_listings
            agencies_found = await enrich_ml_new_listings(new_ml_ids)
            if agencies_found:
                logger.info(f"ML enrich: removed {agencies_found} agency listings")
        except Exception as e:
            logger.error(f"ML enrich failed: {e}")

    # Post-scan rescore: after enrichment updates market_price_ars the scores are stale.
    # Re-scoring immediately ensures deal flags and scores reflect the enriched data
    # instead of waiting up to 4 hours for the next periodic market refresh.
    try:
        from scorer import rescore_all_active_listings
        from database import SessionLocal as _SL
        loop = asyncio.get_running_loop()
        _r, _u = await loop.run_in_executor(None, rescore_all_active_listings, _SL())
        if _u:
            logger.info(f"Post-scan rescore: {_r} rescored, {_u} new deals after enrichment")
    except Exception as e:
        logger.error(f"Post-scan rescore failed: {e}")

    # Individual per-scan alerts disabled — digest sent daily at 8am via scheduler

    t_total = time.time() - t_start

    table = Table(title=f"Scan complete — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    table.add_column("Source", style="cyan", no_wrap=True)
    table.add_column("Fetched", justify="right", style="white")
    table.add_column("New", justify="right", style="green")
    table.add_column("Deals", justify="right", style="yellow")
    table.add_column("Time", justify="right", style="dim")
    table.add_row("MercadoLibre", str(len(ml_listings)), "—", "—", f"{t_ml:.1f}s")
    table.add_row("Autocosmos", str(len(ac_listings)), "—", "—", f"{t_ac:.1f}s")
    table.add_row("Kavak", str(len(kavak_listings)), "—", "—", f"{t_kavak:.1f}s")
    table.add_row("[bold]Total[/bold]", str(len(all_listings)), str(new_count), str(deal_count), f"{t_total:.1f}s")
    console.print(table)

    logger.info(f"Scan complete in {t_total:.1f}s: {new_count} new, {deal_count} deals, {updated_count} updated")


async def run_telegram_digest(telegram_alerter=None):
    # Paused — email digest is sufficient until model quality improves
    logger.info("Telegram digest skipped (paused)")
    return


async def run_email_digest():
    import config as cfg
    if not cfg.SMTP_USER or not cfg.SMTP_PASSWORD:
        logger.info("Email digest skipped: SMTP credentials not set")
        return
    try:
        from alerts.email_digest import send_daily_digest
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(
            None, send_daily_digest, cfg.SMTP_USER, cfg.SMTP_PASSWORD, cfg.DIGEST_RECIPIENT
        )
        if ok:
            logger.info("Daily email digest sent")
    except Exception as e:
        logger.error(f"Email digest failed: {e}")


async def run_db_backup():
    import sqlite3, glob as glob_mod
    import config as cfg
    db_path = cfg.DATABASE_URL.replace("sqlite:///", "")
    if not os.path.exists(db_path):
        return
    os.makedirs("backups", exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y-%m-%d")
    dest  = f"backups/deals_{stamp}.db"
    try:
        # sqlite3.backup() is atomic — waits for active transactions to finish
        src  = sqlite3.connect(db_path)
        bkp  = sqlite3.connect(dest)
        with bkp:
            src.backup(bkp)
        src.close()
        bkp.close()
        logger.info(f"DB backup saved: {dest}")
        all_backups = sorted(glob_mod.glob("backups/deals_*.db"))
        for old in all_backups[:-7]:
            os.remove(old)
            logger.info(f"Removed old backup: {old}")
    except Exception as e:
        logger.error(f"DB backup failed: {e}")


async def run_pricing_model_train():
    from database import SessionLocal
    from pricing_model.pipeline import maybe_retrain
    session = SessionLocal()
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, maybe_retrain, session)
    except Exception as e:
        logger.error(f"Pricing model training failed: {e}")
    finally:
        session.close()


async def run_market_refresh():
    from database import SessionLocal
    from scorer import update_market_references, rescore_all_active_listings, update_segment_velocity

    logger.info("Refreshing market references...")
    session = SessionLocal()
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, update_market_references, session)
        logger.info("Market references refreshed")
        rescored, upgraded = await loop.run_in_executor(None, rescore_all_active_listings, session)
        logger.info(f"Periodic rescore: {rescored} listings rescored, {upgraded} new deals found")
        await loop.run_in_executor(None, update_segment_velocity, session)
    except Exception as e:
        logger.error(f"Market refresh failed: {e}")
    finally:
        session.close()


def run_flask(app):
    try:
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        logger.error(f"Flask server error: {e}")


async def main():
    os.makedirs("logs", exist_ok=True)

    setup_logging()

    import config as cfg
    from database import init_db
    from dashboard.app import create_app
    from alerts.telegram import TelegramAlerter

    init_db()
    logger.info("Database initialized")

    console.print(
        Panel(
            Text.from_markup(
                "[bold cyan]🚗 CarDeal AR — Deal Hunting Bot[/bold cyan]\n\n"
                f"[white]Brands:[/white] [yellow]{cfg.BRANDS}[/yellow]\n"
                f"[white]Min year:[/white] [yellow]{cfg.MIN_YEAR}[/yellow]  "
                f"[white]Max KM:[/white] [yellow]{cfg.MAX_KM:,}[/yellow]\n"
                f"[white]Score threshold:[/white] [yellow]{cfg.DEAL_SCORE_THRESHOLD}[/yellow]  "
                f"[white]Scan interval:[/white] [yellow]{cfg.SCAN_INTERVAL_MINUTES} min[/yellow]\n"
                f"[white]Database:[/white] [dim]{cfg.DATABASE_URL}[/dim]\n"
                f"[white]Dashboard:[/white] [link=http://localhost:5000]http://localhost:5000[/link]"
            ),
            title="Startup",
            border_style="cyan",
        )
    )

    flask_app = create_app()
    flask_thread = threading.Thread(target=run_flask, args=(flask_app,), daemon=True)
    flask_thread.start()
    logger.info("Flask dashboard started on http://0.0.0.0:5000")

    alerter = TelegramAlerter(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)
    await alerter.initialize()

    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        scheduler = AsyncIOScheduler()
        # Scan interval: controlled via SCAN_INTERVAL_MINUTES env var (default 30 min)
        scheduler.add_job(
            run_scan,
            "interval",
            minutes=cfg.SCAN_INTERVAL_MINUTES,
            args=[alerter],
            id="scan_periodic",
        )
        scheduler.add_job(
            run_market_refresh,
            "interval",
            hours=4,
            id="market_refresh",
        )
        scheduler.add_job(
            run_db_backup,
            "cron",
            hour=3, minute=0, timezone="UTC",  # 00hs Argentina
            id="db_backup",
        )
        scheduler.add_job(
            run_pricing_model_train,
            "interval",
            hours=24,
            id="pricing_model_train",
        )
        # Email digests — Argentina UTC-3: 6am=9UTC, 12pm=15UTC, 21hs=0UTC
        for digest_id, utc_hour in [("digest_6am", 9), ("digest_12pm", 15), ("digest_21hs", 0)]:
            scheduler.add_job(
                run_email_digest,
                "cron",
                hour=utc_hour,
                minute=0,
                timezone="UTC",
                id=digest_id,
            )
        # Telegram digest — 8am Argentina = 11:00 UTC
        scheduler.add_job(
            run_telegram_digest,
            "cron",
            hour=11, minute=0,
            timezone="UTC",
            args=[alerter],
            id="telegram_digest_8am",
        )
        scheduler.start()
        logger.info(f"Scheduler started: scans every {cfg.SCAN_INTERVAL_MINUTES}min, market refresh every 4h")
    except Exception as e:
        logger.error(f"Scheduler failed to start: {e}")
        scheduler = None

    console.print(f"\n[bold green]Running first scan...[/bold green]")
    await run_scan(alerter)

    # Rescore immediately after first scan so listings scored with "cold"
    # during the scan get re-evaluated with newly accumulated comparables.
    console.print(f"\n[bold green]Running initial market refresh + rescore...[/bold green]")
    await run_market_refresh()

    console.print(f"\n[bold green]Dashboard running at http://localhost:5000[/bold green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    stop_event = asyncio.Event()

    def _request_shutdown():
        logger.info("Shutdown requested (SIGTERM/SIGINT)")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except (NotImplementedError, OSError):
            pass  # Windows doesn't support add_signal_handler

    try:
        await stop_event.wait()
    except KeyboardInterrupt:
        pass
    console.print("\n[yellow]Shutting down...[/yellow]")
    logger.info("Shutdown requested")
    if scheduler:
        try:
            scheduler.shutdown(wait=False)
        except Exception as e:
            logger.warning(f"Scheduler shutdown failed: {e}")
    await alerter.shutdown()
    console.print("[green]Goodbye.[/green]")


if __name__ == "__main__":
    # Prevent multiple instances — PID-file approach is reliable even if the
    # previous run was SIGKILL'd (flock on re-opened files can be bypassed).
    _PID_FILE = "/tmp/car_deal_bot.pid"
    if os.path.exists(_PID_FILE):
        try:
            _old_pid = int(open(_PID_FILE).read().strip())
            # Check if that PID is still alive
            os.kill(_old_pid, 0)   # signal 0 = existence check only
            print(f"Another instance (PID {_old_pid}) is already running. Exiting.")
            sys.exit(1)
        except (ProcessLookupError, ValueError, PermissionError, OSError):
            pass   # stale PID file — safe to overwrite

    with open(_PID_FILE, "w") as _pf:
        _pf.write(str(os.getpid()))

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        try:
            os.remove(_PID_FILE)
        except FileNotFoundError:
            pass
