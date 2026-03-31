import logging
import re
from typing import Optional

import config
from database import SessionLocal, Listing

logger = logging.getLogger(__name__)

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed; Telegram alerts disabled")


def escape_markdown_v2(text: str) -> str:
    special_chars = r"\_*[]()~`>#+-=|{}.!"
    result = ""
    for ch in str(text):
        if ch in special_chars:
            result += "\\" + ch
        else:
            result += ch
    return result


class TelegramAlerter:
    def __init__(self, token: str, chat_id: str):
        self.token = token or ""
        self.chat_id = chat_id or ""
        self.app = None
        self._initialized = False

    def _is_configured(self) -> bool:
        return bool(self.token and self.chat_id and TELEGRAM_AVAILABLE)

    def format_message(self, listing_obj) -> str:
        usd_part = ""
        if listing_obj.price_usd:
            usd_part = escape_markdown_v2(f" / ${listing_obj.price_usd:,.0f} USD")

        price_str = (
            f"${listing_obj.price_ars:,.0f}" if listing_obj.price_ars else "N/A"
        )
        discount = listing_obj.discount_pct or 0.0
        km_str = f"{listing_obj.km:,}" if listing_obj.km else "N/A"
        transmission = listing_obj.transmission or "N/A"
        fuel = listing_obj.fuel or "N/A"
        year = listing_obj.year or "N/A"
        source_labels = {"mercadolibre": "MercadoLibre", "kavak": "Kavak"}
        source_label = source_labels.get(listing_obj.source or "", listing_obj.source or "")

        title_esc = escape_markdown_v2(listing_obj.title or "")
        price_esc = escape_markdown_v2(f"{price_str} ARS")
        discount_esc = escape_markdown_v2(f"{discount:.0f}")
        km_esc = escape_markdown_v2(str(km_str))
        trans_esc = escape_markdown_v2(transmission)
        fuel_esc = escape_markdown_v2(fuel)
        year_esc = escape_markdown_v2(str(year))
        source_esc = escape_markdown_v2(source_label)
        score_esc = escape_markdown_v2(f"{listing_obj.score:.0f}")
        reason_esc = escape_markdown_v2(listing_obj.deal_reason or "")

        dist_line = ""
        if listing_obj.distance_km is not None:
            dist_esc = escape_markdown_v2(f"{listing_obj.distance_km:.0f} km de Darregueira")
            city_esc = escape_markdown_v2(listing_obj.seller_city or "")
            loc_str = f"{city_esc} — " if city_esc else ""
            dist_line = f"\n📍 {loc_str}{dist_esc}"
        elif listing_obj.seller_city:
            dist_line = f"\n📍 {escape_markdown_v2(listing_obj.seller_city)}"

        text = (
            f"🚗 *DEAL FOUND* — Score: {score_esc}/100\n\n"
            f"*{title_esc}*\n"
            f"💰 {price_esc}{usd_part}\n"
            f"📉 {discount_esc}% below market median\n"
            f"🛣️ {km_esc} km | ⚙️ {trans_esc} | ⛽ {fuel_esc}\n"
            f"📅 {year_esc} | 🏷️ {source_esc}{dist_line}\n\n"
            f"{reason_esc}"
        )
        return text

    async def initialize(self) -> None:
        if not self._is_configured():
            logger.warning(
                "Telegram token or chat_id not set — alerts disabled. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable."
            )
            return

        try:
            self.app = (
                Application.builder()
                .token(self.token)
                .build()
            )

            self.app.add_handler(CommandHandler("status", self.status_command))
            self.app.add_handler(CommandHandler("top10", self.top10_command))
            self.app.add_handler(CommandHandler("search", self.search_command))

            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling(drop_pending_updates=True)
            self._initialized = True
            logger.info("Telegram bot initialized and polling")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            self.app = None
            self._initialized = False

    async def shutdown(self) -> None:
        if self.app and self._initialized:
            try:
                await self.app.updater.stop()
                await self.app.stop()
                await self.app.shutdown()
                logger.info("Telegram bot shut down")
            except Exception as e:
                logger.warning(f"Error during Telegram shutdown: {e}")

    async def send_digest(self) -> bool:
        """Send top-10 deals digest as a single Telegram message."""
        if not self._initialized or not self.app:
            return False

        session = SessionLocal()
        try:
            candidates = (
                session.query(Listing)
                .filter(Listing.is_deal == True, Listing.status == "active",
                        Listing.hidden != True, Listing.is_agency != True,
                        Listing.distance_km <= 300)
                .order_by(Listing.score.desc())
                .limit(50)
                .all()
            )
        finally:
            session.close()

        if not candidates:
            logger.info("Telegram digest: no deals to send")
            return False

        # Balanced selection: max 5 pickups, rest regular cars
        _PICKUP_MODELS = {"hilux","ranger","amarok","saveiro","s10","strada",
                          "oroch","frontier","alaskan","triton","f-150","f 150","ram"}
        def _is_pickup(l):
            txt = ((l.model or "") + " " + (l.title or "")).lower()
            return any(p in txt for p in _PICKUP_MODELS)

        pickups = [l for l in candidates if _is_pickup(l)]
        cars    = [l for l in candidates if not _is_pickup(l)]
        top = (pickups[:5] + cars[:5])
        top.sort(key=lambda l: -(l.score or 0))

        from datetime import datetime as _dt
        date_str = _dt.now().strftime("%d/%m/%Y")
        lines = [f"🚗 *Top 10 Deals — {escape_markdown_v2(date_str)}*\n"]
        for i, d in enumerate(top, 1):
            price = f"${d.price_ars:,.0f}" if d.price_ars else "N/A"
            discount = f"{d.discount_pct:.0f}% OFF" if d.discount_pct else ""
            km = f"{d.km:,} km" if d.km else "N/A"
            city = d.seller_city or "—"
            lines.append(
                f"{i}\\. [{escape_markdown_v2(d.title)}]({d.url})\n"
                f"   💰 {escape_markdown_v2(price)} ARS"
                + (f" — {escape_markdown_v2(discount)}" if discount else "")
                + f"\n   🛣️ {escape_markdown_v2(km)} · 📍 {escape_markdown_v2(city)}"
                + f" · Score: {escape_markdown_v2(str(int(d.score)))}\n"
            )

        text = "\n".join(lines)
        try:
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="MarkdownV2",
                disable_web_page_preview=True,
            )
            logger.info(f"Telegram digest sent with {len(top)} deals")
            return True
        except Exception as e:
            logger.error(f"Telegram digest send failed: {e}")
            return False

    async def send_alerts(self, session=None) -> int:
        if not self._initialized or not self.app:
            return 0

        close_session = False
        if session is None:
            session = SessionLocal()
            close_session = True

        sent_count = 0
        try:
            pending = (
                session.query(Listing)
                .filter(Listing.is_deal == True, Listing.alerted == False)
                .all()
            )

            for listing_obj in pending:
                try:
                    text = self.format_message(listing_obj)

                    keyboard = InlineKeyboardMarkup(
                        [[InlineKeyboardButton("Ver publicación", url=listing_obj.url or "https://www.mercadolibre.com.ar")]]
                    )

                    try:
                        await self.app.bot.send_message(
                            chat_id=self.chat_id,
                            text=text,
                            parse_mode="MarkdownV2",
                            reply_markup=keyboard,
                        )
                    except Exception as md_err:
                        logger.warning(f"MarkdownV2 send failed, trying plain text: {md_err}")
                        _score = f"{listing_obj.score:.0f}" if listing_obj.score is not None else "N/A"
                        _price = f"${listing_obj.price_ars:,.0f} ARS" if listing_obj.price_ars else "N/A"
                        _disc = f"{listing_obj.discount_pct:.0f}%" if listing_obj.discount_pct else "N/A"
                        _km = f"{listing_obj.km:,} km" if listing_obj.km else "N/A"
                        plain_text = (
                            f"DEAL FOUND — Score: {_score}/100\n\n"
                            f"{listing_obj.title}\n"
                            f"Price: {_price}\n"
                            f"{_disc} below market\n"
                            f"{_km} | {listing_obj.transmission or 'N/A'} | {listing_obj.fuel or 'N/A'}\n"
                            f"Year: {listing_obj.year or 'N/A'}\n\n"
                            f"{listing_obj.deal_reason or ''}\n\n"
                            f"{listing_obj.url}"
                        )
                        await self.app.bot.send_message(
                            chat_id=self.chat_id,
                            text=plain_text,
                        )

                    listing_obj.alerted = True
                    sent_count += 1

                except Exception as e:
                    logger.error(f"Failed to send alert for {listing_obj.id}: {e}")

            session.commit()

        except Exception as e:
            logger.error(f"Error in send_alerts: {e}")
            session.rollback()
        finally:
            if close_session:
                session.close()

        return sent_count

    async def status_command(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        session = SessionLocal()
        try:
            from sqlalchemy import func
            last_seen = session.query(func.max(Listing.last_seen)).scalar()
            total_listings = session.query(Listing).count()
            total_deals = session.query(Listing).filter(Listing.is_deal == True).count()

            last_scan_str = last_seen.strftime("%Y-%m-%d %H:%M UTC") if last_seen else "Never"
            msg = (
                f"CarDeal AR — Bot Status\n\n"
                f"Last scan: {last_scan_str}\n"
                f"Total listings: {total_listings:,}\n"
                f"Active deals: {total_deals:,}\n"
            )
            await update.message.reply_text(msg)
        except Exception as e:
            await update.message.reply_text(f"Error fetching status: {e}")
        finally:
            session.close()

    async def top10_command(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        session = SessionLocal()
        try:
            candidates = (
                session.query(Listing)
                .filter(Listing.is_deal == True, Listing.status == "active",
                        Listing.hidden != True, Listing.is_agency != True,
                        Listing.distance_km <= 300)
                .order_by(Listing.score.desc())
                .limit(50)
                .all()
            )
            if not candidates:
                await update.message.reply_text("No hay deals dentro de 300 km de Darregueira por ahora.")
                return

            _PICKUP_MODELS = {"hilux","ranger","amarok","saveiro","s10","strada",
                              "oroch","frontier","alaskan","triton","f-150","f 150","ram"}
            def _is_pickup(l):
                txt = ((l.model or "") + " " + (l.title or "")).lower()
                return any(p in txt for p in _PICKUP_MODELS)

            pickups = [l for l in candidates if _is_pickup(l)]
            cars    = [l for l in candidates if not _is_pickup(l)]
            top = sorted(pickups[:5] + cars[:5], key=lambda l: -(l.score or 0))

            lines = ["🚗 Top 10 Deals (≤300 km de Darregueira):\n"]
            for i, listing in enumerate(top, 1):
                price_str = f"${listing.price_ars:,.0f}" if listing.price_ars else "N/A"
                discount_str = f"{listing.discount_pct:.0f}% OFF" if listing.discount_pct else "N/A"
                dist_str = f"{listing.distance_km:.0f} km" if listing.distance_km else "—"
                lines.append(
                    f"{i}. {listing.title}\n"
                    f"   Score: {listing.score:.0f}/100 | {price_str} ARS | {discount_str}\n"
                    f"   📍 {listing.seller_city or '—'} ({dist_str}) | {listing.url}\n"
                )
            await update.message.reply_text("\n".join(lines))
        except Exception as e:
            await update.message.reply_text(f"Error fetching top 10: {e}")
        finally:
            session.close()

    async def search_command(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        args = context.args or []
        if not args:
            await update.message.reply_text(
                "Usage: /search <brand> [model] [year]\nExample: /search Toyota Hilux 2020"
            )
            return

        session = SessionLocal()
        try:
            brand = args[0] if len(args) >= 1 else None
            model = args[1] if len(args) >= 2 else None
            year_str = args[2] if len(args) >= 3 else None

            query = session.query(Listing)
            if brand:
                query = query.filter(Listing.brand.ilike(f"%{brand}%"))
            if model:
                query = query.filter(Listing.model.ilike(f"%{model}%"))
            if year_str:
                try:
                    year = int(year_str)
                    query = query.filter(Listing.year == year)
                except ValueError:
                    pass

            results = query.order_by(Listing.score.desc()).limit(5).all()

            if not results:
                await update.message.reply_text("No listings found matching your search.")
                return

            lines = [f"Search results for '{' '.join(args)}':\n"]
            for listing in results:
                price_str = f"${listing.price_ars:,.0f}" if listing.price_ars else "N/A"
                lines.append(
                    f"• {listing.title}\n"
                    f"  Score: {listing.score:.0f}/100 | {price_str} ARS\n"
                    f"  {listing.url}\n"
                )
            await update.message.reply_text("\n".join(lines))
        except Exception as e:
            await update.message.reply_text(f"Error searching: {e}")
        finally:
            session.close()
