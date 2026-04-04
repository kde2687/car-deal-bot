"""
Daily email digest — top 10 deals sent at 5am Argentina time.
Uses Gmail SMTP with an app password (no extra dependencies).
"""
import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from database import SessionLocal, Listing

logger = logging.getLogger(__name__)


def _format_price(price_ars, price_usd):
    if price_ars and price_ars > 0:
        if price_ars >= 1_000_000:
            return f"${price_ars / 1_000_000:.1f}M ARS"
        return f"${price_ars:,.0f} ARS"
    if price_usd and price_usd > 0:
        return f"USD {price_usd:,.0f}"
    return "N/A"


def _build_html(deals: list, dashboard_url: str = "https://cardeal.ar") -> str:
    date_str = datetime.now().strftime("%d/%m/%Y")
    rows = ""
    for i, d in enumerate(deals, 1):
        price = _format_price(d.price_ars, d.price_usd)
        km = f"{d.km:,} km" if d.km else "N/A"
        year = d.year or "N/A"
        city = d.seller_city or "—"
        dist = f"{d.distance_km:.0f} km" if d.distance_km else "—"
        discount = f"{d.discount_pct:.0f}% OFF" if d.discount_pct and d.discount_pct > 0 else ""
        source_label = {"mercadolibre": "ML", "kavak": "Kavak", "autocosmos": "Autocosmos"}.get(d.source, d.source)
        rows += f"""
        <tr style="border-bottom:1px solid #eee;">
          <td style="padding:10px 8px;font-weight:bold;color:#666;">#{i}</td>
          <td style="padding:10px 8px;">
            <a href="{d.url}" style="color:#1a73e8;text-decoration:none;font-weight:600;">{d.title}</a>
            <div style="color:#888;font-size:12px;margin-top:2px;">{year} · {km} · {city} ({dist} de Darregueira)</div>
          </td>
          <td style="padding:10px 8px;white-space:nowrap;">
            <span style="font-size:16px;font-weight:bold;color:#222;">{price}</span>
            {"<br><span style='color:#28a745;font-size:12px;font-weight:bold;'>" + discount + "</span>" if discount else ""}
          </td>
          <td style="padding:10px 8px;text-align:center;">
            <span style="background:#{'ffc107' if source_label=='ML' else '17a2b8'};color:{'#333' if source_label=='ML' else '#fff'};padding:2px 8px;border-radius:12px;font-size:12px;">{source_label}</span>
          </td>
          <td style="padding:10px 8px;text-align:center;font-weight:bold;color:#1a73e8;">{d.score:.0f}</td>
        </tr>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family:Arial,sans-serif;max-width:700px;margin:0 auto;padding:20px;color:#333;">
      <div style="background:#1a73e8;color:white;padding:20px;border-radius:8px 8px 0 0;">
        <h1 style="margin:0;font-size:22px;">🚗 CarDeal AR — Top 10 Deals</h1>
        <p style="margin:6px 0 0;opacity:0.85;">{date_str}</p>
      </div>
      <table style="width:100%;border-collapse:collapse;background:#fff;border:1px solid #eee;border-top:none;">
        <thead>
          <tr style="background:#f8f9fa;color:#666;font-size:12px;text-transform:uppercase;">
            <th style="padding:8px;">#</th>
            <th style="padding:8px;text-align:left;">Vehículo</th>
            <th style="padding:8px;text-align:left;">Precio</th>
            <th style="padding:8px;">Fuente</th>
            <th style="padding:8px;">Score</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      <div style="background:#f8f9fa;padding:12px 16px;border:1px solid #eee;border-top:none;border-radius:0 0 8px 8px;font-size:12px;color:#888;">
        Ver todos los deals: <a href="{dashboard_url}" style="color:#1a73e8;">CarDeal AR Dashboard</a>
      </div>
    </body>
    </html>
    """


def send_daily_digest(smtp_user: str, smtp_password: str, recipient: str) -> bool:
    """Fetch top 10 deals and send the digest email. Returns True on success."""
    if not smtp_user or not smtp_password:
        logger.warning("Email digest: SMTP credentials not configured, skipping")
        return False

    session = SessionLocal()
    try:
        deals = (
            session.query(Listing)
            .filter(
                Listing.is_deal == True,
                Listing.status == "active",
                Listing.hidden != True,
                Listing.is_agency != True,
            )
            .order_by(Listing.score.desc())
            .limit(10)
            .all()
        )
    finally:
        session.close()

    if not deals:
        logger.info("Email digest: no deals found, skipping")
        return False

    import config as cfg
    html = _build_html(deals, dashboard_url=cfg.DASHBOARD_URL)
    date_str = datetime.now().strftime("%d/%m/%Y")

    recipients = [r.strip() for r in recipient.split(",") if r.strip()]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🚗 CarDeal AR — Top 10 Deals del {date_str}"
    msg["From"] = smtp_user
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html, "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, recipients, msg.as_string())
        logger.info(f"Email digest sent to {recipients} with {len(deals)} deals")
        return True
    except Exception as e:
        logger.error(f"Email digest send failed: {e}")
        return False
