# CarDeal AR — Car Deal Hunting Bot

Automated bot that scrapes MercadoLibre and Kavak Argentina for used car deals, scores them against market median prices, and sends Telegram alerts. Includes a web dashboard.

---

## Prerequisites

- Python 3.11 or newer
- pip
- A Telegram bot token (optional, for alerts)
- Internet access

---

## Getting a Telegram Bot Token

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts
3. Copy the token BotFather gives you (format: `123456789:ABCdef...`)
4. Set it as `TELEGRAM_BOT_TOKEN` in your `.env`

## Getting Your Telegram Chat ID

**Option A (easiest):**
1. Start your bot (send it any message after the step above)
2. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser
3. Find `"chat":{"id":XXXXXXXXX}` in the JSON — that number is your chat ID

**Option B:**
1. Add `@userinfobot` to Telegram and send `/start`
2. It replies with your numeric ID

Set it as `TELEGRAM_CHAT_ID` in your `.env`.

---

## Installation

```bash
# 1. Enter the project directory
cd /Users/D.Kinderknecht_1/car_deal_bot

# 2. (Recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and edit the environment file
cp .env.example .env
nano .env   # or open with any editor
```

Edit `.env` with your values:

```
TELEGRAM_BOT_TOKEN=123456789:ABCdef...
TELEGRAM_CHAT_ID=987654321
DEAL_SCORE_THRESHOLD=25
SCAN_INTERVAL_MINUTES=30
MIN_YEAR=2010
MAX_KM=200000
BRANDS=Toyota,Ford,Volkswagen,Chevrolet,Renault,Peugeot
DATABASE_URL=sqlite:///deals.db
FLASK_SECRET_KEY=some-random-secret
LOG_LEVEL=INFO
```

---

## Running

```bash
python main.py
```

On first run, the bot will:
1. Initialize the SQLite database (`deals.db`)
2. Start the web dashboard on port 5000
3. Initialize the Telegram bot (if configured)
4. Run the first scan immediately
5. Schedule future scans every `SCAN_INTERVAL_MINUTES` minutes

---

## Dashboard

Open **http://localhost:5000** in your browser.

- `/` — Active deals sorted by score
- `/all` — All scraped listings
- `/listing/<id>` — Detail page for a single listing
- `/api/deals` — JSON API, top 50 deals

---

## Finding the Kavak API Endpoint

Kavak's internal API endpoints change occasionally. To find the current one:

1. Open Chrome / Firefox and navigate to `https://www.kavak.com/ar/seminuevos`
2. Open DevTools → **Network** tab → filter by **Fetch/XHR**
3. Refresh the page and look for requests to `/api/cars`, `/api/v2/cars`, or similar
4. Copy the full request URL and parameters
5. Update `KavakScraper.API_URL` in `scrapers/kavak.py` if needed

The scraper falls back to parsing the `__NEXT_DATA__` JSON embedded in the HTML if the API returns 403/404.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — | Bot token from BotFather |
| `TELEGRAM_CHAT_ID` | — | Your Telegram numeric chat ID |
| `DEAL_SCORE_THRESHOLD` | 25 | Minimum score (0-100) to flag as deal |
| `SCAN_INTERVAL_MINUTES` | 30 | How often to scan (minutes) |
| `MIN_YEAR` | 2010 | Ignore cars older than this year |
| `MAX_KM` | 200000 | Ignore cars with more km than this |
| `BRANDS` | Toyota,Ford,... | Comma-separated brands to search |
| `DATABASE_URL` | sqlite:///deals.db | SQLAlchemy database URL |
| `FLASK_SECRET_KEY` | changeme | Flask session secret (change in prod) |
| `LOG_LEVEL` | INFO | Logging level (DEBUG/INFO/WARNING/ERROR) |

---

## Telegram Bot Commands

Once running, your bot supports:

- `/status` — Shows last scan time, total listings, active deals
- `/top10` — Lists the top 10 deals by score
- `/search Toyota Hilux 2020` — Search listings by brand/model/year

---

## Scoring System

Each listing is scored 0–100:

- **Base score**: percentage below market median (capped at 100)
- **KM modifier**: < 50K km = +10; > 150K km = -15; > 200K km = -25
- **Age modifier**: >= 2020 = +5; < 2010 = -10
- **Source modifier**: Kavak = +5 (certified dealer)
- **Condition modifier**: New = +5

A listing is marked as a **deal** when:
- Score >= `DEAL_SCORE_THRESHOLD` (default 25) AND
- Price is at least 10% below market median

---

## First Run Notes

- On first run, there is no market reference data yet. The scorer uses the average of all scraped prices for that brand as a cold-start estimate. Market references improve after 2–3 scan cycles.
- Kavak may return 0 results if their API changes. Check logs for warnings and update the endpoint if needed.
- The bot respects rate limits with exponential backoff (MercadoLibre) and random delays (Kavak).
- Logs are stored in `logs/bot.log` with rotation at 10MB, keeping 3 backups.

---

## Project Structure

```
car_deal_bot/
├── main.py                  Entry point
├── config.py                Configuration loader
├── database.py              SQLAlchemy models
├── scorer.py                Deal scoring engine
├── scrapers/
│   ├── mercadolibre.py      MercadoLibre scraper
│   └── kavak.py             Kavak scraper
├── alerts/
│   └── telegram.py          Telegram bot + alerts
├── dashboard/
│   ├── app.py               Flask web app
│   └── templates/
│       ├── index.html        Deals dashboard
│       └── detail.html       Listing detail page
├── requirements.txt
├── .env.example
└── logs/                    Auto-created on first run
```
