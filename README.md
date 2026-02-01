# Financial News Crawler & Data Pipeline

A robust Python crawler for collecting HK stock news and correlating with stock prices for sentiment analysis and ontology construction.

## Quick Start

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run the crawler
python main.py
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        config.yaml                               │
│         (stocks, date range, delays, output settings)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      NewsSearcher                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Google News  │→ │Yahoo Finance │→ │  DuckDuckGo  │→ Google   │
│  │     RSS      │  │     RSS      │  │    HTML      │  Search   │
│  └──────────────┘  └──────────────┘  └──────────────┘  (fallback)│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ArticleScraper                               │
│            newspaper3k + BeautifulSoup + tenacity                │
│         (extracts: title, content, date, images)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    StockPriceFetcher                             │
│                   yfinance (hourly data)                         │
│         (matches news timestamp → stock price)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DataSaver                                 │
│              Excel (.xlsx) + JSONL output                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## News Search Fallback Mechanism

The crawler uses **4 search methods** in sequence, stopping early if minimum results are reached:

| Priority | Source | Method | Reliability |
|----------|--------|--------|-------------|
| 1 | **Google News RSS** | RSS feed parsing | High - structured data |
| 2 | **Yahoo Finance RSS** | Stock-specific feed | Medium - stock-focused |
| 3 | **DuckDuckGo HTML** | Web scraping | High - no API limits |
| 4 | **Google Search** | `googlesearch-python` | Low - rate limited |

**Logic:**
```
1. Try Google News RSS → collect URLs
2. If count < min_results → try Yahoo Finance RSS
3. If count < min_results → try DuckDuckGo
4. If count < min_results → try Google Search (last resort)
5. Return all valid URLs found
```

---

## Image Extraction

**Problem:** `newspaper3k` often returns useless images (logos, icons, tracking pixels).

**Solution:** Multi-source extraction with smart filtering:

1. **Collect from multiple sources:**
   - `article.top_image` (newspaper3k)
   - `article.images` (all detected)
   - HTML parsing (`<img>`, `og:image`, `twitter:image`, `srcset`)

2. **Filter out useless images:**
   - Logo/icon patterns: `logo`, `icon`, `avatar`, `favicon`
   - Tracking pixels: `pixel`, `beacon`, `1x1`, `analytics`
   - Social media: `twitter`, `facebook`, `gravatar`
   - Bad extensions: `.svg`, `.ico`

3. **Output:** List of all valid image URLs per article

---

## Stock Price Collection

**Source:** `yfinance` library (free Yahoo Finance data)

**Process:**
1. **Preload:** Fetch hourly price data for lookback period
2. **Match:** Find closest price to news publish timestamp
3. **Calculate:** Open, Close, and % change at that timeframe

**Data cached** per stock to avoid repeated API calls.

---

## Error Handling

| Component | Error Type | Handling |
|-----------|-----------|----------|
| **Search** | Network timeout | Catches exception, tries next source |
| **Search** | Rate limit (Google) | Falls back to DuckDuckGo/RSS |
| **Scraper** | 403 Forbidden | Logs error, skips article |
| **Scraper** | Paywall/parse fail | Logs warning, skips article |
| **Scraper** | Missing title/content | Validates, returns None |
| **Price** | No data found | Returns None values |
| **Price** | Timezone mismatch | Auto-converts to match |
| **Excel** | Timezone datetime | Strips timezone for compatibility |

**Retry:** Article scraping uses `tenacity` with exponential backoff (2 attempts).

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `newspaper3k` | Article extraction (title, text, images) |
| `yfinance` | Stock price data from Yahoo Finance |
| `beautifulsoup4` | HTML parsing for images & dates |
| `feedparser` | RSS feed parsing |
| `googlesearch-python` | Google search fallback |
| `requests` | HTTP requests |
| `pandas` | Data processing |
| `openpyxl` | Excel output |
| `tenacity` | Retry with backoff |
| `PyYAML` | Config file parsing |

---

## Output Fields

| Field | Description |
|-------|-------------|
| `Stock_Code` | e.g., "0700.HK" |
| `Category` | e.g., "Tech", "Bank" |
| `News_Date` | Publication timestamp |
| `News_Title` | Article headline |
| `News_Source` | Domain name (e.g., "YAHOO") |
| `News_Content` | Full article text |
| `Image_URLs` | List of content images (filtered) |
| `Price_At_News_Time` | Close price at news time |
| `Price_Open` | Open price at that hour |
| `Price_Close` | Close price at that hour |
| `Price_Change_Percent` | % change (open→close) |
| `Price_Timestamp` | Actual matched price time |
| `News_URL` | Source article URL |

---

## Configuration

Edit `config.yaml` to customize:

```yaml
search:
  min_results_per_stock: 5  # Keep searching until this many found

request:
  delay_seconds: 2          # Delay between requests (avoid bans)
  timeout: 30               # Request timeout

output:
  format: "both"            # "excel", "jsonl", or "both"
```

---

## File Structure

```
newscrawler/
├── config.yaml       # Configuration
├── main.py           # Main pipeline
├── requirements.txt  # Dependencies
├── crawler.log       # Execution log
├── README.md         # This file
├── venv/             # Virtual environment
└── output/           # Generated files
    ├── financial_news_data_*.xlsx
    └── financial_news_data_*.jsonl
```
