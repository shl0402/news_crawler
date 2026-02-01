# Financial News Crawler Evolution Report

## Project Overview
A robust Financial News Crawler & Data Pipeline for HK stocks (BYD, Tencent, Alibaba, HSBC, SMIC) that correlates news articles with stock prices.

---

## Implementation Comparison

| Approach | File | Time | Articles | Avg Content | Content Quality |
|----------|------|------|----------|-------------|-----------------|
| **newspaper3k** | main.py | ~8 min | 5 | ~50 chars | ❌ Poor (JS pages fail) |
| **LangChain Sequential** | test.py | ~2.4 min | ~15 | ~50 chars | ❌ Poor (static HTML only) |
| **LangChain Parallel** | test2.py | ~44 sec | ~15 | ~50 chars | ❌ Poor (same issue) |
| **Selenium Parallel** | test3.py | **190 sec** | **100** | **1,200 chars** | ✅ Excellent |

---

## Key Findings

### 1. The Content Quality Problem
LangChain's `WebBaseLoader` and `newspaper3k` cannot render JavaScript. Modern news sites (Yahoo Finance, etc.) load article content dynamically via JS. Result:
- **LangChain/newspaper3k**: Gets menus, navigation, headers (~50 chars of garbage)
- **Selenium**: Gets actual article content (avg 1,200+ chars of real text)

### 2. Speed vs Quality Trade-off
| Metric | LangChain | Selenium |
|--------|-----------|----------|
| Speed per article | ~0.3s | ~5s |
| Content retrieved | Empty/garbage | Full article |
| JS rendering | ❌ No | ✅ Yes |

### 3. Parallel Scraping Impact
**test3.py** now uses **3 parallel Selenium WebDrivers** to scrape simultaneously:
- Single driver: ~4-5 sec/article × 100 articles = ~8-10 minutes
- 3 parallel drivers: 190 seconds total = **~63% faster**

---

## Architecture (test3.py)

```
┌─────────────────────────────────────────────────────────────┐
│                    FastSearcher                              │
│  ┌──────────┐  ┌──────┐  ┌───────────┐  ┌─────────────────┐ │
│  │ DuckDuck │  │ Bing │  │ Yahoo RSS │  │ Google News RSS │ │
│  └────┬─────┘  └──┬───┘  └─────┬─────┘  └────────┬────────┘ │
│       └───────────┴────────────┴─────────────────┘          │
│                          ↓ ThreadPoolExecutor               │
│                    Deduplicated URLs                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              ParallelSeleniumScraper (3 drivers)            │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐               │
│  │ Driver #1 │  │ Driver #2 │  │ Driver #3 │               │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘               │
│        └──────────────┴──────────────┘                      │
│                   ThreadPoolExecutor                        │
│                   JS-rendered content                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     PriceFetcher                             │
│              yfinance with parallel preloading              │
│              Correlates news with stock prices              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      DataSaver                               │
│              Excel (.xlsx) + JSONL output                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Final Results (test3.py)

```
Total articles collected: 100
Total time: 190.0 seconds
Average content length: 1,200 chars
Min content length: 219 chars
Max content length: 5,483 chars
Average scrape time: 4.98s per article
```

---

## Recommendation

**Use `test3.py`** (Selenium with parallel scraping) for production:
- ✅ Gets real article content from JS-heavy sites
- ✅ Respects `config.yaml` settings (20 results per stock)
- ✅ 3x faster than sequential Selenium
- ✅ Comprehensive stock price correlation via yfinance
- ✅ Multi-source search (DuckDuckGo, Bing, Yahoo RSS, Google News RSS)

---

## Files Summary

| File | Purpose | Status |
|------|---------|--------|
| `main.py` | Original newspaper3k implementation | ⚠️ Slow, JS issues |
| `test.py` | LangChain test (sequential) | ❌ Content quality issue |
| `test2.py` | LangChain test (parallel search) | ❌ Same content issue |
| `test3.py` | **Selenium parallel (PRODUCTION)** | ✅ Recommended |
| `config.yaml` | Configuration (stocks, limits) | ✅ Working |
