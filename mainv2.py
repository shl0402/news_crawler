"""
=============================================================================
Financial News Crawler - Main Version 2 (mainv2.py)
=============================================================================
Selenium-based parallel scraper for English financial news.
Searches by week ranges for comprehensive date-based coverage.
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote, unquote, parse_qs, urlparse

import yaml
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
import pytz

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options


# =============================================================================
# Configuration & Logging
# =============================================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger('SeleniumCrawler')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# Fast News Searcher (Parallel - English Only)
# =============================================================================

class FastSearcher:
    """Fast parallel news searcher - English only with week-based date-specific searches."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.results_per_week = config.get('search', {}).get('results_per_stock_per_week', 3)
        self.lookback_days = config.get('date_range', {}).get('lookback_days', 30)
        # Calculate total results: results_per_week * number_of_weeks
        self.num_weeks = max(1, self.lookback_days // 7)
        self.total_results = self.results_per_week * self.num_weeks
        # Fetch extra URLs to account for articles that don't meet threshold
        self.fetch_multiplier = 5
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        self.timeout = 10
        self.session = self._create_session()
        
        # Calculate date ranges for week-specific searches
        self.date_ranges = self._calculate_date_ranges()
    
    def _calculate_date_ranges(self) -> List[Dict]:
        """Calculate week-by-week date ranges for targeted searches."""
        ranges = []
        today = datetime.now()
        
        for week in range(self.num_weeks):
            week_end = today - timedelta(days=week * 7)
            week_start = week_end - timedelta(days=7)
            
            # Format for search queries (e.g., "January 2026")
            month_year = week_end.strftime("%B %Y")
            
            ranges.append({
                'start': week_start,
                'end': week_end,
                'month_year': month_year,
                'week_num': week + 1
            })
        
        return ranges
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({'User-Agent': self.user_agent})
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=1)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def search_stock_news(self, stock_code: str, stock_name: str) -> List[str]:
        """Search for English news using parallel requests with date-specific queries."""
        self.logger.info(f"Searching English news for: {stock_name} ({stock_code}) [target: {self.total_results} = {self.results_per_week}/week x {self.num_weeks} weeks]")
        
        start_time = time.time()
        urls = set()
        
        # Build list of search tasks with date-specific queries
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {}
            
            # Standard searches (no date filter)
            futures[executor.submit(self._search_yahoo_rss, stock_code)] = 'Yahoo RSS'
            futures[executor.submit(self._search_google_news_rss, stock_code, stock_name)] = 'Google RSS'
            
            # Date-specific searches for each week in lookback period
            for date_range in self.date_ranges:
                month_year = date_range['month_year']
                futures[executor.submit(self._search_duckduckgo_dated, stock_code, stock_name, month_year)] = f'DDG {month_year}'
                futures[executor.submit(self._search_bing_dated, stock_code, stock_name, month_year)] = f'Bing {month_year}'
            
            # Also do standard searches without dates
            futures[executor.submit(self._search_duckduckgo, stock_code, stock_name)] = 'DuckDuckGo'
            futures[executor.submit(self._search_bing, stock_code, stock_name)] = 'Bing'
            futures[executor.submit(self._search_bing_news, stock_code, stock_name)] = 'Bing News'
            
            for future in as_completed(futures, timeout=25):
                try:
                    result_urls = future.result()
                    if result_urls:
                        urls.update(result_urls)
                except:
                    pass
        
        # Filter and get extra URLs (multiplier) to account for threshold failures
        fetch_count = self.total_results * self.fetch_multiplier
        valid_urls = [url for url in urls if self._is_valid_url(url)][:fetch_count]
        
        elapsed = time.time() - start_time
        self.logger.info(f"Found {len(valid_urls)} candidate URLs in {elapsed:.1f}s (need {self.total_results} valid articles)")
        
        return valid_urls
    
    def _search_duckduckgo_dated(self, stock_code: str, stock_name: str, month_year: str) -> List[str]:
        """Search DuckDuckGo with date-specific query."""
        urls = []
        try:
            query = f"{stock_name} {stock_code} stock news {month_year}"
            search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
            response = self.session.get(search_url, timeout=self.timeout)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.select('a.result__a')[:20]:
                    href = link.get('href', '')
                    if "duckduckgo.com/l/?" in href:
                        parsed = urlparse(href)
                        params = parse_qs(parsed.query)
                        if 'uddg' in params:
                            href = unquote(params['uddg'][0])
                    if href.startswith('http'):
                        urls.append(href)
        except:
            pass
        return urls
    
    def _search_bing_dated(self, stock_code: str, stock_name: str, month_year: str) -> List[str]:
        """Search Bing with date-specific query."""
        urls = []
        try:
            query = f"{stock_name} {stock_code} stock news {month_year}"
            search_url = f"https://www.bing.com/search?q={quote(query)}&setlang=en"
            response = self.session.get(search_url, timeout=self.timeout)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.select('#b_results h2 a')[:20]:
                    href = link.get('href', '')
                    if href.startswith('http') and 'bing.com' not in href and 'microsoft.com' not in href:
                        urls.append(href)
        except:
            pass
        return urls
        
        return valid_urls
    
    def _search_duckduckgo(self, stock_code: str, stock_name: str) -> List[str]:
        urls = []
        try:
            query = f"{stock_name} {stock_code} stock news english"
            search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
            response = self.session.get(search_url, timeout=self.timeout)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.select('a.result__a')[:30]:  # Get more results
                    href = link.get('href', '')
                    if "duckduckgo.com/l/?" in href:
                        parsed = urlparse(href)
                        params = parse_qs(parsed.query)
                        if 'uddg' in params:
                            href = unquote(params['uddg'][0])
                    if href.startswith('http'):
                        urls.append(href)
        except:
            pass
        return urls
    
    def _search_bing(self, stock_code: str, stock_name: str) -> List[str]:
        urls = []
        try:
            query = f"{stock_name} {stock_code} stock news english"
            search_url = f"https://www.bing.com/search?q={quote(query)}&setlang=en"
            response = self.session.get(search_url, timeout=self.timeout)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.select('#b_results h2 a')[:30]:  # Get more results
                    href = link.get('href', '')
                    if href.startswith('http') and 'bing.com' not in href and 'microsoft.com' not in href:
                        urls.append(href)
        except:
            pass
        return urls
    
    def _search_yahoo_rss(self, stock_code: str) -> List[str]:
        urls = []
        try:
            import feedparser
            feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={stock_code}&region=US&lang=en-US"
            response = self.session.get(feed_url, timeout=self.timeout)
            
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                for entry in feed.entries[:20]:  # Get more results
                    url = entry.get('link', '')
                    if url:
                        urls.append(url)
        except:
            pass
        return urls
    
    def _search_google_news_rss(self, stock_code: str, stock_name: str) -> List[str]:
        """Search using Google News RSS - English only."""
        urls = []
        try:
            import feedparser
            query = f"{stock_name} {stock_code} stock english"
            feed_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en&gl=US&ceid=US:en"
            response = self.session.get(feed_url, timeout=self.timeout)
            
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                for entry in feed.entries[:20]:
                    link = entry.get('link', '')
                    # Skip Google redirect URLs that we can't resolve
                    if link and 'news.google.com' not in link:
                        urls.append(link)
        except:
            pass
        return urls
    
    def _search_duckduckgo_finance(self, stock_code: str, stock_name: str) -> List[str]:
        """Search DuckDuckGo with finance-focused query."""
        urls = []
        try:
            query = f"{stock_name} financial analysis earnings report"
            search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
            response = self.session.get(search_url, timeout=self.timeout)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.select('a.result__a')[:30]:
                    href = link.get('href', '')
                    if "duckduckgo.com/l/?" in href:
                        parsed = urlparse(href)
                        params = parse_qs(parsed.query)
                        if 'uddg' in params:
                            href = unquote(params['uddg'][0])
                    if href.startswith('http'):
                        urls.append(href)
        except:
            pass
        return urls
    
    def _search_bing_news(self, stock_code: str, stock_name: str) -> List[str]:
        """Search Bing News specifically."""
        urls = []
        try:
            query = f"{stock_name} {stock_code} stock"
            search_url = f"https://www.bing.com/news/search?q={quote(query)}&setlang=en"
            response = self.session.get(search_url, timeout=self.timeout)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Bing News uses different selectors
                for link in soup.select('a.title')[:30]:
                    href = link.get('href', '')
                    if href.startswith('http') and 'bing.com' not in href and 'microsoft.com' not in href:
                        urls.append(href)
                # Also try news card links
                for link in soup.select('.news-card a')[:30]:
                    href = link.get('href', '')
                    if href.startswith('http') and 'bing.com' not in href:
                        urls.append(href)
        except:
            pass
        return urls
    
    def _is_valid_url(self, url: str) -> bool:
        if not url or not url.startswith('http'):
            return False
        excluded = ['youtube.com', 'twitter.com', 'facebook.com', 'instagram.com',
                   'linkedin.com', 'reddit.com', '.pdf', '.jpg', '.png']
        return not any(ex in url.lower() for ex in excluded)


# =============================================================================
# Parallel Selenium Article Scraper
# =============================================================================

class ParallelSeleniumScraper:
    """
    Parallel article scraper using multiple Selenium WebDrivers.
    Based on scrapfast.py approach but with parallel processing.
    """
    
    USELESS_IMAGE_PATTERNS = [
        'logo', 'icon', 'avatar', 'badge', 'button', 'sprite', 'pixel',
        'tracking', 'ad', 'banner', 'spacer', 'blank', 'transparent',
        'facebook', 'twitter', 'linkedin', 'social', 'share',
        '.svg', 'base64', 'data:image', '1x1', 'placeholder'
    ]
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        # Get parallel drivers from config (default: 3)
        self.num_drivers = config.get('scraping', {}).get('parallel_drivers', 3)
        # Get minimum word count from config (default: 500)
        self.min_words = config.get('scraping', {}).get('min_content_words', 500)
        # Get lookback days for date validation
        self.lookback_days = config.get('date_range', {}).get('lookback_days', 30)
        self.cutoff_date = datetime.now(pytz.UTC) - timedelta(days=self.lookback_days)
        self.drivers = []
        self._setup_drivers()
    
    def _create_driver(self):
        """Create a single Selenium WebDriver."""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-logging')
        chrome_options.add_argument('--log-level=3')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        
        # Disable images for faster loading
        prefs = {"profile.managed_default_content_settings.images": 2}
        chrome_options.add_experimental_option("prefs", prefs)
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(20)
        return driver
    
    def _setup_drivers(self):
        """Setup multiple Selenium WebDrivers for parallel scraping."""
        self.logger.info(f"Setting up {self.num_drivers} Selenium WebDrivers...")
        for i in range(self.num_drivers):
            driver = self._create_driver()
            self.drivers.append(driver)
        self.logger.info(f"{self.num_drivers} Selenium WebDrivers ready")
    
    def scrape_articles_parallel(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape multiple articles in parallel using multiple drivers.
        Each driver gets its own batch of URLs to avoid race conditions.
        """
        if not urls:
            return []
        
        # Split URLs into batches - one batch per driver
        batches = [[] for _ in range(self.num_drivers)]
        for i, url in enumerate(urls):
            batches[i % self.num_drivers].append(url)
        
        all_articles = []
        
        # Each driver processes its own batch sequentially (no race conditions)
        with ThreadPoolExecutor(max_workers=self.num_drivers) as executor:
            futures = []
            for driver_idx, batch in enumerate(batches):
                if batch:  # Only submit if batch has URLs
                    future = executor.submit(
                        self._scrape_batch_with_driver, 
                        batch, 
                        self.drivers[driver_idx],
                        driver_idx
                    )
                    futures.append(future)
            
            # Wait for ALL futures to complete (no timeout that cuts off early)
            for future in futures:
                try:
                    batch_articles = future.result(timeout=300)  # 5 min timeout per batch
                    if batch_articles:
                        all_articles.extend(batch_articles)
                except Exception as e:
                    self.logger.warning(f"Batch failed: {e}")
        
        return all_articles
    
    def _scrape_batch_with_driver(self, urls: List[str], driver, driver_idx: int) -> List[Dict[str, Any]]:
        """Scrape a batch of URLs sequentially with one driver."""
        articles = []
        for url in urls:
            try:
                article = self._scrape_with_driver(url, driver)
                if article:
                    articles.append(article)
            except Exception as e:
                self.logger.debug(f"Driver {driver_idx} failed on {url[:40]}: {e}")
        return articles
    
    def _scrape_with_driver(self, url: str, driver) -> Optional[Dict[str, Any]]:
        """Scrape article using a specific Selenium driver."""
        try:
            start_time = time.time()
            
            # Load page
            driver.get(url)
            
            # Wait for content to load
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except:
                pass
            
            # Extra wait for JS rendering
            time.sleep(1.5)
            
            # Try to find article content using various selectors
            content = self._extract_article_content(driver)
            
            # Check minimum word count (not character count)
            if not content:
                return None
            word_count = len(content.split())
            if word_count < self.min_words:
                return None
            
            # Extract title
            title = self._extract_title(driver)
            
            # Extract date first to validate it's within range
            publish_date = self._extract_date(driver)
            
            # Validate date is within lookback period
            if not self._is_date_valid(publish_date):
                return None
            
            # Extract source
            source = self._extract_source(url)
            
            # Extract images
            images = self._extract_images(driver)
            
            elapsed = time.time() - start_time
            
            word_count = len(content.split())
            
            return {
                'title': title,
                'content': content,
                'publish_date': publish_date,
                'source': source,
                'url': url,
                'images': images,
                'scrape_time': elapsed,
                'content_length': len(content),
                'word_count': word_count
            }
            
        except Exception as e:
            return None
    
    def _is_date_valid(self, publish_date: Optional[datetime]) -> bool:
        """Check if article date is within the lookback period."""
        if publish_date is None:
            return True  # Allow articles without dates (will use current time)
        
        try:
            # Ensure both dates are timezone-aware for comparison
            if publish_date.tzinfo is None:
                publish_date = pytz.UTC.localize(publish_date)
            
            # Article must be after cutoff date (within lookback period)
            # and not in the future
            now = datetime.now(pytz.UTC)
            return self.cutoff_date <= publish_date <= now
        except:
            return True  # Allow on error
    
    def _extract_article_content(self, driver) -> str:
        """Extract article content using multiple selectors."""
        # Priority order of selectors for article content
        selectors = [
            'article',
            '.caas-body',  # Yahoo Finance
            '[data-test-locator="articleBody"]',
            '.article-body',
            '.article-content',
            '.story-body',
            '.post-content',
            'main article',
            '.entry-content',
            '[itemprop="articleBody"]',
            '.content-body',
        ]
        
        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    text = elements[0].text.strip()
                    if len(text) > 200:  # Good enough content
                        return text
            except:
                continue
        
        # Fallback: get main or body text
        try:
            main = driver.find_elements(By.CSS_SELECTOR, 'main')
            if main:
                return main[0].text.strip()
        except:
            pass
        
        try:
            body = driver.find_element(By.TAG_NAME, 'body')
            return body.text.strip()
        except:
            return ""
    
    def _extract_title(self, driver) -> str:
        """Extract article title."""
        selectors = [
            'h1',
            'article h1',
            '.article-title',
            '.headline',
            '[data-test-locator="headline"]',
        ]
        
        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    title = elements[0].text.strip()
                    if 10 < len(title) < 500:
                        return title
            except:
                continue
        
        # Fallback to page title
        try:
            return driver.title
        except:
            return "Untitled"
    
    def _extract_date(self, driver) -> Optional[datetime]:
        """Extract publish date with timezone info (GMT+X format)."""
        parsed_date = None
        
        # Try meta tags first
        try:
            meta_selectors = [
                'meta[property="article:published_time"]',
                'meta[name="pubdate"]',
                'meta[name="date"]',
            ]
            for selector in meta_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    content = elements[0].get_attribute('content')
                    if content:
                        parsed_date = date_parser.parse(content)
                        break
        except:
            pass
        
        # Try time elements
        if not parsed_date:
            try:
                time_elements = driver.find_elements(By.TAG_NAME, 'time')
                if time_elements:
                    datetime_attr = time_elements[0].get_attribute('datetime')
                    if datetime_attr:
                        parsed_date = date_parser.parse(datetime_attr)
            except:
                pass
        
        # Default to current time in UTC
        if not parsed_date:
            parsed_date = datetime.now(pytz.UTC)
        
        # Ensure timezone info exists
        if parsed_date.tzinfo is None:
            # Assume UTC if no timezone
            parsed_date = pytz.UTC.localize(parsed_date)
        
        return parsed_date
    
    def _format_timezone(self, dt: datetime) -> str:
        """Format datetime with GMT+X timezone string."""
        if dt is None:
            return "Unknown"
        
        if dt.tzinfo is None:
            return dt.strftime('%Y-%m-%d %H:%M:%S') + " (GMT+0)"
        
        # Get UTC offset in hours
        offset = dt.utcoffset()
        if offset:
            total_seconds = offset.total_seconds()
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            if minutes:
                tz_str = f"GMT{hours:+d}:{minutes:02d}"
            else:
                tz_str = f"GMT{hours:+d}"
        else:
            tz_str = "GMT+0"
        
        return dt.strftime('%Y-%m-%d %H:%M:%S') + f" ({tz_str})"
    
    def _extract_source(self, url: str) -> str:
        """Extract news source from URL."""
        domain = urlparse(url).netloc.lower().replace('www.', '')
        
        source_map = {
            'yahoo': 'YAHOO', 'reuters': 'REUTERS', 'bloomberg': 'BLOOMBERG',
            'cnbc': 'CNBC', 'cnn': 'CNN', 'bbc': 'BBC', 'fool': 'FOOL',
            'seekingalpha': 'SEEKINGALPHA', 'marketwatch': 'MARKETWATCH',
            'scmp': 'SCMP', 'etnet': 'ETNET', 'aastocks': 'AASTOCKS',
            'investing': 'INVESTING', 'zacks': 'ZACKS'
        }
        
        for key, value in source_map.items():
            if key in domain:
                return value
        
        return domain.split('.')[0].upper()[:10]
    
    def _extract_images(self, driver) -> List[str]:
        """Extract useful images."""
        images = []
        try:
            img_elements = driver.find_elements(By.TAG_NAME, 'img')[:30]
            for img in img_elements:
                src = img.get_attribute('src') or img.get_attribute('data-src')
                if src and self._is_useful_image(src):
                    if src.startswith('//'):
                        src = 'https:' + src
                    if src.startswith('http') and src not in images:
                        images.append(src)
        except:
            pass
        return images[:15]
    
    def _is_useful_image(self, url: str) -> bool:
        url_lower = url.lower()
        return not any(p in url_lower for p in self.USELESS_IMAGE_PATTERNS)
    
    def close(self):
        """Close all WebDrivers."""
        for driver in self.drivers:
            try:
                driver.quit()
            except:
                pass
        self.drivers = []


# =============================================================================
# Data Saver
# =============================================================================

class DataSaver:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.get('output', {}).get('directory', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Use config base filename
        self.base_filename = config.get('output', {}).get('filename', 'financial_news_data')
    
    def save(self, records: List[Dict], output_format: str = 'both'):
        if not records:
            self.logger.warning("No records to save")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{self.base_filename}_{timestamp}"
        
        df = pd.DataFrame(records)
        
        if output_format in ['excel', 'both']:
            self._save_excel(df, base_name)
        
        if output_format in ['jsonl', 'both']:
            self._save_jsonl(records, base_name)
    
    def _save_excel(self, df: pd.DataFrame, base_name: str):
        try:
            for col in df.columns:
                df[col] = df[col].apply(lambda x: '|'.join(x) if isinstance(x, list) else x)
                try:
                    df[col] = df[col].apply(lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo else x)
                except:
                    pass
            
            filepath = self.output_dir / f"{base_name}.xlsx"
            df.to_excel(filepath, index=False, engine='openpyxl')
            self.logger.info(f"Excel saved: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save Excel: {e}")
    
    def _save_jsonl(self, records: List[Dict], base_name: str):
        try:
            filepath = self.output_dir / f"{base_name}.jsonl"
            with open(filepath, 'w', encoding='utf-8') as f:
                for record in records:
                    clean_record = {}
                    for k, v in record.items():
                        if isinstance(v, (datetime, pd.Timestamp)):
                            clean_record[k] = v.isoformat()
                        else:
                            clean_record[k] = v
                    f.write(json.dumps(clean_record, ensure_ascii=False) + '\n')
            self.logger.info(f"JSONL saved: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save JSONL: {e}")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline():
    """Run the Selenium-based English news crawler pipeline."""
    
    config = load_config()
    logger = setup_logging()
    
    # Get config values for logging
    results_per_week = config.get('search', {}).get('results_per_stock_per_week', 3)
    lookback_days = config.get('date_range', {}).get('lookback_days', 30)
    num_weeks = max(1, lookback_days // 7)
    target_per_stock = results_per_week * num_weeks
    min_words = config.get('scraping', {}).get('min_content_words', 500)
    parallel_drivers = config.get('scraping', {}).get('parallel_drivers', 3)
    
    logger.info("=" * 60)
    logger.info("Financial News Crawler v2 (mainv2.py)")
    logger.info("English news only | Week-based date ranges")
    logger.info(f"Settings: {results_per_week} articles/week x {num_weeks} weeks = {target_per_stock} target/stock")
    logger.info(f"Min words: {min_words} | Parallel drivers: {parallel_drivers}")
    logger.info("=" * 60)
    
    total_start = time.time()
    
    # Initialize components (no price fetcher)
    searcher = FastSearcher(config, logger)
    scraper = ParallelSeleniumScraper(config, logger)
    saver = DataSaver(config, logger)
    
    # Get config
    stocks = config.get('stocks', {})
    output_format = config.get('output', {}).get('format', 'both')
    
    all_records = []
    content_lengths = []
    
    try:
        for category, stock_list in stocks.items():
            logger.info(f"\n--- Category: {category} ---")
            
            for stock in stock_list:
                stock_code = stock['code']
                stock_name = stock['name']
                
                logger.info(f"\nProcessing: {stock_name} ({stock_code})")
                
                # Search for news (FastSearcher handles the week-based limit)
                urls = searcher.search_stock_news(stock_code, stock_name)
                
                if not urls:
                    logger.warning(f"No URLs found for {stock_name}")
                    continue
                
                logger.info(f"  Scraping {len(urls)} candidate URLs (need {target_per_stock} valid)...")
                
                # Scrape all URLs in PARALLEL using multiple Selenium drivers
                articles = scraper.scrape_articles_parallel(urls)
                
                # Track seen titles to avoid duplicates (normalize for comparison)
                seen_titles = set()
                
                # Only keep up to target_per_stock valid articles
                stock_records = []
                for article in articles:
                    if len(stock_records) >= target_per_stock:
                        break  # We have enough articles for this stock
                        
                    if article and article.get('title') and article.get('content'):
                        # Normalize title for duplicate check (lowercase, strip whitespace)
                        title_normalized = article['title'].lower().strip()
                        
                        # Skip if we've seen this title before
                        if title_normalized in seen_titles:
                            logger.debug(f"  ✗ Duplicate title skipped: {article['title'][:40]}...")
                            continue
                        
                        seen_titles.add(title_normalized)
                        
                        content_len = article.get('content_length', 0)
                        word_count = article.get('word_count', 0)
                        content_lengths.append(content_len)
                        
                        # Format date with timezone
                        news_date = article.get('publish_date')
                        news_date_formatted = scraper._format_timezone(news_date)
                        
                        # Build record (no pricing fields - friend will handle that)
                        record = {
                            'Stock_Code': stock_code,
                            'Category': category,
                            'News_Date': news_date,  # Keep original datetime for friend's pricing work
                            'News_Date_Formatted': news_date_formatted,  # Human-readable with GMT+X
                            'News_Title': article.get('title', '')[:500],
                            'News_Source': article.get('source', 'UNKNOWN'),
                            'News_Content': article.get('content', '')[:5000],
                            'Content_Length': content_len,
                            'Word_Count': word_count,
                            'Image_URLs': article.get('images', []),
                            'News_URL': article.get('url', ''),
                            'Scrape_Time_Sec': article.get('scrape_time', 0)
                        }
                        
                        stock_records.append(record)
                        logger.info(f"  ✓ [{len(stock_records)}/{target_per_stock}] {article['title'][:35]}... ({word_count} words)")
                
                all_records.extend(stock_records)
                logger.info(f"  Got {len(stock_records)}/{target_per_stock} valid articles for {stock_name}")
    
    finally:
        scraper.close()
    
    total_elapsed = time.time() - total_start
    
    # Save results
    logger.info("\n" + "=" * 60)
    logger.info(f"Pipeline complete. Total records: {len(all_records)}")
    logger.info("=" * 60)
    
    if all_records:
        saver.save(all_records, output_format)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CRAWLER SUMMARY")
    print("=" * 60)
    print(f"Total articles collected: {len(all_records)}")
    print(f"Total time: {total_elapsed:.1f} seconds")
    
    if content_lengths:
        avg_content = sum(content_lengths) / len(content_lengths)
        print(f"\nContent Quality:")
        print(f"  Average content length: {avg_content:.0f} chars")
        print(f"  Min content length: {min(content_lengths)} chars")
        print(f"  Max content length: {max(content_lengths)} chars")
    
    if all_records:
        df = pd.DataFrame(all_records)
        print(f"\nArticles by Source:\n{df['News_Source'].value_counts()}")
        
        if 'Scrape_Time_Sec' in df.columns:
            avg_scrape = df['Scrape_Time_Sec'].mean()
            print(f"\nAverage scrape time: {avg_scrape:.2f}s per article")
    
    # Show sample content
    if all_records:
        print("\n" + "=" * 60)
        print("SAMPLE CONTENT (first article):")
        print("=" * 60)
        sample = all_records[0]
        print(f"Title: {sample['News_Title']}")
        print(f"Date: {sample['News_Date']}")
        print(f"Source: {sample['News_Source']}")
        print(f"Content preview ({sample['Content_Length']} chars):")
        print("-" * 40)
        print(sample['News_Content'][:800])
        print("-" * 40)
    
    print("=" * 60)
    
    return all_records


if __name__ == "__main__":
    run_pipeline()
