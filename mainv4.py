"""
=============================================================================
Financial News Crawler - Main Version 4 (mainv4.py)
=============================================================================
Production version with:
- Graceful shutdown handling (Ctrl+C)
- Auto-save progress after each stock
- Resume capability from progress file
- Failure report for stocks with insufficient data
- Crash recovery with incremental saves
"""

import os
import json
import logging
import time
import random
import sys
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from urllib.parse import quote, urlparse

import yaml
import pandas as pd
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
    logger = logging.getLogger('NewsCrawlerV4')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# Progress Tracker - Handles resume and incremental saves
# =============================================================================

class ProgressTracker:
    """Tracks progress, enables resume, and handles incremental saves."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress file to track what's been scraped
        self.progress_file = self.output_dir / "crawl_progress.json"
        # Failure report file
        self.failure_file = self.output_dir / "crawl_failures.txt"
        # Current session data file (incremental save)
        self.session_file = self.output_dir / "current_session.jsonl"
        
        self.progress = self._load_progress()
        self.failures = []
        self.session_records = []
        
    def _load_progress(self) -> Dict:
        """Load progress from file if exists."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            'completed_stocks': [],  # List of completed stock codes
            'current_stock': None,
            'current_period': 0,
            'total_articles': 0,
            'last_updated': None,
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
    
    def save_progress(self):
        """Save progress to file."""
        self.progress['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2)
    
    def is_stock_completed(self, stock_code: str) -> bool:
        """Check if a stock has been fully scraped."""
        return stock_code in self.progress.get('completed_stocks', [])
    
    def mark_stock_started(self, stock_code: str):
        """Mark a stock as currently being processed."""
        self.progress['current_stock'] = stock_code
        self.progress['current_period'] = 0
        self.save_progress()
    
    def mark_period_completed(self, stock_code: str, period_num: int):
        """Mark a period as completed."""
        self.progress['current_period'] = period_num
        self.save_progress()
    
    def mark_stock_completed(self, stock_code: str, article_count: int):
        """Mark a stock as fully scraped."""
        if stock_code not in self.progress['completed_stocks']:
            self.progress['completed_stocks'].append(stock_code)
        # Remove from blocked list if it was there
        if stock_code in self.progress.get('blocked_stocks', []):
            self.progress['blocked_stocks'].remove(stock_code)
        self.progress['current_stock'] = None
        self.progress['current_period'] = 0
        self.progress['total_articles'] += article_count
        self.save_progress()
    
    def mark_stock_blocked(self, stock_code: str):
        """Mark a stock as possibly blocked (will retry on next run)."""
        if 'blocked_stocks' not in self.progress:
            self.progress['blocked_stocks'] = []
        if stock_code not in self.progress['blocked_stocks']:
            self.progress['blocked_stocks'].append(stock_code)
        # Don't add to completed_stocks so it will be retried
        self.progress['current_stock'] = None
        self.progress['current_period'] = 0
        self.save_progress()
    
    def is_stock_blocked(self, stock_code: str) -> bool:
        """Check if a stock was marked as blocked."""
        return stock_code in self.progress.get('blocked_stocks', [])
    
    def add_failure(self, stock_code: str, stock_name: str, expected: int, actual: int, details: str = ""):
        """Record a failure (stock didn't get enough articles)."""
        failure = {
            'timestamp': datetime.now().isoformat(),
            'stock_code': stock_code,
            'stock_name': stock_name,
            'expected_articles': expected,
            'actual_articles': actual,
            'details': details
        }
        self.failures.append(failure)
        
        # Append to failure file immediately
        with open(self.failure_file, 'a', encoding='utf-8') as f:
            f.write(f"[{failure['timestamp']}] {stock_code} ({stock_name}): Got {actual}/{expected} articles. {details}\n")
    
    def add_record(self, record: Dict):
        """Add a record and save incrementally."""
        self.session_records.append(record)
        
        # Append to session file (JSONL format for crash recovery)
        with open(self.session_file, 'a', encoding='utf-8') as f:
            clean_record = {}
            for k, v in record.items():
                if isinstance(v, (datetime, pd.Timestamp)):
                    clean_record[k] = v.isoformat() if v else None
                elif isinstance(v, list):
                    clean_record[k] = '|'.join(v) if v else ''
                else:
                    clean_record[k] = v
            f.write(json.dumps(clean_record, ensure_ascii=False) + '\n')
    
    def save_stock_checkpoint(self, stock_code: str, stock_name: str, records: List[Dict], base_filename: str):
        """Save checkpoint after completing a stock."""
        if not records:
            return
        
        # Save stock-specific file with stock code and name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Clean stock name for filename (remove special chars)
        clean_name = ''.join(c for c in stock_name if c.isalnum() or c in ' -_').strip().replace(' ', '_')
        stock_file = self.output_dir / f"checkpoint_{stock_code.replace('.', '_')}_{clean_name}_{timestamp}.xlsx"
        
        try:
            df = pd.DataFrame(records)
            for col in df.columns:
                df[col] = df[col].apply(lambda x: '|'.join(x) if isinstance(x, list) else x)
                try:
                    df[col] = df[col].apply(lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo else x)
                except:
                    pass
            
            df.to_excel(stock_file, index=False, engine='openpyxl')
            return stock_file
        except Exception as e:
            return None
    
    def get_session_records(self) -> List[Dict]:
        """Get all records from current session."""
        return self.session_records
    
    def clear_session(self):
        """Clear session data after successful completion."""
        self.progress = {
            'completed_stocks': [],
            'current_stock': None,
            'current_period': 0,
            'total_articles': 0,
            'last_updated': None,
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        self.save_progress()
        
        # Clear session file
        if self.session_file.exists():
            self.session_file.unlink()
    
    def get_summary(self) -> str:
        """Get progress summary."""
        return (
            f"Completed stocks: {len(self.progress.get('completed_stocks', []))}\n"
            f"Total articles: {self.progress.get('total_articles', 0)}\n"
            f"Failures: {len(self.failures)}"
        )


# =============================================================================
# Crawler with Shutdown Handling
# =============================================================================

class RobustCrawler:
    """
    Production crawler with graceful shutdown and crash recovery.
    """
    
    USELESS_IMAGE_PATTERNS = [
        'logo', 'icon', 'avatar', 'badge', 'button', 'sprite', 'pixel',
        'tracking', 'ad', 'banner', 'spacer', 'blank', 'transparent',
        'facebook', 'twitter', 'linkedin', 'social', 'share',
        '.svg', 'base64', 'data:image', '1x1', 'placeholder'
    ]
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, tracker: ProgressTracker):
        self.config = config
        self.logger = logger
        self.tracker = tracker
        
        self.results_per_period = config.get('search', {}).get('results_per_period',
                                    config.get('search', {}).get('results_per_stock_per_week', 2))
        self.lookback_days = config.get('date_range', {}).get('lookback_days', 84)
        self.period_days = config.get('date_range', {}).get('period_days', 7)
        self.num_periods = max(1, self.lookback_days // self.period_days)
        self.min_words = config.get('scraping', {}).get('min_content_words', 500)
        self.target_per_stock = self.results_per_period * self.num_periods
        
        # Shutdown flag
        self.shutdown_requested = False
        self._shutdown_count = 0
        
        # Driver
        self.driver = None
        self._setup_driver()
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self._shutdown_count += 1
        
        if self._shutdown_count == 1:
            self.logger.warning("\n⚠️  Shutdown requested! Finishing current operation and saving...")
            self.logger.warning("    Press Ctrl+C again to force exit immediately.")
            self.shutdown_requested = True
        else:
            self.logger.warning("\n🛑 Force exit requested! Exiting immediately...")
            # Force close driver
            try:
                if self.driver:
                    self.driver.quit()
            except:
                pass
            sys.exit(1)
    
    def _setup_driver(self):
        """Setup Selenium WebDriver."""
        self.logger.info("Setting up Selenium WebDriver...")
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.set_page_load_timeout(30)
        self.logger.info("WebDriver ready")
    
    def crawl_stock(self, stock_code: str, stock_name: str, category: str) -> List[Dict]:
        """Crawl news for a single stock with progress tracking."""
        self.logger.info(f"  Target: {self.target_per_stock} articles ({self.results_per_period}/period x {self.num_periods} periods)")
        
        self.tracker.mark_stock_started(stock_code)
        
        all_records = []
        today = datetime.now()
        seen_titles = set()
        possibly_blocked = False  # Track if Google might be blocking us
        
        for period_num in range(1, self.num_periods + 1):
            # Check for shutdown
            if self.shutdown_requested:
                self.logger.warning(f"  Shutdown requested, stopping at period {period_num}")
                break
            
            # Calculate date range
            period_end = today - timedelta(days=(period_num - 1) * self.period_days)
            period_start = period_end - timedelta(days=self.period_days)
            
            date_min = period_start.strftime("%m/%d/%Y")
            date_max = period_end.strftime("%m/%d/%Y")
            
            self.logger.info(f"  Period {period_num}/{self.num_periods}: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
            sys.stdout.flush()
            
            # Search for URLs
            urls, period_blocked = self._search_google(stock_code, stock_name, date_min, date_max)
            self.logger.info(f"    Found {len(urls)} candidate URLs")
            sys.stdout.flush()
            
            if period_blocked:
                possibly_blocked = True  # Track if any period was blocked
            
            if not urls:
                if period_blocked:
                    self.logger.warning(f"    ⚠️ No URLs found for period {period_num} (possibly blocked by Google)")
                else:
                    self.logger.warning(f"    No URLs found for period {period_num}")
                self.tracker.mark_period_completed(stock_code, period_num)
                continue
            
            # Scrape articles for this period
            period_articles = []
            url_index = 0
            
            while len(period_articles) < self.results_per_period and url_index < len(urls):
                if self.shutdown_requested:
                    break
                
                url = urls[url_index]
                url_index += 1
                
                self.logger.info(f"    Scraping: {url[:80]}...")
                sys.stdout.flush()
                
                article, fail_reason = self._scrape_article(url, period_num, period_start, period_end)
                
                if article:
                    title_normalized = article['title'].lower().strip()
                    if title_normalized in seen_titles:
                        self.logger.info(f"      ✗ Duplicate title")
                        continue
                    
                    seen_titles.add(title_normalized)
                    period_articles.append(article)
                    
                    # Create record
                    record = {
                        'Stock_Code': stock_code,
                        'Stock_Name': stock_name,
                        'Category': category,
                        'News_Date': article.get('publish_date'),
                        'News_Date_Formatted': self._format_timezone(article.get('publish_date')),
                        'Period_Num': period_num,
                        'News_Title': article.get('title', '')[:500],
                        'News_Source': article.get('source', 'UNKNOWN'),
                        'News_Content': article.get('content', '')[:5000],
                        'Content_Length': article.get('content_length', 0),
                        'Word_Count': article.get('word_count', 0),
                        'Image_URLs': article.get('images', []),
                        'News_URL': article.get('url', ''),
                        'Scrape_Time_Sec': article.get('scrape_time', 0)
                    }
                    
                    all_records.append(record)
                    self.tracker.add_record(record)
                    
                    self.logger.info(f"      ✓ [{len(period_articles)}/{self.results_per_period}] {article['title'][:50]}... ({article['word_count']} words)")
                    sys.stdout.flush()
                else:
                    self.logger.info(f"      ✗ {fail_reason}")
                    sys.stdout.flush()
            
            # Log period completion
            if len(period_articles) < self.results_per_period:
                self.logger.warning(f"    ⚠ Period {period_num}: {len(period_articles)}/{self.results_per_period} articles")
            else:
                self.logger.info(f"    ✓ Period {period_num}: {len(period_articles)}/{self.results_per_period} articles")
            
            self.tracker.mark_period_completed(stock_code, period_num)
            time.sleep(random.uniform(0.5, 1))
        
        # Check if we were interrupted
        if self.shutdown_requested:
            # Don't add placeholder or record failure if interrupted
            # The stock will be retried on next run
            self.logger.info(f"  ⚠️ Interrupted with {len(all_records)} articles collected (will retry on resume)")
            # Don't mark as completed so it will be retried
            return all_records, len(all_records), False  # Return tuple (records, real_count, was_blocked)
        
        # Track if we found any real articles
        real_article_count = len(all_records)
        
        # Add placeholder if no articles found (only if NOT interrupted)
        if not all_records:
            if possibly_blocked:
                # Google might be blocking - mark for retry
                placeholder = {
                    'Stock_Code': stock_code,
                    'Stock_Name': stock_name,
                    'Category': category,
                    'News_Date': None,
                    'News_Date_Formatted': 'Possibly blocked - retry later',
                    'Period_Num': 0,
                    'News_Title': 'POSSIBLY_BLOCKED',
                    'News_Source': 'N/A',
                    'News_Content': f'Google may have blocked requests for {stock_name} ({stock_code}). Will retry on next run.',
                    'Content_Length': 0,
                    'Word_Count': 0,
                    'Image_URLs': [],
                    'News_URL': '',
                    'Scrape_Time_Sec': 0
                }
                self.logger.warning(f"  ⚠️ Possibly blocked for {stock_name} - will retry on next run")
                # DON'T mark as completed so it will be retried
                self.tracker.mark_stock_blocked(stock_code)
            else:
                # Genuinely no articles found
                placeholder = {
                    'Stock_Code': stock_code,
                    'Stock_Name': stock_name,
                    'Category': category,
                    'News_Date': None,
                    'News_Date_Formatted': 'No articles found',
                    'Period_Num': 0,
                    'News_Title': 'NO_ARTICLES_FOUND',
                    'News_Source': 'N/A',
                    'News_Content': f'No valid articles found for {stock_name} ({stock_code})',
                    'Content_Length': 0,
                    'Word_Count': 0,
                    'Image_URLs': [],
                    'News_URL': '',
                    'Scrape_Time_Sec': 0
                }
                self.logger.info(f"  ❌ No articles found for {stock_name} - placeholder added")
            all_records.append(placeholder)
            self.tracker.add_record(placeholder)
        else:
            self.logger.info(f"  ✅ Total: {real_article_count}/{self.target_per_stock} articles for {stock_name}")
        
        # Record failure if not enough REAL articles (only if NOT interrupted and not blocked)
        if real_article_count < self.target_per_stock and not possibly_blocked:
            self.tracker.add_failure(
                stock_code, stock_name, 
                self.target_per_stock, real_article_count,
                f"Found {real_article_count} articles (need {self.target_per_stock})"
            )
        
        # Only consider "blocked" if we got NO articles AND detected blocking
        # If we got articles despite blocking warnings, we're fine
        was_actually_blocked = (real_article_count == 0) and possibly_blocked
        
        return all_records, real_article_count, was_actually_blocked
    
    def _search_google(self, stock_code: str, stock_name: str, date_min: str, date_max: str) -> tuple:
        """Search Google with date range. Returns (urls, possibly_blocked)."""
        urls = []
        possibly_blocked = False
        date_filter = f"cdr:1,cd_min:{date_min},cd_max:{date_max}"
        
        queries = [
            f"{stock_name} {stock_code} yfinance",
            f"{stock_name} {stock_code} stock",
        ]
        
        for query_idx, query in enumerate(queries):
            if self.shutdown_requested:
                break
            
            try:
                search_url = f"https://www.google.com/search?q={quote(query)}&tbs={quote(date_filter)}&hl=en"
                self.logger.info(f"    Search: {search_url[:100]}...")
                sys.stdout.flush()
                
                self.driver.get(search_url)
                time.sleep(3)
                
                page_source = self.driver.page_source.lower()
                if 'captcha' in page_source or 'unusual traffic' in page_source:
                    self.logger.warning("    ⚠️ Google may be blocking (CAPTCHA/unusual traffic detected)...")
                    possibly_blocked = True
                    time.sleep(5)
                
                seen = set()
                selectors = ['div.g a[href]', 'div.yuRUbf a', 'h3 a', 'a[data-ved]', 'a[href*="http"]']
                
                for selector in selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for elem in elements:
                            href = elem.get_attribute('href')
                            if href and self._is_valid_news_url(href) and href not in seen:
                                seen.add(href)
                                urls.append(href)
                    except:
                        continue
                
                if not urls:
                    search_url = f"https://www.google.com/search?q={quote(query + ' news')}&tbs={quote(date_filter)}&hl=en"
                    self.logger.info(f"    Retry: {search_url[:100]}...")
                    self.driver.get(search_url)
                    time.sleep(3)
                    
                    # Check for blocking again on retry
                    page_source = self.driver.page_source.lower()
                    if 'captcha' in page_source or 'unusual traffic' in page_source:
                        possibly_blocked = True
                    
                    for selector in selectors:
                        try:
                            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                            for elem in elements:
                                href = elem.get_attribute('href')
                                if href and self._is_valid_news_url(href) and href not in seen:
                                    seen.add(href)
                                    urls.append(href)
                        except:
                            continue
                
                if urls:
                    break
                    
            except Exception as e:
                self.logger.debug(f"Search error: {e}")
                continue
        
        return urls[:30], possibly_blocked
    
    def _is_valid_news_url(self, url: str) -> bool:
        if not url or not url.startswith('http'):
            return False
        
        excluded = [
            'google.com', 'youtube.com', 'twitter.com', 'facebook.com',
            'instagram.com', 'linkedin.com', 'reddit.com', 'wikipedia.org',
            '.pdf', '.jpg', '.png', '.gif', 'webcache.googleusercontent',
            'translate.google', 'maps.google', 'accounts.google'
        ]
        
        url_lower = url.lower()
        return not any(ex in url_lower for ex in excluded)
    
    def _scrape_article(self, url: str, period_num: int, period_start: datetime, period_end: datetime) -> tuple:
        """Scrape a single article."""
        if self.shutdown_requested:
            return None, "Shutdown requested"
        
        try:
            start_time = time.time()
            self.driver.get(url)
            
            if self.shutdown_requested:
                return None, "Shutdown requested"
            
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except:
                return None, f"Timeout"
            
            time.sleep(1)
            
            content = self._extract_content()
            if not content:
                return None, f"No content"
            
            word_count = len(content.split())
            if word_count < self.min_words:
                return None, f"Words: {word_count}/{self.min_words}"
            
            title = self._extract_title()
            publish_date = self._extract_date()
            
            if publish_date is None:
                publish_date = period_end
                if publish_date.tzinfo is None:
                    publish_date = pytz.UTC.localize(publish_date)
            
            source = self._extract_source(url)
            images = self._extract_images()
            
            elapsed = time.time() - start_time
            
            return {
                'title': title,
                'content': content,
                'publish_date': publish_date,
                'source': source,
                'url': url,
                'images': images,
                'scrape_time': elapsed,
                'content_length': len(content),
                'word_count': word_count,
                'period_num': period_num
            }, None
            
        except Exception as e:
            return None, f"Error: {str(e)[:30]}"
    
    def _extract_content(self) -> str:
        selectors = [
            'article', '.caas-body', '[data-test-locator="articleBody"]',
            '.article-body', '.article-content', '.story-body', '.post-content',
            'main article', '.entry-content', '[itemprop="articleBody"]', '.content-body',
        ]
        
        for selector in selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    text = elements[0].text.strip()
                    if len(text) > 200:
                        return text
            except:
                continue
        
        try:
            main = self.driver.find_elements(By.CSS_SELECTOR, 'main')
            if main:
                return main[0].text.strip()
        except:
            pass
        
        try:
            body = self.driver.find_element(By.TAG_NAME, 'body')
            return body.text.strip()
        except:
            return ""
    
    def _extract_title(self) -> str:
        selectors = ['h1', 'article h1', '.article-title', '.headline']
        
        for selector in selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    title = elements[0].text.strip()
                    if 10 < len(title) < 500:
                        return title
            except:
                continue
        
        try:
            return self.driver.title
        except:
            return "Untitled"
    
    def _extract_date(self) -> Optional[datetime]:
        parsed_date = None
        
        try:
            meta_selectors = [
                'meta[property="article:published_time"]',
                'meta[name="pubdate"]', 'meta[name="date"]',
            ]
            for selector in meta_selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    content = elements[0].get_attribute('content')
                    if content:
                        parsed_date = date_parser.parse(content)
                        break
        except:
            pass
        
        if not parsed_date:
            try:
                time_elements = self.driver.find_elements(By.TAG_NAME, 'time')
                if time_elements:
                    datetime_attr = time_elements[0].get_attribute('datetime')
                    if datetime_attr:
                        parsed_date = date_parser.parse(datetime_attr)
            except:
                pass
        
        if parsed_date and parsed_date.tzinfo is None:
            parsed_date = pytz.UTC.localize(parsed_date)
        
        return parsed_date
    
    def _extract_source(self, url: str) -> str:
        domain = urlparse(url).netloc.lower().replace('www.', '')
        
        source_map = {
            'yahoo': 'YAHOO', 'reuters': 'REUTERS', 'bloomberg': 'BLOOMBERG',
            'cnbc': 'CNBC', 'cnn': 'CNN', 'bbc': 'BBC', 'fool': 'FOOL',
            'seekingalpha': 'SEEKINGALPHA', 'marketwatch': 'MARKETWATCH',
            'scmp': 'SCMP', 'etnet': 'ETNET', 'aastocks': 'AASTOCKS',
            'investing': 'INVESTING', 'zacks': 'ZACKS', 'barrons': 'BARRONS'
        }
        
        for key, value in source_map.items():
            if key in domain:
                return value
        
        return domain.split('.')[0].upper()[:10]
    
    def _extract_images(self) -> List[str]:
        images = []
        try:
            img_elements = self.driver.find_elements(By.TAG_NAME, 'img')[:20]
            for img in img_elements:
                src = img.get_attribute('src') or img.get_attribute('data-src')
                if src and self._is_useful_image(src):
                    if src.startswith('//'):
                        src = 'https:' + src
                    if src.startswith('http') and src not in images:
                        images.append(src)
        except:
            pass
        return images[:10]
    
    def _is_useful_image(self, url: str) -> bool:
        url_lower = url.lower()
        return not any(p in url_lower for p in self.USELESS_IMAGE_PATTERNS)
    
    def _format_timezone(self, dt: datetime) -> str:
        if dt is None:
            return "Unknown"
        
        if dt.tzinfo is None:
            return dt.strftime('%Y-%m-%d %H:%M:%S') + " (GMT+0)"
        
        offset = dt.utcoffset()
        if offset:
            total_seconds = offset.total_seconds()
            hours = int(total_seconds // 3600)
            tz_str = f"GMT{hours:+d}"
        else:
            tz_str = "GMT+0"
        
        return dt.strftime('%Y-%m-%d %H:%M:%S') + f" ({tz_str})"
    
    def close(self):
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass


# =============================================================================
# Data Saver
# =============================================================================

class DataSaver:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.get('output', {}).get('directory', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_filename = config.get('output', {}).get('filename', 'financial_news_data')
    
    def save(self, records: List[Dict], output_format: str = 'both'):
        if not records:
            self.logger.warning("No records to save")
            return None, None
        
        self.logger.info(f"Saving {len(records)} records...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{self.base_filename}_{timestamp}"
        
        excel_path = None
        jsonl_path = None
        
        df = pd.DataFrame(records)
        
        if output_format in ['excel', 'both']:
            self.logger.info(f"  Saving Excel file...")
            excel_path = self._save_excel(df, base_name)
        
        if output_format in ['jsonl', 'both']:
            self.logger.info(f"  Saving JSONL file...")
            jsonl_path = self._save_jsonl(records, base_name)
        
        self.logger.info(f"✓ All files saved successfully!")
        return excel_path, jsonl_path
    
    def _save_excel(self, df: pd.DataFrame, base_name: str):
        try:
            df_copy = df.copy()
            for col in df_copy.columns:
                df_copy[col] = df_copy[col].apply(lambda x: '|'.join(x) if isinstance(x, list) else x)
                try:
                    df_copy[col] = df_copy[col].apply(lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo else x)
                except:
                    pass
            
            filepath = self.output_dir / f"{base_name}.xlsx"
            df_copy.to_excel(filepath, index=False, engine='openpyxl')
            self.logger.info(f"Excel saved: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to save Excel: {e}")
            return None
    
    def _save_jsonl(self, records: List[Dict], base_name: str):
        try:
            filepath = self.output_dir / f"{base_name}.jsonl"
            with open(filepath, 'w', encoding='utf-8') as f:
                for record in records:
                    clean_record = {}
                    for k, v in record.items():
                        if isinstance(v, (datetime, pd.Timestamp)):
                            clean_record[k] = v.isoformat() if v else None
                        elif isinstance(v, list):
                            clean_record[k] = v
                        else:
                            clean_record[k] = v
                    f.write(json.dumps(clean_record, ensure_ascii=False) + '\n')
            self.logger.info(f"JSONL saved: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to save JSONL: {e}")
            return None


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline():
    """Run the production news crawler pipeline."""
    
    config = load_config()
    logger = setup_logging()
    
    # Get config values
    results_per_period = config.get('search', {}).get('results_per_period',
                          config.get('search', {}).get('results_per_stock_per_week', 2))
    lookback_days = config.get('date_range', {}).get('lookback_days', 84)
    period_days = config.get('date_range', {}).get('period_days', 7)
    num_periods = max(1, lookback_days // period_days)
    target_per_stock = results_per_period * num_periods
    min_words = config.get('scraping', {}).get('min_content_words', 500)
    
    logger.info("=" * 60)
    logger.info("Financial News Crawler v4 (PRODUCTION)")
    logger.info("With auto-save, resume, and graceful shutdown")
    logger.info("=" * 60)
    logger.info(f"Settings: {results_per_period}/period x {num_periods} periods ({period_days} days) = {target_per_stock}/stock")
    logger.info(f"Min words: {min_words}")
    logger.info("Press Ctrl+C to stop gracefully")
    logger.info("=" * 60)
    
    total_start = time.time()
    
    # Initialize components
    tracker = ProgressTracker(config.get('output', {}).get('directory', 'output'))
    crawler = RobustCrawler(config, logger, tracker)
    saver = DataSaver(config, logger)
    
    # Check for resume
    if tracker.progress.get('completed_stocks'):
        logger.info(f"Resuming from previous session...")
        logger.info(f"Already completed: {len(tracker.progress['completed_stocks'])} stocks")
    if tracker.progress.get('blocked_stocks'):
        logger.info(f"Blocked stocks (will retry): {len(tracker.progress['blocked_stocks'])} stocks")
    
    stocks = config.get('stocks', {})
    output_format = config.get('output', {}).get('format', 'both')
    
    all_records = []
    stocks_processed = 0
    stocks_skipped = 0
    stocks_blocked = 0
    
    try:
        for category, stock_list in stocks.items():
            if crawler.shutdown_requested:
                break
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Category: {category}")
            logger.info(f"{'='*60}")
            
            for stock in stock_list:
                if crawler.shutdown_requested:
                    break
                
                stock_code = stock['code']
                stock_name = stock['name']
                
                # Check if already completed
                if tracker.is_stock_completed(stock_code):
                    logger.info(f"\n--- Skipping: {stock_name} ({stock_code}) [Already completed] ---")
                    stocks_skipped += 1
                    continue
                
                # Check if was blocked before - retry it
                if tracker.is_stock_blocked(stock_code):
                    logger.info(f"\n--- Retrying: {stock_name} ({stock_code}) [Was blocked before] ---")
                else:
                    logger.info(f"\n--- Processing: {stock_name} ({stock_code}) ---")
                
                # Crawl stock - returns (records, real_article_count, was_blocked)
                stock_records, real_count, was_blocked = crawler.crawl_stock(stock_code, stock_name, category)
                all_records.extend(stock_records)
                
                # Check if we were interrupted during this stock
                if crawler.shutdown_requested:
                    # Don't mark as completed - will retry on next run
                    if stock_records:
                        checkpoint_file = tracker.save_stock_checkpoint(stock_code, stock_name, stock_records, saver.base_filename)
                        logger.info(f"  Stock file saved: {checkpoint_file}")
                    logger.info(f"  ⚠️ Stock interrupted - will retry on next run")
                    break  # Exit the stock loop
                
                # Save individual stock file (not just checkpoint)
                if stock_records:
                    checkpoint_file = tracker.save_stock_checkpoint(stock_code, stock_name, stock_records, saver.base_filename)
                    logger.info(f"  Stock file saved: {checkpoint_file}")
                
                # Handle completion based on results
                if was_blocked:
                    # Don't mark as completed, will retry
                    stocks_blocked += 1
                    logger.info(f"  Progress: {stocks_blocked} stocks blocked (will retry)")
                elif real_count > 0:
                    tracker.mark_stock_completed(stock_code, real_count)
                    stocks_processed += 1
                    total_real_articles = sum(1 for r in all_records if r.get('News_Title') not in ['NO_ARTICLES_FOUND', 'POSSIBLY_BLOCKED'])
                    logger.info(f"  Progress: {stocks_processed} stocks with articles, {total_real_articles} real articles")
                else:
                    # No articles found (not blocked) - genuinely no results
                    tracker.mark_stock_completed(stock_code, 0)
                    stocks_no_articles = sum(1 for r in all_records if r.get('News_Title') == 'NO_ARTICLES_FOUND')
                    logger.info(f"  Progress: {stocks_no_articles} stocks with no articles found")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.info("Saving progress before exit...")
    
    finally:
        logger.info("Closing WebDriver...")
        crawler.close()
        logger.info("WebDriver closed.")
    
    total_elapsed = time.time() - total_start
    
    # Final save
    logger.info("\n" + "=" * 60)
    if crawler.shutdown_requested:
        logger.info("⚠️  Shutdown requested - saving all collected data...")
    else:
        logger.info("Pipeline complete!")
    logger.info("=" * 60)
    
    # Save progress file
    logger.info("Saving progress tracker...")
    tracker.save_progress()
    logger.info(f"  Progress saved to: {tracker.progress_file}")
    
    # Save final results
    if all_records:
        excel_path, jsonl_path = saver.save(all_records, output_format)
    else:
        logger.info("No records collected in this session.")
    
    # Print summary - separate real articles from placeholders/blocked
    real_articles = [r for r in all_records if r.get('News_Title') not in ['NO_ARTICLES_FOUND', 'POSSIBLY_BLOCKED']]
    no_articles_count = sum(1 for r in all_records if r.get('News_Title') == 'NO_ARTICLES_FOUND')
    blocked_count = sum(1 for r in all_records if r.get('News_Title') == 'POSSIBLY_BLOCKED')
    
    print("\n" + "=" * 60)
    print("CRAWLER v4 SUMMARY")
    print("=" * 60)
    print(f"Real articles collected: {len(real_articles)}")
    print(f"Stocks with articles: {stocks_processed}")
    print(f"Stocks with NO articles: {no_articles_count}")
    print(f"Stocks BLOCKED (will retry): {blocked_count}")
    print(f"Stocks skipped (already done): {stocks_skipped}")
    print(f"Total time: {total_elapsed:.1f} seconds")
    print(f"\nProgress tracker summary:")
    print(tracker.get_summary())
    
    if tracker.failures:
        print(f"\n⚠️  FAILURES ({len(tracker.failures)} stocks with insufficient data):")
        for f in tracker.failures:
            print(f"  - {f['stock_code']} ({f['stock_name']}): {f['actual_articles']}/{f['expected_articles']}")
        print(f"\nSee details in: {tracker.failure_file}")
    
    if all_records:
        df = pd.DataFrame(all_records)
        
        # Filter out placeholder records for stats
        df_valid = df[df['Word_Count'] > 0]
        
        if len(df_valid) > 0:
            word_counts = df_valid['Word_Count'].tolist()
            print(f"\nContent Quality (valid articles only):")
            print(f"  Average word count: {sum(word_counts)/len(word_counts):.0f} words")
            print(f"  Min word count: {min(word_counts)} words")
            print(f"  Max word count: {max(word_counts)} words")
        
        # By stock
        print(f"\nArticles per Stock:")
        for stock_code in df['Stock_Code'].unique():
            count = len(df[(df['Stock_Code'] == stock_code) & (df['Word_Count'] > 0)])
            status = "✓" if count >= target_per_stock else "✗"
            print(f"  {stock_code}: {count}/{target_per_stock} {status}")
    
    print("=" * 60)
    
    if not crawler.shutdown_requested:
        # Clear progress on successful completion
        tracker.clear_session()
        print("\n✓ Session completed successfully. Progress cleared.")
    else:
        print(f"\n⚠️  Session interrupted. Resume by running again.")
        print(f"Progress saved in: {tracker.progress_file}")
    
    return all_records


if __name__ == "__main__":
    run_pipeline()
