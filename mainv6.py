"""
=============================================================================
Financial News Crawler - Main Version 6 (mainv6.py)
=============================================================================
PARALLEL version with UNDETECTED CHROMEDRIVER to bypass Google bot detection.

Features:
- PARALLEL scraping of multiple stocks simultaneously
- Each stock has its own progress file (no file lock issues)
- Uses undetected-chromedriver to avoid CAPTCHA/blocking
- Better search queries: "小米集團 (1810.HK)" then "小米集團 (1810.HK) news"
- Google News search (tbm=nws)
- Graceful shutdown handling (Ctrl+C)
- Auto-save progress per stock
- Resume capability

Config setting:
  scraping:
    parallel_drivers: 3  # Number of parallel browser instances

Install: pip install undetected-chromedriver
"""

import os
import json
import logging
import time
import random
import sys
import signal
import atexit
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from urllib.parse import quote, urlparse
from queue import Queue
import traceback

import yaml
import pandas as pd
from dateutil import parser as date_parser
import pytz

# Suppress undetected-chromedriver cleanup warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.excepthook = lambda *args: None  # Suppress unhandled exception messages at exit

# Undetected ChromeDriver - bypasses bot detection
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# =============================================================================
# Global shutdown event for clean interrupts
# =============================================================================
_shutdown_event = threading.Event()
_shutdown_count = 0
_print_lock = threading.Lock()
_driver_init_lock = threading.Lock()  # Lock for driver initialization


def safe_print(msg: str):
    """Thread-safe printing."""
    with _print_lock:
        print(msg, flush=True)


def force_print(msg: str):
    """Force print without lock (for emergency messages)."""
    try:
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()
    except:
        pass


# =============================================================================
# Configuration & Logging
# =============================================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger('NewsCrawlerV6')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(threadName)s] %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# Per-Stock Progress Tracker
# =============================================================================

class StockProgressTracker:
    """
    Tracks progress for a SINGLE stock.
    Each stock has its own progress file to avoid lock issues.
    
    File: output/progress/progress_1810.HK.txt
    Format:
    2026-01-19|2026-02-02|2/2|DONE
    2026-01-05|2026-01-19|0/2|PENDING
    ...
    """
    
    def __init__(self, stock_code: str, stock_name: str, category: str, 
                 config: Dict, output_dir: str = "output"):
        self.stock_code = stock_code
        self.stock_name = stock_name
        self.category = category
        self.config = config
        
        self.output_dir = Path(output_dir)
        self.progress_dir = self.output_dir / "progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress file for this stock
        safe_code = stock_code.replace('.', '_').replace(':', '_')
        self.progress_file = self.progress_dir / f"progress_{safe_code}.txt"
        
        # Settings
        self.period_days = config.get('date_range', {}).get('period_days', 7)
        self.lookback_days = config.get('date_range', {}).get('lookback_days', 14)
        self.results_per_period = config.get('search', {}).get('results_per_period', 2)
        self.num_periods = self.lookback_days // self.period_days
        
        # Lock for thread-safe file operations
        self._lock = threading.Lock()
        
        # Load or initialize tasks for this stock
        self.tasks = self._load_or_init_progress()
    
    def _load_or_init_progress(self) -> List[Dict]:
        """Load existing progress or initialize new task list."""
        if self.progress_file.exists():
            return self._load_progress()
        else:
            return self._init_progress()
    
    def _init_progress(self) -> List[Dict]:
        """Initialize all period tasks for this stock."""
        tasks = []
        today = datetime.now()
        
        for period_num in range(1, self.num_periods + 1):
            period_end = today - timedelta(days=(period_num - 1) * self.period_days)
            period_start = period_end - timedelta(days=self.period_days)
            
            task = {
                'period_start': period_start.strftime('%Y-%m-%d'),
                'period_end': period_end.strftime('%Y-%m-%d'),
                'target': self.results_per_period,
                'scraped': 0,
                'status': 'PENDING'
            }
            tasks.append(task)
        
        # DON'T save here - only save when actual progress is made
        # This prevents creating empty progress files for stocks that never started
        return tasks
    
    def _load_progress(self) -> List[Dict]:
        """Load progress from file."""
        tasks = []
        try:
            with self._lock:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        parts = line.split('|')
                        if len(parts) >= 4:
                            scraped_target = parts[2].split('/')
                            task = {
                                'period_start': parts[0],
                                'period_end': parts[1],
                                'scraped': int(scraped_target[0]),
                                'target': int(scraped_target[1]),
                                'status': parts[3]
                            }
                            tasks.append(task)
        except Exception as e:
            tasks = self._init_progress()
        
        return tasks
    
    def _save_progress(self, tasks: List[Dict] = None):
        """Save progress to file."""
        if tasks is None:
            tasks = self.tasks
        
        with self._lock:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                f.write(f"# Progress for {self.stock_code} ({self.stock_name})\n")
                f.write(f"# Updated: {datetime.now().isoformat()}\n")
                f.write("# Format: period_start|period_end|scraped/target|status\n")
                f.write("#\n")
                
                for task in tasks:
                    line = f"{task['period_start']}|{task['period_end']}|"
                    line += f"{task['scraped']}/{task['target']}|{task['status']}\n"
                    f.write(line)
    
    def get_next_task(self) -> Optional[Dict]:
        """Get next task that needs work."""
        for task in self.tasks:
            if task['status'] != 'DONE':
                return task
        return None
    
    def update_task(self, period_start: str, scraped: int, status: str):
        """Update a specific task's progress."""
        for task in self.tasks:
            if task['period_start'] == period_start:
                task['scraped'] = scraped
                task['status'] = status
                self._save_progress()
                return
    
    def is_all_done(self) -> bool:
        """Check if all tasks for this stock are done."""
        return all(t['status'] == 'DONE' for t in self.tasks)
    
    def get_summary(self) -> Tuple[int, int, int]:
        """Get summary: (done, total, articles_scraped)."""
        done = sum(1 for t in self.tasks if t['status'] == 'DONE')
        total = len(self.tasks)
        articles = sum(t['scraped'] for t in self.tasks)
        return done, total, articles


# =============================================================================
# Stock Worker - Handles one stock in a thread
# =============================================================================

class StockWorker:
    """
    Worker that scrapes a single stock using its own WebDriver.
    """
    
    def __init__(self, worker_id: int, config: Dict):
        self.worker_id = worker_id
        self.config = config
        self.logger = logging.getLogger('NewsCrawlerV6')
        
        # Settings
        self.min_words = config.get('scraping', {}).get('min_content_words', 500)
        self.headless = config.get('scraping', {}).get('headless', False)
        self.chrome_version = config.get('scraping', {}).get('chrome_version', None)
        
        # Driver
        self.driver = None
        self._driver_active = False
        
        # Output dir
        self.output_dir = Path(config.get('output', {}).get('directory', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_driver(self):
        """Setup Undetected ChromeDriver with lock to prevent race conditions."""
        safe_print(f"[Worker-{self.worker_id}] Waiting for driver setup...")
        
        # Use lock to prevent multiple drivers being initialized at once
        with _driver_init_lock:
            safe_print(f"[Worker-{self.worker_id}] Setting up ChromeDriver...")
            
            options = uc.ChromeOptions()
            
            if self.headless:
                options.add_argument('--headless=new')
            
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--disable-blink-features=AutomationControlled')
            
            try:
                if self.chrome_version:
                    self.driver = uc.Chrome(options=options, use_subprocess=True, version_main=self.chrome_version)
                else:
                    self.driver = uc.Chrome(options=options, use_subprocess=True)
                
                self._driver_active = True
                self.driver.set_page_load_timeout(30)
                
                try:
                    self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                except:
                    pass
                
                safe_print(f"[Worker-{self.worker_id}] ChromeDriver ready")
                
                # Small delay after setup to let the driver stabilize
                time.sleep(1)
                
            except Exception as e:
                safe_print(f"[Worker-{self.worker_id}] Driver setup error: {e}")
                raise
    
    def close_driver(self):
        """Close WebDriver safely."""
        if self._driver_active and self.driver:
            try:
                self._driver_active = False
                self.driver.quit()
            except:
                pass
            finally:
                self.driver = None
    
    def process_stock(self, stock_code: str, stock_name: str, category: str) -> Tuple[List[Dict], bool]:
        """
        Process all periods for a single stock.
        Returns: (list of scraped records, driver_dead flag)
        """
        # Check shutdown BEFORE doing anything
        if _shutdown_event.is_set():
            safe_print(f"[Worker-{self.worker_id}] Shutdown - skipping {stock_code}")
            return [], False
        
        safe_print(f"[Worker-{self.worker_id}] Starting: {stock_name} ({stock_code})")
        
        # Create progress tracker for this stock
        tracker = StockProgressTracker(stock_code, stock_name, category, self.config)
        
        if tracker.is_all_done():
            safe_print(f"[Worker-{self.worker_id}] {stock_code} already completed, skipping")
            return [], False
        
        all_records = []
        seen_titles = set()
        driver_dead = False
        
        while not _shutdown_event.is_set() and not driver_dead:
            task = tracker.get_next_task()
            if task is None:
                break
            
            period_start = datetime.strptime(task['period_start'], '%Y-%m-%d')
            period_end = datetime.strptime(task['period_end'], '%Y-%m-%d')
            target = task['target']
            already_scraped = task['scraped']
            
            safe_print(f"[Worker-{self.worker_id}] {stock_code}: {task['period_start']} → {task['period_end']} ({already_scraped}/{target})")
            
            # Check shutdown before search
            if _shutdown_event.is_set():
                safe_print(f"[Worker-{self.worker_id}] Shutdown detected, stopping...")
                tracker.update_task(task['period_start'], already_scraped, 'PARTIAL' if already_scraped > 0 else 'PENDING')
                break
            
            # Search for URLs
            date_min = period_start.strftime("%m/%d/%Y")
            date_max = period_end.strftime("%m/%d/%Y")
            
            urls, possibly_blocked, driver_error = self._search_google_news(stock_code, stock_name, date_min, date_max)
            
            # Check if driver died or shutdown requested
            if driver_error:
                safe_print(f"[Worker-{self.worker_id}] Driver error detected, stopping worker")
                driver_dead = True
                tracker.update_task(task['period_start'], already_scraped, 'PENDING')
                break
            
            if _shutdown_event.is_set():
                safe_print(f"[Worker-{self.worker_id}] Shutdown detected, saving progress...")
                tracker.update_task(task['period_start'], already_scraped, 'PARTIAL' if already_scraped > 0 else 'PENDING')
                break
            
            if possibly_blocked and not urls:
                safe_print(f"[Worker-{self.worker_id}] {stock_code}: ⚠️ BLOCKED - will retry later")
                tracker.update_task(task['period_start'], already_scraped, 'BLOCKED')
                continue
            
            if not urls:
                safe_print(f"[Worker-{self.worker_id}] {stock_code}: No URLs found for this period")
                tracker.update_task(task['period_start'], already_scraped, 'DONE')
                continue
            
            # Scrape articles
            records = []
            needed = target - already_scraped
            
            for url in urls:
                if _shutdown_event.is_set() or driver_dead:
                    break
                
                if len(records) >= needed:
                    break
                
                article, fail_reason = self._scrape_article(url, period_start, period_end)
                
                # Check if driver died
                if fail_reason and 'connection' in fail_reason.lower():
                    driver_dead = True
                    break
                
                if article:
                    title_normalized = article['title'].lower().strip()
                    if title_normalized in seen_titles:
                        continue
                    
                    seen_titles.add(title_normalized)
                    
                    record = {
                        'Stock_Code': stock_code,
                        'Stock_Name': stock_name,
                        'Category': category,
                        'News_Date': article.get('publish_date'),
                        'News_Date_Formatted': self._format_timezone(article.get('publish_date')),
                        'Period_Start': task['period_start'],
                        'Period_End': task['period_end'],
                        'News_Title': article.get('title', '')[:500],
                        'News_Source': article.get('source', 'UNKNOWN'),
                        'News_Content': article.get('content', '')[:5000],
                        'Content_Length': article.get('content_length', 0),
                        'Word_Count': article.get('word_count', 0),
                        'Image_URLs': article.get('images', []),
                        'News_URL': article.get('url', ''),
                        'Scrape_Time_Sec': article.get('scrape_time', 0)
                    }
                    
                    records.append(record)
                    all_records.append(record)
                    
                    safe_print(f"[Worker-{self.worker_id}] {stock_code}: ✓ [{already_scraped + len(records)}/{target}] {article['title'][:40]}...")
            
            # Update progress
            total_scraped = already_scraped + len(records)
            
            if _shutdown_event.is_set() or driver_dead:
                # Interrupted - mark as PARTIAL so we can resume later
                status = 'PARTIAL' if total_scraped > 0 else 'PENDING'
            else:
                # Completed scraping attempt - mark as DONE even if we didn't get enough
                # (we tried all URLs, no point retrying the same search)
                status = 'DONE'
            
            tracker.update_task(task['period_start'], total_scraped, status)
            
            # Small delay between periods
            if not _shutdown_event.is_set() and not driver_dead:
                _shutdown_event.wait(timeout=random.uniform(0.5, 1.5))
        
        # Save Excel file for this stock (single file, not checkpoint)
        if all_records:
            self._save_stock_excel(stock_code, stock_name, all_records)
        
        done, total, articles = tracker.get_summary()
        safe_print(f"[Worker-{self.worker_id}] {stock_code} finished: {done}/{total} periods, {articles} articles")
        
        return all_records, driver_dead
    
    def _search_google_news(self, stock_code: str, stock_name: str, date_min: str, date_max: str) -> Tuple[List[str], bool, bool]:
        """
        Search Google for English news.
        Query format: {name} news "{code}" (e.g., "XPENG news "9868 HK"")
        
        Returns: (urls, possibly_blocked, driver_error)
        """
        urls = []
        possibly_blocked = False
        driver_error = False
        
        # Check shutdown first
        if _shutdown_event.is_set():
            return [], False, False
        
        # Convert stock code: "1810.HK" -> "1810 HK" for better search results
        code_for_search = stock_code.replace('.', ' ')
        
        # Date filter combined with English language filter
        # Format: lr:lang_1en,cdr:1,cd_min:1/19/2026,cd_max:2/2/2026
        date_filter = f"lr:lang_1en,cdr:1,cd_min:{date_min},cd_max:{date_max}"
        
        # Single query: stock name + news + quoted code
        # No fallback - stock code alone is too generic and matches unrelated results
        queries = [
            f'{stock_name} news "{code_for_search}"',
        ]
        
        for query_idx, query in enumerate(queries):
            if _shutdown_event.is_set():
                safe_print(f"[Worker-{self.worker_id}] Shutdown detected, stopping search")
                break
            
            try:
                # Google NEWS search with English language filter
                # tbm=nws = News tab only
                search_url = f"https://www.google.com/search?q={quote(query)}&tbm=nws&lr=lang_en&hl=en&tbs={quote(date_filter)}"
                
                safe_print(f"[Worker-{self.worker_id}] Search NEWS: {query}")
                
                self.driver.get(search_url)
                
                # Interruptible wait
                if _shutdown_event.wait(timeout=random.uniform(2, 4)):
                    safe_print(f"[Worker-{self.worker_id}] Shutdown during search wait")
                    break
                
                # Check for blocking
                page_source = self.driver.page_source.lower()
                if 'captcha' in page_source or 'unusual traffic' in page_source:
                    safe_print(f"[Worker-{self.worker_id}] ⚠️ Google blocking detected")
                    possibly_blocked = True
                    if _shutdown_event.wait(timeout=random.uniform(5, 10)):
                        break
                
                # Extract URLs from news results
                seen = set()
                selectors = [
                    'div.SoaBEf a',  # News card links
                    'a[data-ved]',
                    'div.g a[href]',
                    'article a[href]',
                    'a[href*="http"]'
                ]
                
                for selector in selectors:
                    if _shutdown_event.is_set():
                        break
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
                    safe_print(f"[Worker-{self.worker_id}] Found {len(urls)} URLs with query: {query}")
                    break  # Found URLs, stop trying queries
                
                if query_idx < len(queries) - 1 and not _shutdown_event.is_set():
                    _shutdown_event.wait(timeout=random.uniform(1, 2))
                    
            except Exception as e:
                error_str = str(e).lower()
                # Check if this is a connection error (driver died)
                if 'connection' in error_str or 'refused' in error_str or 'max retries' in error_str:
                    if _shutdown_event.is_set():
                        safe_print(f"[Worker-{self.worker_id}] Driver closed due to shutdown")
                    else:
                        safe_print(f"[Worker-{self.worker_id}] Driver connection lost: {str(e)[:50]}")
                    driver_error = True
                    break
                else:
                    safe_print(f"[Worker-{self.worker_id}] Search error: {e}")
                continue
        
        return urls[:30], possibly_blocked, driver_error
    
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
    
    def _scrape_article(self, url: str, period_start: datetime, period_end: datetime) -> Tuple[Optional[Dict], Optional[str]]:
        """Scrape a single article."""
        if _shutdown_event.is_set():
            return None, "Shutdown"
        
        try:
            start_time = time.time()
            self.driver.get(url)
            
            if _shutdown_event.is_set():
                return None, "Shutdown"
            
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except:
                return None, "Timeout"
            
            _shutdown_event.wait(timeout=random.uniform(1, 2))
            
            if _shutdown_event.is_set():
                return None, "Shutdown"
            
            content = self._extract_content()
            if not content:
                return None, "No content"
            
            word_count = len(content.split())
            if word_count < self.min_words:
                return None, f"Words: {word_count}/{self.min_words}"
            
            title = self._extract_title()
            publish_date = self._extract_date(period_start, period_end)
            source = urlparse(url).netloc.replace('www.', '')
            images = self._extract_images()
            
            elapsed = round(time.time() - start_time, 2)
            
            return {
                'title': title,
                'content': content,
                'content_length': len(content),
                'word_count': word_count,
                'publish_date': publish_date,
                'source': source,
                'url': url,
                'images': images[:5],
                'scrape_time': elapsed
            }, None
            
        except Exception as e:
            return None, f"Error: {str(e)[:50]}"
    
    def _extract_content(self) -> str:
        """Extract main content from page."""
        selectors = [
            'article', 'main', '.article-content', '.post-content',
            '.entry-content', '.content', '#content', '.story-body',
            '.article-body', '[itemprop="articleBody"]'
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
            body = self.driver.find_element(By.TAG_NAME, 'body')
            paragraphs = body.find_elements(By.TAG_NAME, 'p')
            texts = [p.text.strip() for p in paragraphs if len(p.text.strip()) > 50]
            if texts:
                return '\n\n'.join(texts)
        except:
            pass
        
        return ""
    
    def _extract_title(self) -> str:
        """Extract article title."""
        selectors = [
            'h1', 'article h1', '.article-title', '.post-title',
            '.entry-title', '[itemprop="headline"]', 'title'
        ]
        
        for selector in selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    title = elements[0].text.strip()
                    if title and len(title) > 10:
                        return title
            except:
                continue
        
        return self.driver.title or "Unknown Title"
    
    def _extract_date(self, period_start: datetime, period_end: datetime) -> Optional[datetime]:
        """Extract publish date from article."""
        selectors = [
            'time[datetime]', '[itemprop="datePublished"]',
            '.publish-date', '.post-date', '.article-date',
            '.date', 'time', '.timestamp'
        ]
        
        for selector in selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for elem in elements:
                    date_str = elem.get_attribute('datetime') or elem.text
                    if date_str:
                        try:
                            parsed = date_parser.parse(date_str, fuzzy=True)
                            return parsed
                        except:
                            continue
            except:
                continue
        
        return period_start + (period_end - period_start) / 2
    
    def _extract_images(self) -> List[str]:
        """Extract image URLs from article."""
        images = []
        try:
            img_elements = self.driver.find_elements(By.CSS_SELECTOR, 'article img, .article img, main img')
            for img in img_elements[:10]:
                src = img.get_attribute('src')
                if src and src.startswith('http') and not any(x in src for x in ['icon', 'logo', 'avatar']):
                    images.append(src)
        except:
            pass
        return images
    
    def _format_timezone(self, dt: Optional[datetime]) -> str:
        """Format datetime for display."""
        if not dt:
            return "Unknown"
        try:
            if dt.tzinfo is None:
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            return dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        except:
            return str(dt)
    
    def _save_stock_excel(self, stock_code: str, stock_name: str, records: List[Dict]):
        """Save a single Excel file for this stock (overwrites if exists)."""
        if not records:
            return
        
        clean_name = ''.join(c for c in stock_name if c.isalnum() or c in ' -_').strip().replace(' ', '_')
        safe_code = stock_code.replace('.', '_')
        # Single file per stock (no timestamp - will be overwritten on resume)
        excel_file = self.output_dir / f"stock_{safe_code}_{clean_name}.xlsx"
        
        try:
            df = pd.DataFrame(records)
            for col in df.columns:
                df[col] = df[col].apply(lambda x: '|'.join(str(i) for i in x) if isinstance(x, list) else x)
                try:
                    df[col] = df[col].apply(lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo else x)
                except:
                    pass
            
            df.to_excel(excel_file, index=False, engine='openpyxl')
            safe_print(f"[Worker-{self.worker_id}] Saved: {excel_file.name}")
        except Exception as e:
            safe_print(f"[Worker-{self.worker_id}] Error saving Excel: {e}")


# =============================================================================
# Parallel Crawler Manager
# =============================================================================

class ParallelCrawlerManager:
    """
    Manages parallel stock scraping with multiple workers.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger('NewsCrawlerV6')
        
        self.num_workers = config.get('scraping', {}).get('parallel_drivers', 3)
        self.output_dir = Path(config.get('output', {}).get('directory', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all stocks
        self.stocks = self._get_all_stocks()
        
        # Results
        self.all_records = []
        self._records_lock = threading.Lock()
        
        # Workers
        self.workers = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        global _shutdown_count
        _shutdown_count += 1
        
        if _shutdown_count == 1:
            force_print("\n" + "=" * 50)
            force_print("⚠️  SHUTDOWN REQUESTED - stopping all workers...")
            force_print("    Waiting for current operations to finish...")
            force_print("    Press Ctrl+C again to FORCE EXIT immediately.")
            force_print("=" * 50)
            _shutdown_event.set()
        else:
            force_print("\n" + "=" * 50)
            force_print("🛑 FORCE EXIT - killing all drivers...")
            force_print("=" * 50)
            # Kill all drivers
            for worker in self.workers:
                try:
                    worker.close_driver()
                except:
                    pass
            # Hard exit
            os._exit(1)
    
    def _get_all_stocks(self) -> List[Dict]:
        """Get all stocks from config as flat list."""
        stocks = []
        for category, stock_list in self.config.get('stocks', {}).items():
            for stock in stock_list:
                # Avoid duplicates
                if not any(s['code'] == stock['code'] for s in stocks):
                    stocks.append({
                        'code': stock['code'],
                        'name': stock['name'],
                        'category': category
                    })
        return stocks
    
    def _worker_task(self, worker_id: int, stock_queue: Queue) -> List[Dict]:
        """Worker task that processes stocks from queue."""
        # Stagger worker starts to avoid driver init conflicts
        time.sleep(worker_id * 3)  # Each worker waits 3 seconds more than the previous
        
        worker = StockWorker(worker_id, self.config)
        self.workers.append(worker)
        
        records = []
        driver_dead = False
        
        try:
            worker.setup_driver()
            
            while not _shutdown_event.is_set() and not driver_dead:
                try:
                    stock = stock_queue.get_nowait()
                except:
                    break  # Queue empty
                
                stock_records, driver_dead = worker.process_stock(
                    stock['code'], stock['name'], stock['category']
                )
                
                records.extend(stock_records)
                
                with self._records_lock:
                    self.all_records.extend(stock_records)
                
                stock_queue.task_done()
                
                # If driver died, stop processing more stocks
                if driver_dead:
                    safe_print(f"[Worker-{worker_id}] Driver died, worker stopping")
                    break
        
        except Exception as e:
            safe_print(f"[Worker-{worker_id}] Error: {e}")
            traceback.print_exc()
        
        finally:
            worker.close_driver()
        
        return records
    
    def run(self) -> List[Dict]:
        """Run parallel crawling."""
        safe_print(f"\nStarting {self.num_workers} parallel workers for {len(self.stocks)} stocks...")
        safe_print("=" * 60)
        
        # Create stock queue
        stock_queue = Queue()
        for stock in self.stocks:
            stock_queue.put(stock)
        
        # Run workers
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(self.num_workers):
                future = executor.submit(self._worker_task, i, stock_queue)
                futures.append(future)
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    safe_print(f"Worker error: {e}")
        
        return self.all_records
    
    def save_final_results(self):
        """Save all collected records to final output files."""
        if not self.all_records:
            safe_print("No records to save.")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        safe_print(f"\nSaving {len(self.all_records)} total records...")
        
        df = pd.DataFrame(self.all_records)
        
        # Clean data
        for col in df.columns:
            df[col] = df[col].apply(lambda x: '|'.join(str(i) for i in x) if isinstance(x, list) else x)
            try:
                df[col] = df[col].apply(lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo else x)
            except:
                pass
        
        output_format = self.config.get('output', {}).get('format', 'both')
        
        if output_format in ['excel', 'both']:
            excel_path = self.output_dir / f"financial_news_all_{timestamp}.xlsx"
            df.to_excel(excel_path, index=False, engine='openpyxl')
            safe_print(f"✓ Excel saved: {excel_path}")
        
        if output_format in ['jsonl', 'both']:
            jsonl_path = self.output_dir / f"financial_news_all_{timestamp}.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    f.write(json.dumps(row.to_dict(), ensure_ascii=False, default=str) + '\n')
            safe_print(f"✓ JSONL saved: {jsonl_path}")
    
    def get_summary(self) -> str:
        """Get overall progress summary."""
        total_stocks = len(self.stocks)
        progress_dir = self.output_dir / "progress"
        
        completed = 0
        total_articles = 0
        
        if progress_dir.exists():
            for progress_file in progress_dir.glob("progress_*.txt"):
                try:
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
                        all_done = all('DONE' in l for l in lines)
                        if all_done:
                            completed += 1
                        for line in lines:
                            parts = line.split('|')
                            if len(parts) >= 3:
                                scraped = int(parts[2].split('/')[0])
                                total_articles += scraped
                except:
                    pass
        
        return (f"Stocks: {completed}/{total_stocks} completed\n"
                f"Total articles: {total_articles}")


# =============================================================================
# Main
# =============================================================================

def test_shutdown():
    """Test shutdown functionality - manually test Ctrl+C within 15 seconds."""
    safe_print("=" * 60)
    safe_print("SHUTDOWN TEST MODE")
    safe_print("=" * 60)
    safe_print("Press Ctrl+C within 15 seconds to test shutdown!")
    safe_print("You should see a clear shutdown message appear.")
    safe_print("")
    
    # Setup the signal handler (same as real crawler)
    def test_signal_handler(signum, frame):
        global _shutdown_count
        _shutdown_count += 1
        if _shutdown_count == 1:
            force_print("\n" + "=" * 50)
            force_print("✅ SHUTDOWN TEST PASSED!")
            force_print("   The shutdown message appeared correctly.")
            force_print("   force_print() is working!")
            force_print("=" * 50)
            _shutdown_event.set()
        else:
            force_print("\n🛑 FORCE EXIT (second Ctrl+C)")
            os._exit(1)
    
    signal.signal(signal.SIGINT, test_signal_handler)
    signal.signal(signal.SIGTERM, test_signal_handler)
    
    # Simulate work - wait for user to press Ctrl+C
    for i in range(15, 0, -1):
        if _shutdown_event.is_set():
            break
        safe_print(f"  Waiting for Ctrl+C... {i} seconds remaining")
        time.sleep(1)
    
    if _shutdown_event.is_set():
        safe_print("\nShutdown test PASSED! ✅")
        safe_print("The crawler will now properly show shutdown messages.")
    else:
        safe_print("\nTest timed out - you didn't press Ctrl+C.")
        safe_print("Run the test again and press Ctrl+C to verify shutdown works.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Financial News Crawler v6')
    parser.add_argument('--test-shutdown', action='store_true', 
                        help='Test shutdown functionality (runs for 10 seconds)')
    args = parser.parse_args()
    
    if args.test_shutdown:
        test_shutdown()
        return
    
    logger = setup_logging()
    
    safe_print("=" * 60)
    safe_print("Financial News Crawler v6 (PARALLEL)")
    safe_print("Multi-threaded with per-stock progress tracking")
    safe_print("=" * 60)
    
    config = load_config()
    
    # Display settings
    period_days = config.get('date_range', {}).get('period_days', 7)
    results_per_period = config.get('search', {}).get('results_per_period', 2)
    lookback_days = config.get('date_range', {}).get('lookback_days', 14)
    num_periods = lookback_days // period_days
    min_words = config.get('scraping', {}).get('min_content_words', 500)
    parallel_workers = config.get('scraping', {}).get('parallel_drivers', 3)
    headless = config.get('scraping', {}).get('headless', False)
    
    safe_print(f"Settings:")
    safe_print(f"  Parallel workers: {parallel_workers}")
    safe_print(f"  Lookback: {lookback_days} days, Period: {period_days} days = {num_periods} periods")
    safe_print(f"  Articles per period: {results_per_period}")
    safe_print(f"  Min words: {min_words}")
    safe_print(f"  Headless: {headless}")
    safe_print("")
    safe_print("Press Ctrl+C to stop gracefully")
    safe_print("=" * 60)
    
    # Initialize manager
    manager = ParallelCrawlerManager(config)
    
    # Show initial progress
    safe_print(f"\n{manager.get_summary()}")
    
    total_start = time.time()
    
    try:
        # Run parallel crawling
        records = manager.run()
        
    except Exception as e:
        safe_print(f"Error: {e}")
        traceback.print_exc()
    
    total_elapsed = time.time() - total_start
    
    # Final save
    safe_print("\n" + "=" * 60)
    if _shutdown_event.is_set():
        safe_print("⚠️  Session interrupted - saving collected data...")
    else:
        safe_print("Pipeline complete!")
    safe_print("=" * 60)
    
    manager.save_final_results()
    
    # Summary
    print("\n" + "=" * 60)
    print("CRAWLER v6 SUMMARY (PARALLEL)")
    print("=" * 60)
    print(f"Session articles: {len(manager.all_records)}")
    print(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"\n{manager.get_summary()}")
    print("=" * 60)
    
    if _shutdown_event.is_set():
        print("\n⚠️  Session was interrupted. Run again to continue.")
        print("   Progress is saved per-stock in: output/progress/")
    else:
        print("\n✅ Session completed!")
        print("   To clear progress and start fresh, delete: output/progress/")


if __name__ == '__main__':
    main()
