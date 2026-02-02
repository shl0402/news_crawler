"""
=============================================================================
Financial News Crawler - Main Version 5 (mainv5.py)
=============================================================================
Production version with UNDETECTED CHROMEDRIVER to bypass Google bot detection.

Features:
- Uses undetected-chromedriver to avoid CAPTCHA/blocking
- GRANULAR progress tracking (per stock + period combination)
- Human-readable progress file (scrape_progress.txt)
- Graceful shutdown handling (Ctrl+C)
- Auto-save progress after each task
- Resume capability from progress file
- Clean WebDriver cleanup (fixes WinError 6)

Progress File Format:
1810.HK|小米集團|Auto|2026-01-26|2026-02-02|2/2|DONE
1810.HK|小米集團|Auto|2026-01-19|2026-01-26|0/2|PENDING  ← will retry this
1211.HK|比亞迪股份|Auto|2026-01-26|2026-02-02|1/2|PARTIAL

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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from urllib.parse import quote, urlparse

import yaml
import pandas as pd
from dateutil import parser as date_parser
import pytz

# Undetected ChromeDriver - bypasses bot detection
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# =============================================================================
# Configuration & Logging
# =============================================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger('NewsCrawlerV5')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# Progress Tracker - Granular tracking per stock + period
# =============================================================================

class ProgressTracker:
    """
    Tracks progress at granular level: each stock + period combination.
    
    Progress file format (scrape_progress.txt):
    1810.HK|小米集團|Auto|2026-01-26|2026-02-02|2/2|DONE
    1810.HK|小米集團|Auto|2026-01-19|2026-01-26|0/2|PENDING
    1211.HK|比亞迪股份|Auto|2026-01-26|2026-02-02|1/2|PARTIAL
    ...
    
    Status: PENDING, PARTIAL, DONE, BLOCKED, ERROR
    """
    
    def __init__(self, config: Dict, output_dir: str = "output"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.progress_file = self.output_dir / "scrape_progress.txt"
        self.failure_file = self.output_dir / "scrape_failures.txt"
        self.session_file = self.output_dir / "current_session.jsonl"
        
        self.logger = logging.getLogger('NewsCrawlerV5')
        
        # Settings from config
        self.period_days = config.get('date_range', {}).get('period_days', 7)
        self.lookback_days = config.get('date_range', {}).get('lookback_days', 14)
        self.results_per_period = config.get('search', {}).get('results_per_period', 2)
        self.num_periods = self.lookback_days // self.period_days
        
        # Load or initialize progress
        self.tasks = self._load_or_init_progress()
        self.session_records = []
    
    def _load_or_init_progress(self) -> List[Dict]:
        """Load existing progress or initialize new task list."""
        if self.progress_file.exists():
            return self._load_progress()
        else:
            return self._init_progress()
    
    def _init_progress(self) -> List[Dict]:
        """Initialize all tasks based on config."""
        tasks = []
        today = datetime.now()
        stocks = self.config.get('stocks', {})
        
        self.logger.info("Initializing progress tracker (first run)...")
        
        for category, stock_list in stocks.items():
            for stock in stock_list:
                stock_code = stock['code']
                stock_name = stock['name']
                
                # Create tasks from most recent to oldest period
                for period_num in range(1, self.num_periods + 1):
                    period_end = today - timedelta(days=(period_num - 1) * self.period_days)
                    period_start = period_end - timedelta(days=self.period_days)
                    
                    task = {
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'category': category,
                        'period_start': period_start.strftime('%Y-%m-%d'),
                        'period_end': period_end.strftime('%Y-%m-%d'),
                        'target': self.results_per_period,
                        'scraped': 0,
                        'status': 'PENDING'  # PENDING, PARTIAL, DONE, BLOCKED, ERROR
                    }
                    tasks.append(task)
        
        self._save_progress(tasks)
        
        # Count unique stocks
        unique_stocks = len(set(t['stock_code'] for t in tasks))
        self.logger.info(f"Initialized {len(tasks)} tasks ({unique_stocks} stocks x {self.num_periods} periods)")
        return tasks
    
    def _load_progress(self) -> List[Dict]:
        """Load progress from file."""
        tasks = []
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split('|')
                    if len(parts) >= 7:
                        scraped_target = parts[5].split('/')
                        task = {
                            'stock_code': parts[0],
                            'stock_name': parts[1],
                            'category': parts[2],
                            'period_start': parts[3],
                            'period_end': parts[4],
                            'scraped': int(scraped_target[0]),
                            'target': int(scraped_target[1]),
                            'status': parts[6]
                        }
                        tasks.append(task)
            
            self.logger.info(f"Loaded {len(tasks)} tasks from progress file")
            
            # Count status
            done = sum(1 for t in tasks if t['status'] == 'DONE')
            pending = sum(1 for t in tasks if t['status'] == 'PENDING')
            partial = sum(1 for t in tasks if t['status'] == 'PARTIAL')
            blocked = sum(1 for t in tasks if t['status'] == 'BLOCKED')
            error = sum(1 for t in tasks if t['status'] == 'ERROR')
            
            self.logger.info(f"  Status: DONE={done}, PENDING={pending}, PARTIAL={partial}, BLOCKED={blocked}, ERROR={error}")
            
        except Exception as e:
            self.logger.error(f"Error loading progress: {e}")
            tasks = self._init_progress()
        
        return tasks
    
    def _save_progress(self, tasks: List[Dict] = None):
        """Save progress to file."""
        if tasks is None:
            tasks = self.tasks
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            f.write("# Stock Scraping Progress - DO NOT EDIT MANUALLY\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write("# Format: stock_code|stock_name|category|period_start|period_end|scraped/target|status\n")
            f.write("# Status: PENDING=not started, PARTIAL=incomplete, DONE=complete, BLOCKED=Google blocked, ERROR=scrape error\n")
            f.write("#\n")
            
            for task in tasks:
                line = f"{task['stock_code']}|{task['stock_name']}|{task['category']}|"
                line += f"{task['period_start']}|{task['period_end']}|"
                line += f"{task['scraped']}/{task['target']}|{task['status']}\n"
                f.write(line)
    
    def get_next_task(self) -> Optional[Dict]:
        """Get next task that needs work (from top to bottom, never skip)."""
        for task in self.tasks:
            # Check tasks in order - DONE tasks are skipped, everything else needs work
            if task['status'] != 'DONE':
                return task
        return None
    
    def update_task(self, stock_code: str, period_start: str, scraped: int, status: str):
        """Update a specific task's progress."""
        for task in self.tasks:
            if task['stock_code'] == stock_code and task['period_start'] == period_start:
                task['scraped'] = scraped
                task['status'] = status
                self._save_progress()
                return
    
    def mark_task_done(self, stock_code: str, period_start: str, scraped: int):
        """Mark a task as completed."""
        self.update_task(stock_code, period_start, scraped, 'DONE')
    
    def mark_task_partial(self, stock_code: str, period_start: str, scraped: int):
        """Mark a task as partially completed."""
        self.update_task(stock_code, period_start, scraped, 'PARTIAL')
    
    def mark_task_blocked(self, stock_code: str, period_start: str, scraped: int = 0):
        """Mark a task as blocked by Google."""
        self.update_task(stock_code, period_start, scraped, 'BLOCKED')
    
    def mark_task_error(self, stock_code: str, period_start: str, scraped: int = 0):
        """Mark a task as error."""
        self.update_task(stock_code, period_start, scraped, 'ERROR')
    
    def add_record(self, record: Dict):
        """Add a record and save incrementally to session file."""
        self.session_records.append(record)
        
        with open(self.session_file, 'a', encoding='utf-8') as f:
            clean_record = {}
            for k, v in record.items():
                if isinstance(v, (datetime, pd.Timestamp)):
                    clean_record[k] = v.isoformat() if v else None
                elif isinstance(v, list):
                    clean_record[k] = '|'.join(str(x) for x in v) if v else ''
                else:
                    clean_record[k] = v
            f.write(json.dumps(clean_record, ensure_ascii=False) + '\n')
    
    def add_failure(self, stock_code: str, stock_name: str, period: str, reason: str):
        """Record a failure."""
        with open(self.failure_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().isoformat()
            f.write(f"[{timestamp}] {stock_code} ({stock_name}) - {period}: {reason}\n")
    
    def save_checkpoint(self, stock_code: str, stock_name: str, records: List[Dict]) -> Optional[Path]:
        """Save checkpoint file for current scraping batch."""
        if not records:
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        clean_name = ''.join(c for c in stock_name if c.isalnum() or c in ' -_').strip().replace(' ', '_')
        checkpoint_file = self.output_dir / f"checkpoint_{stock_code.replace('.', '_')}_{clean_name}_{timestamp}.xlsx"
        
        try:
            df = pd.DataFrame(records)
            for col in df.columns:
                df[col] = df[col].apply(lambda x: '|'.join(str(i) for i in x) if isinstance(x, list) else x)
                try:
                    df[col] = df[col].apply(lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo else x)
                except:
                    pass
            
            df.to_excel(checkpoint_file, index=False, engine='openpyxl')
            return checkpoint_file
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            return None
    
    def get_summary(self) -> str:
        """Get progress summary."""
        total = len(self.tasks)
        done = sum(1 for t in self.tasks if t['status'] == 'DONE')
        pending = sum(1 for t in self.tasks if t['status'] == 'PENDING')
        partial = sum(1 for t in self.tasks if t['status'] == 'PARTIAL')
        blocked = sum(1 for t in self.tasks if t['status'] == 'BLOCKED')
        error = sum(1 for t in self.tasks if t['status'] == 'ERROR')
        
        total_scraped = sum(t['scraped'] for t in self.tasks)
        total_target = sum(t['target'] for t in self.tasks)
        
        pct = (done / total * 100) if total > 0 else 0
        art_pct = (total_scraped / total_target * 100) if total_target > 0 else 0
        
        return (f"Tasks: {done}/{total} done ({pct:.1f}%)\n"
                f"  DONE={done}, PENDING={pending}, PARTIAL={partial}, BLOCKED={blocked}, ERROR={error}\n"
                f"  Articles: {total_scraped}/{total_target} ({art_pct:.1f}%)")
    
    def is_all_done(self) -> bool:
        """Check if all tasks are completed."""
        return all(t['status'] == 'DONE' for t in self.tasks)
    
    def reset_blocked_and_errors(self):
        """Reset BLOCKED and ERROR tasks to allow retry."""
        count = 0
        for task in self.tasks:
            if task['status'] in ['BLOCKED', 'ERROR']:
                task['status'] = 'PENDING' if task['scraped'] == 0 else 'PARTIAL'
                count += 1
        
        if count > 0:
            self._save_progress()
            self.logger.info(f"Reset {count} BLOCKED/ERROR tasks for retry")
        return count
    
    def delete_progress_file(self):
        """Delete progress files after successful completion."""
        if self.progress_file.exists():
            self.progress_file.unlink()
            self.logger.info(f"Deleted: {self.progress_file}")
        if self.session_file.exists():
            self.session_file.unlink()
            self.logger.info(f"Deleted: {self.session_file}")


# =============================================================================
# Data Saver
# =============================================================================

class DataSaver:
    """Handles saving data in multiple formats."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_filename = f"financial_news_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = logging.getLogger('NewsCrawlerV5')
    
    def save(self, records: List[Dict], output_format: str = "both"):
        """Save records to file(s)."""
        if not records:
            self.logger.warning("No records to save")
            return None, None
        
        self.logger.info(f"Saving {len(records)} records...")
        
        df = pd.DataFrame(records)
        
        # Clean data
        for col in df.columns:
            df[col] = df[col].apply(lambda x: '|'.join(str(i) for i in x) if isinstance(x, list) else x)
            try:
                df[col] = df[col].apply(lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo else x)
            except:
                pass
        
        excel_path = None
        jsonl_path = None
        
        if output_format in ['excel', 'both']:
            excel_path = self.output_dir / f"{self.base_filename}.xlsx"
            self.logger.info(f"  Saving Excel file...")
            df.to_excel(excel_path, index=False, engine='openpyxl')
            self.logger.info(f"  ✓ Excel saved: {excel_path}")
        
        if output_format in ['jsonl', 'both']:
            jsonl_path = self.output_dir / f"{self.base_filename}.jsonl"
            self.logger.info(f"  Saving JSONL file...")
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    f.write(json.dumps(row.to_dict(), ensure_ascii=False, default=str) + '\n')
            self.logger.info(f"  ✓ JSONL saved: {jsonl_path}")
        
        return excel_path, jsonl_path


# =============================================================================
# Robust Crawler with Undetected ChromeDriver
# =============================================================================

# Global shutdown event for clean interrupts
_shutdown_event = threading.Event()
_shutdown_count = 0


class RobustCrawler:
    """
    Web crawler using undetected-chromedriver to bypass Google bot detection.
    """
    
    def __init__(self, config: Dict, tracker: ProgressTracker):
        self.config = config
        self.tracker = tracker
        self.logger = logging.getLogger('NewsCrawlerV5')
        
        # Settings
        self.min_words = config.get('scraping', {}).get('min_content_words', 500)
        self.headless = config.get('scraping', {}).get('headless', False)
        self.chrome_version = config.get('scraping', {}).get('chrome_version', None)
        
        # Driver state
        self.driver = None
        self._driver_active = False
        
        # Setup driver
        self._setup_driver()
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Register cleanup on exit (prevents WinError 6)
        atexit.register(self._safe_cleanup)
    
    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown was requested."""
        return _shutdown_event.is_set()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        global _shutdown_count
        _shutdown_count += 1
        
        if _shutdown_count == 1:
            print("\n" + "="*50, flush=True)
            print("⚠️  SHUTDOWN REQUESTED - saving progress...", flush=True)
            print("    Press Ctrl+C again to force exit.", flush=True)
            print("="*50, flush=True)
            _shutdown_event.set()
        else:
            print("\n🛑 FORCE EXIT!", flush=True)
            self._safe_cleanup()
            os._exit(1)  # Hard exit
    
    def _safe_cleanup(self):
        """Safely cleanup resources (prevents WinError 6)."""
        if self._driver_active and self.driver:
            try:
                self._driver_active = False  # Prevent double cleanup
                self.driver.quit()
            except Exception:
                pass  # Ignore all errors during cleanup
            finally:
                self.driver = None
    
    def _setup_driver(self):
        """Setup Undetected ChromeDriver - bypasses bot detection."""
        self.logger.info("Setting up Undetected ChromeDriver...")
        
        options = uc.ChromeOptions()
        
        if self.headless:
            options.add_argument('--headless=new')
            self.logger.info("  Mode: Headless")
        else:
            self.logger.info("  Mode: Visible browser (recommended)")
        
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-blink-features=AutomationControlled')
        
        # Chrome version
        if self.chrome_version:
            self.logger.info(f"  Chrome version: {self.chrome_version}")
            self.driver = uc.Chrome(options=options, use_subprocess=True, version_main=self.chrome_version)
        else:
            self.driver = uc.Chrome(options=options, use_subprocess=True)
        
        self._driver_active = True
        self.driver.set_page_load_timeout(30)
        
        try:
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except:
            pass
        
        self.logger.info("✓ Undetected ChromeDriver ready")
    
    def crawl_task(self, task: Dict) -> Tuple[List[Dict], int, str]:
        """
        Crawl a single task (one stock + one period).
        Returns: (records, new_scraped_count, final_status)
        """
        stock_code = task['stock_code']
        stock_name = task['stock_name']
        category = task['category']
        period_start = datetime.strptime(task['period_start'], '%Y-%m-%d')
        period_end = datetime.strptime(task['period_end'], '%Y-%m-%d')
        target = task['target']
        already_scraped = task['scraped']
        
        self.logger.info(f"  Period: {task['period_start']} → {task['period_end']}")
        self.logger.info(f"  Need: {target - already_scraped} more articles (have {already_scraped}/{target})")
        
        # Search for URLs
        date_min = period_start.strftime("%m/%d/%Y")
        date_max = period_end.strftime("%m/%d/%Y")
        
        urls, possibly_blocked = self._search_google(stock_code, stock_name, date_min, date_max)
        self.logger.info(f"    Found {len(urls)} candidate URLs")
        sys.stdout.flush()
        
        # Check for blocking
        if possibly_blocked and not urls:
            self.logger.warning(f"    ⚠️ Possibly blocked by Google - will retry later")
            self.tracker.add_failure(stock_code, stock_name, 
                                     f"{task['period_start']} to {task['period_end']}", 
                                     "Blocked by Google")
            return [], 0, 'BLOCKED'
        
        # No URLs but not blocked = genuinely no results for this period
        if not urls:
            self.logger.info(f"    No URLs found for this period")
            return [], 0, 'DONE'
        
        # Scrape articles
        records = []
        seen_titles = set()
        needed = target - already_scraped
        
        for url in urls:
            if self.shutdown_requested:
                break
            
            if len(records) >= needed:
                break
            
            self.logger.info(f"    Scraping: {url[:70]}...")
            sys.stdout.flush()
            
            article, fail_reason = self._scrape_article(url, period_start, period_end)
            
            if article:
                title_normalized = article['title'].lower().strip()
                if title_normalized in seen_titles:
                    self.logger.info(f"      ✗ Duplicate title")
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
                self.tracker.add_record(record)
                
                self.logger.info(f"      ✓ [{already_scraped + len(records)}/{target}] {article['title'][:50]}... ({article['word_count']} words)")
                sys.stdout.flush()
            else:
                self.logger.info(f"      ✗ {fail_reason}")
                sys.stdout.flush()
        
        # Determine final status
        total_scraped = already_scraped + len(records)
        
        if self.shutdown_requested:
            # Interrupted - save partial progress
            status = 'PARTIAL' if total_scraped > 0 else 'PENDING'
        elif total_scraped >= target:
            status = 'DONE'
        elif total_scraped > 0:
            status = 'PARTIAL'
        elif possibly_blocked:
            status = 'BLOCKED'
        else:
            # Tried but found nothing valid = treat as done for this period
            status = 'DONE'
        
        return records, len(records), status
    
    def _search_google(self, stock_code: str, stock_name: str, date_min: str, date_max: str) -> Tuple[List[str], bool]:
        """Search Google with date range. Returns (urls, possibly_blocked)."""
        urls = []
        possibly_blocked = False
        date_filter = f"cdr:1,cd_min:{date_min},cd_max:{date_max}"
        
        queries = [
            f"{stock_name} {stock_code} stock news",
            f"{stock_name} {stock_code} financial news",
            f"{stock_code} news",
        ]
        
        for query_idx, query in enumerate(queries):
            if self.shutdown_requested:
                break
            
            try:
                search_url = f"https://www.google.com/search?q={quote(query)}&tbs={quote(date_filter)}&hl=en"
                
                self.driver.get(search_url)
                self._interruptible_sleep(random.uniform(2, 4))  # Human-like delay
                
                # Check for blocking
                page_source = self.driver.page_source.lower()
                if 'captcha' in page_source or 'unusual traffic' in page_source:
                    self.logger.warning("    ⚠️ Google blocking detected...")
                    possibly_blocked = True
                    self._interruptible_sleep(random.uniform(5, 10))
                
                # Extract URLs
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
                
                if urls:
                    break  # Found URLs, stop trying queries
                
                if query_idx < len(queries) - 1:
                    self._interruptible_sleep(random.uniform(1, 2))
                    
            except Exception as e:
                self.logger.debug(f"Search error: {e}")
                continue
        
        return urls[:30], possibly_blocked
    
    def _interruptible_sleep(self, seconds: float):
        """Sleep that can be interrupted by shutdown signal."""
        _shutdown_event.wait(timeout=seconds)
    
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
                return None, "Timeout"
            
            self._interruptible_sleep(random.uniform(1, 2))
            
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
        
        # Default to middle of period
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
    
    def close(self):
        """Close the WebDriver safely."""
        self._safe_cleanup()


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Financial News Crawler v5 (UNDETECTED CHROME)")
    logger.info("Granular progress tracking (per stock + period)")
    logger.info("=" * 60)
    
    config = load_config()
    
    # Display settings
    period_days = config.get('date_range', {}).get('period_days', 7)
    results_per_period = config.get('search', {}).get('results_per_period', 2)
    lookback_days = config.get('date_range', {}).get('lookback_days', 14)
    num_periods = lookback_days // period_days
    min_words = config.get('scraping', {}).get('min_content_words', 500)
    headless = config.get('scraping', {}).get('headless', False)
    
    logger.info(f"Settings:")
    logger.info(f"  Lookback: {lookback_days} days, Period: {period_days} days = {num_periods} periods")
    logger.info(f"  Articles per period: {results_per_period}")
    logger.info(f"  Min words: {min_words}")
    logger.info(f"  Headless: {headless}")
    logger.info("Press Ctrl+C to stop gracefully")
    logger.info("=" * 60)
    
    # Initialize components
    tracker = ProgressTracker(config)
    saver = DataSaver()
    crawler = RobustCrawler(config, tracker)
    
    # Handle --reset flag
    if '--reset' in sys.argv:
        reset_count = tracker.reset_blocked_and_errors()
        if reset_count == 0:
            logger.info("No BLOCKED/ERROR tasks to reset")
    
    total_start = time.time()
    all_records = []
    tasks_done = 0
    tasks_blocked = 0
    current_stock = None
    stock_records = []
    
    logger.info(f"\n{tracker.get_summary()}")
    logger.info("")
    
    try:
        while True:
            if crawler.shutdown_requested:
                break
            
            # Get next task
            task = tracker.get_next_task()
            if task is None:
                logger.info("\n✅ All tasks completed!")
                break
            
            # Log stock change
            if task['stock_code'] != current_stock:
                # Save checkpoint for previous stock
                if current_stock and stock_records:
                    checkpoint = tracker.save_checkpoint(current_stock, stock_records[0]['Stock_Name'], stock_records)
                    if checkpoint:
                        logger.info(f"  Checkpoint saved: {checkpoint.name}")
                
                current_stock = task['stock_code']
                stock_records = []
                logger.info(f"\n{'='*50}")
                logger.info(f"Stock: {task['stock_name']} ({task['stock_code']}) [{task['category']}]")
                logger.info(f"{'='*50}")
            
            # Crawl the task
            records, new_count, status = crawler.crawl_task(task)
            all_records.extend(records)
            stock_records.extend(records)
            
            # Update progress
            total_scraped = task['scraped'] + new_count
            tracker.update_task(task['stock_code'], task['period_start'], total_scraped, status)
            
            # Log result
            if status == 'DONE':
                tasks_done += 1
                logger.info(f"    ✅ DONE: {total_scraped}/{task['target']} articles")
            elif status == 'BLOCKED':
                tasks_blocked += 1
                logger.warning(f"    ⚠️ BLOCKED - will retry")
            elif status == 'PARTIAL':
                logger.warning(f"    ⚠ PARTIAL: {total_scraped}/{task['target']} articles")
            elif status == 'ERROR':
                logger.error(f"    ❌ ERROR: {total_scraped}/{task['target']} articles")
            
            # Small delay between tasks
            if not crawler.shutdown_requested:
                _shutdown_event.wait(timeout=random.uniform(0.5, 1.5))
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save last stock checkpoint
        if current_stock and stock_records:
            checkpoint = tracker.save_checkpoint(current_stock, stock_records[0]['Stock_Name'], stock_records)
            if checkpoint:
                logger.info(f"  Checkpoint saved: {checkpoint.name}")
        
        logger.info("\nClosing WebDriver...")
        crawler.close()
        logger.info("WebDriver closed.")
    
    total_elapsed = time.time() - total_start
    
    # Final save
    logger.info("\n" + "=" * 60)
    if crawler.shutdown_requested:
        logger.info("⚠️  Session interrupted - progress saved")
    else:
        logger.info("Pipeline finished!")
    logger.info("=" * 60)
    
    # Save final results
    output_format = config.get('output', {}).get('format', 'both')
    if all_records:
        excel_path, jsonl_path = saver.save(all_records, output_format)
    else:
        logger.info("No new records collected in this session.")
    
    # Summary
    print("\n" + "=" * 60)
    print("CRAWLER v5 SUMMARY")
    print("=" * 60)
    print(f"Session articles: {len(all_records)}")
    print(f"Tasks done this session: {tasks_done}")
    print(f"Tasks blocked: {tasks_blocked}")
    print(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"\n{tracker.get_summary()}")
    print("=" * 60)
    
    if tracker.is_all_done():
        print("\n✅ ALL TASKS COMPLETED!")
        print(f"   You can delete the progress file if you want to start fresh.")
        print(f"   Progress file: {tracker.progress_file}")
    else:
        print("\n⚠️  Some tasks incomplete. Run again to continue.")
        print(f"   Progress file: {tracker.progress_file}")
        print("\n   To retry BLOCKED/ERROR tasks: python mainv5.py --reset")


if __name__ == '__main__':
    main()
