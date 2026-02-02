"""
=============================================================================
Financial News Crawler - Main Version 5 (mainv5.py)
=============================================================================
Production version with UNDETECTED CHROMEDRIVER to bypass Google bot detection.

Features:
- Uses undetected-chromedriver to avoid CAPTCHA/blocking
- Graceful shutdown handling (Ctrl+C)
- Auto-save progress after each stock
- Resume capability from progress file
- Failure report for stocks with insufficient data
- Crash recovery with incremental saves

Install: pip install undetected-chromedriver
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
            'completed_stocks': [],
            'blocked_stocks': [],
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
    
    def is_stock_blocked(self, stock_code: str) -> bool:
        """Check if a stock was marked as blocked."""
        return stock_code in self.progress.get('blocked_stocks', [])
    
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
        self.progress['current_stock'] = None
        self.progress['current_period'] = 0
        self.save_progress()
    
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
        
        with open(self.failure_file, 'a', encoding='utf-8') as f:
            f.write(f"[{failure['timestamp']}] {stock_code} ({stock_name}): Got {actual}/{expected} articles. {details}\n")
    
    def add_record(self, record: Dict):
        """Add a record and save incrementally."""
        self.session_records.append(record)
        
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
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
    
    def get_summary(self) -> str:
        """Get progress summary."""
        return f"Completed stocks: {len(self.progress.get('completed_stocks', []))}\nTotal articles: {self.progress.get('total_articles', 0)}\nFailures: {len(self.failures)}"
    
    def clear_progress(self):
        """Clear progress file after successful completion."""
        if self.progress_file.exists():
            self.progress_file.unlink()
        if self.session_file.exists():
            self.session_file.unlink()


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
            df[col] = df[col].apply(lambda x: '|'.join(x) if isinstance(x, list) else x)
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
            self.logger.info(f"Excel saved: {excel_path}")
        
        if output_format in ['jsonl', 'both']:
            jsonl_path = self.output_dir / f"{self.base_filename}.jsonl"
            self.logger.info(f"  Saving JSONL file...")
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    f.write(json.dumps(row.to_dict(), ensure_ascii=False, default=str) + '\n')
            self.logger.info(f"JSONL saved: {jsonl_path}")
        
        self.logger.info("✓ All files saved successfully!")
        return excel_path, jsonl_path


# =============================================================================
# Robust Crawler with Undetected ChromeDriver
# =============================================================================

class RobustCrawler:
    """
    Web crawler using undetected-chromedriver to bypass Google bot detection.
    """
    
    def __init__(self, config: Dict, tracker: ProgressTracker):
        self.config = config
        self.tracker = tracker
        self.logger = logging.getLogger('NewsCrawlerV5')
        
        # Settings
        self.period_days = config.get('scraping', {}).get('period_days', 7)
        self.results_per_period = config.get('scraping', {}).get('results_per_period', 2)
        lookback_days = config.get('scraping', {}).get('lookback_days', 14)
        self.num_periods = lookback_days // self.period_days
        self.min_words = config.get('scraping', {}).get('min_content_words', 500)
        self.target_per_stock = self.results_per_period * self.num_periods
        
        # Headless mode setting
        self.headless = config.get('scraping', {}).get('headless', False)
        
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
            try:
                if self.driver:
                    self.driver.quit()
            except:
                pass
            sys.exit(1)
    
    def _setup_driver(self):
        """Setup Undetected ChromeDriver - bypasses bot detection."""
        self.logger.info("Setting up Undetected ChromeDriver...")
        
        options = uc.ChromeOptions()
        
        # Headless mode (v2 for undetected-chromedriver)
        if self.headless:
            options.add_argument('--headless=new')
            self.logger.info("  Mode: Headless")
        else:
            self.logger.info("  Mode: Visible browser (better for avoiding detection)")
        
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-blink-features=AutomationControlled')
        
        # Use undetected_chromedriver
        self.driver = uc.Chrome(options=options, use_subprocess=True)
        self.driver.set_page_load_timeout(30)
        
        # Add some human-like behavior
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        self.logger.info("✓ Undetected ChromeDriver ready")
    
    def crawl_stock(self, stock_code: str, stock_name: str, category: str) -> tuple:
        """Crawl news for a single stock with progress tracking."""
        self.logger.info(f"  Target: {self.target_per_stock} articles ({self.results_per_period}/period x {self.num_periods} periods)")
        
        self.tracker.mark_stock_started(stock_code)
        
        all_records = []
        today = datetime.now()
        seen_titles = set()
        possibly_blocked = False
        
        for period_num in range(1, self.num_periods + 1):
            if self.shutdown_requested:
                self.logger.warning(f"  Shutdown requested, stopping at period {period_num}")
                break
            
            period_end = today - timedelta(days=(period_num - 1) * self.period_days)
            period_start = period_end - timedelta(days=self.period_days)
            
            date_min = period_start.strftime("%m/%d/%Y")
            date_max = period_end.strftime("%m/%d/%Y")
            
            self.logger.info(f"  Period {period_num}/{self.num_periods}: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
            sys.stdout.flush()
            
            urls, period_blocked = self._search_google(stock_code, stock_name, date_min, date_max)
            self.logger.info(f"    Found {len(urls)} candidate URLs")
            sys.stdout.flush()
            
            if period_blocked:
                possibly_blocked = True
            
            if not urls:
                if period_blocked:
                    self.logger.warning(f"    ⚠️ No URLs found for period {period_num} (possibly blocked by Google)")
                else:
                    self.logger.warning(f"    No URLs found for period {period_num}")
                self.tracker.mark_period_completed(stock_code, period_num)
                continue
            
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
            
            if len(period_articles) < self.results_per_period:
                self.logger.warning(f"    ⚠ Period {period_num}: {len(period_articles)}/{self.results_per_period} articles")
            else:
                self.logger.info(f"    ✓ Period {period_num}: {len(period_articles)}/{self.results_per_period} articles")
            
            self.tracker.mark_period_completed(stock_code, period_num)
            time.sleep(random.uniform(1, 2))  # Random delay between periods
        
        # Check if we were interrupted
        if self.shutdown_requested:
            self.logger.info(f"  ⚠️ Interrupted with {len(all_records)} articles collected (will retry on resume)")
            return all_records, len(all_records), False
        
        real_article_count = len(all_records)
        
        if not all_records:
            if possibly_blocked:
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
                self.tracker.mark_stock_blocked(stock_code)
            else:
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
        
        if real_article_count < self.target_per_stock and not possibly_blocked:
            self.tracker.add_failure(
                stock_code, stock_name, 
                self.target_per_stock, real_article_count,
                f"Found {real_article_count} articles (need {self.target_per_stock})"
            )
        
        was_actually_blocked = (real_article_count == 0) and possibly_blocked
        
        return all_records, real_article_count, was_actually_blocked
    
    def _search_google(self, stock_code: str, stock_name: str, date_min: str, date_max: str) -> tuple:
        """Search Google with date range. Returns (urls, possibly_blocked)."""
        urls = []
        possibly_blocked = False
        date_filter = f"cdr:1,cd_min:{date_min},cd_max:{date_max}"
        
        queries = [
            f"{stock_name} {stock_code} yfinance",
            f"{stock_name} {stock_code} stock news",
            f"{stock_name} {stock_code} financial news",
        ]
        
        for query_idx, query in enumerate(queries):
            if self.shutdown_requested:
                break
            
            try:
                search_url = f"https://www.google.com/search?q={quote(query)}&tbs={quote(date_filter)}&hl=en"
                self.logger.info(f"    Search: {search_url[:100]}...")
                sys.stdout.flush()
                
                self.driver.get(search_url)
                
                # Random human-like delay
                time.sleep(random.uniform(2, 4))
                
                page_source = self.driver.page_source.lower()
                if 'captcha' in page_source or 'unusual traffic' in page_source:
                    self.logger.warning("    ⚠️ Google blocking detected...")
                    possibly_blocked = True
                    time.sleep(random.uniform(5, 10))
                
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
                
                # If no results, try with different query
                if not urls and query_idx < len(queries) - 1:
                    time.sleep(random.uniform(1, 2))
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
                return None, "Timeout"
            
            time.sleep(random.uniform(1, 2))
            
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
    
    def close(self):
        """Close the WebDriver."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Financial News Crawler v5 (UNDETECTED CHROME)")
    logger.info("With bot-detection bypass for Google searches")
    logger.info("=" * 60)
    
    config = load_config()
    
    period_days = config.get('scraping', {}).get('period_days', 7)
    results_per_period = config.get('scraping', {}).get('results_per_period', 2)
    lookback_days = config.get('scraping', {}).get('lookback_days', 14)
    num_periods = lookback_days // period_days
    target_per_stock = results_per_period * num_periods
    min_words = config.get('scraping', {}).get('min_content_words', 500)
    headless = config.get('scraping', {}).get('headless', False)
    
    logger.info(f"Settings: {results_per_period}/period x {num_periods} periods ({period_days} days) = {target_per_stock}/stock")
    logger.info(f"Min words: {min_words}")
    logger.info(f"Headless mode: {headless}")
    logger.info("Press Ctrl+C to stop gracefully")
    logger.info("=" * 60)
    
    tracker = ProgressTracker()
    saver = DataSaver()
    crawler = RobustCrawler(config, tracker)
    
    total_start = time.time()
    
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
                
                if tracker.is_stock_completed(stock_code):
                    logger.info(f"\n--- Skipping: {stock_name} ({stock_code}) [Already completed] ---")
                    stocks_skipped += 1
                    continue
                
                if tracker.is_stock_blocked(stock_code):
                    logger.info(f"\n--- Retrying: {stock_name} ({stock_code}) [Was blocked before] ---")
                else:
                    logger.info(f"\n--- Processing: {stock_name} ({stock_code}) ---")
                
                stock_records, real_count, was_blocked = crawler.crawl_stock(stock_code, stock_name, category)
                all_records.extend(stock_records)
                
                if crawler.shutdown_requested:
                    if stock_records:
                        checkpoint_file = tracker.save_stock_checkpoint(stock_code, stock_name, stock_records, saver.base_filename)
                        logger.info(f"  Stock file saved: {checkpoint_file}")
                    logger.info(f"  ⚠️ Stock interrupted - will retry on next run")
                    break
                
                if stock_records:
                    checkpoint_file = tracker.save_stock_checkpoint(stock_code, stock_name, stock_records, saver.base_filename)
                    logger.info(f"  Stock file saved: {checkpoint_file}")
                
                if was_blocked:
                    stocks_blocked += 1
                    logger.info(f"  Progress: {stocks_blocked} stocks blocked (will retry)")
                elif real_count > 0:
                    tracker.mark_stock_completed(stock_code, real_count)
                    stocks_processed += 1
                    total_real_articles = sum(1 for r in all_records if r.get('News_Title') not in ['NO_ARTICLES_FOUND', 'POSSIBLY_BLOCKED'])
                    logger.info(f"  Progress: {stocks_processed} stocks with articles, {total_real_articles} real articles")
                else:
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
    
    logger.info("\n" + "=" * 60)
    if crawler.shutdown_requested:
        logger.info("⚠️  Shutdown requested - saving all collected data...")
    else:
        logger.info("Pipeline complete!")
    logger.info("=" * 60)
    
    logger.info("Saving progress tracker...")
    tracker.save_progress()
    logger.info(f"  Progress saved to: {tracker.progress_file}")
    
    if all_records:
        excel_path, jsonl_path = saver.save(all_records, output_format)
    else:
        logger.info("No records collected in this session.")
    
    # Summary
    real_articles = [r for r in all_records if r.get('News_Title') not in ['NO_ARTICLES_FOUND', 'POSSIBLY_BLOCKED']]
    no_articles_count = sum(1 for r in all_records if r.get('News_Title') == 'NO_ARTICLES_FOUND')
    blocked_count = sum(1 for r in all_records if r.get('News_Title') == 'POSSIBLY_BLOCKED')
    
    print("\n" + "=" * 60)
    print("CRAWLER v5 SUMMARY (Undetected Chrome)")
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
    
    if real_articles:
        df = pd.DataFrame(real_articles)
        if 'Word_Count' in df.columns:
            valid_wc = df[df['Word_Count'] > 0]['Word_Count']
            if len(valid_wc) > 0:
                print(f"\nContent Quality (valid articles only):")
                print(f"  Average word count: {int(valid_wc.mean())} words")
                print(f"  Min word count: {int(valid_wc.min())} words")
                print(f"  Max word count: {int(valid_wc.max())} words")
        
        if 'Stock_Code' in df.columns:
            print(f"\nArticles per Stock:")
            for code in df['Stock_Code'].unique():
                count = len(df[df['Stock_Code'] == code])
                status = "✓" if count >= target_per_stock else "✗"
                print(f"  {code}: {count}/{target_per_stock} {status}")
    
    print("=" * 60)
    
    if not crawler.shutdown_requested and not tracker.progress.get('blocked_stocks'):
        tracker.clear_progress()
        print("\n✓ Session completed successfully. Progress cleared.")
    else:
        print("\n⚠️  Session interrupted or has blocked stocks. Resume by running again.")
        print(f"Progress saved in: {tracker.progress_file}")


if __name__ == '__main__':
    main()
