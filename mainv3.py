"""
=============================================================================
Financial News Crawler - Main Version 3 (mainv3.py)
=============================================================================
Uses Google Search with DATE RANGE filtering for targeted news scraping.
Processes WEEK BY WEEK - ensures exactly results_per_stock_per_week articles per week.
"""

import os
import json
import logging
import time
import random
import sys
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
    logger = logging.getLogger('GoogleSearchCrawler')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# Google Search with Date Range - Week by Week
# =============================================================================

class WeekByWeekCrawler:
    """
    Crawls news week by week, ensuring exactly N articles per week.
    Uses Google Search with date range filtering.
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
        self.results_per_period = config.get('search', {}).get('results_per_period', 
                                    config.get('search', {}).get('results_per_stock_per_week', 2))
        self.lookback_days = config.get('date_range', {}).get('lookback_days', 84)
        self.period_days = config.get('date_range', {}).get('period_days', 7)
        self.num_periods = max(1, self.lookback_days // self.period_days)
        self.min_words = config.get('scraping', {}).get('min_content_words', 500)
        
        # Single driver for search + scrape (sequential processing)
        self.driver = None
        self._setup_driver()
    
    def _setup_driver(self):
        """Setup a single Selenium WebDriver."""
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
    
    def crawl_stock(self, stock_code: str, stock_name: str) -> List[Dict]:
        """
        Crawl news for a single stock, period by period.
        Returns list of article records.
        """
        total_target = self.results_per_period * self.num_periods
        self.logger.info(f"Target: {total_target} articles ({self.results_per_period}/period x {self.num_periods} periods of {self.period_days} days)")
        
        all_articles = []
        today = datetime.now()
        
        # Track seen titles across all periods to avoid duplicates
        seen_titles = set()
        
        for period_num in range(1, self.num_periods + 1):
            # Calculate date range for this period
            period_end = today - timedelta(days=(period_num - 1) * self.period_days)
            period_start = period_end - timedelta(days=self.period_days)
            
            date_min = period_start.strftime("%m/%d/%Y")
            date_max = period_end.strftime("%m/%d/%Y")
            
            self.logger.info(f"\n  Period {period_num}/{self.num_periods}: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
            sys.stdout.flush()
            
            # Get URLs for this period from Google
            urls = self._search_google_week(stock_code, stock_name, date_min, date_max)
            self.logger.info(f"    Found {len(urls)} candidate URLs")
            sys.stdout.flush()
            
            if not urls:
                self.logger.warning(f"    No URLs found for period {period_num}, skipping...")
                sys.stdout.flush()
                continue
            
            # Scrape URLs one by one until we have enough for this period
            period_articles = []
            url_index = 0
            
            while len(period_articles) < self.results_per_period and url_index < len(urls):
                url = urls[url_index]
                url_index += 1
                
                article, fail_reason = self._scrape_article(url, period_num, period_start, period_end)
                
                if article:
                    # Check for duplicate title
                    title_normalized = article['title'].lower().strip()
                    if title_normalized in seen_titles:
                        self.logger.info(f"    ✗ Duplicate title, skipping...")
                        sys.stdout.flush()
                        continue
                    
                    seen_titles.add(title_normalized)
                    period_articles.append(article)
                    
                    # LOG IMMEDIATELY when article is found
                    current_total = len(all_articles) + len(period_articles)
                    self.logger.info(f"    ✓ [{len(period_articles)}/{self.results_per_period}] {article['title'][:50]}... ({article['word_count']} words)")
                    sys.stdout.flush()  # Force flush to show immediately
                else:
                    # Log the failure reason
                    self.logger.info(f"    ✗ {fail_reason}")
                    sys.stdout.flush()
            
            all_articles.extend(period_articles)
            
            if len(period_articles) < self.results_per_period:
                self.logger.warning(f"    ⚠ Only got {len(period_articles)}/{self.results_per_period} articles for period {period_num}")
            else:
                self.logger.info(f"    ✓ Period {period_num} complete: {len(period_articles)}/{self.results_per_period} articles")
            sys.stdout.flush()
            
            # Small delay between periods
            time.sleep(random.uniform(0.5, 1))
        
        self.logger.info(f"\n  Total: {len(all_articles)}/{total_target} articles for {stock_name}")
        sys.stdout.flush()
        return all_articles
    
    def _search_google_week(self, stock_code: str, stock_name: str, date_min: str, date_max: str) -> List[str]:
        """
        Search Google for a specific week's date range.
        Returns list of URLs.
        Strategy: Try with 'yfinance' first, fallback to basic query.
        """
        urls = []
        date_filter = f"cdr:1,cd_min:{date_min},cd_max:{date_max}"
        
        # Query strategies: try yfinance first, then fallback
        queries = [
            f"{stock_name} {stock_code} yfinance",  # Primary: with yfinance
            f"{stock_name} {stock_code} stock",           # Fallback: basic query
        ]
        
        for query_idx, query in enumerate(queries):
            try:
                # Use regular Google search with date range (not News tab)
                search_url = f"https://www.google.com/search?q={quote(query)}&tbs={quote(date_filter)}&hl=en"
                
                self.logger.info(f"    Searching: {search_url}")
                sys.stdout.flush()
                
                self.driver.get(search_url)
                time.sleep(3)
                
                # Debug: Check if we got blocked
                page_source = self.driver.page_source.lower()
                if 'captcha' in page_source or 'unusual traffic' in page_source:
                    self.logger.warning("    Google may be blocking - trying alternative...")
                    time.sleep(5)
                
                # Extract URLs from regular Google search results
                seen = set()
                selectors = [
                    'div.g a[href]',
                    'div.yuRUbf a',
                    'h3 a',
                    'a[data-ved]',
                    'a[href*="http"]'
                ]
                
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
                
                # If no results, try with 'news' keyword added
                if not urls:
                    search_url = f"https://www.google.com/search?q={quote(query + ' news')}&tbs={quote(date_filter)}&hl=en"
                    self.logger.info(f"    Retry with news keyword: {search_url}")
                    sys.stdout.flush()
                    self.driver.get(search_url)
                    time.sleep(3)
                    
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
                
                # If we found URLs, break out of query loop
                if urls:
                    if query_idx == 0:
                        self.logger.debug(f"    Found results with yfinance query")
                    break
                else:
                    if query_idx == 0:
                        self.logger.debug(f"    No results with yfinance, trying fallback...")
                
            except Exception as e:
                self.logger.debug(f"Google search error: {e}")
                continue
        
        return urls[:30]
    
    def _is_valid_news_url(self, url: str) -> bool:
        """Check if URL is a valid news article."""
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
        """Scrape a single article. Returns (article_dict, fail_reason) tuple."""
        try:
            start_time = time.time()
            
            self.logger.info(f"    Scraping: {url}")
            sys.stdout.flush()
            
            self.driver.get(url)
            
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except:
                return None, f"Page load timeout: {url[:50]}..."
            
            time.sleep(1)
            
            # Extract content
            content = self._extract_content()
            if not content:
                return None, f"No content found: {url[:50]}..."
            
            word_count = len(content.split())
            if word_count < self.min_words:
                return None, f"Not enough words: {word_count}/{self.min_words}"
            
            title = self._extract_title()
            publish_date = self._extract_date()
            
            # Use period's date if no date found
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
            return None, f"Scrape error: {str(e)[:50]}"
    
    def _extract_content(self) -> str:
        """Extract article content."""
        selectors = [
            'article',
            '.caas-body',
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
        """Extract article title."""
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
        """Extract publish date."""
        parsed_date = None
        
        try:
            meta_selectors = [
                'meta[property="article:published_time"]',
                'meta[name="pubdate"]',
                'meta[name="date"]',
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
        """Extract news source from URL."""
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
        """Extract useful images."""
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
        """Format datetime with GMT+X timezone string."""
        if dt is None:
            return "Unknown"
        
        if dt.tzinfo is None:
            return dt.strftime('%Y-%m-%d %H:%M:%S') + " (GMT+0)"
        
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
    
    def close(self):
        """Close WebDriver."""
        if self.driver:
            self.driver.quit()


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
    """Run the week-by-week news crawler pipeline."""
    
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
    logger.info("Financial News Crawler v3 (mainv3.py)")
    logger.info("Period-by-Period Processing with Google Date Range")
    logger.info(f"Settings: {results_per_period} articles/period x {num_periods} periods ({period_days} days each) = {target_per_stock} target/stock")
    logger.info(f"Min words: {min_words}")
    logger.info("=" * 60)
    
    total_start = time.time()
    
    # Initialize crawler
    crawler = WeekByWeekCrawler(config, logger)
    saver = DataSaver(config, logger)
    
    stocks = config.get('stocks', {})
    output_format = config.get('output', {}).get('format', 'both')
    
    all_records = []
    
    try:
        for category, stock_list in stocks.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Category: {category}")
            logger.info(f"{'='*60}")
            
            for stock in stock_list:
                stock_code = stock['code']
                stock_name = stock['name']
                
                logger.info(f"\n--- Processing: {stock_name} ({stock_code}) ---")
                
                # Crawl week by week
                articles = crawler.crawl_stock(stock_code, stock_name)
                
                # If no articles found, add a placeholder record so stock still appears in report
                if not articles:
                    placeholder_record = {
                        'Stock_Code': stock_code,
                        'Category': category,
                        'News_Date': None,
                        'News_Date_Formatted': 'No articles found',
                        'Period_Num': 0,
                        'News_Title': 'NO ARTICLES FOUND',
                        'News_Source': 'N/A',
                        'News_Content': f'No valid articles found for {stock_name} ({stock_code}) within the specified date range and word count threshold.',
                        'Content_Length': 0,
                        'Word_Count': 0,
                        'Image_URLs': [],
                        'News_URL': '',
                        'Scrape_Time_Sec': 0
                    }
                    all_records.append(placeholder_record)
                    logger.warning(f"  Added placeholder record for {stock_name} - no articles found")
                else:
                    # Convert to records
                    for article in articles:
                        news_date = article.get('publish_date')
                        news_date_formatted = crawler._format_timezone(news_date)
                        
                        record = {
                            'Stock_Code': stock_code,
                            'Category': category,
                            'News_Date': news_date,
                            'News_Date_Formatted': news_date_formatted,
                            'Period_Num': article.get('period_num', 0),
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
    
    finally:
        crawler.close()
    
    total_elapsed = time.time() - total_start
    
    # Save results
    logger.info("\n" + "=" * 60)
    logger.info(f"Pipeline complete. Total records: {len(all_records)}")
    logger.info("=" * 60)
    
    if all_records:
        saver.save(all_records, output_format)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CRAWLER v3 SUMMARY (Period-by-Period)")
    print("=" * 60)
    print(f"Total articles collected: {len(all_records)}")
    print(f"Total time: {total_elapsed:.1f} seconds")
    
    if all_records:
        df = pd.DataFrame(all_records)
        
        # Content stats
        word_counts = df['Word_Count'].tolist()
        print(f"\nContent Quality:")
        print(f"  Average word count: {sum(word_counts)/len(word_counts):.0f} words")
        print(f"  Min word count: {min(word_counts)} words")
        print(f"  Max word count: {max(word_counts)} words")
        
        # By source
        print(f"\nArticles by Source:\n{df['News_Source'].value_counts()}")
        
        # By period - should show exact counts per period
        print(f"\nArticles by Period:")
        period_counts = df['Period_Num'].value_counts().sort_index()
        for period, count in period_counts.items():
            status = "✓" if count >= results_per_period else "✗"
            print(f"  Period {period}: {count}/{results_per_period} {status}")
        
        # By stock
        print(f"\nArticles per Stock:")
        for stock_code in df['Stock_Code'].unique():
            count = len(df[df['Stock_Code'] == stock_code])
            print(f"  {stock_code}: {count}/{target_per_stock}")
    
    print("=" * 60)
    
    return all_records


if __name__ == "__main__":
    run_pipeline()
