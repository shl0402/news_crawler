"""
=============================================================================
Financial News Crawler - Selenium Version (test3.py)
=============================================================================
Uses Selenium for scraping (like scrapfast.py) to handle JavaScript-rendered pages.
Compares speed and content quality vs main.py (newspaper3k).
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
import yfinance as yf
from dateutil import parser as date_parser

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
# Fast News Searcher (Parallel - same as test2.py)
# =============================================================================

class FastSearcher:
    """Fast parallel news searcher."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.num_results = config.get('search', {}).get('results_per_stock', 5)
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        self.timeout = 10
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({'User-Agent': self.user_agent})
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=1)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def search_stock_news(self, stock_code: str, stock_name: str) -> List[str]:
        """Search for news using parallel requests."""
        self.logger.info(f"Searching news for: {stock_name} ({stock_code}) [target: {self.num_results}]")
        
        start_time = time.time()
        urls = set()
        
        # Parallel search from multiple sources
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._search_duckduckgo, stock_code, stock_name): 'DuckDuckGo',
                executor.submit(self._search_yahoo_rss, stock_code): 'Yahoo RSS',
                executor.submit(self._search_bing, stock_code, stock_name): 'Bing',
                executor.submit(self._search_google_news_rss, stock_code, stock_name): 'Google RSS',
            }
            
            for future in as_completed(futures, timeout=15):
                try:
                    result_urls = future.result()
                    if result_urls:
                        urls.update(result_urls)
                except:
                    pass
        
        # Filter and limit
        valid_urls = [url for url in urls if self._is_valid_url(url)][:self.num_results]
        
        elapsed = time.time() - start_time
        self.logger.info(f"Found {len(valid_urls)} URLs in {elapsed:.1f}s")
        
        return valid_urls
    
    def _search_duckduckgo(self, stock_code: str, stock_name: str) -> List[str]:
        urls = []
        try:
            query = f"{stock_name} {stock_code} stock news"
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
            query = f"{stock_name} {stock_code} stock news"
            search_url = f"https://www.bing.com/search?q={quote(query)}"
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
            feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={stock_code}&region=HK&lang=en-HK"
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
        """Search using Google News RSS."""
        urls = []
        try:
            import feedparser
            query = f"{stock_name} {stock_code} stock"
            feed_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-HK&gl=HK&ceid=HK:en"
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
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, num_drivers: int = 3):
        self.config = config
        self.logger = logger
        self.num_drivers = num_drivers
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
        """Scrape multiple articles in parallel using multiple drivers."""
        articles = []
        
        # Split URLs among drivers
        with ThreadPoolExecutor(max_workers=self.num_drivers) as executor:
            future_to_url = {}
            for i, url in enumerate(urls):
                driver_idx = i % self.num_drivers
                future = executor.submit(self._scrape_with_driver, url, self.drivers[driver_idx])
                future_to_url[future] = url
            
            for future in as_completed(future_to_url, timeout=120):
                url = future_to_url[future]
                try:
                    article = future.result()
                    if article:
                        articles.append(article)
                except Exception as e:
                    self.logger.debug(f"Failed {url[:40]}...: {e}")
        
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
            
            if not content or len(content) < 100:
                return None
            
            # Extract title
            title = self._extract_title(driver)
            
            # Extract date
            publish_date = self._extract_date(driver)
            
            # Extract source
            source = self._extract_source(url)
            
            # Extract images
            images = self._extract_images(driver)
            
            elapsed = time.time() - start_time
            
            return {
                'title': title,
                'content': content,
                'publish_date': publish_date,
                'source': source,
                'url': url,
                'images': images,
                'scrape_time': elapsed,
                'content_length': len(content)
            }
            
        except Exception as e:
            return None
    
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
        """Extract publish date."""
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
                        return date_parser.parse(content)
        except:
            pass
        
        # Try time elements
        try:
            time_elements = driver.find_elements(By.TAG_NAME, 'time')
            if time_elements:
                datetime_attr = time_elements[0].get_attribute('datetime')
                if datetime_attr:
                    return date_parser.parse(datetime_attr)
        except:
            pass
        
        return datetime.now()
    
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
# Stock Price Fetcher
# =============================================================================

class PriceFetcher:
    """Stock price fetcher with caching."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.cache = {}
    
    def preload_prices(self, stocks: Dict, lookback_days: int):
        """Preload all stock prices in parallel."""
        self.logger.info("Preloading stock prices...")
        start_time = time.time()
        
        all_codes = [stock['code'] for stock_list in stocks.values() for stock in stock_list]
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self._load_price, code, lookback_days): code for code in all_codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        self.cache[code] = df
                except:
                    pass
        
        elapsed = time.time() - start_time
        self.logger.info(f"Preloaded {len(self.cache)} stocks in {elapsed:.1f}s")
    
    def _load_price(self, stock_code: str, lookback_days: int) -> Optional[pd.DataFrame]:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 30)
            
            ticker = yf.Ticker(stock_code)
            df = ticker.history(start=start_date, end=end_date, interval='1h')
            
            if df.empty:
                df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if not df.empty:
                df = df.reset_index()
                if 'Datetime' in df.columns:
                    df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
                elif 'Date' in df.columns:
                    df['Datetime'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                return df
        except:
            pass
        return None
    
    def get_price_at_time(self, stock_code: str, news_time: datetime) -> Optional[Dict]:
        if stock_code not in self.cache:
            return None
        
        df = self.cache[stock_code]
        if df.empty:
            return None
        
        try:
            if news_time.tzinfo:
                news_time = news_time.replace(tzinfo=None)
            
            df['time_diff'] = abs(df['Datetime'] - news_time)
            closest_idx = df['time_diff'].idxmin()
            row = df.loc[closest_idx]
            
            return {
                'price': float(row['Close']),
                'open': float(row['Open']),
                'close': float(row['Close']),
                'change_pct': float((row['Close'] - row['Open']) / row['Open'] * 100),
                'timestamp': row['Datetime']
            }
        except:
            return None


# =============================================================================
# Data Saver
# =============================================================================

class DataSaver:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.get('output', {}).get('directory', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, records: List[Dict], output_format: str = 'both'):
        if not records:
            self.logger.warning("No records to save")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"selenium_news_{timestamp}"
        
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

def run_selenium_pipeline():
    """Run the Selenium-based news crawler pipeline."""
    
    config = load_config()
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("SELENIUM Financial News Crawler (test3.py)")
    logger.info("Testing: Selenium with PARALLEL scraping")
    logger.info("=" * 60)
    
    total_start = time.time()
    
    # Initialize components
    searcher = FastSearcher(config, logger)
    scraper = ParallelSeleniumScraper(config, logger)
    price_fetcher = PriceFetcher(config, logger)
    saver = DataSaver(config, logger)
    
    # Get config
    stocks = config.get('stocks', {})
    lookback_days = config.get('date_range', {}).get('lookback_days', 180)
    output_format = config.get('output', {}).get('format', 'both')
    results_per_stock = config.get('results_per_stock', 20)
    
    # Preload prices
    price_fetcher.preload_prices(stocks, lookback_days)
    
    all_records = []
    content_lengths = []
    
    try:
        for category, stock_list in stocks.items():
            logger.info(f"\n--- Category: {category} ---")
            
            for stock in stock_list:
                stock_code = stock['code']
                stock_name = stock['name']
                
                logger.info(f"\nProcessing: {stock_name} ({stock_code})")
                
                # Search for news
                urls = searcher.search_stock_news(stock_code, stock_name)
                
                if not urls:
                    logger.warning(f"No URLs found for {stock_name}")
                    continue
                
                # Limit to configured results_per_stock
                urls = urls[:results_per_stock]
                logger.info(f"  Scraping {len(urls)} URLs in parallel...")
                
                # Scrape all URLs in PARALLEL using multiple Selenium drivers
                articles = scraper.scrape_articles_parallel(urls)
                
                for article in articles:
                    if article and article.get('title') and article.get('content'):
                        content_len = article.get('content_length', 0)
                        content_lengths.append(content_len)
                        
                        # Get price
                        price_info = None
                        if article.get('publish_date'):
                            price_info = price_fetcher.get_price_at_time(stock_code, article['publish_date'])
                        
                        record = {
                            'Stock_Code': stock_code,
                            'Category': category,
                            'News_Date': article.get('publish_date'),
                            'News_Title': article.get('title', '')[:500],
                            'News_Source': article.get('source', 'UNKNOWN'),
                            'News_Content': article.get('content', '')[:5000],
                            'Content_Length': content_len,
                            'Image_URLs': article.get('images', []),
                            'Price_At_News_Time': price_info.get('price') if price_info else None,
                            'Price_Open': price_info.get('open') if price_info else None,
                            'Price_Close': price_info.get('close') if price_info else None,
                            'Price_Change_Percent': price_info.get('change_pct') if price_info else None,
                            'Price_Timestamp': price_info.get('timestamp') if price_info else None,
                            'News_URL': article.get('url', ''),
                            'Scrape_Time_Sec': article.get('scrape_time', 0)
                        }
                        
                        all_records.append(record)
                        logger.info(f"  ✓ {article['title'][:40]}... ({content_len} chars, {article.get('scrape_time', 0):.1f}s)")
                    else:
                        logger.debug(f"  ✗ Failed to extract content")
    
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
    print("SELENIUM CRAWLER SUMMARY")
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
        print(f"Source: {sample['News_Source']}")
        print(f"Content preview ({sample['Content_Length']} chars):")
        print("-" * 40)
        print(sample['News_Content'][:800])
        print("-" * 40)
    
    print("=" * 60)
    
    return all_records


if __name__ == "__main__":
    run_selenium_pipeline()
