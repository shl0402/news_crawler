"""
=============================================================================
Financial News Crawler - Ultra Fast Version (Parallel Search + LangChain)
=============================================================================
Optimized for speed using:
- Concurrent/parallel news searching
- DuckDuckGo as primary (fastest response)
- ThreadPoolExecutor for parallel operations
- LangChain WebBaseLoader for fast scraping
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

# LangChain for fast scraping
from langchain_community.document_loaders import WebBaseLoader


# =============================================================================
# Configuration & Logging
# =============================================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    logger = logging.getLogger('UltraFastCrawler')
    logger.setLevel(log_level)
    logger.handlers = []
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# Ultra Fast News Searcher (Parallel Search)
# =============================================================================

class UltraFastSearcher:
    """
    Ultra-fast news searcher using parallel requests.
    Inspired by scrapfast.py multi-engine approach.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.num_results = config.get('search', {}).get('results_per_stock', 5)
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        self.timeout = 10  # Shorter timeout for speed
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a session with connection pooling for faster requests."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        # Connection pooling for speed
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=1
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def search_stock_news(self, stock_code: str, stock_name: str, 
                          lookback_days: int) -> List[str]:
        """Search for news using parallel requests to multiple sources."""
        self.logger.info(f"Searching news for: {stock_name} ({stock_code}) [target: {self.num_results}]")
        
        start_time = time.time()
        urls = set()
        
        # Run all searches in parallel for speed
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._search_duckduckgo, stock_code, stock_name): 'DuckDuckGo',
                executor.submit(self._search_yahoo_finance_rss, stock_code): 'Yahoo RSS',
                executor.submit(self._search_bing, stock_code, stock_name): 'Bing',
                executor.submit(self._search_google_news_rss, stock_code, stock_name): 'Google RSS',
            }
            
            for future in as_completed(futures, timeout=15):
                source = futures[future]
                try:
                    result_urls = future.result()
                    if result_urls:
                        urls.update(result_urls)
                        self.logger.debug(f"  {source}: found {len(result_urls)} URLs")
                except Exception as e:
                    self.logger.debug(f"  {source} failed: {e}")
        
        # Filter and limit
        valid_urls = [url for url in urls if self._is_valid_news_url(url)][:self.num_results]
        
        elapsed = time.time() - start_time
        self.logger.info(f"Found {len(valid_urls)} URLs in {elapsed:.1f}s")
        
        return valid_urls
    
    def _search_duckduckgo(self, stock_code: str, stock_name: str) -> List[str]:
        """DuckDuckGo HTML search - usually fastest."""
        urls = []
        try:
            query = f"{stock_name} {stock_code} stock news"
            search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
            
            response = self.session.get(search_url, timeout=self.timeout)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.select('a.result__a')[:15]:
                    href = link.get('href', '')
                    # Extract actual URL from DuckDuckGo redirect
                    if "duckduckgo.com/l/?" in href:
                        parsed = urlparse(href)
                        params = parse_qs(parsed.query)
                        if 'uddg' in params:
                            href = unquote(params['uddg'][0])
                    if href.startswith('http'):
                        urls.append(href)
        except Exception as e:
            pass
        return urls
    
    def _search_bing(self, stock_code: str, stock_name: str) -> List[str]:
        """Bing search - often fast and reliable."""
        urls = []
        try:
            query = f"{stock_name} {stock_code} stock news"
            search_url = f"https://www.bing.com/search?q={quote(query)}"
            
            response = self.session.get(search_url, timeout=self.timeout)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.select('#b_results h2 a')[:15]:
                    href = link.get('href', '')
                    if href.startswith('http') and 'bing.com' not in href and 'microsoft.com' not in href:
                        urls.append(href)
        except Exception as e:
            pass
        return urls
    
    def _search_yahoo_finance_rss(self, stock_code: str) -> List[str]:
        """Yahoo Finance RSS - direct finance news."""
        urls = []
        try:
            import feedparser
            feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={stock_code}&region=HK&lang=en-HK"
            
            response = self.session.get(feed_url, timeout=self.timeout)
            
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                for entry in feed.entries[:10]:
                    url = entry.get('link', '')
                    if url:
                        urls.append(url)
        except Exception as e:
            pass
        return urls
    
    def _search_google_news_rss(self, stock_code: str, stock_name: str) -> List[str]:
        """Google News RSS - comprehensive but slower."""
        urls = []
        try:
            import feedparser
            query = f"{stock_name} {stock_code} stock"
            feed_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-HK&gl=HK&ceid=HK:en"
            
            response = self.session.get(feed_url, timeout=self.timeout)
            
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                for entry in feed.entries[:10]:
                    url = self._extract_google_url(entry.get('link', ''))
                    if url:
                        urls.append(url)
        except Exception as e:
            pass
        return urls
    
    def _extract_google_url(self, google_news_url: str) -> Optional[str]:
        """Extract actual URL from Google News redirect."""
        if not google_news_url or 'news.google.com' not in google_news_url:
            return google_news_url
        
        try:
            # Try to follow redirect
            response = self.session.get(google_news_url, timeout=5, allow_redirects=True)
            if response.url and 'news.google.com' not in response.url:
                return response.url
        except:
            pass
        
        return None  # Skip if can't resolve
    
    def _is_valid_news_url(self, url: str) -> bool:
        """Check if URL is a valid news source."""
        if not url or not url.startswith('http'):
            return False
        
        excluded = ['youtube.com', 'twitter.com', 'facebook.com', 'instagram.com',
                   'linkedin.com', 'reddit.com', '.pdf', '.jpg', '.png', '.gif',
                   'google.com/search', 'bing.com/search']
        return not any(ex in url.lower() for ex in excluded)


# =============================================================================
# Ultra Fast Article Scraper (Parallel Scraping)
# =============================================================================

class UltraFastScraper:
    """
    Ultra-fast article scraper using LangChain with parallel processing.
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
        self.timeout = 10
    
    def scrape_articles_parallel(self, urls: List[str], max_workers: int = 3) -> List[Dict]:
        """Scrape multiple articles in parallel."""
        articles = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self._scrape_single, url): url for url in urls}
            
            for future in as_completed(future_to_url, timeout=60):
                url = future_to_url[future]
                try:
                    article = future.result()
                    if article:
                        articles.append(article)
                except Exception as e:
                    self.logger.debug(f"Failed to scrape {url}: {e}")
        
        return articles
    
    def _scrape_single(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape a single article using LangChain."""
        try:
            start_time = time.time()
            
            loader = WebBaseLoader(
                web_paths=[url],
                requests_kwargs={'timeout': self.timeout}
            )
            docs = loader.load()
            
            if not docs:
                return None
            
            full_content = "\n".join(doc.page_content for doc in docs)
            content = self._clean_content(full_content)
            
            if len(content) < 100:
                return None
            
            title = docs[0].metadata.get('title', '') or self._extract_title(content)
            publish_date = self._extract_date(docs[0].metadata, content)
            source = self._extract_source(url)
            images = self._extract_images(url)
            
            elapsed = time.time() - start_time
            
            return {
                'title': title,
                'content': content,
                'publish_date': publish_date,
                'source': source,
                'url': url,
                'images': images,
                'scrape_time': elapsed
            }
            
        except Exception as e:
            return None
    
    def _clean_content(self, content: str) -> str:
        """Clean extracted content."""
        import re
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Remove boilerplate
        boilerplate = [r'Cookie', r'Privacy Policy', r'Terms', r'Subscribe', r'Advertisement']
        for pattern in boilerplate:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content.strip()
    
    def _extract_title(self, content: str) -> str:
        """Extract title from content."""
        lines = content.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if 20 < len(line) < 200:
                return line
        return content[:100] if content else "Untitled"
    
    def _extract_date(self, metadata: Dict, content: str) -> Optional[datetime]:
        """Extract publish date."""
        date_fields = ['date', 'publishedTime', 'datePublished', 'article:published_time']
        for field in date_fields:
            if field in metadata:
                try:
                    return date_parser.parse(metadata[field])
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
            'scmp': 'SCMP', 'etnet': 'ETNET', 'aastocks': 'AASTOCKS'
        }
        
        for key, value in source_map.items():
            if key in domain:
                return value
        
        return domain.split('.')[0].upper()[:10]
    
    def _extract_images(self, url: str) -> List[str]:
        """Extract useful images from URL."""
        images = []
        try:
            response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for img in soup.find_all('img')[:30]:
                    src = img.get('src') or img.get('data-src')
                    if src and self._is_useful_image(src):
                        if src.startswith('//'):
                            src = 'https:' + src
                        elif src.startswith('/'):
                            parsed = urlparse(url)
                            src = f"{parsed.scheme}://{parsed.netloc}{src}"
                        if src.startswith('http') and src not in images:
                            images.append(src)
        except:
            pass
        return images[:20]
    
    def _is_useful_image(self, url: str) -> bool:
        """Check if image URL is useful."""
        url_lower = url.lower()
        return not any(p in url_lower for p in self.USELESS_IMAGE_PATTERNS)


# =============================================================================
# Stock Price Fetcher (Cached)
# =============================================================================

class CachedPriceFetcher:
    """Stock price fetcher with caching."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.cache = {}
    
    def preload_all_prices(self, stocks: Dict, lookback_days: int):
        """Preload all stock prices in parallel."""
        self.logger.info("Preloading all stock prices...")
        start_time = time.time()
        
        all_stocks = []
        for category, stock_list in stocks.items():
            for stock in stock_list:
                all_stocks.append(stock['code'])
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self._load_price, code, lookback_days): code 
                      for code in all_stocks}
            
            for future in as_completed(futures):
                code = futures[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        self.cache[code] = df
                except Exception as e:
                    self.logger.warning(f"Failed to load {code}: {e}")
        
        elapsed = time.time() - start_time
        self.logger.info(f"Preloaded {len(self.cache)} stocks in {elapsed:.1f}s")
    
    def _load_price(self, stock_code: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """Load price data for a single stock."""
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
        """Get stock price closest to news time."""
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
    """Save collected data."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.get('output', {}).get('directory', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, records: List[Dict], output_format: str = 'both'):
        """Save records."""
        if not records:
            self.logger.warning("No records to save")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"ultrafast_news_{timestamp}"
        
        df = pd.DataFrame(records)
        
        if output_format in ['excel', 'both']:
            self._save_excel(df, base_name)
        
        if output_format in ['jsonl', 'both']:
            self._save_jsonl(records, base_name)
    
    def _save_excel(self, df: pd.DataFrame, base_name: str):
        """Save to Excel."""
        try:
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = df[col].apply(
                            lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo else x
                        )
                    except:
                        pass
            
            for col in df.columns:
                df[col] = df[col].apply(lambda x: '|'.join(x) if isinstance(x, list) else x)
            
            filepath = self.output_dir / f"{base_name}.xlsx"
            df.to_excel(filepath, index=False, engine='openpyxl')
            self.logger.info(f"Excel saved: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save Excel: {e}")
    
    def _save_jsonl(self, records: List[Dict], base_name: str):
        """Save to JSONL."""
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

def run_ultrafast_pipeline():
    """Run the ultra-fast news crawler pipeline."""
    
    config = load_config()
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("ULTRA FAST Financial News Crawler")
    logger.info("(Parallel Search + Parallel Scraping + LangChain)")
    logger.info("=" * 60)
    
    total_start = time.time()
    
    # Initialize components
    searcher = UltraFastSearcher(config, logger)
    scraper = UltraFastScraper(config, logger)
    price_fetcher = CachedPriceFetcher(config, logger)
    saver = DataSaver(config, logger)
    
    # Get config values
    stocks = config.get('stocks', {})
    lookback_days = config.get('date_range', {}).get('lookback_days', 180)
    output_format = config.get('output', {}).get('format', 'both')
    
    # Preload all prices in parallel (big speedup!)
    price_fetcher.preload_all_prices(stocks, lookback_days)
    
    all_records = []
    
    # Process each stock
    for category, stock_list in stocks.items():
        logger.info(f"\n--- Category: {category} ---")
        
        for stock in stock_list:
            stock_code = stock['code']
            stock_name = stock['name']
            
            logger.info(f"\nProcessing: {stock_name} ({stock_code})")
            
            # Fast parallel search
            urls = searcher.search_stock_news(stock_code, stock_name, lookback_days)
            
            if not urls:
                logger.warning(f"No URLs found for {stock_name}")
                continue
            
            # Fast parallel scraping
            articles = scraper.scrape_articles_parallel(urls)
            
            for article in articles:
                if article and article.get('title') and article.get('content'):
                    price_info = None
                    if article.get('publish_date'):
                        price_info = price_fetcher.get_price_at_time(
                            stock_code, article['publish_date']
                        )
                    
                    record = {
                        'Stock_Code': stock_code,
                        'Category': category,
                        'News_Date': article.get('publish_date'),
                        'News_Title': article.get('title', '')[:500],
                        'News_Source': article.get('source', 'UNKNOWN'),
                        'News_Content': article.get('content', '')[:5000],
                        'Image_URLs': article.get('images', []),
                        'Price_At_News_Time': price_info.get('price') if price_info else None,
                        'Price_Open': price_info.get('open') if price_info else None,
                        'Price_Close': price_info.get('close') if price_info else None,
                        'Price_Change_Percent': price_info.get('change_pct') if price_info else None,
                        'Price_Timestamp': price_info.get('timestamp') if price_info else None,
                        'News_URL': article.get('url'),
                        'Scrape_Time_Sec': article.get('scrape_time', 0)
                    }
                    
                    all_records.append(record)
                    logger.info(f"  ✓ {article['title'][:50]}... ({article.get('scrape_time', 0):.1f}s)")
    
    # Save results
    total_elapsed = time.time() - total_start
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Pipeline complete. Total records: {len(all_records)}")
    logger.info("=" * 60)
    
    if all_records:
        saver.save(all_records, output_format)
    
    # Print summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total articles collected: {len(all_records)}")
    print(f"Total time: {total_elapsed:.1f} seconds")
    
    if all_records:
        df = pd.DataFrame(all_records)
        print(f"\nArticles by Category:\n{df['Category'].value_counts()}")
        print(f"\nArticles by Stock:\n{df['Stock_Code'].value_counts()}")
        print(f"\nArticles by Source:\n{df['News_Source'].value_counts()}")
        
        if 'Scrape_Time_Sec' in df.columns:
            avg_scrape = df['Scrape_Time_Sec'].mean()
            print(f"\nAverage scrape time: {avg_scrape:.2f}s per article")
    
    print("=" * 60)
    
    return all_records


if __name__ == "__main__":
    run_ultrafast_pipeline()
