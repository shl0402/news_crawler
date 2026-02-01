"""
=============================================================================
Financial News Crawler - Fast Test Version (Using LangChain)
=============================================================================
Alternative implementation using LangChain WebBaseLoader for faster scraping.
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import yaml
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential
from dateutil import parser as date_parser

# LangChain imports for faster scraping
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =============================================================================
# Configuration & Logging
# =============================================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    logger = logging.getLogger('FastNewsCrawler')
    logger.setLevel(log_level)
    logger.handlers = []
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# Fast News Searcher (Same search logic, faster scraping)
# =============================================================================

class FastNewsSearcher:
    """Search for news using RSS feeds and direct search."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.delay = config.get('request', {}).get('delay_seconds', 2)
        self.num_results = config.get('search', {}).get('results_per_stock', 5)
        self.user_agent = config.get('request', {}).get('user_agent', 
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        self.timeout = config.get('request', {}).get('timeout', 30)
    
    def search_stock_news(self, stock_code: str, stock_name: str, 
                          lookback_days: int) -> List[str]:
        """Search for news articles using multiple methods."""
        self.logger.info(f"Searching news for: {stock_name} ({stock_code}) [target: {self.num_results}]")
        
        urls = set()
        
        # Method 1: Google News RSS
        urls.update(self._search_google_news_rss(stock_code, stock_name))
        
        # Method 2: Yahoo Finance RSS
        if len(urls) < self.num_results:
            urls.update(self._search_yahoo_finance_rss(stock_code))
        
        # Method 3: DuckDuckGo HTML
        if len(urls) < self.num_results:
            urls.update(self._search_duckduckgo(stock_code, stock_name))
        
        valid_urls = [url for url in urls if self._is_valid_news_url(url)][:self.num_results]
        self.logger.info(f"Found {len(valid_urls)} news URLs for {stock_name}")
        return valid_urls
    
    def _search_google_news_rss(self, stock_code: str, stock_name: str) -> List[str]:
        """Search using Google News RSS feed."""
        urls = []
        try:
            import feedparser
            query = f"{stock_name} {stock_code} stock"
            feed_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-HK&gl=HK&ceid=HK:en"
            
            headers = {'User-Agent': self.user_agent}
            response = requests.get(feed_url, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                for entry in feed.entries[:20]:
                    url = self._extract_actual_url(entry.get('link', ''))
                    if url:
                        urls.append(url)
                        
        except Exception as e:
            self.logger.warning(f"Google News RSS failed: {e}")
        
        return urls
    
    def _search_yahoo_finance_rss(self, stock_code: str) -> List[str]:
        """Search using Yahoo Finance RSS."""
        urls = []
        try:
            import feedparser
            feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={stock_code}&region=HK&lang=en-HK"
            
            headers = {'User-Agent': self.user_agent}
            response = requests.get(feed_url, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                for entry in feed.entries[:15]:
                    url = entry.get('link', '')
                    if url and 'yahoo' in url.lower():
                        urls.append(url)
                        
        except Exception as e:
            self.logger.warning(f"Yahoo Finance RSS failed: {e}")
        
        return urls
    
    def _search_duckduckgo(self, stock_code: str, stock_name: str) -> List[str]:
        """Search using DuckDuckGo HTML."""
        urls = []
        try:
            from urllib.parse import quote, unquote, parse_qs, urlparse
            
            query = f"{stock_name} {stock_code} stock news"
            search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
            
            headers = {'User-Agent': self.user_agent}
            response = requests.get(search_url, headers=headers, timeout=self.timeout)
            
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
            self.logger.warning(f"DuckDuckGo search failed: {e}")
        
        return urls
    
    def _extract_actual_url(self, google_news_url: str) -> Optional[str]:
        """Extract actual URL from Google News redirect."""
        if not google_news_url:
            return None
        
        # Direct URL
        if 'news.google.com' not in google_news_url:
            return google_news_url
        
        try:
            # Try following redirect
            headers = {'User-Agent': self.user_agent}
            response = requests.get(google_news_url, headers=headers, 
                                   timeout=10, allow_redirects=True)
            if response.url and 'news.google.com' not in response.url:
                return response.url
        except:
            pass
        
        # Try base64 decode
        try:
            import base64
            if '/articles/' in google_news_url:
                article_id = google_news_url.split('/articles/')[-1].split('?')[0]
                padding = 4 - len(article_id) % 4
                if padding != 4:
                    article_id += '=' * padding
                decoded = base64.urlsafe_b64decode(article_id)
                decoded_str = decoded.decode('utf-8', errors='ignore')
                
                import re
                url_match = re.search(r'https?://[^\s<>"{}|\\^`\[\]]+', decoded_str)
                if url_match:
                    return url_match.group(0)
        except:
            pass
        
        return None
    
    def _is_valid_news_url(self, url: str) -> bool:
        """Check if URL is a valid news source."""
        if not url or not url.startswith('http'):
            return False
        
        excluded = ['youtube.com', 'twitter.com', 'facebook.com', 'instagram.com',
                   'linkedin.com', 'reddit.com', '.pdf', '.jpg', '.png', '.gif']
        return not any(ex in url.lower() for ex in excluded)


# =============================================================================
# Fast Article Scraper (Using LangChain WebBaseLoader)
# =============================================================================

class FastArticleScraper:
    """Fast article scraper using LangChain WebBaseLoader."""
    
    # Patterns for useless images
    USELESS_IMAGE_PATTERNS = [
        'logo', 'icon', 'avatar', 'badge', 'button', 'sprite', 'pixel',
        'tracking', 'ad', 'banner', 'spacer', 'blank', 'transparent',
        'facebook', 'twitter', 'linkedin', 'instagram', 'youtube', 'social',
        'share', 'follow', 'like', 'comment', 'emoji', 'smiley',
        '1x1', '2x2', 'placeholder', 'default', 'generic',
        'gravatar', 'profile-pic', 'user-icon',
        '.svg', 'base64', 'data:image'
    ]
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.timeout = config.get('request', {}).get('timeout', 30)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=200
        )
    
    def scrape_article(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape article using LangChain WebBaseLoader."""
        try:
            start_time = time.time()
            
            # Use LangChain WebBaseLoader for fast scraping with timeout
            loader = WebBaseLoader(
                web_paths=[url],
                requests_kwargs={'timeout': 15}
            )
            docs = loader.load()
            
            if not docs:
                return None
            
            # Get full content
            full_content = "\n".join(doc.page_content for doc in docs)
            
            # Clean content
            content = self._clean_content(full_content)
            
            if len(content) < 100:
                return None
            
            # Extract title from metadata or content
            title = docs[0].metadata.get('title', '')
            if not title:
                title = self._extract_title_from_content(content)
            
            # Extract publish date
            publish_date = self._extract_date(docs[0].metadata, content)
            
            # Extract source
            source = self._extract_source(url)
            
            # Extract images from page
            images = self._extract_images_from_url(url)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Scraped in {elapsed:.1f}s: {title[:50]}... ({len(images)} images)")
            
            return {
                'title': title,
                'content': content,
                'publish_date': publish_date,
                'source': source,
                'url': url,
                'images': images
            }
            
        except Exception as e:
            self.logger.error(f"Failed to scrape {url}: {e}")
            return None
    
    def _clean_content(self, content: str) -> str:
        """Clean extracted content."""
        import re
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Remove common boilerplate
        boilerplate = [
            r'Cookie[s]? (Policy|Settings|Consent)',
            r'Privacy Policy',
            r'Terms (of|and) (Service|Use)',
            r'Subscribe to',
            r'Sign up for',
            r'Advertisement',
            r'Read more:',
            r'Related Articles',
        ]
        for pattern in boilerplate:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content.strip()
    
    def _extract_title_from_content(self, content: str) -> str:
        """Extract title from content."""
        lines = content.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if 20 < len(line) < 200:
                return line
        return content[:100] if content else "Untitled"
    
    def _extract_date(self, metadata: Dict, content: str) -> Optional[datetime]:
        """Extract publish date from metadata or content."""
        # Try metadata
        date_fields = ['date', 'publishedTime', 'datePublished', 'article:published_time']
        for field in date_fields:
            if field in metadata:
                try:
                    return date_parser.parse(metadata[field])
                except:
                    pass
        
        # Try to find date in content
        import re
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, content[:1000])
            if match:
                try:
                    return date_parser.parse(match.group(0))
                except:
                    pass
        
        return datetime.now()
    
    def _extract_source(self, url: str) -> str:
        """Extract news source from URL."""
        from urllib.parse import urlparse
        
        domain = urlparse(url).netloc.lower()
        domain = domain.replace('www.', '')
        
        source_map = {
            'yahoo': 'YAHOO', 'reuters': 'REUTERS', 'bloomberg': 'BLOOMBERG',
            'cnbc': 'CNBC', 'cnn': 'CNN', 'bbc': 'BBC', 'fool': 'FOOL',
            'seekingalpha': 'SEEKINGALPHA', 'marketwatch': 'MARKETWATCH',
            'wsj': 'WSJ', 'ft.com': 'FT', 'scmp': 'SCMP', 'hkej': 'HKEJ',
            'etnet': 'ETNET', 'aastocks': 'AASTOCKS'
        }
        
        for key, value in source_map.items():
            if key in domain:
                return value
        
        return domain.split('.')[0].upper()[:10]
    
    def _extract_images_from_url(self, url: str) -> List[str]:
        """Extract useful images from URL."""
        images = []
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all images
                for img in soup.find_all('img'):
                    src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                    if src and self._is_useful_image(src):
                        # Normalize URL
                        if src.startswith('//'):
                            src = 'https:' + src
                        elif src.startswith('/'):
                            from urllib.parse import urlparse
                            parsed = urlparse(url)
                            src = f"{parsed.scheme}://{parsed.netloc}{src}"
                        
                        if src.startswith('http') and src not in images:
                            images.append(src)
                            
        except Exception as e:
            pass
        
        return images[:20]  # Limit to 20 images
    
    def _is_useful_image(self, url: str) -> bool:
        """Check if image URL is useful (not logo/icon/etc)."""
        url_lower = url.lower()
        return not any(pattern in url_lower for pattern in self.USELESS_IMAGE_PATTERNS)


# =============================================================================
# Stock Price Fetcher (Same as original)
# =============================================================================

class StockPriceFetcher:
    """Fetch stock prices using yfinance."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.price_cache = {}
    
    def load_price_data(self, stock_code: str, lookback_days: int) -> pd.DataFrame:
        """Load historical price data for a stock."""
        if stock_code in self.price_cache:
            return self.price_cache[stock_code]
        
        self.logger.info(f"Loading price data for {stock_code}")
        
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
                
                self.price_cache[stock_code] = df
                self.logger.info(f"Loaded {len(df)} price records for {stock_code}")
                return df
                
        except Exception as e:
            self.logger.error(f"Failed to load price data for {stock_code}: {e}")
        
        return pd.DataFrame()
    
    def get_price_at_time(self, stock_code: str, news_time: datetime) -> Optional[Dict]:
        """Get stock price closest to news publication time."""
        if stock_code not in self.price_cache:
            return None
        
        df = self.price_cache[stock_code]
        if df.empty:
            return None
        
        try:
            if news_time.tzinfo:
                news_time = news_time.replace(tzinfo=None)
            
            df['time_diff'] = abs(df['Datetime'] - news_time)
            closest_idx = df['time_diff'].idxmin()
            closest_row = df.loc[closest_idx]
            
            return {
                'price': float(closest_row['Close']),
                'open': float(closest_row['Open']),
                'close': float(closest_row['Close']),
                'change_pct': float((closest_row['Close'] - closest_row['Open']) / closest_row['Open'] * 100),
                'timestamp': closest_row['Datetime']
            }
            
        except Exception as e:
            return None


# =============================================================================
# Data Saver (Same as original)
# =============================================================================

class DataSaver:
    """Save collected data to various formats."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.get('output', {}).get('directory', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, records: List[Dict], output_format: str = 'both'):
        """Save records to specified format(s)."""
        if not records:
            self.logger.warning("No records to save")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"fast_news_data_{timestamp}"
        
        df = pd.DataFrame(records)
        
        if output_format in ['excel', 'both']:
            self._save_excel(df, base_name)
        
        if output_format in ['jsonl', 'both']:
            self._save_jsonl(records, base_name)
    
    def _save_excel(self, df: pd.DataFrame, base_name: str):
        """Save to Excel."""
        try:
            # Handle datetime columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = df[col].apply(
                            lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo else x
                        )
                    except:
                        pass
            
            # Convert lists to strings for Excel
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
                    # Convert datetime objects to ISO strings
                    clean_record = {}
                    for k, v in record.items():
                        if isinstance(v, datetime):
                            clean_record[k] = v.isoformat()
                        elif isinstance(v, pd.Timestamp):
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

def run_fast_pipeline():
    """Run the fast news crawler pipeline."""
    
    # Load config
    config = load_config()
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("Starting FAST Financial News Crawler (LangChain)")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Initialize components
    searcher = FastNewsSearcher(config, logger)
    scraper = FastArticleScraper(config, logger)
    price_fetcher = StockPriceFetcher(config, logger)
    saver = DataSaver(config, logger)
    
    # Get config values
    stocks = config.get('stocks', {})
    lookback_days = config.get('date_range', {}).get('lookback_days', 180)
    delay = config.get('request', {}).get('delay_seconds', 2)
    output_format = config.get('output', {}).get('format', 'both')
    
    all_records = []
    
    # Process each category and stock
    for category, stock_list in stocks.items():
        logger.info(f"\n--- Processing Category: {category} ---")
        
        for stock in stock_list:
            stock_code = stock['code']
            stock_name = stock['name']
            
            logger.info(f"\nProcessing: {stock_name} ({stock_code})")
            
            # Load price data
            price_df = price_fetcher.load_price_data(stock_code, lookback_days)
            
            # Search for news
            urls = searcher.search_stock_news(stock_code, stock_name, lookback_days)
            
            # Scrape each article
            for url in urls:
                article = scraper.scrape_article(url)
                
                if article and article.get('title') and article.get('content'):
                    # Get price at news time
                    price_info = None
                    if article.get('publish_date'):
                        price_info = price_fetcher.get_price_at_time(
                            stock_code, article['publish_date']
                        )
                    
                    # Create record
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
                        'News_URL': url
                    }
                    
                    all_records.append(record)
                    logger.info(f"Collected: {article['title'][:50]}... ({len(article.get('images', []))} images)")
                else:
                    logger.warning(f"Incomplete article: {url}")
                
                time.sleep(delay)
    
    # Save results
    logger.info("\n" + "=" * 60)
    logger.info(f"Pipeline complete. Total records: {len(all_records)}")
    logger.info("=" * 60)
    
    if all_records:
        saver.save(all_records, output_format)
        logger.info(f"Saved {len(all_records)} records to output")
    
    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total articles collected: {len(all_records)}")
    print(f"Total time: {elapsed:.1f} seconds")
    
    if all_records:
        df = pd.DataFrame(all_records)
        print(f"\nArticles by Category:\n{df['Category'].value_counts()}")
        print(f"\nArticles by Stock:\n{df['Stock_Code'].value_counts()}")
        print(f"\nArticles by Source:\n{df['News_Source'].value_counts()}")
    
    print("=" * 60)
    
    return all_records


if __name__ == "__main__":
    run_fast_pipeline()
