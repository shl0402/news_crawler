"""
=============================================================================
Financial News Crawler & Data Pipeline
=============================================================================
A robust crawler for collecting financial news and correlating with stock prices.
Designed for the "Ontology Chatbot" university project.

Author: Data Engineering Team
Version: 1.0.0
=============================================================================
"""

import os
import sys
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
from newspaper import Article, Config as NewspaperConfig
from googlesearch import search as google_search
from tenacity import retry, stop_after_attempt, wait_exponential
from dateutil import parser as date_parser


# =============================================================================
# Configuration Loader
# =============================================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging based on configuration."""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_file = log_config.get('file', 'crawler.log')
    
    # Create logger
    logger = logging.getLogger('FinancialNewsCrawler')
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# News Search Module
# =============================================================================

class NewsSearcher:
    """Search for news articles using multiple sources including RSS feeds and direct APIs."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.delay = config.get('request', {}).get('delay_seconds', 2)
        self.num_results = config.get('search', {}).get('results_per_stock', 5)
        self.language = config.get('search', {}).get('language', 'en')
        self.user_agent = config.get('request', {}).get('user_agent', '')
        self.timeout = config.get('request', {}).get('timeout', 30)
        
        # News RSS feeds for HK stocks
        self.rss_feeds = [
            "https://news.google.com/rss/search?q={query}&hl=en-HK&gl=HK&ceid=HK:en",
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s={stock_code}&region=HK&lang=en-HK",
        ]
    
    def search_stock_news(self, stock_code: str, stock_name: str, 
                          lookback_days: int) -> List[str]:
        """Search for news articles about a specific stock using multiple methods."""
        self.logger.info(f"Searching news for: {stock_name} ({stock_code}) [target: {self.num_results}]")
        
        urls = set()  # Use set to avoid duplicates
        
        # Method 1: Google News RSS
        urls.update(self._search_google_news_rss(stock_code, stock_name))
        
        # Method 2: Yahoo Finance RSS (if not enough)
        if len(urls) < self.num_results:
            urls.update(self._search_yahoo_finance_rss(stock_code))
        
        # Method 3: Direct search via DuckDuckGo HTML (if not enough)
        if len(urls) < self.num_results:
            urls.update(self._search_duckduckgo(stock_code, stock_name))
        
        # Method 4: Google Search fallback (if still not enough)
        if len(urls) < self.num_results:
            urls.update(self._search_google(stock_code, stock_name))
        
        # Filter valid URLs and limit to exact number requested
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
                for entry in feed.entries[:20]:  # Get more entries to ensure minimum
                    # Google News redirects - get actual URL
                    link = entry.get('link', '')
                    if link:
                        # Try to extract actual URL from Google redirect
                        actual_url = self._extract_actual_url(link)
                        if actual_url:
                            urls.append(actual_url)
                
                self.logger.debug(f"Google News RSS found {len(urls)} URLs")
                
        except Exception as e:
            self.logger.debug(f"Google News RSS search failed: {str(e)}")
        
        return urls
    
    def _search_yahoo_finance_rss(self, stock_code: str) -> List[str]:
        """Search using Yahoo Finance RSS feed."""
        urls = []
        try:
            import feedparser
            # Yahoo Finance RSS for stock news
            feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={stock_code}&region=HK&lang=en-HK"
            
            headers = {'User-Agent': self.user_agent}
            response = requests.get(feed_url, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                for entry in feed.entries[:20]:  # Get more entries
                    link = entry.get('link', '')
                    if link and self._is_valid_news_url(link):
                        urls.append(link)
                
                self.logger.debug(f"Yahoo Finance RSS found {len(urls)} URLs")
                
        except Exception as e:
            self.logger.debug(f"Yahoo Finance RSS search failed: {str(e)}")
        
        return urls
    
    def _search_duckduckgo(self, stock_code: str, stock_name: str) -> List[str]:
        """Search using DuckDuckGo HTML (more reliable, no API needed)."""
        urls = []
        try:
            query = f"{stock_name} {stock_code} stock news"
            search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            response = requests.get(search_url, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'lxml')
                # Find result links
                for link in soup.find_all('a', class_='result__url'):
                    href = link.get('href', '')
                    if href and self._is_valid_news_url(href):
                        urls.append(href)
                
                # Also try result__a links
                for link in soup.find_all('a', class_='result__a'):
                    href = link.get('href', '')
                    if href and href.startswith('http') and self._is_valid_news_url(href):
                        urls.append(href)
                
                self.logger.debug(f"DuckDuckGo found {len(urls)} URLs")
                
        except Exception as e:
            self.logger.debug(f"DuckDuckGo search failed: {str(e)}")
        
        return urls
    
    def _search_google(self, stock_code: str, stock_name: str) -> List[str]:
        """Fallback to Google Search library."""
        urls = []
        try:
            query = f'"{stock_name}" OR "{stock_code}" stock news Hong Kong'
            
            search_results = google_search(
                query, 
                num_results=20,  # Get more results
                lang=self.language,
                advanced=False
            )
            
            for url in search_results:
                if self._is_valid_news_url(url):
                    urls.append(url)
                time.sleep(0.5)
                
        except Exception as e:
            self.logger.debug(f"Google search failed: {str(e)}")
        
        return urls
    
    def _extract_actual_url(self, google_redirect_url: str) -> Optional[str]:
        """Extract actual URL from Google News redirect."""
        try:
            # Google News uses redirect URLs - try multiple methods
            if 'news.google.com' in google_redirect_url:
                # Method 1: Try to follow redirect with GET request
                headers = {
                    'User-Agent': self.user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                }
                try:
                    response = requests.get(google_redirect_url, headers=headers, 
                                           allow_redirects=True, timeout=15)
                    if response.url and 'news.google.com' not in response.url:
                        return response.url
                except:
                    pass
                
                # Method 2: Parse the redirect URL for embedded link
                import base64
                import re
                
                # Try to extract URL from the path
                if '/articles/' in google_redirect_url:
                    # The article ID is base64 encoded
                    try:
                        path_part = google_redirect_url.split('/articles/')[-1].split('?')[0]
                        # Pad the base64 string if needed
                        padding = 4 - len(path_part) % 4
                        if padding != 4:
                            path_part += '=' * padding
                        decoded = base64.urlsafe_b64decode(path_part)
                        # Look for URL patterns in decoded content
                        urls = re.findall(rb'https?://[^\s<>"\']+', decoded)
                        for url in urls:
                            url_str = url.decode('utf-8', errors='ignore')
                            if 'google.com' not in url_str and len(url_str) > 20:
                                return url_str
                    except:
                        pass
                
                # If all methods fail, skip this URL
                return None
            return google_redirect_url
        except:
            return None
    
    def _is_valid_news_url(self, url: str) -> bool:
        """Check if URL is a valid news article URL."""
        if not url or not url.startswith('http'):
            return False
        
        # Skip Google redirect URLs that weren't resolved
        if 'news.google.com' in url:
            return False
            
        # Skip social media, video sites, etc.
        skip_domains = [
            'youtube.com', 'twitter.com', 'facebook.com', 'instagram.com',
            'linkedin.com', 'tiktok.com', 'reddit.com', 'wikipedia.org',
            'investopedia.com/terms', 'google.com/search', 'bing.com/search',
            'duckduckgo.com', 'yahoo.com/search'
        ]
        
        url_lower = url.lower()
        for domain in skip_domains:
            if domain in url_lower:
                return False
        
        return True


# =============================================================================
# Article Scraper Module
# =============================================================================

class ArticleScraper:
    """Scrape and parse news articles."""
    
    # Domains/patterns for useless images (logos, icons, tracking pixels)
    USELESS_IMAGE_PATTERNS = [
        'logo', 'icon', 'avatar', 'favicon', 'sprite', 'pixel', 'tracking',
        'advertisement', 'banner', 'button', 'widget', 'share', 'social',
        'twitter', 'facebook', 'linkedin', 'instagram', 'youtube',
        'gravatar', 'placeholder', 'default', 'blank', 'spacer',
        's.yimg.com', 'sb.scorecardresearch', 'pixel.quantserve',
        'b.scorecardresearch', 'analytics', 'beacon', '/ads/',
        '1x1', '2x2', 'transparent.gif', 'blank.gif'
    ]
    
    # Minimum dimensions for useful images
    MIN_IMAGE_WIDTH = 200
    MIN_IMAGE_HEIGHT = 150
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.delay = config.get('request', {}).get('delay_seconds', 2)
        self.timeout = config.get('request', {}).get('timeout', 30)
        self.user_agent = config.get('request', {}).get('user_agent', '')
        
        # Setup newspaper config
        self.newspaper_config = NewspaperConfig()
        self.newspaper_config.browser_user_agent = self.user_agent
        self.newspaper_config.request_timeout = self.timeout
        self.newspaper_config.fetch_images = True
        self.newspaper_config.memoize_articles = False
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
    def scrape_article(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape a single article and extract relevant information."""
        self.logger.debug(f"Scraping: {url}")
        
        try:
            # Download and parse article
            article = Article(url, config=self.newspaper_config)
            article.download()
            article.parse()
            
            # Extract all useful images
            all_images = self._extract_useful_images(article, url)
            
            # Extract data
            data = {
                'url': url,
                'title': article.title or '',
                'content': article.text or '',
                'publish_date': article.publish_date,
                'source': self._extract_source(url),
                'image_urls': all_images,  # List of all useful images
                'authors': article.authors or []
            }
            
            # Validate - must have title and content
            if not data['title'] or not data['content']:
                self.logger.warning(f"Incomplete article data from: {url}")
                return None
            
            # If no publish date, try to extract from page
            if not data['publish_date']:
                data['publish_date'] = self._try_extract_date(article.html)
            
            self.logger.info(f"Successfully scraped: {data['title'][:50]}... ({len(all_images)} images)")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to scrape {url}: {str(e)}")
            return None
    
    def _extract_useful_images(self, article: Article, url: str) -> List[str]:
        """Extract all useful images from article, filtering out logos/icons."""
        useful_images = []
        seen_urls = set()
        
        # Collect all image candidates
        candidates = []
        
        # 1. Top image from newspaper3k
        if article.top_image:
            candidates.append(article.top_image)
        
        # 2. All images from newspaper3k
        if hasattr(article, 'images') and article.images:
            candidates.extend(list(article.images))
        
        # 3. Extract from HTML using BeautifulSoup for more thorough search
        if article.html:
            candidates.extend(self._extract_images_from_html(article.html, url))
        
        # Filter and deduplicate
        for img_url in candidates:
            if not img_url or img_url in seen_urls:
                continue
            
            # Normalize URL
            img_url = self._normalize_image_url(img_url, url)
            if not img_url:
                continue
            
            # Check if it's a useful image
            if self._is_useful_image(img_url):
                seen_urls.add(img_url)
                useful_images.append(img_url)
        
        return useful_images
    
    def _extract_images_from_html(self, html: str, base_url: str) -> List[str]:
        """Extract image URLs from HTML content."""
        images = []
        try:
            soup = BeautifulSoup(html, 'lxml')
            
            # Find all img tags
            for img in soup.find_all('img'):
                # Try different attributes
                for attr in ['src', 'data-src', 'data-lazy-src', 'data-original']:
                    src = img.get(attr, '')
                    if src and src.startswith(('http', '//')):
                        images.append(src)
                        break
                
                # Also check srcset
                srcset = img.get('srcset', '')
                if srcset:
                    # Parse srcset and get the largest image
                    parts = srcset.split(',')
                    for part in parts:
                        src = part.strip().split(' ')[0]
                        if src.startswith(('http', '//')):
                            images.append(src)
            
            # Find og:image meta tags
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                images.append(og_image['content'])
            
            # Find twitter:image meta tags  
            twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
            if twitter_image and twitter_image.get('content'):
                images.append(twitter_image['content'])
                
        except Exception as e:
            self.logger.debug(f"Error extracting images from HTML: {e}")
        
        return images
    
    def _normalize_image_url(self, img_url: str, base_url: str) -> Optional[str]:
        """Normalize image URL to absolute URL."""
        try:
            from urllib.parse import urljoin, urlparse
            
            # Handle protocol-relative URLs
            if img_url.startswith('//'):
                img_url = 'https:' + img_url
            
            # Handle relative URLs
            if not img_url.startswith('http'):
                img_url = urljoin(base_url, img_url)
            
            # Validate URL structure
            parsed = urlparse(img_url)
            if not parsed.scheme or not parsed.netloc:
                return None
            
            return img_url
            
        except Exception:
            return None
    
    def _is_useful_image(self, img_url: str) -> bool:
        """Check if image URL is likely a useful content image (not logo/icon)."""
        if not img_url:
            return False
        
        img_url_lower = img_url.lower()
        
        # Check against useless patterns
        for pattern in self.USELESS_IMAGE_PATTERNS:
            if pattern in img_url_lower:
                return False
        
        # Check file extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
        has_valid_ext = any(ext in img_url_lower for ext in valid_extensions)
        
        # URLs without extensions might still be valid (dynamic images)
        # So we only reject if it clearly has a bad extension
        bad_extensions = ['.svg', '.ico', '.cur']
        has_bad_ext = any(ext in img_url_lower for ext in bad_extensions)
        
        if has_bad_ext:
            return False
        
        # Check for very small image indicators in URL
        small_indicators = ['_thumb', '_small', '_tiny', '_mini', '50x50', '100x100', '32x32', '16x16']
        for indicator in small_indicators:
            if indicator in img_url_lower:
                return False
        
        return True
    
    def _extract_source(self, url: str) -> str:
        """Extract source name from URL."""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            # Get main domain name
            parts = domain.split('.')
            if len(parts) >= 2:
                return parts[-2].upper()
            return domain.upper()
        except:
            return "UNKNOWN"
    
    def _try_extract_date(self, html: str) -> Optional[datetime]:
        """Try to extract publish date from HTML if newspaper3k fails."""
        if not html:
            return None
        
        try:
            soup = BeautifulSoup(html, 'lxml')
            
            # Common date meta tags
            date_selectors = [
                ('meta', {'property': 'article:published_time'}),
                ('meta', {'name': 'pubdate'}),
                ('meta', {'name': 'publishdate'}),
                ('meta', {'name': 'date'}),
                ('meta', {'property': 'og:updated_time'}),
                ('time', {'datetime': True}),
            ]
            
            for tag, attrs in date_selectors:
                element = soup.find(tag, attrs)
                if element:
                    date_str = element.get('content') or element.get('datetime')
                    if date_str:
                        try:
                            return date_parser.parse(date_str)
                        except:
                            continue
            
            return None
            
        except Exception:
            return None


# =============================================================================
# Stock Price Module
# =============================================================================

class StockPriceFetcher:
    """Fetch stock price data using yfinance."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.price_cache: Dict[str, pd.DataFrame] = {}
    
    def preload_stock_data(self, stock_code: str, lookback_days: int) -> None:
        """Preload stock data for a given stock."""
        if stock_code in self.price_cache:
            return
        
        self.logger.info(f"Loading price data for {stock_code}")
        
        try:
            ticker = yf.Ticker(stock_code)
            
            # Get data with some buffer
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 7)
            
            # Get historical data
            df = ticker.history(start=start_date, end=end_date, interval='1h')
            
            if df.empty:
                # Try daily data if hourly not available
                df = ticker.history(start=start_date, end=end_date, interval='1d')
                self.logger.warning(f"Using daily data for {stock_code} (hourly unavailable)")
            
            if not df.empty:
                self.price_cache[stock_code] = df
                self.logger.info(f"Loaded {len(df)} price records for {stock_code}")
            else:
                self.logger.warning(f"No price data found for {stock_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to load price data for {stock_code}: {str(e)}")
    
    def get_price_at_time(self, stock_code: str, 
                          news_time: Optional[datetime]) -> Dict[str, Any]:
        """Get stock price closest to the news publication time."""
        result = {
            'price_at_news_time': None,
            'price_open': None,
            'price_close': None,
            'price_change_percent': None,
            'price_timestamp': None
        }
        
        if stock_code not in self.price_cache:
            self.logger.warning(f"No cached price data for {stock_code}")
            return result
        
        df = self.price_cache[stock_code]
        
        if news_time is None:
            # Use most recent price if no news time
            self.logger.debug("No news time, using most recent price")
            if not df.empty:
                latest = df.iloc[-1]
                result['price_at_news_time'] = float(latest['Close'])
                result['price_timestamp'] = str(df.index[-1])
            return result
        
        try:
            # Make news_time timezone-aware if needed
            if news_time.tzinfo is None:
                # Assume HK timezone
                import pytz
                hk_tz = pytz.timezone('Asia/Hong_Kong')
                news_time = hk_tz.localize(news_time)
            
            # Convert index to comparable format
            if df.index.tz is not None:
                news_time = news_time.astimezone(df.index.tz)
            else:
                news_time = news_time.replace(tzinfo=None)
            
            # Find closest time
            time_diffs = abs(df.index - news_time)
            closest_idx = time_diffs.argmin()
            closest_row = df.iloc[closest_idx]
            
            result['price_at_news_time'] = float(closest_row['Close'])
            result['price_open'] = float(closest_row['Open'])
            result['price_close'] = float(closest_row['Close'])
            result['price_timestamp'] = str(df.index[closest_idx])
            
            # Calculate price change
            if result['price_open'] and result['price_open'] > 0:
                change = ((result['price_close'] - result['price_open']) 
                          / result['price_open'] * 100)
                result['price_change_percent'] = round(change, 4)
            
            self.logger.debug(f"Found price {result['price_at_news_time']} at {result['price_timestamp']}")
            
        except Exception as e:
            self.logger.error(f"Error getting price at time: {str(e)}")
        
        return result


# =============================================================================
# Data Saver Module
# =============================================================================

class DataSaver:
    """Save collected data to Excel and/or JSONL."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        output_config = config.get('output', {})
        self.output_dir = Path(output_config.get('directory', 'output'))
        self.filename = output_config.get('filename', 'financial_news_data')
        self.output_format = output_config.get('format', 'both')
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, data: List[Dict[str, Any]]) -> None:
        """Save data to configured format(s)."""
        if not data:
            self.logger.warning("No data to save!")
            return
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Reorder columns
        column_order = [
            'Stock_Code', 'Category', 'News_Date', 'News_Title', 
            'News_Source', 'News_Content', 'Image_URLs', 
            'Price_At_News_Time', 'Price_Open', 'Price_Close',
            'Price_Change_Percent', 'Price_Timestamp', 'News_URL'
        ]
        
        # Only include columns that exist
        columns = [col for col in column_order if col in df.columns]
        df = df[columns]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save based on format
        if self.output_format in ['excel', 'both']:
            self._save_excel(df, timestamp)
        
        if self.output_format in ['jsonl', 'both']:
            self._save_jsonl(data, timestamp)
        
        self.logger.info(f"Saved {len(data)} records to {self.output_dir}")
    
    def _save_excel(self, df: pd.DataFrame, timestamp: str) -> None:
        """Save data to Excel file."""
        filepath = self.output_dir / f"{self.filename}_{timestamp}.xlsx"
        
        try:
            # Convert timezone-aware datetimes to timezone-naive for Excel compatibility
            df_excel = df.copy()
            for col in df_excel.columns:
                if df_excel[col].dtype == 'object':
                    # Check if column contains datetime objects
                    sample = df_excel[col].dropna().iloc[0] if not df_excel[col].dropna().empty else None
                    if isinstance(sample, datetime):
                        df_excel[col] = df_excel[col].apply(
                            lambda x: x.replace(tzinfo=None) if isinstance(x, datetime) and x is not None and hasattr(x, 'tzinfo') and x.tzinfo else x
                        )
                elif hasattr(df_excel[col].dtype, 'tz') and df_excel[col].dtype.tz is not None:
                    # For datetime64[ns, tz] columns
                    df_excel[col] = df_excel[col].dt.tz_localize(None)
            
            # Create Excel writer with formatting
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df_excel.to_excel(writer, sheet_name='News_Data', index=False)
                
                # Auto-adjust column widths
                worksheet = writer.sheets['News_Data']
                for idx, col in enumerate(df_excel.columns):
                    max_length = max(
                        df_excel[col].astype(str).apply(len).max(),
                        len(col)
                    )
                    # Cap at 50 characters
                    adjusted_width = min(max_length + 2, 50)
                    # Use column letter properly (handles columns beyond Z)
                    from openpyxl.utils import get_column_letter
                    col_letter = get_column_letter(idx + 1)
                    worksheet.column_dimensions[col_letter].width = adjusted_width
            
            self.logger.info(f"Excel saved: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save Excel: {str(e)}")
    
    def _save_jsonl(self, data: List[Dict[str, Any]], timestamp: str) -> None:
        """Save data to JSONL file."""
        filepath = self.output_dir / f"{self.filename}_{timestamp}.jsonl"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for record in data:
                    # Convert datetime objects to strings
                    record_clean = {}
                    for k, v in record.items():
                        if isinstance(v, datetime):
                            record_clean[k] = v.isoformat()
                        else:
                            record_clean[k] = v
                    f.write(json.dumps(record_clean, ensure_ascii=False) + '\n')
            
            self.logger.info(f"JSONL saved: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save JSONL: {str(e)}")


# =============================================================================
# Main Pipeline
# =============================================================================

class FinancialNewsPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config)
        
        self.searcher = NewsSearcher(self.config, self.logger)
        self.scraper = ArticleScraper(self.config, self.logger)
        self.price_fetcher = StockPriceFetcher(self.config, self.logger)
        self.saver = DataSaver(self.config, self.logger)
        
        self.delay = self.config.get('request', {}).get('delay_seconds', 2)
        self.lookback_days = self.config.get('date_range', {}).get('lookback_days', 180)
    
    def run(self) -> None:
        """Run the complete pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("Starting Financial News Crawler Pipeline")
        self.logger.info("=" * 60)
        
        all_data = []
        stocks_config = self.config.get('stocks', {})
        
        # Process each category
        for category, stocks in stocks_config.items():
            self.logger.info(f"\n--- Processing Category: {category} ---")
            
            for stock in stocks:
                stock_code = stock['code']
                stock_name = stock['name']
                
                self.logger.info(f"\nProcessing: {stock_name} ({stock_code})")
                
                # Preload stock price data
                self.price_fetcher.preload_stock_data(stock_code, self.lookback_days)
                
                # Search for news
                news_urls = self.searcher.search_stock_news(
                    stock_code, stock_name, self.lookback_days
                )
                
                # Process each news article
                for url in news_urls:
                    # Respect rate limiting
                    time.sleep(self.delay)
                    
                    # Scrape article
                    article_data = self.scraper.scrape_article(url)
                    
                    if article_data is None:
                        continue
                    
                    # Get price at news time
                    price_data = self.price_fetcher.get_price_at_time(
                        stock_code, article_data.get('publish_date')
                    )
                    
                    # Build final record
                    record = {
                        'Stock_Code': stock_code,
                        'Category': category,
                        'News_Date': article_data.get('publish_date'),
                        'News_Title': article_data.get('title', ''),
                        'News_Source': article_data.get('source', ''),
                        'News_Content': article_data.get('content', ''),
                        'Image_URLs': article_data.get('image_urls', []),  # List of all useful images
                        'Price_At_News_Time': price_data.get('price_at_news_time'),
                        'Price_Open': price_data.get('price_open'),
                        'Price_Close': price_data.get('price_close'),
                        'Price_Change_Percent': price_data.get('price_change_percent'),
                        'Price_Timestamp': price_data.get('price_timestamp'),
                        'News_URL': url
                    }
                    
                    all_data.append(record)
                    img_count = len(record['Image_URLs'])
                    self.logger.info(f"Collected: {record['News_Title'][:40]}... ({img_count} images)")
        
        # Save all data
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"Pipeline complete. Total records: {len(all_data)}")
        self.logger.info(f"{'=' * 60}")
        
        self.saver.save(all_data)
        
        # Print summary
        self._print_summary(all_data)
    
    def _print_summary(self, data: List[Dict[str, Any]]) -> None:
        """Print a summary of collected data."""
        if not data:
            return
        
        df = pd.DataFrame(data)
        
        print("\n" + "=" * 60)
        print("COLLECTION SUMMARY")
        print("=" * 60)
        print(f"Total articles collected: {len(data)}")
        print("\nArticles by Category:")
        print(df['Category'].value_counts().to_string())
        print("\nArticles by Stock:")
        print(df['Stock_Code'].value_counts().to_string())
        print("\nArticles by Source:")
        print(df['News_Source'].value_counts().head(10).to_string())
        print("=" * 60)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point."""
    try:
        pipeline = FinancialNewsPipeline()
        pipeline.run()
    except FileNotFoundError:
        print("ERROR: config.yaml not found. Please create the configuration file.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Pipeline failed - {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
