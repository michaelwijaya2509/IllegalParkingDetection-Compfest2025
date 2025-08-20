import asyncio
import json
import logging
import signal
import sys
import atexit
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import feedparser
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp import types
import argparse
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time
import re
import os


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("mcp-events-server")


cleanup_done = False
shutdown_in_progress = False

def cleanup_resources():

    global cleanup_done, shutdown_in_progress
    if not cleanup_done and not shutdown_in_progress:
        shutdown_in_progress = True
        logger.info("Cleaning up resources")
        cleanup_done = True

def signal_handler(signum, frame):

    global shutdown_in_progress
    if not shutdown_in_progress:
        logger.info(f"Received signal {signum}, shutting down system.")
        cleanup_resources()
        time.sleep(0.1)
        os._exit(0)  


atexit.register(cleanup_resources)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class EventsServer:
    def __init__(self, gemini_api_key: str):

        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # RSS Feed URLs untuk Indonesian news - focus pada upcoming events
        self.rss_feeds = [
            "https://news.google.com/rss/search?q=konser+Indonesia+event+acara+2025&hl=id&gl=ID&ceid=ID:id",
            "https://news.google.com/rss/search?q=festival+Indonesia+acara+besar+2025&hl=id&gl=ID&ceid=ID:id",
            "https://news.google.com/rss/search?q=pameran+Indonesia+exhibition+2025&hl=id&gl=ID&ceid=ID:id",
            "https://news.google.com/rss/search?q=pertunjukan+Indonesia+show+2025&hl=id&gl=ID&ceid=ID:id",
            "https://news.google.com/rss/search?q=\"minggu+ini\"+\"minggu+depan\"+event+Indonesia&hl=id&gl=ID&ceid=ID:id",
            "https://news.google.com/rss/search?q=\"akhir+pekan\"+\"weekend\"+konser+Indonesia&hl=id&gl=ID&ceid=ID:id",
            "https://news.google.com/rss/search?q=\"segera\"+\"akan+datang\"+event+Indonesia&hl=id&gl=ID&ceid=ID:id",
            "https://news.google.com/rss/search?q=\"GBK\"+konser+event+acara+2025&hl=id&gl=ID&ceid=ID:id",
            "https://news.google.com/rss/search?q=\"JCC\"+\"Jakarta+Convention\"+event+2025&hl=id&gl=ID&ceid=ID:id",
            "https://news.google.com/rss/search?q=\"Istora+Senayan\"+konser+acara+2025&hl=id&gl=ID&ceid=ID:id",
            "https://news.google.com/rss/search?q=\"Indonesia+Arena\"+event+konser+2025&hl=id&gl=ID&ceid=ID:id",
            "https://news.google.com/rss/search?q=Jakarta+event+konser+festival+2025&hl=id&gl=ID&ceid=ID:id",
            "https://news.google.com/rss/search?q=Bandung+konser+festival+acara+2025&hl=id&gl=ID&ceid=ID:id",
            "https://news.google.com/rss/search?q=Surabaya+festival+acara+event+2025&hl=id&gl=ID&ceid=ID:id",
            "https://news.google.com/rss/search?q=Bali+acara+event+festival+2025&hl=id&gl=ID&ceid=ID:id",
        ]
        
        self.events_cache = []
        self.last_update = None
        self.cache_duration = timedelta(hours=1)  
        
        self.executor = ThreadPoolExecutor(max_workers=12, thread_name_prefix="events-worker")
        atexit.register(self.cleanup_executor)
        
    def cleanup_executor(self):
        try:
            if hasattr(self, 'executor') and self.executor:
                logger.info("Shutting down thread pool.")
                try:
                    future = self.executor.submit(lambda: None)
                    future.result(timeout=1.0) 
                except:
                    pass
                
                self.executor.shutdown(wait=False)
                self.executor = None
                
        except Exception as e:
            pass
        
        
    def is_within_7_days(self, date_str: str) -> bool:

        """
        Cek kalau tanggal event dalam 7 hari ke depan atau TBA
        """

        if not date_str or date_str.upper() == 'TBA' or 'TBA' in date_str:
            return True  
            
        try:
            today = datetime.now().date()
            seven_days_later = today + timedelta(days=7)
            
            if 'TBA-SOON' in date_str:
                return True  
            
            if re.match(r'\d{4}-\d{2}-\d{2}', date_str): 
                event_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                return today <= event_date <= seven_days_later
            elif re.match(r'\d{4}-\d{2}-TBA', date_str): 

                year_month = date_str[:7] 
                current_year_month = today.strftime('%Y-%m')
                next_month = (today.replace(day=1) + timedelta(days=32)).strftime('%Y-%m')
                return year_month in [current_year_month, next_month]
            
        except Exception as e:
            logger.error(f"Error checking date {date_str}: {str(e)}")
            return True 
            
        return False
        
    def fetch_single_rss(self, feed_url: str) -> List[Dict[str, Any]]:

        """
        Fetch single RSS feed untuk parallel processing
        """

        articles = []
        try:
            logger.debug(f"Fetching RSS from: {feed_url}")
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:12]: 
                article = {
                    'title': entry.title,
                    'link': entry.link,
                    'published': entry.published if hasattr(entry, 'published') else '',
                    'summary': entry.summary if hasattr(entry, 'summary') else '',
                    'source': feed.feed.title if hasattr(feed.feed, 'title') else 'Google News'
                }
                articles.append(article)
                
        except Exception as e:
            logger.error(f"Error fetching RSS {feed_url}: {str(e)}")
            
        return articles

    def fetch_article_content(self, url: str) -> str:

        """
        Fetch full article content from URL
        """

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Ambil context text
            text = soup.get_text()
            
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:5000]  
            
        except Exception as e:
            logger.debug(f"Error fetching article content from {url}: {str(e)}")
            return ""

    def parse_indonesian_date_flexible(self, date_text: str, current_year: int = 2025) -> str:

        """
        Parse Indonesian date formats 
        """

        if not date_text:
            return "TBA"
            
        if date_text.upper() == 'TBA' or 'tba' in date_text.lower():
            return "TBA"
            
        try:

            months_id = {
                'januari': '01', 'jan': '01',
                'februari': '02', 'feb': '02', 
                'maret': '03', 'mar': '03',
                'april': '04', 'apr': '04',
                'mei': '05',
                'juni': '06', 'jun': '06',
                'juli': '07', 'jul': '07',
                'agustus': '08', 'agu': '08', 'aug': '08',
                'september': '09', 'sep': '09',
                'oktober': '10', 'okt': '10', 'oct': '10',
                'november': '11', 'nov': '11',
                'desember': '12', 'des': '12', 'dec': '12'
            }
            
            date_text = date_text.lower().strip()
            
            if any(word in date_text for word in ['segera', 'akan datang', 'coming soon', 'soon']):
                return "TBA-SOON"
            if any(word in date_text for word in ['minggu ini', 'this week', 'akhir pekan']):
                return "2025-08-TBA"  
            if any(word in date_text for word in ['minggu depan', 'next week']):
                return "2025-08-TBA"  
            

            for month_name, month_num in months_id.items():
                if month_name in date_text:
                    day_pattern = r'(\d{1,2})[\s\-]*' + re.escape(month_name)
                    year_pattern = r'(20\d{2})'
                    
                    day_match = re.search(day_pattern, date_text)
                    year_match = re.search(year_pattern, date_text)
                    
                    if day_match and year_match:
                        day = day_match.group(1).zfill(2)
                        year = year_match.group(1)
                        
                        day_int = int(day)
                        if 1 <= day_int <= 31:
                            return f"{year}-{month_num}-{day}"
                    
                    if year_match:
                        year = year_match.group(1)
                        return f"{year}-{month_num}-TBA"
                    else:
                        return f"{current_year}-{month_num}-TBA"
            
            exact_match = re.search(r'(20\d{2})-(\d{2})-(\d{2})', date_text)
            if exact_match:
                year, month, day = exact_match.groups()
                month_int, day_int = int(month), int(day)
                if 1 <= month_int <= 12 and 1 <= day_int <= 31:
                    return f"{year}-{month}-{day}"
                
            slash_match = re.search(r'(\d{1,2})/(\d{1,2})/(20\d{2})', date_text)
            if slash_match:
                day, month, year = slash_match.groups()
                month_int, day_int = int(month), int(day)
                if 1 <= month_int <= 12 and 1 <= day_int <= 31:
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
        except Exception as e:
            logger.debug(f"Error parsing date '{date_text}': {str(e)}")
            
        return "TBA"
    
    async def fetch_news_from_rss(self) -> List[Dict[str, Any]]:

        """
        Ambil berita dari RSS feeds dengan parallel processing
        """

        logger.info("Starting parallel RSS fetch for 7-day events.")
        start_time = time.time()
        
        loop = asyncio.get_event_loop()
        tasks = []
        for feed_url in self.rss_feeds:
            if not shutdown_in_progress:  
                task = loop.run_in_executor(self.executor, self.fetch_single_rss, feed_url)
                tasks.append(task)
        
        if not tasks:
            return []
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        seen_titles = set()
        all_articles = []
        for result in results:
            if isinstance(result, list): 
                for article in result:
                    title_key = article['title'].lower().strip()
                    if title_key not in seen_titles:
                        seen_titles.add(title_key)
                        all_articles.append(article)
        
        fetch_time = time.time() - start_time
        logger.info(f"RSS fetch completed in {fetch_time:.2f}s, got {len(all_articles)} unique articles")
        return all_articles
    
    async def extract_event_info_with_gemini(self, article: Dict[str, Any], full_content: str = "") -> List[Dict[str, Any]]:

        """
        Extract informasi event menggunakan Gemini AI 
        """

        try:
            if shutdown_in_progress:
                return []
                
            content_to_analyze = full_content if full_content.strip() else article['summary']
            
            if not content_to_analyze.strip():
                return []
            
            prompt = f"""
            Analisis artikel Indonesia berikut dan cari EVENT yang akan segera terjadi (dalam 7 hari ke depan) atau yang tanggalnya TBA:
            
            Judul: {article['title']}
            Konten: {content_to_analyze[:4000]}
            
            ATURAN PRIORITAS:
            1. PRIORITAS UTAMA: Event yang akan terjadi dalam 7 hari ke depan
            2. TERIMA JUGA: Event dengan tanggal TBA, "segera", "akan datang", "minggu ini/depan"
            3. Terima semua jenis event: konser, festival, pameran, seminar, workshop, bazaar, pertunjukan
            4. Fokus pada event yang akan segera berlangsung
            5. Jangan ambil event yang sudah lewat atau terlalu jauh di masa depan
            
            Format JSON array:
            [
                {{
                    "event_name": "nama event (wajib ada)",
                    "event_type": "konser/festival/pameran/pertunjukan/seminar/workshop/bazaar/lainnya",
                    "date": "tanggal lengkap (YYYY-MM-DD) atau TBA/TBA-SOON",
                    "time": "waktu atau TBA", 
                    "location": "kota/wilayah atau TBA",
                    "venue": "nama venue atau TBA"
                }}
            ]
            
            CONTOH YANG DIPRIORITASKAN:
            - "Konser besok" → date: "2025-08-13"
            - "Festival akhir pekan" → date: "2025-08-TBA"
            - "Pameran minggu ini" → date: "2025-08-TBA"
            - "Event segera" → date: "TBA-SOON"
            
            Jika TIDAK ada event yang relevan: []
            
            Response HANYA JSON array!
            """
            
            response = self.model.generate_content(prompt)
            
            result_text = response.text.strip()
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            
            result_text = result_text.strip()
            
            events = []
            if result_text.startswith('[') and result_text.endswith(']'):
                event_list = json.loads(result_text)
                
                for event_info in event_list:
                    if isinstance(event_info, dict) and event_info.get('event_name'):

                        raw_date = event_info.get('date', 'TBA')
                        parsed_date = self.parse_indonesian_date_flexible(raw_date)
                        event_info['date'] = parsed_date
                        
                        event_info.setdefault('location', 'TBA')
                        event_info.setdefault('venue', 'TBA')
                        event_info.setdefault('time', 'TBA')
                        event_info.setdefault('event_type', 'Lainnya')
                        
                        # Metadata
                        event_info.update({
                            'source_title': article['title'],
                            'source_link': article['link'],
                            'source_published': article['published'],
                            'extracted_at': datetime.now().isoformat()
                        })
                        
                        events.append(event_info)
            
            return events
            
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {str(e)}")
        except Exception as e:
            logger.debug(f"Error extracting event info: {str(e)}")
            
        return []
    
    async def process_articles_enhanced(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        all_events = []
        processed_count = 0
        
        for i, article in enumerate(articles[:120]):  
            if shutdown_in_progress:
                break
                
            logger.info(f"Processing article {i+1}/{min(len(articles), 120)}: {article['title'][:60]}...")
            
            try:
                loop = asyncio.get_event_loop()
                full_content = await loop.run_in_executor(
                    self.executor, 
                    self.fetch_article_content, 
                    article['link']
                )
                
                events_from_article = await self.extract_event_info_with_gemini(article, full_content)
                
                for event_info in events_from_article:
                    if self.is_within_7_days(event_info.get('date', 'TBA')):
                        all_events.append(event_info)
                        logger.info(f"Found 7-day event #{len(all_events)}: {event_info.get('event_name')} on {event_info.get('date')}")
                
                processed_count += 1
                

                if len(all_events) >= 80:  
                    logger.info(f"Target reached: {len(all_events)} events within 7 days")
                    break
                
                await asyncio.sleep(0.05)
                
            except Exception as e:
                logger.debug(f"Error processing article {i+1}: {str(e)}")
                continue
        
        logger.info(f"Processing completed: {len(all_events)} events within 7 days from {processed_count} articles")
        return all_events
    
    async def get_events(self, force_refresh: bool = False) -> List[Dict[str, Any]]:

        if shutdown_in_progress:
            return []
        
        if (not force_refresh and 
            self.last_update and 
            datetime.now() - self.last_update < self.cache_duration and 
            self.events_cache):
            logger.info("Returning cached 7-day events")
            return self.events_cache
        
        logger.info("Fetching events within 7 days.")
        start_time = time.time()
        
        articles = await self.fetch_news_from_rss()
        
        if shutdown_in_progress:
            return []
        
        events = await self.process_articles_enhanced(articles)
        
        unique_events = []
        seen_events = set()
        
        for event in events:
            event_key = f"{event.get('event_name', '')}_{event.get('date', '')}"
            if event_key not in seen_events:
                seen_events.add(event_key)
                
                # Clean event data 
                filtered_event = {
                    'event_name': event.get('event_name', 'TBA'),
                    'event_type': event.get('event_type', 'TBA'),
                    'date': event.get('date', 'TBA'),
                    'time': event.get('time', 'TBA'),
                    'location': event.get('location', 'TBA'),
                    'venue': event.get('venue', 'TBA'),
                    'source_title': event.get('source_title', ''),
                    'source_link': event.get('source_link', ''),
                    'extracted_at': event.get('extracted_at', '')
                }
                unique_events.append(filtered_event)
        
        # Sort by date
        def sort_key(event):
            date_str = event.get('date', 'TBA')
            if date_str == 'TBA':
                return 'ZZZZ-12-31'  
            elif 'TBA-SOON' in date_str:
                return '2025-08-13'  
            elif 'TBA' in date_str:
                return date_str.replace('-TBA', '-01')  
            else:
                return date_str
        
        unique_events.sort(key=sort_key)
        
        # Update cache
        self.events_cache = unique_events
        self.last_update = datetime.now()
        
        total_time = time.time() - start_time
        logger.info(f"7-day focused processing completed in {total_time:.2f}s")
        logger.info(f"Found {len(unique_events)} unique events within 7 days (including TBA)")
        return unique_events
    
    async def search_events_by_location(self, location: str) -> List[Dict[str, Any]]:

        """
        Cari event berdasarkan lokasi dalam 7 hari
        """

        events = await self.get_events()
        filtered_events = []
        
        location_lower = location.lower()
        for event in events:
            event_location = event.get('location', '').lower()
            event_venue = event.get('venue', '').lower()
            event_name = event.get('event_name', '').lower()
            
            if (location_lower in event_location or 
                location_lower in event_venue or
                location_lower in event_name):
                filtered_events.append(event)
        
        return filtered_events


server = Server("indonesian-events-7day-server")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:

    """
    List available tools
    """

    return [
        types.Tool(
            name="get_events",
            description="Ambil event/acara yang akan terjadi dalam 7 hari ke depan di Indonesia (termasuk yang TBA)",
            inputSchema={
                "type": "object",
                "properties": {
                    "force_refresh": {
                        "type": "boolean",
                        "description": "Force refresh data (skip cache)",
                        "default": False
                    }
                }
            }
        ),
        types.Tool(
            name="search_events_by_location",
            description="Cari event dalam 7 hari berdasarkan lokasi/kota",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Nama kota atau lokasi"
                    }
                },
                "required": ["location"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:
    """
    Handle tool calls
    """
    
    if name == "get_events":
        force_refresh = arguments.get("force_refresh", False) if arguments else False
        events = await events_server.get_events(force_refresh=force_refresh)
        
        return [types.TextContent(
            type="text",
            text=json.dumps(events, indent=2, ensure_ascii=False)
        )]
    
    elif name == "search_events_by_location":
        if not arguments or "location" not in arguments:
            raise ValueError("Location parameter is required")
        
        location = arguments["location"]
        events = await events_server.search_events_by_location(location)
        
        return [types.TextContent(
            type="text", 
            text=json.dumps(events, indent=2, ensure_ascii=False)
        )]
    
    else:
        raise ValueError(f"Unknown tool: {name}")


events_server = None

async def main():
    parser = argparse.ArgumentParser(description="Indonesian Events MCP Server - 7 Days Focus")
    parser.add_argument("--gemini-api-key", required=True, 
                       help="Gemini API Key")
    args = parser.parse_args()
    
    global events_server
    events_server = EventsServer(args.gemini_api_key)
    
    try:
        import mcp.server.stdio
        
        logger.info("Starting Indonesian Events MCP Server (7-Day Focus)")
        
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="indonesian-events-7day-server",
                    server_version="7.0.1",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        cleanup_resources()
        await asyncio.sleep(0.1)
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        cleanup_resources()
        os._exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        cleanup_resources()
        os._exit(1)
