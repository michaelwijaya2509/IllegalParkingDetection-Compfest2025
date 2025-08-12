#!/usr/bin/env python3
"""
Enhanced MCP Server untuk ambil informasi event/acara besar di Indonesia
Dengan full article content fetching untuk lebih akurat
"""

import asyncio
import json
import logging
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-events-server")

class EventsServer:
    def __init__(self, gemini_api_key: str):
        # Setup Gemini AI
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # RSS Feed URLs untuk Indonesian news
        self.rss_feeds = [
            "https://news.google.com/rss/search?q=konser+Indonesia+2025&hl=id&gl=ID&ceid=ID:id",
            "https://news.google.com/rss/search?q=festival+Indonesia+2025&hl=id&gl=ID&ceid=ID:id", 
            "https://news.google.com/rss/search?q=event+Jakarta+2025&hl=id&gl=ID&ceid=ID:id",
            "https://news.google.com/rss/search?q=acara+musik+Indonesia+2025&hl=id&gl=ID&ceid=ID:id"
        ]
        
        # Cache untuk menyimpan hasil
        self.events_cache = []
        self.last_update = None
        self.cache_duration = timedelta(hours=4)  # Cache 4 jam
        
        # Thread pool untuk parallel processing
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    def fetch_single_rss(self, feed_url: str) -> List[Dict[str, Any]]:
        """Fetch single RSS feed - untuk parallel processing"""
        articles = []
        try:
            logger.info(f"Fetching RSS from: {feed_url}")
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:3]:  # Kurangi jadi 3 per feed untuk lebih fokus
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
        """Fetch full article content from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit text length untuk Gemini
            return text[:3000]  # Max 3000 chars
            
        except Exception as e:
            logger.error(f"Error fetching article content from {url}: {str(e)}")
            return ""

    async def fetch_news_from_rss(self) -> List[Dict[str, Any]]:
        """Ambil berita dari RSS feeds dengan parallel processing"""
        logger.info("Starting parallel RSS fetch...")
        start_time = time.time()
        
        # Jalankan fetch RSS secara parallel
        loop = asyncio.get_event_loop()
        tasks = []
        for feed_url in self.rss_feeds:
            task = loop.run_in_executor(self.executor, self.fetch_single_rss, feed_url)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Gabungkan semua artikel
        all_articles = []
        for articles in results:
            all_articles.extend(articles)
        
        fetch_time = time.time() - start_time
        logger.info(f"RSS fetch completed in {fetch_time:.2f}s, got {len(all_articles)} articles")
        return all_articles
    
    async def extract_event_info_with_gemini(self, article: Dict[str, Any], full_content: str = "") -> Optional[Dict[str, Any]]:
        """Extract informasi event menggunakan Gemini AI dengan full content"""
        try:
            # Gunakan full content jika ada, fallback ke summary
            content_to_analyze = full_content if full_content.strip() else article['summary']
            
            if not content_to_analyze.strip():
                return None
            
            # Enhanced prompt dengan lebih spesifik
            prompt = f"""
            Analisis artikel berita Indonesia berikut untuk extract informasi event/konser/festival:
            
            Judul: {article['title']}
            Konten: {content_to_analyze[:2000]}
            
            Jika ini tentang event/konser/festival yang akan datang di Indonesia, ekstrak info dalam JSON format:
            {{
                "is_event": true,
                "event_name": "nama event lengkap",
                "event_type": "konser/festival/pameran/pertunjukan/acara_resmi",
                "date": "tanggal dalam format YYYY-MM-DD jika ada, atau 'TBA' jika tidak ada",
                "time": "waktu acara jika disebutkan",
                "location": "kota/provinsi lokasi acara",
                "venue": "nama venue/tempat spesifik jika ada",
                "organizer": "penyelenggara jika disebutkan",
                "description": "deskripsi singkat 1-2 kalimat",
                "ticket_info": "info tiket jika ada (harga/penjualan)",
                "artists": "nama artis/performer jika ada",
                "expected_crowd": "perkiraan jumlah pengunjung jika disebutkan",
                "traffic_impact": "tinggi/sedang/rendah berdasarkan ukuran event dan lokasi"
            }}
            
            PENTING:
            - Cari tanggal dengan format seperti "15 Agustus 2025", "20-21 Agustus", dll
            - Cari nama venue seperti "Istana Negara", "GBK", "Indonesia Arena", dll  
            - Jika bukan event masa depan atau tidak ada info lengkap, return {{"is_event": false}}
            - Fokus pada event yang benar-benar akan terjadi, bukan berita lama
            
            Response hanya JSON, tidak ada teks lain.
            """
            
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            result_text = response.text.strip()
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            
            # Clean up any extra text
            result_text = result_text.strip()
            if result_text.startswith('{') and result_text.endswith('}'):
                event_info = json.loads(result_text)
                
                if event_info.get('is_event', False):
                    # Tambahkan metadata
                    event_info.update({
                        'source_title': article['title'],
                        'source_link': article['link'],
                        'source_published': article['published'],
                        'extracted_at': datetime.now().isoformat()
                    })
                    return event_info
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Response text: {result_text[:200]}...")
        except Exception as e:
            logger.error(f"Error extracting event info: {str(e)}")
            
        return None
    
    async def process_articles_enhanced(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process articles dengan full content fetching"""
        events = []
        
        for i, article in enumerate(articles[:15]):  # Process max 15 articles untuk menghindari timeout
            logger.info(f"Processing article {i+1}/{min(len(articles), 15)}: {article['title'][:50]}...")
            
            try:
                # Fetch full article content
                loop = asyncio.get_event_loop()
                full_content = await loop.run_in_executor(
                    self.executor, 
                    self.fetch_article_content, 
                    article['link']
                )
                
                # Extract event info dengan full content
                event_info = await self.extract_event_info_with_gemini(article, full_content)
                
                if event_info:
                    events.append(event_info)
                    logger.info(f"✅ Found event: {event_info.get('event_name', 'Unknown')}")
                
                # Small delay to be nice to servers
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing article {i+1}: {str(e)}")
                continue
        
        return events
    
    async def get_events(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Ambil daftar event dengan enhanced processing"""
        # Cek cache
        if (not force_refresh and 
            self.last_update and 
            datetime.now() - self.last_update < self.cache_duration and 
            self.events_cache):
            logger.info("Returning cached events")
            return self.events_cache
        
        logger.info("Fetching fresh events data with enhanced processing...")
        start_time = time.time()
        
        # Ambil artikel dari RSS
        articles = await self.fetch_news_from_rss()
        
        # Process articles dengan full content
        events = await self.process_articles_enhanced(articles)
        
        # Update cache
        self.events_cache = events
        self.last_update = datetime.now()
        
        total_time = time.time() - start_time
        logger.info(f"Enhanced processing completed in {total_time:.2f}s, found {len(events)} events")
        return events
    
    async def search_events_by_location(self, location: str) -> List[Dict[str, Any]]:
        """Cari event berdasarkan lokasi"""
        events = await self.get_events()
        filtered_events = []
        
        location_lower = location.lower()
        for event in events:
            event_location = event.get('location', '').lower()
            event_venue = event.get('venue', '').lower()
            
            if (location_lower in event_location or 
                location_lower in event_venue):
                filtered_events.append(event)
        
        return filtered_events
    
    async def get_high_impact_events(self) -> List[Dict[str, Any]]:
        """Ambil event dengan dampak lalu lintas tinggi"""
        events = await self.get_events()
        high_impact_events = []
        
        for event in events:
            traffic_impact = event.get('traffic_impact', '').lower()
            if 'tinggi' in traffic_impact:
                high_impact_events.append(event)
        
        return high_impact_events

# Setup MCP Server
server = Server("indonesian-events-server")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="get_events",
            description="Ambil daftar event/acara besar yang akan datang di Indonesia dengan info lengkap",
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
            description="Cari event berdasarkan lokasi/kota tertentu",
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
        ),
        types.Tool(
            name="get_high_impact_events",
            description="Ambil event dengan dampak lalu lintas tinggi",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:
    """Handle tool calls"""
    
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
    
    elif name == "get_high_impact_events":
        events = await events_server.get_high_impact_events()
        
        return [types.TextContent(
            type="text",
            text=json.dumps(events, indent=2, ensure_ascii=False)
        )]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

# Global events server instance
events_server = None

async def main():
    parser = argparse.ArgumentParser(description="Enhanced Indonesian Events MCP Server")
    parser.add_argument("--gemini-api-key", required=True, 
                       help="Gemini API Key")
    args = parser.parse_args()
    
    global events_server
    events_server = EventsServer(args.gemini_api_key)
    
    # Run server with stdio
    import mcp.server.stdio
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="indonesian-events-server",
                server_version="2.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())