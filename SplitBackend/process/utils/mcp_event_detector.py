import os
import json
import asyncio
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any
from openai import AsyncOpenAI, OpenAIError
import aiohttp
from bs4 import BeautifulSoup
import feedparser

logger = logging.getLogger("mcp-events-server")

class MCPEventDetector:
    def __init__(self, openai_api_key: str):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = "gpt-4o-mini"
        self.rss_feeds = [
            "https://news.google.com/rss/search?q=konser+Indonesia+event+acara+when:7d&hl=id&gl=ID&ceid=ID:id",
        ]

    async def fetch_full_contents(self, articles: List[Dict[str, Any]]) -> List[str]:
        """
        Fetch full content of articles asynchronously
        """
        async def fetch_content(article: Dict[str, Any]) -> str:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(article['link'], headers={'User-Agent': 'Mozilla/5.0'}) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            text = ' '.join(soup.get_text().split())
                            return text[:10000]  # Limit content size
            except Exception as e:
                logger.error(f"Error fetching content for {article['title']}: {str(e)}")
            return ""

        tasks = [fetch_content(article) for article in articles]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def parse_indonesian_date_flexible(self, date_str: str) -> str:
        """
        Parse Indonesian date strings into YYYY-MM-DD, TBA, TBA-SOON, or YYYY-MM-TBA
        """
        date_str = date_str.lower().strip()
        today = datetime.now()
        
        if date_str in ['tba', 'tba-soon']:
            return date_str
        if 'segera' in date_str or 'akan datang' in date_str:
            return 'TBA-SOON'
        if 'minggu ini' in date_str or 'akhir pekan' in date_str:
            return today.strftime('%Y-%m') + '-TBA'
        if 'minggu depan' in date_str:
            next_week = today + timedelta(days=7)
            return next_week.strftime('%Y-%m') + '-TBA'
        
        try:
            # Try parsing specific dates (e.g., "24-28 Sept 2025" or "24 September 2025")
            months = {
                'januari': '01', 'februari': '02', 'maret': '03', 'april': '04',
                'mei': '05', 'juni': '06', 'juli': '07', 'agustus': '08',
                'september': '09', 'oktober': '10', 'november': '11', 'desember': '12'
            }
            for month, num in months.items():
                if month in date_str.lower():
                    parts = date_str.replace('-', ' ').replace(',', ' ').split()
                    day = next((p for p in parts if p.isdigit() and 1 <= int(p) <= 31), '01')
                    year = next((p for p in parts if p.isdigit() and 2020 <= int(p) <= 2030), today.year)
                    return f"{year}-{num}-{day.zfill(2)}"
            return 'TBA'
        except:
            return 'TBA'

    async def extract_event_info_with_openai(self, article: Dict[str, Any], full_content: str = "") -> List[Dict[str, Any]]:
        """
        Extract event information using OpenAI GPT-4o-mini with rate limiting and retries
        """
        try:
            # Keyword filter to skip irrelevant articles
            event_keywords = ['konser', 'festival', 'pameran', 'akhir pekan', 'minggu ini', 'segera', 'event', 'acara']
            if not any(keyword in article['title'].lower() or keyword in article['summary'].lower() for keyword in event_keywords):
                logger.debug(f"Skipping article '{article['title']}' (no event keywords)")
                return []

            # Check cache
            script_dir = os.path.dirname(__file__)
            cache_file = os.path.join(script_dir, "openai_cache.json")
            cache = json.load(open(cache_file)) if os.path.exists(cache_file) else {}
            article_key = f"{article['link']}_{article['title']}"
            
            if article_key in cache:
                logger.debug(f"Using cached OpenAI response for '{article['title']}'")
                return cache[article_key]

            # Prepare content and prompt
            content_to_analyze = full_content if full_content.strip() else article['summary']
            
            if not content_to_analyze.strip():
                logger.debug(f"No content to analyze for article: {article['title']}")
                return []

            prompt = f"""
            Analisis artikel Indonesia berikut dan cari EVENT yang akan terjadi (prioritas dalam 7 hari ke depan dari {datetime.now().date()}, tapi terima juga yang lebih jauh atau TBA):

            Judul: {article['title']}
            Konten: {content_to_analyze[:4000]}

            ATURAN PRIORITAS:
            1. PRIORITAS UTAMA: Event dalam 7 hari ke depan dari {datetime.now().date()}
            2. TERIMA JUGA: Event dengan tanggal TBA, "segera", "akan datang", "minggu ini/depan", atau date ranges (gunakan start date untuk date ranges)
            3. Terima semua jenis event: konser, festival, pameran, seminar, workshop, bazaar, pertunjukan, olahraga, atau lainnya
            4. Fokus pada event yang belum lewat; abaikan event sebelum {datetime.now().date()}
            5. Jika tidak ada tanggal spesifik, gunakan "TBA-SOON" untuk "segera" atau "akan datang", dan "YYYY-MM-TBA" untuk "minggu ini/depan"
            6. Ekstrak SEMUA event yang relevan dari artikel, termasuk jika artikel berisi daftar event
            7. Pastikan event_name jelas dan spesifik (misalnya, "Konser Dewa 19" bukan "Konser")
            
            Format JSON array:
            [
                {{
                    "event_name": "nama event (wajib ada, spesifik)",
                    "event_type": "konser/festival/pameran/pertunjukan/seminar/workshop/bazaar/olahraga/lainnya",
                    "date": "YYYY-MM-DD atau TBA/TBA-SOON/YYYY-MM-TBA",
                    "time": "waktu atau TBA",
                    "location": "kota/wilayah atau TBA",
                    "venue": "nama venue atau TBA"
                }}
            ]
            
            CONTOH:
            - "Konser besok di Jakarta" → [{{"event_name": "Konser Besok", "event_type": "konser", "date": "{(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}", "time": "TBA", "location": "Jakarta", "venue": "TBA"}}]
            - "Festival akhir pekan" → [{{"event_name": "Festival Akhir Pekan", "event_type": "festival", "date": "{datetime.now().strftime('%Y-%m')}-TBA", "time": "TBA", "location": "TBA", "venue": "TBA"}}]
            - "Pameran minggu ini di GBK" → [{{"event_name": "Pameran GBK", "event_type": "pameran", "date": "{datetime.now().strftime('%Y-%m')}-TBA", "time": "TBA", "location": "Jakarta", "venue": "GBK"}}]
            - "Event segera di Bali" → [{{"event_name": "Event Bali", "event_type": "lainnya", "date": "TBA-SOON", "time": "TBA", "location": "Bali", "venue": "TBA"}}]
            - "Konser 24-28 Sept 2025" → [{{"event_name": "Konser September", "event_type": "konser", "date": "2025-09-24", "time": "TBA", "location": "TBA", "venue": "TBA"}}]
            
            Jika TIDAK ada event yang relevan: []
            
            Response HANYA JSON array!
            """
            
            max_retries = 3
            base_delay = 5  # Start with 5 seconds for OpenAI rate limits
            for attempt in range(max_retries):
                try:
                    # Rate limiting: sleep to respect API limits
                    await asyncio.sleep(2)  # 2-second delay between requests
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are an event extraction assistant. Return only a JSON array as specified."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1500,
                        temperature=0.3
                    )
                    result_text = response.choices[0].message.content.strip()
                    logger.debug(f"OpenAI response for '{article['title']}': {result_text}")
                    
                    # Clean up response if wrapped in code fences
                    if result_text.startswith('```json'):
                        result_text = result_text[7:]
                    if result_text.endswith('```'):
                        result_text = result_text[:-3]
                    result_text = result_text.strip()
                    
                    events = []
                    if result_text.startswith('[') and result_text.endswith(']'):
                        try:
                            event_list = json.loads(result_text)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error for '{article['title']}': {str(e)}, response: {result_text}")
                            return []
                        
                        for event_info in event_list:
                            if isinstance(event_info, dict) and event_info.get('event_name'):
                                raw_date = event_info.get('date', 'TBA')
                                parsed_date = self.parse_indonesian_date_flexible(raw_date)
                                event_info['date'] = parsed_date
                                
                                event_info.setdefault('location', 'TBA')
                                event_info.setdefault('venue', 'TBA')
                                event_info.setdefault('time', 'TBA')
                                event_info.setdefault('event_type', 'lainnya')
                                
                                event_info.update({
                                    'source_title': article['title'],
                                    'source_link': article['link'],
                                    'source_published': article['published'],
                                    'extracted_at': datetime.now().isoformat()
                                })
                                
                                events.append(event_info)
                    
                    # Cache the result
                    cache[article_key] = events
                    with open(cache_file, 'w') as f:
                        json.dump(cache, f, ensure_ascii=False)
                    
                    logger.debug(f"Extracted events for '{article['title']}': {json.dumps(events, ensure_ascii=False)}")
                    return events
                    
                except OpenAIError as e:
                    if "rate_limit_exceeded" in str(e) and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff: 5s, 10s, 20s
                        logger.warning(f"Rate limit exceeded for '{article['title']}', retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                        continue
                    logger.error(f"Error extracting event info for '{article['title']}': {str(e)}")
                    return []
            
            logger.error(f"Failed to extract events for '{article['title']}' after {max_retries} retries")
            return []
            
        except Exception as e:
            logger.error(f"Unexpected error extracting event info for '{article['title']}': {str(e)}")
            return []

    async def fetch_articles(self) -> List[Dict[str, Any]]:
        """
        Fetch articles from RSS feeds
        """
        articles = []
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    articles.append({
                        'title': entry.get('title', ''),
                        'link': entry.get('link', ''),
                        'summary': entry.get('summary', ''),
                        'published': entry.get('published', '')
                    })
            except Exception as e:
                logger.error(f"Error fetching feed {feed_url}: {str(e)}")
        return articles

    async def process_articles_enhanced(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process articles to extract events
        """
        all_events = []
        for i, article in enumerate(articles[:5]):  # Limit to 5 articles for testing
            logger.info(f"Processing article {i+1}/{len(articles[:5])}: {article['title']}")
            full_content = (await self.fetch_full_contents([article]))[0]
            events = await self.extract_event_info_with_openai(article, full_content)
            all_events.extend(events)
        return all_events