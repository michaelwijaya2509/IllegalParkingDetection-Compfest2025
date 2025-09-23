import asyncio
import argparse
import json
import logging
from mcp_event_detector import MCPEventDetector

async def main():

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("mcp-events-server")


    parser = argparse.ArgumentParser(description="MCP Event Detection Client")
    parser.add_argument("--openai-api-key", required=True, help="OpenAI API key")
    args = parser.parse_args()

    try:
        detector = MCPEventDetector(openai_api_key=args.openai_api_key)
        
        # Fetch articles
        logger.info("Fetching articles from RSS feeds...")
        articles = await detector.fetch_articles()
        logger.info(f"Fetched {len(articles)} articles")

        # Process articles
        logger.info("Processing articles...")
        events = await detector.process_articles_enhanced(articles)

        # Output results
        logger.info(f"Found {len(events)} unique events within 7 days (including TBA)")
        print(json.dumps(events, ensure_ascii=False, indent=2))

    except KeyboardInterrupt:
        logger.info("Received shutdown signal, cleaning up...")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())