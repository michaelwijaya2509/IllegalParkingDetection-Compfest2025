# File: Backend/mcp_events_client.py

import json
import asyncio
import sys
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters
from mcp.client import stdio
import argparse

async def fetch_events_for_web():


    server_params = StdioServerParameters(
        command="python",
        args=["mcp_event_detector.py", "--gemini-api-key", "AIzaSyDb8HvnyDX1orqYMGerKL7z7-OZNWidQqo"]
    )

    try:
        async with stdio.stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Fetch all events
                result_all = await session.call_tool("get_events", {"force_refresh": True})
                all_events = json.loads(result_all.content[0].text)

                final_output = {
                    "all_events": all_events,
                }
                print(json.dumps(final_output, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"error": f"An error occurred while communicating with the MCP server: {str(e)}"}))
        sys.exit(1)

if __name__ == "__main__":
    
    try:
        asyncio.run(fetch_events_for_web())
    except Exception as e:
        print(json.dumps({"error": f"Fatal client error: {str(e)}"}))
        sys.exit(1)