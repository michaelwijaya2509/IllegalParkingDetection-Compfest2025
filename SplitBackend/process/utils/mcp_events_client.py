
import json
import asyncio
import sys
import traceback 
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters
from mcp.client import stdio
import argparse

async def fetch_events_for_web(api_key: str):

    server_params = StdioServerParameters(
        command="python",
        args=["utils/mcp_event_detector.py", "--gemini-api-key", api_key]
    )

    try:
        async with stdio.stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result_all = await session.call_tool("get_events", {"force_refresh": True})
                all_events = json.loads(result_all.content[0].text)

                final_output = {
                    "all_events": all_events,
                }
                print(json.dumps(final_output, ensure_ascii=False))

    except Exception as e:

        print(json.dumps({"error": f"An error occurred while communicating with the MCP server: {str(e)}"}))
        
        traceback.print_exc(file=sys.stderr)
        
        sys.exit(1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fetch MCP events for web display.")
    parser.add_argument("--gemini-api-key", type=str, required=True, help="Gemini API key to use")
    args = parser.parse_args()
    
    try:
        asyncio.run(fetch_events_for_web(api_key=args.gemini_api_key))
    except Exception as e:

        print(json.dumps({"error": f"Fatal client error: {str(e)}"}))
        
        traceback.print_exc(file=sys.stderr)
        
        sys.exit(1)
        