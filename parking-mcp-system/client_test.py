import json
import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters
from mcp.client import stdio

async def get_events():
    server_params = StdioServerParameters(
        command="python", 
        args=["mcp_events_server.py", "--gemini-api-key", "AIzaSyDb8HvnyDX1orqYMGerKL7z7-OZNWidQqo"]
    )
    
    try:
        async with stdio.stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
   
                print("Initializing MCP session...")
                await session.initialize()
                
                print("Fetching events...")
                result = await session.call_tool("get_events", {"force_refresh": True})
                events = json.loads(result.content[0].text)
                
                print(f"\nFound {len(events)} events")
                for i, event in enumerate(events, 1):
                    print(f"\n{i}. {event.get('event_name', 'Unknown Event')}")
                    print(f"   📍 Location: {event.get('location', 'Unknown')}")
                    print(f"   📅 Date: {event.get('date', 'TBA')}")
                    print(f"   🏢 Venue: {event.get('venue', 'TBA')}")
                    print(f"   🚗 Traffic Impact: {event.get('traffic_impact', 'Unknown')}")
                    print(f"   👥 Expected Crowd: {event.get('expected_crowd', 'Unknown')}")
                    print(f"   📰 Source: {event.get('source_title', '')[:50]}...")
                
                # Test location search
                print(f"\n=== Events in Jakarta ===")
                result = await session.call_tool("search_events_by_location", {"location": "Jakarta"})
                jakarta_events = json.loads(result.content[0].text)
                
                for event in jakarta_events:
                    print(f"- {event.get('event_name', 'Unknown')} ({event.get('date', 'TBA')})")
                
                # Test high impact events
                print(f"\n=== High Impact Events ===")
                result = await session.call_tool("get_high_impact_events", {})
                high_impact_events = json.loads(result.content[0].text)
                
                for event in high_impact_events:
                    print(f"- {event.get('event_name', 'Unknown')} at {event.get('location', 'Unknown')}")
                    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed all requirements and the MCP server file exists")

if __name__ == "__main__":
    # Run the test
    asyncio.run(get_events())