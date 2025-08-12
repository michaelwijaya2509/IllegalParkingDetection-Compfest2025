import json
import asyncio
import signal
import sys
import os
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters
from mcp.client import stdio


def display_event(event, index):

    event_name = event.get('event_name', 'Unknown Event')
    location = event.get('location', 'TBA')
    date = event.get('date', 'TBA')
    venue = event.get('venue', 'TBA')
    event_type = event.get('event_type', 'TBA')
    source_title = event.get('source_title', '')
    
    if len(source_title) > 60:
        source_title = source_title[:57] + "."
    
    print(f"\n{index}. {event_name}")
    print(f"   📍 Location: {location}")
    print(f"   📅 Date: {date}")
    print(f"   🏢 Venue: {venue}")
    print(f"   🎭 Type: {event_type}")
    if source_title:
        print(f"   📰 Source: {source_title}")


def signal_handler(signum, frame):
    os._exit(0)  


async def get_all_events():
 
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    server_params = StdioServerParameters(
        command="python", 
        args=["mcp_events_server.py", "--gemini-api-key", "AIzaSyDb8HvnyDX1orqYMGerKL7z7-OZNWidQqo"]
    )
    
    session = None
    try:
        async with stdio.stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                print("Starting MCP Session for 7-Day Events\n")
                await session.initialize()
                
                print("Fetching events within 7 days")
                result = await session.call_tool("get_events", {"force_refresh": True})
                events = json.loads(result.content[0].text)
                
                if not events:
                    print("No events found within the next 7 days!")
                    return
                
                print(f"\nFound {len(events)} events within 7 days")

                for i, event in enumerate(events, 1):
                    display_event(event, i)
                
                print("Event Statistics\n")
                
                # Categorize by data completeness
                complete_events = []
                partial_events = []  
                tba_events = []
                
                # Count by different categories
                locations = {}
                event_types = {}
                venues = {}
                dates_status = {
                    'Complete Dates (YYYY-MM-DD)': 0,
                    'Partial Dates (YYYY-MM-TBA)': 0, 
                    'Soon Dates (TBA-SOON)': 0,
                    'Complete TBA': 0
                }
                
                for event in events:
                    # Location counting
                    loc = event.get('location', 'TBA')
                    if loc != 'TBA':
                        locations[loc] = locations.get(loc, 0) + 1
                    
                    # Venue counting
                    venue = event.get('venue', 'TBA')
                    if venue != 'TBA':
                        venues[venue] = venues.get(venue, 0) + 1
                    
                    # Event type counting  
                    event_type = event.get('event_type', 'TBA')
                    event_types[event_type] = event_types.get(event_type, 0) + 1
                    
                    # Date status categorization
                    date_str = event.get('date', 'TBA')
                    if date_str == 'TBA':
                        dates_status['Complete TBA'] += 1
                        tba_events.append(event)
                    elif 'TBA-SOON' in date_str:
                        dates_status['Soon Dates (TBA-SOON)'] += 1
                        partial_events.append(event)
                    elif 'TBA' in date_str:
                        dates_status['Partial Dates (YYYY-MM-TBA)'] += 1
                        partial_events.append(event)
                    else:
                        dates_status['Complete Dates (YYYY-MM-DD)'] += 1
                        complete_events.append(event)
                
                print(f"\nData Status Breakdown:")
                for category, count in dates_status.items():
                    percentage = (count / len(events)) * 100 if events else 0
                    print(f"   • {category}: {count} events ({percentage:.1f}%)")
                
                if locations:
                    print(f"\nTop Locations:")
                    for loc, count in sorted(locations.items(), key=lambda x: x[1], reverse=True)[:10]:
                        print(f"   • {loc}: {count} events")
                else:
                    print(f"\nLocations: Most events have TBA locations")
                
                if venues:
                    print(f"\nTop Venues:")
                    for venue, count in sorted(venues.items(), key=lambda x: x[1], reverse=True)[:10]:
                        print(f"   • {venue}: {count} events")
                else:
                    print(f"\nVenues: Most events have TBA venues")
                
                print(f"\nEvent Types:")
                for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"   • {event_type}: {count} events")
                
                # Test Jakarta search 

                print("\nJakarta Events Search Test")
                
                result = await session.call_tool("search_events_by_location", {"location": "Jakarta"})
                jakarta_events = json.loads(result.content[0].text)
                
                if jakarta_events:
                    print(f"\nFound {len(jakarta_events)} events in Jakarta:")
                    
                    for i, event in enumerate(jakarta_events[:15], 1):  
                        display_event(event, i)
                    
                    if len(jakarta_events) > 15:
                        print(f"\n   ... and {len(jakarta_events) - 15} more Jakarta events!")
                else:
                    print("No Jakarta events found within 7 days")
                
                print("SUMMARY")
                print(f"Total Events Found: {len(events)}")
                print(f"Time Range: Next 7 days + TBA events")
                print(f"Jakarta Events: {len(jakarta_events)}")
                print(f"Events with Complete Dates: {dates_status['Complete Dates (YYYY-MM-DD)']}")
                print(f"Events with TBA Dates: {dates_status['Complete TBA']}")
                print(f"Events Coming Soon: {dates_status['Soon Dates (TBA-SOON)']}")
                
    except KeyboardInterrupt:
        os._exit(0)
    except Exception as e:
        print("Make sure you have installed all requirements and the MCP server file exists")
    finally:
        print("\nSession ended")


if __name__ == "__main__":
    try:
        asyncio.run(get_all_events())
    except KeyboardInterrupt:
        os._exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        os._exit(1)