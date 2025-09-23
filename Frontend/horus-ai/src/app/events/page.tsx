/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";
import { useState, useEffect } from "react";
import Navigation from "@/components/Navigation";
import {
  FiCalendar,
  FiMapPin,
  FiTag,
  FiHome,
  FiLoader,
  FiXCircle,
  FiGlobe,
  FiRefreshCw,
  FiSearch,
  FiX,
} from "react-icons/fi";
import Chart from "chart.js/auto";

const PROCESS_URL = process.env.NEXT_PUBLIC_PROCESS_URL;

interface Event {
  event_name: string;
  location: string;
  date: string;
  venue: string;
  event_type: string;
  source_link: string;
}
interface EventsData {
  all_events: Event[];
  error?: string;
}

const EventCard = ({ event }: { event: Event }) => (
  <div className="bg-tile2 border border-gray-700 rounded-lg p-4 flex flex-col justify-between hover:border-blue-500 transition-all duration-300">
    <div>
      <h3 className="font-bold text-white mb-2">{event.event_name}</h3>
      <div className="space-y-1.5 text-sm text-gray-300">
        <p className="flex items-center">
          <FiCalendar className="w-4 h-4 mr-2 text-blue-400" /> {event.date}
        </p>
        <p className="flex items-center">
          <FiMapPin className="w-4 h-4 mr-2 text-blue-400" /> {event.location}
        </p>
        <p className="flex items-center">
          <FiHome className="w-4 h-4 mr-2 text-blue-400" /> {event.venue}
        </p>
        <p className="flex items-center">
          <FiTag className="w-4 h-4 mr-2 text-blue-400" /> {event.event_type}
        </p>
      </div>
    </div>
    {event.source_link && (
      <a
        href={event.source_link}
        target="_blank"
        rel="noopener noreferrer"
        className="mt-4 text-center bg-gray-600 hover:bg-gray-500 text-white text-xs font-semibold py-2 px-3 rounded-md transition-colors"
      >
        View Source
      </a>
    )}
  </div>
);

const EventTypePieChart = ({ events }: { events: Event[] }) => {
  useEffect(() => {
    const eventTypes = events.reduce(
      (acc: Record<string, number>, event: Event) => {
        acc[event.event_type] = (acc[event.event_type] || 0) + 1;
        return acc;
      },
      {}
    );

    const ctx = document.getElementById("eventTypeChart") as HTMLCanvasElement;
    const chart = new Chart(ctx, {
      type: "pie",
      data: {
        labels: Object.keys(eventTypes),
        datasets: [
          {
            data: Object.values(eventTypes),
            backgroundColor: [
              "#3B82F6",
              "#10B981",
              "#F59E0B",
              "#EF4444",
              "#8B5CF6",
              "#EC4899",
            ],
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: "#D1D5DB" },
          },
        },
      },
    });

    return () => chart.destroy();
  }, [events]);

  return <canvas id="eventTypeChart" className="max-w-xs mx-auto" />;
};

const EventTimeline = ({ events }: { events: Event[] }) => {
  const groupedByDate = events.reduce(
    (acc: Record<string, Event[]>, event: Event) => {
      const date = event.date || "TBA";
      if (!acc[date]) acc[date] = [];
      acc[date].push(event);
      return acc;
    },
    {}
  );

  const sortedDates = Object.keys(groupedByDate).sort((a, b) => {
    if (a === "TBA" || a === "TBA-SOON") return 1;
    if (b === "TBA" || b === "TBA-SOON") return -1;
    return a.localeCompare(b);
  });

  return (
    <div className="space-y-4">
      {sortedDates.map((date) => (
        <div
          key={date}
          className="bg-tile1 p-4 rounded-lg border border-gray-700"
        >
          <h3 className="text-lg font-semibold text-white mb-2">{date}</h3>
          <div className="space-y-2">
            {groupedByDate[date].map((event: Event, index: number) => (
              <div
                key={`timeline-${date}-${index}`}
                className="text-sm text-gray-300"
              >
                {event.event_name} ({event.event_type}) - {event.location}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

export default function UpcomingEvents() {
  const [data, setData] = useState<EventsData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [isCacheStale, setIsCacheStale] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCity, setSelectedCity] = useState("All Cities");

  const fetchEventsFromCache = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${PROCESS_URL}/events/upcoming`, {
        headers: { "Content-Type": "application/json" },
      });
      if (!response.ok) {
        throw new Error(
          `HTTP error ${response.status}: ${await response.text()}`
        );
      }
      const eventsData: EventsData = await response.json();
      console.log("Fetched events:", eventsData);
      if (eventsData.error && eventsData.error !== "Cache is empty.") {
        throw new Error(eventsData.error);
      }
      setData(eventsData);
    } catch (err: any) {
      console.error("Fetch events error:", err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const checkCacheStatus = async () => {
    try {
      const response = await fetch(`${PROCESS_URL}/events/cache_status`, {
        headers: { "Content-Type": "application/json" },
      });
      if (!response.ok) {
        throw new Error(
          `HTTP error ${response.status}: ${await response.text()}`
        );
      }
      const status = await response.json();
      console.log("Cache status:", status);
      if (status.last_updated) {
        setLastUpdated(status.last_updated);
        const lastUpdateDate = new Date(status.last_updated);
        const now = new Date();
        const hoursDiff =
          (now.getTime() - lastUpdateDate.getTime()) / (1000 * 60 * 60);
        setIsCacheStale(hoursDiff > 1);
      } else {
        setIsCacheStale(true);
      }
    } catch (err) {
      console.error("Could not check cache status:", err);
      setIsCacheStale(true);
    }
  };

  useEffect(() => {
    checkCacheStatus();
    fetchEventsFromCache();
  }, []);

  const handleRefreshCache = async () => {
    setIsRefreshing(true);
    setError(null);
    try {
      const response = await fetch(`${PROCESS_URL}/events/refresh_cache`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!response.ok) {
        throw new Error(
          `HTTP error ${response.status}: ${await response.text()}`
        );
      }
      const result = await response.json();
      console.log("Refresh cache result:", result);
      if (!result.ok) {
        throw new Error(result.error || "Failed to refresh cache.");
      }
      await fetchEventsFromCache();
      setLastUpdated(result.updated_at);
      setIsCacheStale(false);
    } catch (err: any) {
      console.error("Refresh cache error:", err);
      setError(err.message);
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleClearFilters = () => {
    setSearchQuery("");
    setSelectedCity("All Cities");
  };

  const filteredEvents =
    data?.all_events?.filter(
      (event) =>
        event.event_name.toLowerCase().includes(searchQuery.toLowerCase()) &&
        (selectedCity === "All Cities" || event.location === selectedCity)
    ) ?? [];

  const uniqueCities = [
    "All Cities",
    ...new Set(data?.all_events?.map((event) => event.location) ?? []),
  ].filter((city) => city !== "TBA");

  const totalEvents = filteredEvents.length;

  return (
    <div className="min-h-screen bg-primary">
      <Navigation />
      <main className="pt-28 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">
              Upcoming Events
            </h1>
            <p className="text-gray-400">Events in Indonesia</p>
          </div>

          <div className="bg-tile1 border border-gray-700 rounded-lg p-4 mb-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div>
              <p className="text-sm text-gray-400">Data Last Updated:</p>
              <p className="text-white font-semibold">
                {lastUpdated ? new Date(lastUpdated).toLocaleString() : "Never"}
              </p>
            </div>
            <button
              onClick={handleRefreshCache}
              disabled={isRefreshing || !isCacheStale}
              className="flex cursor-pointer items-center gap-2 px-4 py-2 bg-blue-600 text-white font-semibold rounded-md transition-colors hover:bg-blue-700 disabled:bg-gray-500 disabled:cursor-not-allowed"
            >
              {isRefreshing ? (
                <FiLoader className="animate-spin" />
              ) : (
                <FiRefreshCw />
              )}
              {isRefreshing ? "Updating..." : "Update Now"}
            </button>
          </div>

          <div className="mb-8 flex flex-col sm:flex-row gap-4">
            <div className="relative flex-1">
              <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search events by name..."
                className="w-full pl-10 pr-4 py-2 bg-tile2 border border-gray-700 rounded-md text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
              />
            </div>
            <select
              value={selectedCity}
              onChange={(e) => setSelectedCity(e.target.value)}
              className="w-full sm:w-48 bg-tile2 border border-gray-700 rounded-md py-2 px-3 text-white focus:outline-none focus:border-blue-500"
            >
              {uniqueCities.map((city) => (
                <option key={city} value={city}>
                  {city}
                </option>
              ))}
            </select>
            <button
              onClick={handleClearFilters}
              className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-500"
            >
              <FiX />
              Clear Filters
            </button>
          </div>

          {isLoading ? (
            <div className="flex justify-center items-center h-64 text-gray-400">
              <FiLoader className="animate-spin w-8 h-8 mr-4" />
              <p>Loading events from cache...</p>
            </div>
          ) : error || !data || !data.all_events ? (
            <div className="flex justify-center items-center h-64 text-red-400 bg-red-500/10 rounded-lg p-6 text-center">
              <FiXCircle className="w-8 h-8 mr-4 flex-shrink-0" />
              <div>
                <p className="font-bold">Failed to load events.</p>
                <p className="text-sm">
                  {error || "Cache might be empty. Try updating now."}
                </p>
              </div>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
                <div className="bg-tile1 border border-gray-700 rounded-lg p-6">
                  <p className="text-gray-400 text-sm">Total Events Found</p>
                  <p className="text-3xl font-bold text-white">{totalEvents}</p>
                </div>
                <div className="bg-tile1 border border-gray-700 rounded-lg p-6">
                  <p className="text-gray-400 text-sm">Data Freshness</p>
                  <p
                    className={`text-3xl font-bold ${
                      isCacheStale ? "text-yellow-400" : "text-green-400"
                    }`}
                  >
                    {isCacheStale ? "Stale" : "Fresh"}
                  </p>
                </div>
                <div className="bg-tile1 border border-gray-700 rounded-lg p-6">
                  <p className="text-gray-400 text-sm">Event Types</p>
                  <EventTypePieChart events={filteredEvents} />
                </div>
              </div>

              <div className="mb-12">
                <h2 className="text-2xl font-semibold text-white mb-4 flex items-center">
                  <FiCalendar className="mr-3 text-green-400" />
                  Event Timeline
                </h2>
                <EventTimeline events={filteredEvents} />
              </div>

              <div>
                <h2 className="text-2xl font-semibold text-white mb-4 flex items-center">
                  <FiGlobe className="mr-3 text-green-400" />
                  All Upcoming Events
                </h2>
                {filteredEvents.length === 0 ? (
                  <div className="text-center text-gray-400 p-6">
                    No events match your search or filter criteria.
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                    {filteredEvents.map((event, index) => (
                      <EventCard key={`all-${index}`} event={event} />
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
