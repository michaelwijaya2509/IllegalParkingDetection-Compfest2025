/* eslint-disable @typescript-eslint/no-explicit-any */
// File: app/upcoming-events/page.tsx

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
} from "react-icons/fi";

const BACKEND_URL =
  "https://horus-backend-395725017559.asia-southeast2.run.app";

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
  jakarta_events: Event[];
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

export default function UpcomingEvents() {
  const [data, setData] = useState<EventsData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [isCacheStale, setIsCacheStale] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchEventsFromCache = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${BACKEND_URL}/events/upcoming`);
      const eventsData: EventsData = await response.json();
      if (eventsData.error && eventsData.error !== "Cache is empty.") {
        throw new Error(eventsData.error);
      }
      setData(eventsData);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    // Cek status cache saat halaman dimuat
    const checkCacheStatus = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/events/cache_status`);
        const status = await response.json();
        if (status.last_updated) {
          setLastUpdated(status.last_updated);
          const lastUpdateDate = new Date(status.last_updated);
          const now = new Date();
          const hoursDiff =
            (now.getTime() - lastUpdateDate.getTime()) / (1000 * 60 * 60);
          if (hoursDiff > 24) {
            setIsCacheStale(true);
          }
        } else {
          // refresh cache
          setIsCacheStale(true);
        }
      } catch (err) {
        console.error("Could not check cache status:", err);
        setIsCacheStale(true);
      }
    };

    checkCacheStatus();
    fetchEventsFromCache();
  }, []);

  const handleRefreshCache = async () => {
    setIsRefreshing(true);
    setError(null);
    try {
      const response = await fetch(`${BACKEND_URL}/events/refresh_cache`, {
        method: "POST",
      });
      const result = await response.json();
      if (!response.ok || !result.ok) {
        throw new Error(result.error || "Failed to refresh cache.");
      }
      await fetchEventsFromCache();
      setLastUpdated(result.updated_at);
      setIsCacheStale(false);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsRefreshing(false);
    }
  };

  const totalEvents = data?.all_events?.length ?? 0;

  return (
    <div className="min-h-screen bg-primary">
      <Navigation />
      <main className="pt-20 p-8 md:p-12 mt-10">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">
              Upcoming Events
            </h1>
            <p className="text-gray-400">Events in Indonesia</p>
          </div>

          <div className="bg-tile1 border border-gray-700 rounded-lg p-4 mb-8 flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-400">Data Last Updated:</p>
              <p className="text-white font-semibold">
                {lastUpdated ? new Date(lastUpdated).toLocaleString() : "Never"}
              </p>
            </div>
            <button
              onClick={handleRefreshCache}
              disabled={isRefreshing || !isCacheStale}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white font-semibold rounded-md transition-colors hover:bg-blue-700 disabled:bg-gray-500 disabled:cursor-not-allowed"
            >
              {isRefreshing ? (
                <FiLoader className="animate-spin" />
              ) : (
                <FiRefreshCw />
              )}
              {isRefreshing ? "Updating..." : "Update Now"}
            </button>
          </div>

          {isLoading ? (
            <div className="flex justify-center items-center h-64 text-gray-400">
              <FiLoader className="animate-spin w-8 h-8 mr-4" />
              <p>Loading events from cache...</p>
            </div>
          ) : error || !data ? (
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
              </div>

              <div>
                <h2 className="text-2xl font-semibold text-white mb-4 flex items-center">
                  <FiGlobe className="mr-3 text-green-400" />
                  All Upcoming Events
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                  {data.all_events.map((event, index) => (
                    <EventCard key={`all-${index}`} event={event} />
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
