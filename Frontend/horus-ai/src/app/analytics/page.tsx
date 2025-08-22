/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";
import { useState, useEffect } from "react";
import Navigation from "@/components/Navigation";
import {
  FiTrendingUp,
  FiMapPin,
  FiClock,
  FiAlertTriangle,
  FiLoader,
  FiXCircle,
  FiTag,
  FiMessageSquare,
} from "react-icons/fi";

const BACKEND_URL =
  "https://horus-backend-395725017559.asia-southeast1.run.app";

interface TrendData {
  date: string;
  count: number;
}
interface HotspotData {
  location: string;
  count: number;
}
interface CategoryData {
  category: string;
  count: number;
}
interface ReasonData {
  reason: string;
  count: number;
}

interface AnalyticsData {
  totalApprovedIncidents: number;
  todayApprovedIncidents: number;
  activeCameras: number;
  incidentTrends: TrendData[];
  locationHotspots: HotspotData[];
  violationTypes: CategoryData[];
  commonReasons: ReasonData[];
}

export default function Analytics() {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalyticsData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`${BACKEND_URL}/analytics/summary`);
        if (!response.ok) {
          throw new Error("Failed to fetch analytics summary from the server.");
        }
        const summaryData: AnalyticsData = await response.json();
        setData(summaryData);
      } catch (err: any) {
        setError(err.message || "An unknown error occurred.");
        console.error("Fetch error:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchAnalyticsData();
  }, []);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
    });
  };

  return (
    <div className="min-h-screen bg-primary">
      <Navigation />
      <main className="pt-20 p-12 mt-10">
        <div className="max-w-10xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">
              Analytics Dashboard
            </h1>
            <p className="text-gray-400">
              System performance and trends from approved incidents
            </p>
          </div>

          {isLoading ? (
            <div className="flex justify-center items-center h-64 text-gray-400">
              <FiLoader className="animate-spin w-8 h-8 mr-4" />
              <p>Loading analytics data...</p>
            </div>
          ) : error || !data ? (
            <div className="flex justify-center items-center h-64 text-red-400 bg-red-500/10 rounded-lg">
              <FiXCircle className="w-8 h-8 mr-4" />
              <p>{error || "No data available to display."}</p>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <div className="bg-tile1 border border-gray-700 rounded-lg p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-gray-400 text-sm">Total Approved</p>
                      <p className="text-3xl font-bold text-white">
                        {data.totalApprovedIncidents}
                      </p>
                    </div>
                    <div className="p-3 bg-red-600/80 rounded-lg flex-shrink-0">
                      <FiAlertTriangle className="w-6 h-6 text-white" />
                    </div>
                  </div>
                </div>

                <div className="bg-tile1 border border-gray-700 rounded-lg p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-gray-400 text-sm">Approved Today</p>
                      <p className="text-3xl font-bold text-white">
                        {data.todayApprovedIncidents}
                      </p>
                    </div>
                    <div className="p-3 bg-blue-600/80 rounded-lg flex-shrink-0">
                      <FiClock className="w-6 h-6 text-white" />
                    </div>
                  </div>
                </div>

                <div className="bg-tile1 border border-gray-700 rounded-lg p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-gray-400 text-sm">Active Cameras</p>
                      <p className="text-3xl font-bold text-white">
                        {data.activeCameras}
                      </p>
                    </div>
                    <div className="p-3 bg-green-600/80 rounded-lg flex-shrink-0">
                      <FiTrendingUp className="w-6 h-6 text-white" />
                    </div>
                  </div>
                </div>

                <div className="bg-tile1 border border-gray-700 rounded-lg p-6">
                  <div className="flex items-center justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <p className="text-gray-400 text-sm">Top Hotspot</p>
                      <p
                        className="text-xl font-bold text-white truncate"
                        title={data.locationHotspots[0]?.location ?? "N/A"}
                      >
                        {data.locationHotspots[0]?.location ?? "N/A"}
                      </p>
                    </div>
                    <div className="p-3 bg-purple-600/80 rounded-lg flex-shrink-0">
                      <FiMapPin className="w-6 h-6 text-white" />
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-tile1 border border-gray-700 rounded-lg p-6">
                  <h2 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <FiTrendingUp className="mr-2" />
                    Incident Trends (Last 7 Days)
                  </h2>
                  <div className="space-y-3">
                    {data.incidentTrends.map((trend) => (
                      <div
                        key={trend.date}
                        className="flex items-center justify-between text-sm"
                      >
                        <p className="text-gray-300">
                          {formatDate(trend.date)}
                        </p>
                        <div className="flex items-center gap-2">
                          <div className="w-32 bg-gray-700 rounded-full h-2.5">
                            <div
                              className="bg-blue-500 h-2.5 rounded-full"
                              style={{
                                width: `${Math.max(
                                  1,
                                  (trend.count /
                                    Math.max(
                                      ...data.incidentTrends.map(
                                        (t) => t.count
                                      ),
                                      1
                                    )) *
                                    100
                                )}%`,
                              }}
                            ></div>
                          </div>
                          <p className="font-semibold text-white w-8 text-right">
                            {trend.count}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-tile1 border border-gray-700 rounded-lg p-6">
                  <h2 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <FiMapPin className="mr-2" />
                    Location Hotspots
                  </h2>
                  <div className="space-y-2 text-sm">
                    {data.locationHotspots.map((spot, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-2 rounded-md hover:bg-tile2"
                      >
                        <p
                          className="text-gray-300 truncate pr-4"
                          title={spot.location}
                        >
                          {index + 1}. {spot.location}
                        </p>
                        <p className="font-bold text-white bg-blue-600/50 px-2 py-0.5 rounded-md">
                          {spot.count}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                <div className="bg-tile1 border border-gray-700 rounded-lg p-6">
                  <h2 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <FiTag className="mr-2" />
                    Violation by Category
                  </h2>
                  <div className="space-y-2 text-sm">
                    {data.violationTypes.map((cat, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-2 rounded-md hover:bg-tile2"
                      >
                        <p className="text-gray-300">{cat.category}</p>
                        <p className="font-semibold text-white">
                          {cat.count} cases
                        </p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-tile1 border border-gray-700 rounded-lg p-6">
                  <h2 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <FiMessageSquare className="mr-2" />
                    Common Violation Factors
                  </h2>
                  <div className="space-y-2 text-sm">
                    {data.commonReasons.map((reason, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-2 rounded-md hover:bg-tile2"
                      >
                        <p className="text-gray-300">{reason.reason}</p>
                        <p className="font-semibold text-white">
                          {reason.count} times
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
