"use client";

import HlsPlayer from "@/components/HLSPlayer";
import { cctvLocations, illegalParkingLocations } from "@/utilities/DummyData";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { BiSort } from "react-icons/bi";
import { FiClock, FiMapPin, FiAlertTriangle, FiCamera } from "react-icons/fi";
import Navigation from "@/components/Navigation";

const Map = dynamic(() => import("@/components/Map"), {
  ssr: false,
});

// ini nanti ubah aja sesuain sama urgency scoring
const getSeverityInfo = (index: number) => {
  const severities = [
    {
      level: "CRITICAL",
      color: "bg-red-500",
      textColor: "text-red-400",
      icon: FiAlertTriangle,
    },
    {
      level: "HIGH",
      color: "bg-orange-500",
      textColor: "text-orange-400",
      icon: FiClock,
    },
    {
      level: "MEDIUM",
      color: "bg-yellow-500",
      textColor: "text-yellow-400",
      icon: FiClock,
    },
    {
      level: "HIGH",
      color: "bg-orange-500",
      textColor: "text-orange-400",
      icon: FiClock,
    },
    {
      level: "CRITICAL",
      color: "bg-red-500",
      textColor: "text-red-400",
      icon: FiAlertTriangle,
    },
  ];
  return severities[index % severities.length];
};

export default function Home() {
  const [previewCoordinates, setPreviewCoordinates] = useState<
    [number, number] | null
  >(null);
  const [selectedCCTVIndex, setSelectedCCTVIndex] = useState<number | null>(
    null
  );
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  // ini buat koneksi SSE
  useEffect(() => {
    let es: EventSource | null = null;

    try {
      es = new EventSource("");

      es.onopen = () => {
        console.log("SSE connected");
        setIsConnected(true);
      };

      es.onmessage = (e) => {
        console.log("Message:", e.data);
      };

      es.onerror = () => {
        console.warn("SSE connection failed");
        setIsConnected(false);
        if (es) es.close();
      };
    } catch {
      console.warn("SSE not available");
      setIsConnected(false);
    }

    return () => {
      if (es) es.close();
    };
  }, []);

  const handleEventClick = (index: number | null) => {
    if (selectedIndex === index || index === null) {
      setSelectedIndex(null);
      setPreviewCoordinates(null);
    } else {
      setSelectedIndex(index);
      setPreviewCoordinates(illegalParkingLocations[index].coordinates);
    }
    setSelectedCCTVIndex(null);
  };

  const viewOnGoogleMaps = (coordinates: [number, number]) => {
    window.open(
      `https://www.google.com/maps/search/?api=1&query=${coordinates[0]},${coordinates[1]}`,
      "_blank"
    );
  };

  return (
    <div className="min-h-screen bg-primary">
      <Navigation />

      <main className="flex flex-row justify-center w-full min-h-screen space-x-10 p-10 pt-24 mt-4">
        <div>
          <div className="flex items-center justify-between w-[60vh] mb-8">
            <div className="flex items-center space-x-3">
              <div className="text-3xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
                ILLEGAL PARKING ALERTS
              </div>
              <div className="px-3 py-1 bg-red-500/20 border border-red-500/30 rounded-full">
                <span className="text-red-400 text-xs font-mono font-semibold">
                  {illegalParkingLocations.length} ACTIVE
                </span>
              </div>
            </div>
            <div
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg border ${
                isConnected
                  ? "bg-green-500/10 border-green-500/30 text-green-400"
                  : "bg-yellow-500/10 border-yellow-500/30 text-yellow-400"
              }`}
            >
              <div
                className={`w-2 h-2 rounded-full ${
                  isConnected ? "bg-green-400" : "bg-yellow-400"
                } animate-pulse`}
              ></div>
              <span className="text-sm font-mono font-semibold">
                {isConnected ? "LIVE CONNECTED" : "SSE NOT CONNECTED"}
              </span>
            </div>
          </div>

          {illegalParkingLocations.length === 0 ? (
            <div className="text-center text-gray-500">
              No illegal parking cases found.
            </div>
          ) : (
            <div className="w-full">
              <div className="flex mb-6 items-center justify-between p-4 bg-tile1/50 border border-gray-700/50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <BiSort className="text-xl text-gray-400" />
                  <span className="text-sm font-mono text-gray-300">
                    SORT BY:
                  </span>
                  <select className="bg-tile2 border border-gray-600 rounded-md px-3 py-1 text-sm text-white font-mono focus:outline-none focus:border-blue-500">
                    <option value="urgency">URGENCY LEVEL</option>
                    <option value="time">TIMESTAMP</option>
                    <option value="location">LOCATION</option>
                  </select>
                </div>
                <div className="text-xs text-gray-400 font-mono">
                  LAST UPDATE: {new Date().toLocaleTimeString()}
                </div>
              </div>

              <div className="space-y-3">
                {illegalParkingLocations.map((location, index) => {
                  const severity = getSeverityInfo(index);
                  const SeverityIcon = severity.icon;

                  return (
                    <div
                      key={index}
                      className={`group relative overflow-hidden rounded-lg border transition-all duration-300 cursor-pointer ${
                        selectedIndex === index
                          ? "border-blue-500 bg-blue-500/5 shadow-lg shadow-blue-500/20"
                          : "border-gray-700/50 bg-tile1/30 hover:border-gray-600 hover:bg-tile1/50"
                      }`}
                      onClick={() => handleEventClick(index)}
                    >
                      <div
                        className={`absolute left-0 top-0 w-1 h-full ${severity.color}`}
                      ></div>

                      <div className="p-5 pl-6">
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center space-x-3">
                            <div
                              className={`p-2 rounded-lg ${severity.color}/20 border border-${severity.color}/30`}
                            >
                              <SeverityIcon
                                className={`w-4 h-4 ${severity.textColor}`}
                              />
                            </div>
                            <div>
                              <div className="flex items-center space-x-2 mb-1">
                                <h3 className="font-semibold text-white text-lg">
                                  {location.locationName}
                                </h3>
                                <span
                                  className={`px-2 py-1 rounded-full text-xs font-mono font-bold ${severity.color} text-black`}
                                >
                                  {severity.level}
                                </span>
                              </div>
                              <div className="flex items-center space-x-4 text-xs text-gray-400 font-mono">
                                <div className="flex items-center space-x-1">
                                  <FiCamera className="w-3 h-3" />
                                  <span>
                                    {/* ini tar ganti sama id cam asli waktu ud bikin json */}
                                    CAM-
                                    {String((index % 6) + 1).padStart(2, "0")}
                                  </span>
                                </div>
                                <div className="flex items-center space-x-1">
                                  <FiClock className="w-3 h-3" />
                                  <span>
                                    {new Date(
                                      location.timestamp
                                    ).toLocaleTimeString()}
                                  </span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="mb-3">
                          <p className="text-gray-300 text-sm leading-relaxed">
                            {location.narration}
                          </p>
                        </div>

                        <div className="flex items-center justify-between pt-3 border-t border-gray-700/30">
                          <div className="flex items-center space-x-4 text-xs text-gray-400">
                            <span>
                              COORDINATES: {location.coordinates[0].toFixed(4)},{" "}
                              {location.coordinates[1].toFixed(4)}
                            </span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                            <span className="text-xs text-green-400 font-mono">
                              EVIDENCE RECORDED
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="absolute inset-0 bg-gradient-to-r from-blue-500/0 to-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        <Map
          className="h-[calc(100vh-120px)] w-full z-50 rounded-lg border border-gray-700"
          illegalParkingLocations={illegalParkingLocations}
          defaultViewingCoordinates={
            previewCoordinates ?? [-6.921817208463581, 107.6070564264402]
          }
          cctvLocations={cctvLocations}
          zoomLevel={previewCoordinates ? 18 : 14}
          onMarkerClick={(index) => {
            if (index !== null) {
              setSelectedCCTVIndex(index);
              setSelectedIndex(null);
              setPreviewCoordinates(cctvLocations[index].coordinates);
            } else {
              setSelectedCCTVIndex(null);
              setSelectedIndex(null);
              setPreviewCoordinates(null);
            }
          }}
        />

        {selectedIndex !== null && (
          <div className="fixed z-100 top-20 right-30 bg-tile1 border border-gray-600 p-6 rounded-lg shadow-2xl max-w-100 space-y-4">
            <h2 className="text-xl font-semibold">
              {illegalParkingLocations[selectedIndex].locationName}
            </h2>
            <div className="relative w-full pb-[56.25%] h-0 rounded-lg overflow-hidden">
              <iframe
                src={illegalParkingLocations[selectedIndex].videoUrl}
                className="absolute top-0 left-0 w-full h-full"
                allow="autoplay"
              />
            </div>
            <p className="text-sm text-gray-300">
              {illegalParkingLocations[selectedIndex].narration}
            </p>
            <p className="text-sm text-gray-400">
              Timestamp:{" "}
              {new Date(
                illegalParkingLocations[selectedIndex].timestamp
              ).toLocaleString()}
            </p>
            <p className="font-secondary text-sm">
              Coordinates:{" "}
              {illegalParkingLocations[selectedIndex].coordinates[0].toFixed(6)}
              ,{" "}
              {illegalParkingLocations[selectedIndex].coordinates[1].toFixed(6)}
            </p>
            <div className="flex flex-row space-x-2">
              <button
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 cursor-pointer text-white rounded transition"
                onClick={() => handleEventClick(selectedIndex)}
              >
                Close
              </button>
              <button
                className="px-4 py-2 bg-red-600 hover:bg-red-700 cursor-pointer text-white rounded w-full transition"
                onClick={() =>
                  viewOnGoogleMaps(
                    illegalParkingLocations[selectedIndex].coordinates
                  )
                }
              >
                Open On Google Maps
              </button>
            </div>
          </div>
        )}

        {selectedCCTVIndex !== null && (
          <div className="fixed z-100 top-20 right-30 bg-tile1 border border-gray-600 p-6 rounded-lg shadow-2xl w-150 space-y-4">
            <h2 className="text-xl font-semibold">
              {cctvLocations[selectedCCTVIndex].description}
            </h2>
            <div className="relative w-full pb-[75%] h-0 rounded-lg overflow-hidden">
              {cctvLocations[selectedCCTVIndex].streamUrl && (
                <HlsPlayer
                  src={cctvLocations[selectedCCTVIndex].streamUrl ?? ""}
                />
              )}
            </div>
            <button
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 cursor-pointer text-white rounded transition"
              onClick={() => setSelectedCCTVIndex(null)}
            >
              Close
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
