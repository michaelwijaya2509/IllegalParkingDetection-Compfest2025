"use client";

import HlsPlayer from "@/components/HLSPlayer";
import {
  cctvLocations,
  illegalParkingLocations as initialIncidents,
} from "@/utilities/DummyData";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import {
  FiClock,
  FiMapPin,
  FiAlertTriangle,
  FiCamera,
  FiArrowLeft,
} from "react-icons/fi";
import Navigation from "@/components/Navigation";

const Map = dynamic(() => import("@/components/Map"), {
  ssr: false,
});

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
  ];
  return severities[index % severities.length];
};

const IncidentList = ({
  incidents,
  onIncidentClick,
}: {
  incidents: any[];
  onIncidentClick: (index: number) => void;
}) => (
  <div>
    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-6">
      <div className="flex items-center space-x-3 mb-4 sm:mb-0">
        <h2 className="text-2xl font-bold text-white">
          Illegal Parking Alerts
        </h2>
        <div className="px-3 py-1 bg-red-500/20 border border-red-500/30 rounded-full">
          <span className="text-red-400 text-xs font-mono font-semibold">
            {incidents.length} ACTIVE
          </span>
        </div>
      </div>
    </div>
    <div className="space-y-4 max-h-[calc(100vh-200px)] overflow-y-auto pr-2">
      {incidents.length > 0 ? (
        incidents.map((location, index) => {
          const severity = getSeverityInfo(index);
          const SeverityIcon = severity.icon;
          return (
            <div
              key={index}
              className="group relative overflow-hidden rounded-lg border border-gray-700 bg-tile1/80 hover:border-gray-600 hover:bg-tile1 transition-all duration-300 cursor-pointer"
              onClick={() => onIncidentClick(index)}
            >
              <div
                className={`absolute left-0 top-0 w-1 h-full ${severity.color}`}
              ></div>
              <div className="p-4 pl-5">
                <div className="flex items-center space-x-3 mb-2">
                  <div className={`p-2 rounded-lg ${severity.color}/20`}>
                    <SeverityIcon className={`w-4 h-4 ${severity.textColor}`} />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">
                      {location.locationName || `Incident #${index + 1}`}
                    </h3>
                    <div className="flex items-center space-x-3 text-xs text-gray-400 font-mono">
                      <div className="flex items-center space-x-1">
                        <FiCamera className="w-3 h-3" />
                        <span>
                          {location.cam_id || `CAM-0${(index % 6) + 1}`}
                        </span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <FiClock className="w-3 h-3" />
                        <span>
                          {new Date(location.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
                <p className="text-gray-300 text-sm leading-relaxed pl-10">
                  {location.narration ||
                    `Vehicle (ID: ${location.track_id}) detected in zone ${location.zone_name}.`}
                </p>
              </div>
            </div>
          );
        })
      ) : (
        <div className="text-center text-gray-500 py-10">
          No active incidents.
        </div>
      )}
    </div>
  </div>
);

const CctvPreview = ({
  cctv,
  onBackClick,
}: {
  cctv: any;
  onBackClick: () => void;
}) => (
  <div>
    <div className="flex items-center mb-6">
      <button
        onClick={onBackClick}
        className="mr-4 p-2 rounded-full hover:bg-tile2 transition-colors"
      >
        <FiArrowLeft className="w-6 h-6 text-white" />
      </button>
      <div>
        <h2 className="text-2xl font-bold text-white">{cctv.description}</h2>
        <p className="text-sm text-gray-400">{cctv.location}</p>
      </div>
    </div>
    <div className="aspect-video w-full rounded-lg overflow-hidden bg-black border border-gray-700">
      {cctv.streamUrl && <HlsPlayer src={cctv.streamUrl} />}
    </div>
    <div className="mt-4 text-xs text-gray-500 font-mono">
      COORDINATES: {cctv.coordinates[0].toFixed(4)},{" "}
      {cctv.coordinates[1].toFixed(4)}
    </div>
  </div>
);

export default function Home() {
  const [incidents, setIncidents] = useState<any[]>(initialIncidents);
  const [previewCoordinates, setPreviewCoordinates] = useState<
    [number, number] | null
  >(null);
  const [selectedCCTV, setSelectedCCTV] = useState<any | null>(null);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const eventSource = new EventSource("http://localhost:5001/events");

    eventSource.onopen = () => {
      console.log("SSE Connection Opened!");
    };

    eventSource.onmessage = (event) => {
      try {
        const eventData = JSON.parse(event.data);

        if (eventData.type === "violation") {
          console.log("New violation incident received:", eventData);
          setIncidents((prevIncidents) => [eventData, ...prevIncidents]);
        }
      } catch (error) {
        console.error("Failed to parse SSE data:", error);
      }
    };

    eventSource.onerror = (err) => {
      console.error("SSE Error:", err);
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, []);

  const handleMarkerClick = (index: number | null) => {
    if (index !== null) {
      const cctv = cctvLocations[index];
      setSelectedCCTV(cctv);
      setPreviewCoordinates(cctv.coordinates);
    } else {
      setSelectedCCTV(null);
    }
  };

  const handleIncidentClick = (index: number) => {
    const incident = incidents[index];
    setSelectedCCTV(null);
    setPreviewCoordinates(incident.coordinates);
  };

  const handleBackToList = () => {
    setSelectedCCTV(null);
    setPreviewCoordinates(null);
  };

  if (!isClient) {
    return null;
  }

  return (
    <div className="min-h-screen bg-primary">
      <Navigation />

      <main className="flex flex-col lg:flex-row w-full min-h-screen pt-24 mt-4 px-4 sm:px-6 lg:px-8 gap-6">
        <div className="w-full lg:w-2/5 xl:w-1/3 flex-shrink-0 transition-all duration-300">
          {selectedCCTV ? (
            <CctvPreview cctv={selectedCCTV} onBackClick={handleBackToList} />
          ) : (
            <IncidentList
              incidents={incidents}
              onIncidentClick={handleIncidentClick}
            />
          )}
        </div>

        <div className="w-full lg:w-3/5 xl:w-2/3 h-[50vh] lg:h-[calc(100vh-120px)] rounded-lg border border-gray-700 overflow-hidden">
          <Map
            illegalParkingLocations={incidents}
            defaultViewingCoordinates={previewCoordinates ?? [-6.9218, 107.607]}
            cctvLocations={cctvLocations}
            zoomLevel={previewCoordinates ? 18 : 14}
            onMarkerClick={handleMarkerClick}
          />
        </div>
      </main>
    </div>
  );
}
