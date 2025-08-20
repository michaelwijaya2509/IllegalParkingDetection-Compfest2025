"use client";

import HlsPlayer from "@/components/HLSPlayer";
import dynamic from "next/dynamic";
import { useEffect, useState, useRef } from "react";
import {
  FiClock,
  FiAlertTriangle,
  FiCamera,
  FiArrowLeft,
  FiZap,
  FiMapPin,
  FiShield,
  FiCheckCircle,
  FiUsers,
  FiTruck,
  FiWifiOff,
} from "react-icons/fi";
import Navigation from "@/components/Navigation";

interface Track {
  track_id: number;
  bbox: [number, number, number, number];
  class_name: string;
  stationary_s: number;
  is_close_to_violation: boolean;
  is_violation: boolean;
}

interface Zone {
  name: string;
  polygon: [number, number][];
}

interface TrackingData {
  tracks: Track[];
  zones: Zone[];
  timestamp: number;
  video_width: number;
  video_height: number;
}

const Map = dynamic(() => import("@/components/Map"), {
  ssr: false,
});

const BACKEND_URL = "http://localhost:5001";

const getUrgencyInfo = (score: number) => {
  if (score >= 80) {
    return {
      level: "CRITICAL",
      color: "bg-red-500",
      textColor: "text-red-400",
      borderColor: "border-red-500",
      bgColor: "bg-red-500/20",
      icon: FiAlertTriangle,
    };
  }
  if (score >= 60) {
    return {
      level: "HIGH",
      color: "bg-orange-500",
      textColor: "text-orange-400",
      borderColor: "border-orange-500",
      bgColor: "bg-orange-500/20",
      icon: FiAlertTriangle,
    };
  }
  return {
    level: "MEDIUM",
    color: "bg-yellow-500",
    textColor: "text-yellow-400",
    borderColor: "border-yellow-500",
    bgColor: "bg-yellow-500/20",
    icon: FiClock,
  };
};

const getPriorityColor = (label: string) => {
  switch (label?.toLowerCase()) {
    case "critical":
      return "text-red-400 bg-red-500/20 border-red-500/30";
    case "high":
      return "text-orange-400 bg-orange-500/20 border-orange-500/30";
    case "medium":
      return "text-yellow-400 bg-yellow-500/20 border-yellow-500/30";
    case "low":
      return "text-green-400 bg-green-500/20 border-green-500/30";
    default:
      return "text-gray-400 bg-gray-500/20 border-gray-500/30";
  }
};

const IncidentList = ({
  incidents,
  onIncidentClick,
  onAccept,
  onDecline,
}: {
  incidents: any[];
  onIncidentClick: (index: number) => void;
  onAccept: (incident: any) => void;
  onDecline: (incident: any) => void;
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
        incidents.map((incident, index) => {
          const urgency = getUrgencyInfo(incident.urgency_score);
          const UrgencyIcon = urgency.icon;
          const eventDetails = incident.event;
          const llmData = incident.llm_data;

          return (
            <div
              key={eventDetails.event_id}
              className="group relative overflow-hidden rounded-lg border border-gray-700 bg-tile1/80 hover:border-gray-600 hover:bg-tile1 transition-all duration-300"
            >
              <div
                className={`absolute left-0 top-0 w-1 h-full ${urgency.color}`}
              ></div>
              <div
                className="p-4 pl-5 cursor-pointer"
                onClick={() => onIncidentClick(index)}
              >
                <div className="flex items-start space-x-3 mb-4">
                  <div
                    className={`p-2 rounded-lg ${urgency.bgColor} flex-shrink-0`}
                  >
                    <UrgencyIcon className={`w-4 h-4 ${urgency.textColor}`} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h3 className="font-semibold text-white text-sm leading-tight mb-1">
                          <FiMapPin className="inline w-3 h-3 mr-1" />
                          {incident.address || "Lokasi tidak diketahui"}
                        </h3>
                        <div className="flex items-center space-x-3 text-xs text-gray-400 font-mono">
                          <div className="flex items-center space-x-1">
                            <FiCamera className="w-3 h-3" />
                            <span>{eventDetails.cam_id}</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <FiClock className="w-3 h-3" />
                            <span>
                              {new Date(
                                eventDetails.started_at
                              ).toLocaleTimeString()}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="flex flex-col items-end space-y-1">
                        <span
                          className={`px-2 py-0.5 rounded-full text-xs font-bold border ${getPriorityColor(
                            llmData?.priority_label
                          )}`}
                        >
                          {llmData?.priority_label?.toUpperCase() || "UNKNOWN"}
                        </span>
                        <div className="flex items-center space-x-1 text-xs">
                          <FiZap className={`w-3 h-3 ${urgency.textColor}`} />
                          <span className={`font-bold ${urgency.textColor}`}>
                            {incident.urgency_score}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="pl-9 space-y-3">
                  {llmData?.narrative && (
                    <div className="bg-gray-800/50 rounded-md p-3 border border-gray-700/50">
                      <p className="text-gray-300 text-sm leading-relaxed italic">
                        "{llmData.narrative}"
                      </p>
                    </div>
                  )}

                  {llmData?.reasons && llmData.reasons.length > 0 && (
                    <div>
                      <div className="text-gray-400 font-mono text-xs mb-2 flex items-center">
                        <FiShield className="w-3 h-3 mr-1" />
                        VIOLATION FACTORS
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {llmData.reasons.map((reason: string, idx: number) => (
                          <span
                            key={idx}
                            className="px-2 py-1 bg-blue-500/20 border border-blue-500/30 text-blue-400 text-xs rounded-full"
                          >
                            {reason}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {llmData?.recommended_actions &&
                    llmData.recommended_actions.length > 0 && (
                      <div>
                        <div className="text-gray-400 font-mono text-xs mb-2 flex items-center">
                          <FiCheckCircle className="w-3 h-3 mr-1" />
                          RECOMMENDED ACTIONS
                        </div>
                        <div className="space-y-1">
                          {llmData.recommended_actions.map(
                            (action: string, idx: number) => (
                              <div
                                key={idx}
                                className="flex items-center space-x-2 text-xs text-gray-300"
                              >
                                <div className="w-1 h-1 bg-green-400 rounded-full"></div>
                                <span>{action}</span>
                              </div>
                            )
                          )}
                        </div>
                      </div>
                    )}
                  <div className="flex items-center justify-between text-xs pt-2 border-t border-gray-700/50">
                    <div className="flex items-center space-x-3">
                      <div className="flex items-center space-x-1">
                        <span className="text-gray-400">CONFIDENCE:</span>
                        <span
                          className={`font-semibold ${
                            llmData?.confidence === "high"
                              ? "text-green-400"
                              : llmData?.confidence === "medium"
                              ? "text-yellow-400"
                              : "text-red-400"
                          }`}
                        >
                          {llmData?.confidence?.toUpperCase() || "UNKNOWN"}
                        </span>
                      </div>
                      {llmData?.category && (
                        <div className="flex items-center space-x-1">
                          <FiTruck className="w-3 h-3 text-gray-400" />
                          <span className="text-gray-300">
                            {llmData.category}
                          </span>
                        </div>
                      )}
                    </div>

                    <div className="text-gray-400 font-mono">
                      {new Date(eventDetails.started_at).toLocaleDateString()}
                    </div>
                  </div>
                </div>
              </div>

              <div className="px-4 pb-4 pl-14">
                {eventDetails.snapshot_url && (
                  <div className="pt-2">
                    <img
                      src={eventDetails.snapshot_url}
                      alt={`Snapshot for ${eventDetails.event_id}`}
                      className="w-full rounded-md border border-gray-600 bg-gray-800"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.style.display = "none";
                      }}
                    />
                  </div>
                )}

                <div className="flex items-center space-x-3 mt-4">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDecline(incident);
                    }}
                    className="w-full text-center py-2 px-4 bg-red-500/20 hover:bg-red-500/40 text-red-400 border border-red-500/30 rounded-lg transition-all duration-200 font-semibold"
                  >
                    Decline
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onAccept(incident);
                    }}
                    className="w-full text-center py-2 px-4 bg-green-500/20 hover:bg-green-500/40 text-green-400 border border-green-500/30 rounded-lg transition-all duration-200 font-semibold"
                  >
                    Accept
                  </button>
                </div>
              </div>
            </div>
          );
        })
      ) : (
        <div className="text-center text-gray-500 py-10">
          <FiUsers className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>No active incidents detected</p>
          <p className="text-xs mt-1">Monitoring for violations...</p>
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
}) => {
  const [trackingData, setTrackingData] = useState<TrackingData | null>(null);
  const [isFrozen, setIsFrozen] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLImageElement>(null);
  const [streamUrl, setStreamUrl] = useState<string>("");

  useEffect(() => {
    setIsFrozen(false);
    setTrackingData(null);
  }, [cctv]);

  const drawOverlays = (data: TrackingData) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const videoRect = video.getBoundingClientRect();
    canvas.width = videoRect.width;
    canvas.height = videoRect.height;

    const scaleX = canvas.width / (data.video_width || 1);
    const scaleY = canvas.height / (data.video_height || 1);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = "rgba(255, 255, 0, 0.7)";
    ctx.lineWidth = 2;
    data.zones.forEach((zone) => {
      ctx.beginPath();
      zone.polygon.forEach((point, index) => {
        const [x, y] = point;
        if (index === 0) ctx.moveTo(x * scaleX, y * scaleY);
        else ctx.lineTo(x * scaleX, y * scaleY);
      });
      ctx.closePath();
      ctx.stroke();
    });

    data.tracks.forEach((track) => {
      const [x1, y1, x2, y2] = track.bbox;
      const color = track.is_violation
        ? "red"
        : track.is_close_to_violation
        ? "orange"
        : "cyan";
      ctx.strokeStyle = color;
      ctx.lineWidth = track.is_violation || track.is_close_to_violation ? 3 : 2;
      ctx.strokeRect(
        x1 * scaleX,
        y1 * scaleY,
        (x2 - x1) * scaleX,
        (y2 - y1) * scaleY
      );

      ctx.fillStyle = "white";
      ctx.font = "12px Arial";
      const label = `${track.class_name}:${track.track_id} (${track.stationary_s}s)`;
      ctx.fillText(label, x1 * scaleX, y1 * scaleY - 10);
    });
  };

  useEffect(() => {
    if (!cctv?.is_running || !cctv?.stream_endpoint || isFrozen) {
      return;
    }

    const intervalId = setInterval(async () => {
      try {
        const response = await fetch(
          `${BACKEND_URL}/detector/tracking_data/${cctv.cam_id}`
        );
        const data: TrackingData = await response.json();
        setTrackingData(data);
        setStreamUrl(
          `${BACKEND_URL}${cctv.stream_endpoint}?t=${new Date().getTime()}`
        );

        const hasViolation = data.tracks.some((track) => track.is_violation);
        if (hasViolation) {
          console.log(`Violation detected on ${cctv.cam_id}. Freezing stream.`);
          setIsFrozen(true);
        }
      } catch (error) {
        console.error("Failed to fetch tracking data:", error);
        setTrackingData(null);
      }
    }, 500);

    return () => clearInterval(intervalId);
  }, [cctv, isFrozen]);

  useEffect(() => {
    if (trackingData) {
      drawOverlays(trackingData);
    }
  }, [trackingData]);

  return (
    <div>
      <div className="flex items-center mb-6">
        <button
          onClick={onBackClick}
          className="mr-4 p-2 rounded-full hover:bg-tile2 transition-colors"
        >
          <FiArrowLeft className="w-6 h-6 text-white" />
        </button>
        <div>
          <h2 className="text-2xl font-bold text-white">{cctv.name}</h2>
          <p className="text-sm text-gray-400">{cctv.address}</p>
        </div>
      </div>
      <div className="relative aspect-video w-full rounded-lg overflow-hidden bg-black border border-gray-700">
        {cctv?.is_running && cctv?.stream_endpoint ? (
          <>
            {streamUrl && (
              <>
                <img
                  ref={videoRef}
                  src={streamUrl}
                  alt="Live video feed of incident"
                  className="absolute top-0 left-0 w-full h-full object-contain"
                />
                <canvas
                  ref={canvasRef}
                  className="absolute top-0 left-0 w-full h-full"
                />
              </>
            )}
          </>
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-500">
            <div className="text-center">
              <FiWifiOff className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p>Live feed not available for this camera.</p>
              <p className="text-xs mt-1">Camera: {cctv.cam_id}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default function Home() {
  const [cctvLocations, setCctvLocations] = useState<any[]>([]);
  const [incidents, setIncidents] = useState<any[]>([]);
  const [previewCoordinates, setPreviewCoordinates] = useState<
    [number, number] | null
  >(null);
  const [selectedCCTV, setSelectedCCTV] = useState<any | null>(null);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);

    const fetchCCTVData = async () => {
      try {
        console.log("Fetching CCTV data from backend...");
        const response = await fetch(`${BACKEND_URL}/cameras`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const rawData = await response.json();
        console.log("Raw CCTV data received from backend:", rawData);

        const formattedData = rawData.map((camera: any) => ({
          ...camera,
          coordinates: camera.coordinates || [camera.lat || 0, camera.lon || 0],
        }));

        if (formattedData && formattedData.length > 0) {
          console.log(
            "SUCCESS: Transformed CCTV data with coordinates:",
            formattedData
          );
          setCctvLocations(formattedData);
        } else {
          console.log("There are no cameras fetched");
        }
      } catch (error) {
        console.error("Failed to fetch CCTV data from backend:", error);
      }
    };

    const fetchPendingIncidents = async () => {
      try {
        console.log("Fetching pending incidents from backend...");
        const response = await fetch(`${BACKEND_URL}/incidents/pending`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        const formattedIncidents = data.map((scoredEvent: any) => ({
          urgency_score: scoredEvent.scored?.priority_score || 0,
          address:
            scoredEvent.camera?.address ||
            scoredEvent.event.location?.address ||
            "Lokasi tidak diketahui",
          coordinates: [scoredEvent.camera.lat, scoredEvent.camera.lon],
          cam_id: scoredEvent.event.cam_id,
          event: scoredEvent.event,
          llm_data: {
            priority_score: scoredEvent.scored?.priority_score || 0,
            priority_label: scoredEvent.scored?.priority_label || "unknown",
            narrative: scoredEvent.scored?.narrative || "",
            reasons: scoredEvent.scored?.reasons || [],
            recommended_actions: scoredEvent.scored?.recommended_actions || [],
            confidence: scoredEvent.scored?.confidence || "unknown",
            category: scoredEvent.scored?.category || "",
          },
          timestamp: scoredEvent.event.started_at,
          scored_at: scoredEvent.scored_at,
        }));
        setIncidents(formattedIncidents);
        console.log(
          "SUCCESS: Fetched initial pending incidents:",
          formattedIncidents
        );
      } catch (error) {
        console.error("Failed to fetch pending incidents:", error);
      }
    };

    fetchCCTVData();
    fetchPendingIncidents();
  }, []);

  useEffect(() => {
    if (!isClient) return;

    console.log("Connecting to SSE...");
    const eventSource = new EventSource(`${BACKEND_URL}/events`);

    eventSource.onopen = () => {
      console.log("SSE Connection Established!");
    };

    eventSource.onmessage = (event) => {
      try {
        const eventData = JSON.parse(event.data);

        if (eventData.type === "violation_event") {
          const scoredEvent = eventData.data;
          const urgencyScore = scoredEvent.scored?.priority_score || 0;

          const incidentWithCoords = {
            urgency_score: urgencyScore,
            address:
              scoredEvent.camera?.address ||
              scoredEvent.event.location?.address ||
              "Lokasi tidak diketahui",
            coordinates: [scoredEvent.camera.lat, scoredEvent.camera.lon],
            cam_id: scoredEvent.event.cam_id,
            event: scoredEvent.event,
            llm_data: {
              priority_score: scoredEvent.scored?.priority_score || 0,
              priority_label: scoredEvent.scored?.priority_label || "unknown",
              narrative: scoredEvent.scored?.narrative || "",
              reasons: scoredEvent.scored?.reasons || [],
              recommended_actions:
                scoredEvent.scored?.recommended_actions || [],
              confidence: scoredEvent.scored?.confidence || "unknown",
              category: scoredEvent.scored?.category || "",
            },
            timestamp: scoredEvent.event.started_at,
            scored_at: scoredEvent.scored_at,
          };

          setIncidents((prev) => {
            const existingIndex = prev.findIndex(
              (existing) =>
                existing.event.event_id === incidentWithCoords.event.event_id
            );

            if (existingIndex !== -1) {
              const updated = [...prev];
              updated[existingIndex] = incidentWithCoords;
              return updated;
            } else {
              return [incidentWithCoords, ...prev];
            }
          });
        }
      } catch (error) {
        console.error(
          "❌ Failed to parse SSE data:",
          error,
          "Raw data:",
          event.data
        );
      }
    };

    eventSource.onerror = (err) => {
      console.error("❌ SSE Error:", err);
    };

    return () => {
      console.log("🔌 Closing SSE connection.");
      eventSource.close();
    };
  }, [isClient]);

  const handleMarkerClick = (index: number | null) => {
    if (index !== null) {
      const cctv = cctvLocations[index];
      setSelectedCCTV(cctv);
      if (cctv.coordinates) {
        setPreviewCoordinates(cctv.coordinates as [number, number]);
      }
    } else {
      setSelectedCCTV(null);
    }
  };

  const handleIncidentClick = (index: number) => {
    const incident = incidents[index];
    if (!incident) return;

    const matchingCCTV = cctvLocations.find(
      (cctv) => cctv.cam_id === incident.cam_id
    );

    if (matchingCCTV) {
      setSelectedCCTV(matchingCCTV);
    } else {
      setSelectedCCTV({ cam_id: incident.cam_id, name: incident.address });
    }

    if (incident.coordinates) {
      setPreviewCoordinates(incident.coordinates as [number, number]);
    }
  };

  const handleBackToList = () => {
    setSelectedCCTV(null);
    setPreviewCoordinates(null);
  };

  const handleAccept = async (incident: any) => {
    try {
      const response = await fetch(`${BACKEND_URL}/incident/accept`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ incident_data: incident }),
      });
      if (!response.ok) throw new Error("Failed to accept incident.");

      setIncidents((prev) =>
        prev.filter((i) => i.event.event_id !== incident.event.event_id)
      );
    } catch (error) {
      console.error("Error accepting incident:", error);
      alert("Failed to accept the incident. Please try again.");
    }
  };

  const handleDecline = async (incident: any) => {
    try {
      const response = await fetch(`${BACKEND_URL}/incident/decline`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ incident_data: incident }),
      });
      if (!response.ok) throw new Error("Failed to decline incident.");

      setIncidents((prev) =>
        prev.filter((i) => i.event.event_id !== incident.event.event_id)
      );
    } catch (error) {
      console.error("Error declining incident:", error);
      alert("Failed to decline the incident. Please try again.");
    }
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
              onAccept={handleAccept}
              onDecline={handleDecline}
            />
          )}
        </div>

        <div className="w-full lg:w-3/5 xl:w-2/3 h-[50vh] lg:h-[calc(100vh-120px)] rounded-lg border border-gray-700 overflow-hidden">
          {cctvLocations.length > 0 ? (
            <Map
              illegalParkingLocations={incidents}
              defaultViewingCoordinates={
                previewCoordinates ?? [-6.9218, 107.607]
              }
              cctvLocations={cctvLocations}
              zoomLevel={previewCoordinates ? 18 : 14}
              onMarkerClick={handleMarkerClick}
              onIncidentClick={handleIncidentClick}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-tile1 text-gray-400">
              Loading map and CCTV data...
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
