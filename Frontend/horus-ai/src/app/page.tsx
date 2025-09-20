/* eslint-disable @next/next/no-img-element */
/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

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
  FiLoader,
} from "react-icons/fi";
import Navigation from "@/components/Navigation";
import { Loader } from "@/components/spinner";
import { toast } from "react-toastify";
import { ToastProps } from "@/utilities/Props";
import { LuListFilter } from "react-icons/lu";
import FilterModal from "@/components/FilterModal";
import Image from "next/image";

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

const INFERENCE_URL = process.env.NEXT_PUBLIC_INFERENCE_URL;
const PROCESS_URL = process.env.NEXT_PUBLIC_PROCESS_URL;

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
  selectedCameraId,
  camIdFilter,
  setFilterState,
  onDecline,
  isSubmitting,
}: {
  incidents: any[];
  onIncidentClick: (index: number) => void;
  setFilterState: () => void;
  selectedCameraId: string | null;
  onAccept: (incident: any) => void;
  camIdFilter?: string;
  onDecline: (incident: any) => void;
  isSubmitting: boolean;
}) => (
  <div>
    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-6">
      <div className="flex w-full items-center space-x-3 mb-4 sm:mb-0">
        <h2 className="text-2xl font-bold text-white">
          Illegal Parking Alerts
        </h2>
        <div className="px-3 py-1 bg-red-500/20 border border-red-500/30 rounded-full">
          <span className="text-red-400 text-xs font-mono font-semibold">
            {selectedCameraId ? incidents.filter(incident => incident.event.cam_id === selectedCameraId).length : incidents.length} ACTIVE
          </span>
        </div>
        <button className="flex gap-2 ml-auto cursor-pointer items-center justify-center" onClick={() => setFilterState()}>
          <LuListFilter className="text-gray-400" />
          <span className="text-gray-400 text-sm font-semibold"> Filter</span>
        </button>
      </div>
    </div>
    <div className="space-y-4 max-h-[calc(100vh-200px)] overflow-y-auto pr-2">
      {incidents.filter((incident) => {
        if (!camIdFilter) return true;
        return incident.event.cam_id === camIdFilter;
      }).length > 0 ? (
        incidents
          .filter((incident) => {
            if (!camIdFilter) return true;
            return incident.event.cam_id === camIdFilter;
          })
          .sort(
            (a, b) =>
              new Date(a.event.started_at).getTime() -
              new Date(b.event.started_at).getTime()
          )
          .reverse()
          .map((incident, index) => {
            const urgency = getUrgencyInfo(incident.urgency_score);
            const UrgencyIcon = urgency.icon;
            const eventDetails = incident.event;
            const llmData = incident.llm_data;

            return (
              <div
                key={eventDetails.event_id + String(index)}
                className="group p-5 relative overflow-hidden rounded-lg border border-gray-700 bg-tile1/80 hover:border-gray-600 hover:bg-tile1 transition-all duration-300"
              >
                <div
                  className={`absolute left-0 top-0 w-1 h-full ${urgency.color}`}
                ></div>
                <div
                  className="flex flex-col cursor-pointer"
                  onClick={() => onIncidentClick(index)}
                >
                  <div>
                    <div className="flex items-start space-x-3 mb-4">
                      <div
                        className={`p-2 rounded-lg ${urgency.bgColor} flex-shrink-0`}
                      >
                        <UrgencyIcon className={`w-4 h-4 ${urgency.textColor}`} />
                      </div>
                      <div className="flex-1">
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
                                  {/* {new Date(eventDetails.started_at).getTime()} */}
                                </span>
                              </div>
                            </div>
                          </div>
                          <div className="flex flex-row items-center justify-center gap-3">
                            <div className="flex items-center space-x-1 text-xs">
                              <FiZap className={`w-3 h-3 ${urgency.textColor}`} />
                              <span className={`font-bold ${urgency.textColor}`}>
                                {incident.urgency_score}
                              </span>
                            </div>
                            <span
                              className={`px-2 py-0.5 rounded-full text-xs font-bold border ${getPriorityColor(
                                llmData?.priority_label
                              )}`}
                            >
                              {llmData?.priority_label?.toUpperCase() ||
                                "UNKNOWN"}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="flex flex-row portrait:flex-col portrait:items-center gap-5">
                  <div className="space-y-3 w-full">
                    {llmData?.narrative && (
                      <div className="bg-gray-800/50 rounded-md p-3 border border-gray-700/50">
                        <p className="text-gray-300 text-xs leading-relaxed italic">
                          {`"${llmData.narrative}"`}
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
                          {llmData.reasons.map(
                            (reason: string, idx: number) => (
                              <span
                                key={idx}
                                className="px-2 py-1 bg-blue-500/20 border border-blue-500/30 text-blue-400 text-xs rounded-lg md:rounded-full"
                              >
                                {reason}
                              </span>
                            )
                          )}
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
                    <div className="flex items-center justify-between text-xs border-t pt-1 border-gray-700/50">
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
                  {/* snapshot gambar */}
                  <div className="landscape:ml-auto">
                    {eventDetails.snapshot_url && (
                      <Image
                        src={eventDetails.snapshot_url}
                        alt={`Snapshot for ${eventDetails.event_id}`}
                        width={200}
                        height={0}
                        className="landscape:max-w-[200px] rounded-md border border-gray-600 bg-gray-800"
                        onError={(e) => {
                          const target = e.target as HTMLImageElement;
                          target.style.display = "none";
                        }}
                      />
                    )}

                    {
                      !isSubmitting ? (
                        <div className="flex items-center space-x-3 mt-4">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onDecline(incident);
                            }}
                            className="w-full text-center py-2 px-4 bg-red-500/20 hover:bg-red-500/40 text-red-400 border border-red-500/30 rounded-lg transition-all duration-200 font-semibold hover:cursor-pointer"
                          >
                            Decline
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onAccept(incident);
                            }}
                            className="w-full text-center py-2 px-4 bg-green-500/20 hover:bg-green-500/40 text-green-400 border border-green-500/30 rounded-lg transition-all duration-200 font-semibold hover:cursor-pointer"
                          >
                            Accept
                          </button>
                        </div>
                      ) : (
                        <div className="flex items-center justify-center mt-4">
                          <FiLoader className="animate-spin" />
                          <span className="ml-2 text-gray-400">Submitting...</span>
                        </div>
                      )
                    }
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
  incident,
  onBackClick,
}: {
  cctv: any;
  incident?: any;
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
          `${INFERENCE_URL}/detector/tracking_data/${cctv.cam_id}`
        );
        const data: TrackingData = await response.json();
        setTrackingData(data);
        setStreamUrl(
          `${PROCESS_URL}${cctv.stream_endpoint}?t=${new Date().getTime()}`
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
        {incident?.event?.snapshot_url ? (
          <img
            src={incident.event.snapshot_url}
            alt="Incident snapshot"
            className="absolute top-0 left-0 w-full h-full object-contain"
          />
        ) : cctv?.is_running && cctv?.stream_endpoint ? (
          <>
            {streamUrl && (
              <>
                <img
                  ref={videoRef}
                  src={streamUrl}
                  alt="Live video feed"
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
              <p>Live feed or snapshot not available.</p>
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
  const [selectedIncident, setSelectedIncident] = useState<any | null>(null);
  const [wholePageLoading, setWholePageLoading] = useState(false);
  const [isFilterModalOpen, setIsFilterModalOpen] = useState<boolean>(false);
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
  const [incidentIdSubmitting, setIncidentIdSubmitting] = useState<string | null>(null);


  useEffect(() => {
    setIsClient(true);

    const fetchCCTVData = async () => {
      try {
        setWholePageLoading(true);
        console.log("Fetching CCTV data from backend...");
        const response = await fetch(`${PROCESS_URL}/cameras`);
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
        const response = await fetch(`${PROCESS_URL}/incidents/pending`);
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
      } catch (error) {
        console.error("Failed to fetch pending incidents:", error);
      } finally {
        setWholePageLoading(false);
      }
    };

    fetchCCTVData();
    fetchPendingIncidents();
  }, []);

  useEffect(() => {
    if (!isClient) return;

    let eventSource: EventSource | null = null;
    const timeoutId = setTimeout(() => {
      eventSource = new EventSource(`${INFERENCE_URL}/events`);

      eventSource.onopen = () => {
        console.log("✅ SSE Connection Established!");
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
    }, 1000);

    return () => {
      clearTimeout(timeoutId);
      if (eventSource) {
        console.log("🔌 Closing SSE connection.");
        eventSource.close();
      }
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

    setSelectedIncident(incident);

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
    setSelectedIncident(null);
    setPreviewCoordinates(null);
  };

  const handleAccept = async (incident: any) => {
    try {
      setIncidentIdSubmitting(incident.event.event_id);
      const response = await fetch(`${PROCESS_URL}/incident/accept`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ incident_data: incident }),
      });
      if (!response.ok) throw new Error("Failed to accept incident.");

      setIncidents((prev) =>
        prev.filter((i) => i.event.event_id !== incident.event.event_id)
      );

      toast.success("Incident recorded. View details in the incidents tab.", ToastProps);
    } catch (error) {
      console.error("Error accepting incident:", error);
      alert("Failed to accept the incident. Please try again.");
    } finally {
      setIncidentIdSubmitting(null);
    }
  };

  const handleDecline = async (incident: any) => {
    try {
      setIncidentIdSubmitting(incident.event.event_id);
      const response = await fetch(`${PROCESS_URL}/incident/decline`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ incident_data: incident }),
      });
      if (!response.ok) throw new Error("Failed to decline incident.");

      setIncidents((prev) =>
        prev.filter((i) => i.event.event_id !== incident.event.event_id)
      );

      toast.success("The false alarm has been logged successfully for further system improvements.", ToastProps);
    } catch (error) {
      console.error("Error declining incident:", error);
      alert("Failed to decline the incident. Please try again.");
    } finally {
      setIncidentIdSubmitting(null);
    }
  };

  if (!isClient) {
    return null;
  }

  const filterByCamId = (cameraId: string | null) => {
    setSelectedCameraId(cameraId);
    setIsFilterModalOpen(false);
    console.log("Selected Camera", cameraId);
  }
  

  return (
    <div className="min-h-screen bg-primary">
      <Navigation />
      {wholePageLoading && <Loader />}
      { 
        isFilterModalOpen &&
        <FilterModal
          isLoading={false}
          setOnBackgroundClick={() => setIsFilterModalOpen(false)}
          cameraIds={cctvLocations.length > 0 ? cctvLocations.map((cctv) => cctv.cam_id) : []}
          buttonLabel="Apply Filters"
          setState={(camId: string | null) => filterByCamId(camId)}
          title="Camera Location Filters"
        />
      }

      <main className="flex flex-row lg:flex-row w-full min-h-screen pt-28 px-4 sm:px-6 lg:px-8 gap-6">
        <div className="w-full lg:w-1/2 flex-shrink-0 transition-all duration-300 overflow-hidden">
          {selectedCCTV ? (
            <CctvPreview
              cctv={selectedCCTV}
              incident={selectedIncident}
              onBackClick={handleBackToList}
            />
          ) : (
            <IncidentList
              incidents={incidents}
              onIncidentClick={handleIncidentClick}
              onAccept={handleAccept}
              selectedCameraId={selectedCameraId}
              setFilterState={() => setIsFilterModalOpen(true)}
              camIdFilter={selectedCameraId || undefined}
              onDecline={handleDecline}
              isSubmitting={incidentIdSubmitting === selectedIncident?.event.event_id}
            />
          )}
        </div>

        <div className="w-full ml-auto lg:w-1/2 h-[50vh] lg:h-[calc(100vh-150px)] rounded-lg border border-gray-700 overflow-hidden">
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
              <FiLoader className="animate-spin mr-2" />
              Loading map and CCTV data...
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
