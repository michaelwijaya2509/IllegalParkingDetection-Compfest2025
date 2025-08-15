// app/live-feed/page.tsx (MODIFIED)

"use client";
import { useState, useEffect, useRef } from "react";
import Navigation from "@/components/Navigation";
import {
  FiMaximize2,
  FiMinimize2,
  FiCamera,
  FiWifi,
  FiWifiOff,
  FiPlayCircle,
} from "react-icons/fi";

// --- PERUBAHAN 1: Definisikan tipe data dari API ---
interface Camera {
  cam_id: string;
  name: string;
  address: string;
  stream_url: string;
  zones: string[];
  is_running: boolean;
  stream_endpoint: string | null;
}

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

const API_BASE_URL = "http://localhost:5001";

export default function LiveFeed() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<Camera | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [trackingData, setTrackingData] = useState<TrackingData | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    async function fetchCameras() {
      try {
        const response = await fetch(`${API_BASE_URL}/cameras`);
        const data: Camera[] = await response.json();
        setCameras(data);
        if (data.length > 0) {
          setSelectedCamera(data[0]);
        }
      } catch (error) {
        console.error("Failed to fetch cameras:", error);
      }
    }
    fetchCameras();
  }, []);

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

    ctx.strokeStyle = "yellow";
    ctx.lineWidth = 2;
    data.zones.forEach((zone) => {
      ctx.beginPath();
      zone.polygon.forEach((point, index) => {
        const [x, y] = point;
        if (index === 0) {
          ctx.moveTo(x * scaleX, y * scaleY);
        } else {
          ctx.lineTo(x * scaleX, y * scaleY);
        }
      });
      ctx.closePath();
      ctx.stroke();

      ctx.fillStyle = "yellow";
      ctx.font = "14px Arial";
      ctx.fillText(
        zone.name,
        zone.polygon[0][0] * scaleX,
        zone.polygon[0][1] * scaleY - 5
      );
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
      ctx.fillText(label, x1 * scaleX, y1 * scaleY - 5);
    });
  };

  useEffect(() => {
    if (!selectedCamera?.is_running) {
      setTrackingData(null);
      return;
    }

    const intervalId = setInterval(async () => {
      try {
        const response = await fetch(
          `${API_BASE_URL}/detector/tracking_data/${selectedCamera.cam_id}`
        );
        const data: TrackingData = await response.json();
        setTrackingData(data);
      } catch (error) {
        console.error("Failed to fetch tracking data:", error);
        setTrackingData(null);
      }
    }, 500); // Ambil data setiap 500ms

    return () => clearInterval(intervalId);
  }, [selectedCamera]);

  useEffect(() => {
    if (trackingData) {
      drawOverlays(trackingData);
    }
  }, [trackingData]);

  const startCameraDetector = async (cam_id: string) => {
    try {
      await fetch(`${API_BASE_URL}/detector/start_by_id`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cam_id: cam_id }),
      });
      // Refresh camera list to update status
      const response = await fetch(`${API_BASE_URL}/cameras`);
      const data = await response.json();
      setCameras(data);
      // Set selected camera to the newly started one
      const newSelected = data.find((c: Camera) => c.cam_id === cam_id);
      if (newSelected) setSelectedCamera(newSelected);
    } catch (error) {
      console.error("Failed to start camera detector:", error);
    }
  };

  const handleSelectCamera = (camera: Camera) => {
    if (!camera.is_running) {
      startCameraDetector(camera.cam_id);
    } else {
      setSelectedCamera(camera);
    }
  };

  return (
    <div className="min-h-screen bg-primary">
      <Navigation />
      <main className="pt-20 p-6 mt-10">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">
              Live Camera Feed
            </h1>
            <p className="text-gray-400">
              Monitor all CCTV cameras in real-time
            </p>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            <div className="lg:col-span-1">
              <div className="bg-tile1 rounded-lg border border-gray-700 p-4">
                <h2 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <FiCamera className="mr-2" /> Cameras ({cameras.length})
                </h2>
                <div className="space-y-2">
                  {cameras.map((camera) => (
                    <button
                      key={camera.cam_id}
                      onClick={() => handleSelectCamera(camera)}
                      className={`w-full text-left p-3 rounded-lg transition-colors ${
                        selectedCamera?.cam_id === camera.cam_id
                          ? "bg-blue-600 text-white"
                          : "bg-tile2 text-gray-300 hover:bg-gray-700"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium text-sm">{camera.name}</p>
                          <p className="text-xs opacity-75 truncate">
                            {camera.address}
                          </p>
                        </div>
                        <div className="flex items-center">
                          {camera.is_running ? (
                            <FiWifi className="w-4 h-4 text-green-400" />
                          ) : (
                            <FiPlayCircle className="w-4 h-4 text-gray-400" />
                          )}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <div className="lg:col-span-3">
              <div className="bg-tile1 rounded-lg border border-gray-700 p-4">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h2 className="text-lg font-semibold text-white">
                      {selectedCamera?.name ?? "No Camera Selected"}
                    </h2>
                    <p className="text-gray-400 text-sm">
                      {selectedCamera?.address}
                    </p>
                  </div>
                  <button
                    onClick={() => setIsFullscreen(!isFullscreen)}
                    className="p-2 bg-tile2 hover:bg-gray-700 rounded-lg transition-colors"
                  >
                    {isFullscreen ? (
                      <FiMinimize2 className="w-5 h-5 text-white" />
                    ) : (
                      <FiMaximize2 className="w-5 h-5 text-white" />
                    )}
                  </button>
                </div>

                {/* --- PERUBAHAN 7: Ganti HLSPlayer dengan <img> dan <canvas> --- */}
                <div
                  className={`relative bg-black rounded-lg overflow-hidden ${
                    isFullscreen
                      ? "fixed inset-0 z-50 rounded-none"
                      : "aspect-video"
                  }`}
                >
                  {selectedCamera?.is_running &&
                  selectedCamera?.stream_endpoint ? (
                    <>
                      <img
                        ref={videoRef}
                        src={`${API_BASE_URL}${
                          selectedCamera.stream_endpoint
                        }?t=${new Date().getTime()}`} // Tambah timestamp untuk bust cache
                        alt="Live video feed"
                        className="w-full h-full object-contain"
                      />
                      <canvas
                        ref={canvasRef}
                        className="absolute top-0 left-0 w-full h-full"
                      />
                    </>
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center">
                        <FiWifiOff className="w-16 h-16 text-gray-500 mx-auto mb-4" />
                        <p className="text-gray-400">
                          {selectedCamera
                            ? "Camera offline. Click to start."
                            : "Select a camera to view feed."}
                        </p>
                      </div>
                    </div>
                  )}

                  {isFullscreen && (
                    <button
                      onClick={() => setIsFullscreen(false)}
                      className="absolute top-4 right-4 p-2 bg-black bg-opacity-50 hover:bg-opacity-75 rounded-lg transition-colors"
                    >
                      <FiMinimize2 className="w-6 h-6 text-white" />
                    </button>
                  )}
                </div>

                <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-tile2 p-3 rounded-lg">
                    <p className="text-gray-400 text-xs">Status</p>
                    <p className="text-white font-medium">
                      {selectedCamera?.is_running ? "Online" : "Offline"}
                    </p>
                  </div>
                  {/* Info lain bisa ditambahkan dari data tracking jika perlu */}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
