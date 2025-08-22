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
  FiLoader,
} from "react-icons/fi";
import HlsPlayer from "@/components/HLSPlayer";

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

const API_BASE_URL =
  "https://horus-backend-395725017559.asia-southeast2.run.app";

export default function LiveFeed() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<Camera | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const [resolvedHlsUrl, setResolvedHlsUrl] = useState<string | null>(null);
  const [isHlsLoading, setIsHlsLoading] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    async function fetchCameras() {
      try {
        const response = await fetch(`${API_BASE_URL}/cameras`);
        const data: Camera[] = await response.json();
        setCameras(data);
        if (data.length > 0 && !selectedCamera) {
          handleSelectCamera(data[0]);
        }
      } catch (error) {
        console.error("Failed to fetch cameras:", error);
      }
    }
    fetchCameras();
  }, []);

  useEffect(() => {
    const resolveUrlForPlayer = async () => {
      if (
        !selectedCamera ||
        selectedCamera.cam_id === "pasteur1" ||
        selectedCamera.cam_id === "viet1"
      ) {
        setResolvedHlsUrl(null);
        return;
      }

      setIsHlsLoading(true);
      setResolvedHlsUrl(null);
      try {
        const response = await fetch(`${API_BASE_URL}/detector/resolve_url`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: selectedCamera.stream_url }),
        });
        const data = await response.json();
        if (data.ok) {
          setResolvedHlsUrl(data.stream_url);
        } else {
          console.error("Failed to resolve HLS URL:", data.error);
        }
      } catch (error) {
        console.error("Error resolving HLS URL:", error);
      } finally {
        setIsHlsLoading(false);
      }
    };

    if (selectedCamera?.is_running) {
      resolveUrlForPlayer();
    }
  }, [selectedCamera?.cam_id, selectedCamera?.is_running]);

  const startCameraDetector = async (cam_id: string) => {
    try {
      const camToStart = cameras.find((c) => c.cam_id === cam_id);
      if (camToStart) setSelectedCamera(camToStart);

      await fetch(`${API_BASE_URL}/detector/start_by_id`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cam_id: cam_id }),
      });

      const response = await fetch(`${API_BASE_URL}/cameras`);
      const data = await response.json();
      setCameras(data);

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

  const isLocalCamera =
    selectedCamera?.cam_id === "pasteur1" || selectedCamera?.cam_id === "viet1";

  return (
    <div className="min-h-screen bg-primary">
      <Navigation />
      <main className="pt-20 p-12 mt-10">
        <div className="max-w-10xl mx-auto">
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
                </div>

                <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                  {selectedCamera?.is_running ? (
                    <>
                      {isLocalCamera ? (
                        <img
                          ref={videoRef}
                          src={`${API_BASE_URL}${selectedCamera.stream_endpoint}`}
                          alt="Live video feed"
                          className="absolute top-0 left-0 w-full h-full object-contain"
                        />
                      ) : isHlsLoading ? (
                        <div className="w-full h-full flex items-center justify-center">
                          <FiLoader className="w-10 h-10 text-gray-400 animate-spin" />
                        </div>
                      ) : resolvedHlsUrl ? (
                        <HlsPlayer src={resolvedHlsUrl} />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-gray-400">
                          Failed to load stream.
                        </div>
                      )}

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
                            ? "Camera offline. Click list to start."
                            : "Select a camera to view feed."}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
