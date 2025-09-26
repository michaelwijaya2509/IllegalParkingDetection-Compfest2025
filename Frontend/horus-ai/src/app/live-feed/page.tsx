/* eslint-disable @next/next/no-img-element */
/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";
import { useState, useEffect, useRef } from "react";
import Navigation from "@/components/Navigation";
import { FiCamera, FiWifiOff, FiTrash2 } from "react-icons/fi";
import HlsPlayer from "@/components/HLSPlayer";
import { Loader } from "@/components/spinner";
import { FaCircleDot } from "react-icons/fa6";

interface Camera {
  cam_id: string;
  name: string;
  address: string;
  stream_url: string;
  is_running: boolean;
  stream_endpoint: string | null;
}

interface Zone {
  name: string;
  polygon: [number, number][];
}

interface Track {
  track_id: number;
  class_name: string;
  bbox: [number, number, number, number];
  stationary_s: number;
  is_violation: boolean;
  is_close_to_violation: boolean;
}

interface TrackingData {
  tracks: Track[];
  zones: Zone[];
  timestamp: number;
  video_width: number;
  video_height: number;
}

const PROCESS_URL = process.env.NEXT_PUBLIC_PROCESS_URL;
const INFERENCE_URL = process.env.NEXT_PUBLIC_INFERENCE_URL;

export default function LiveFeed() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<Camera | null>(null);
  const [wholePageLoading, setWholePageLoading] = useState(false);
  const [trackingData, setTrackingData] = useState<TrackingData | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLImageElement>(null);

  const drawOverlays = (data: TrackingData | null) => {
    const canvas = canvasRef.current;
    const videoElement = videoRef.current;
    let scaleX = null;
    let scaleY = null;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (data) {
      if (!videoElement && data.video_width === 0) return;
      
      
      if (!ctx) return;

      const videoRect = videoElement?.getBoundingClientRect();
      const displayWidth =
        videoRect?.width || canvas.parentElement?.clientWidth || 0;
      const displayHeight =
        videoRect?.height || canvas.parentElement?.clientHeight || 0;

      canvas.width = displayWidth;
      canvas.height = displayHeight;

      scaleX = displayWidth / (data.video_width || 1);
      scaleY = displayHeight / (data.video_height || 1);
    }


    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!data || !scaleX || !scaleY || !data.zones) return;

    // Draw zones
    ctx.strokeStyle = "yellow";
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
      ctx.fillStyle = "yellow";
      ctx.font = "14px Arial";
      ctx.fillText(
        zone.name,
        zone.polygon[0][0] * scaleX,
        zone.polygon[0][1] * scaleY - 5
      );
    });

    // Draw tracks (bounding boxes)
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
    async function fetchCameras() {
      try {
        console.log("Fetching cameras ...");
        setWholePageLoading(true);
        const response = await fetch(`${PROCESS_URL}/cameras`);
        const data: Camera[] = await response.json();
        setCameras(data);
        if (data.length > 0 && !selectedCamera) {
          setSelectedCamera(data[0]);
          handleSelectCamera(data[0]);
        }
      } catch (error) {
        console.error("Failed to fetch cameras:", error);
      } finally {
        console.log("done fetching camera");
        setWholePageLoading(false);
      }
    }
    fetchCameras();
  }, []);

  useEffect(() => {
    if (!selectedCamera?.is_running) {
      setTrackingData(null);
      return;
    }
    const intervalId = setInterval(async () => {
      try {
        const response = await fetch(
          `${INFERENCE_URL}/detector/tracking_data/${selectedCamera.cam_id}`
        );
        const data: TrackingData = await response.json();
        setTrackingData(data);
      } catch (error) {
        console.error("Failed to fetch tracking data:", error);
        setTrackingData(null);
      }
    }, 500);
    return () => clearInterval(intervalId);
  }, [selectedCamera?.cam_id, selectedCamera?.is_running]);

  useEffect(() => {
    drawOverlays(trackingData);
    
  }, [trackingData]);

  const startCameraDetector = async (cam_id: string) => {
    try {
      const camToStart = cameras.find((c) => c.cam_id === cam_id);
      if (camToStart) setSelectedCamera(camToStart);

      await fetch(`${PROCESS_URL}/detector/start_by_id`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cam_id: cam_id }),
      });

      const response = await fetch(`${PROCESS_URL}/cameras`);
      const data = await response.json();
      setCameras(data);

      const newSelected = data.find((c: Camera) => c.cam_id === cam_id);
      if (newSelected) setSelectedCamera(newSelected);
    } catch (error) {
      console.error("Failed to start camera detector:", error);
    } finally {
      console.log("finished");
    }
  };

  const handleSelectCamera = (camera: Camera) => {
    if (!camera.is_running) {
      console.log("Starting camera detector");
      startCameraDetector(camera.cam_id);
    } else {
      setSelectedCamera(camera);
    }
  };

  const handleDeleteCamera = async (cam_idToDelete: string) => {
    if (!window.confirm("Are you sure you want to delete this camera?")) {
      return;
    }

    try {
      const response = await fetch(`${PROCESS_URL}/cameras/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cam_id: cam_idToDelete }),
      });

      const result = await response.json();

      if (response.ok && result.ok) {
        setCameras((currentCameras) =>
          currentCameras.filter((cam) => cam.cam_id !== cam_idToDelete)
        );

        if (selectedCamera?.cam_id === cam_idToDelete) {
          setSelectedCamera(null);
        }
        alert("Camera deleted successfully.");
      } else {
        throw new Error(result.error || "Failed to delete camera.");
      }
    } catch (error: any) {
      console.error("Failed to delete camera:", error);
      alert(`Error: ${error.message}`);
    }
  };

  const [isHlsStream, setIsHlsStream] = useState<boolean>(false);
  const [isLocalStream, setIsLocalStream] = useState<boolean>(false);

  useEffect(() => {
    if (
      selectedCamera?.is_running &&
      selectedCamera.stream_url &&
      selectedCamera?.stream_url.startsWith("http")
    ) {
      setIsHlsStream(true);
    } else setIsHlsStream(false);
  }, [selectedCamera]);

  useEffect(() => {
    if (selectedCamera?.is_running && selectedCamera.stream_endpoint) {
      setIsLocalStream(true);
      drawOverlays(null);
    }
      
    else setIsLocalStream(false);
  }, [selectedCamera]);

  useEffect(() => {
    if (!isLocalStream) return;
    console.log(`${PROCESS_URL}${selectedCamera?.stream_endpoint}`);
  }, [isLocalStream]);

  return (
    <div className="min-h-screen bg-primary">
      {wholePageLoading && <Loader />}
      <Navigation />
      <main className="pt-28 lg:px-40 p-12">
        <div className="max-w-10xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">
              Live Camera Feed
            </h1>
            <p className="text-gray-400">
              Monitor all CCTV cameras in real-time
            </p>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            <div className="lg:col-span-2">
              <div className="bg-tile1 rounded-lg border border-gray-700 p-4">
                <h2 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <FiCamera className="mr-2" /> Cameras ({cameras.length})
                </h2>
                <div className="space-y-2">
                  {cameras.map((camera) => (
                    <div
                      key={camera.cam_id}
                      onClick={() => handleSelectCamera(camera)}
                      className={`w-full text-left p-3 rounded-lg transition cursor-pointer ${
                        selectedCamera?.cam_id === camera.cam_id
                          ? "bg-blue-600 text-white"
                          : "bg-tile2 text-gray-300 hover:bg-gray-700"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          {camera.is_running ? (
                            <FaCircleDot className="w-4 h-4 text-green-400 flex-shrink-0" />
                          ) : (
                            <FaCircleDot className="w-4 h-4 text-yellow-400 flex-shrink-0" />
                          )}
                          <div>
                            <p className="font-medium text-sm">{camera.name}</p>
                            <p className="text-xs opacity-75">
                              {camera.address}
                            </p>
                          </div>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteCamera(camera.cam_id);
                          }}
                          className="p-2 rounded-full hover:bg-red-500/20 text-gray-400 hover:text-red-500 transition cursor-pointer"
                          aria-label="Delete camera"
                        >
                          <FiTrash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
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
                      {isLocalStream && !isHlsStream && (
                        <img
                          ref={videoRef} // Added ref to the image element
                          src={`${PROCESS_URL}${selectedCamera.stream_endpoint}`}
                          alt="Live video feed"
                          className="absolute top-0 left-0 w-full h-full object-contain"
                        />
                      )}

                      {isHlsStream && (
                        <HlsPlayer src={selectedCamera.stream_url} />
                      )}

                      <canvas
                        ref={canvasRef}
                        className="absolute top-0 left-0 w-full h-full pointer-events-none"
                      />

                      {!isLocalStream && !isHlsStream && (
                        <div className="w-full h-full flex items-center justify-center text-gray-400">
                          Error: Stream running but URL is invalid.
                        </div>
                      )}
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
