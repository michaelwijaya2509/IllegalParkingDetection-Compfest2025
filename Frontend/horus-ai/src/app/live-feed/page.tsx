/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";
import { useState, useEffect, useRef } from "react";
import Navigation from "@/components/Navigation";
import {
  FiCamera,
  FiWifi,
  FiWifiOff,
  FiPlayCircle,
  FiTrash2,
  FiLoader,
} from "react-icons/fi";
import HlsPlayer from "@/components/HLSPlayer";
import Image from "next/image";
import { Loader } from "@/components/spinner";

interface Camera {
  cam_id: string;
  name: string;
  address: string;
  stream_url: string;
  is_running: boolean;
  stream_endpoint: string | null;
}

const API_BASE_URL =
  "https://horus-backend-395725017559.asia-southeast1.run.app/";

export default function LiveFeed() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<Camera | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [wholePageLoading, setWholePageLoading] = useState(false);
  const [resolvedHlsUrl, setResolvedHlsUrl] = useState<string | null>(null);
  const [isHlsLoading, setIsHlsLoading] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    async function fetchCameras() {
      try {
        console.log("Fetching cameras ...");
        setWholePageLoading(true);
        const response = await fetch(`${API_BASE_URL}/cameras`);
        const data: Camera[] = await response.json();
        setCameras(data);
        if (data.length > 0 && !selectedCamera) {
          setSelectedCamera(data[0]);
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

  const startCameraDetector = async (cam_id: string) => {
    try {
      setIsHlsLoading(true);
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
    } finally {
      console.log("finished");
      setIsHlsLoading(false);
    }
  };

  const handleSelectCamera = (camera: Camera) => {
    if (!camera.is_running) {
      console.log("Starting camera detector");
      startCameraDetector(camera.cam_id);
    } else {
      setSelectedCamera(camera);
    }
    setIsHlsLoading(true);
  };

  const handleDeleteCamera = async (cam_idToDelete: string) => {
    if (!window.confirm("Are you sure you want to delete this camera?")) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/cameras/delete`, {
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

  const isHlsStream =
    selectedCamera?.is_running &&
    selectedCamera.stream_url &&
    selectedCamera.stream_url.startsWith("http");

  const isLocalStream =
    selectedCamera?.is_running && selectedCamera.stream_endpoint;

  return (
    <div className="min-h-screen bg-primary">
      {wholePageLoading && <Loader />}
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
                        <div className="flex items-center gap-3">
                          {camera.is_running ? (
                            <FiWifi className="w-4 h-4 text-green-400 flex-shrink-0" />
                          ) : (
                            <FiPlayCircle className="w-4 h-4 text-gray-400 flex-shrink-0" />
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
                          className="p-2 rounded-full hover:bg-red-500/20 text-gray-400 hover:text-red-500 transition-colors"
                          aria-label="Delete camera"
                        >
                          <FiTrash2 className="w-4 h-4" />
                        </button>
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
                      {isLocalStream && (
                        <img
                          src={`${API_BASE_URL}${selectedCamera.stream_endpoint}`}
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
