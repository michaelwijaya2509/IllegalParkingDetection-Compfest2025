"use client";
import { useState } from "react";
import Navigation from "@/components/Navigation";
import HlsPlayer from "@/components/HLSPlayer";
import { cctvLocations } from "@/utilities/DummyData";
import {
  FiMaximize2,
  FiMinimize2,
  FiCamera,
  FiWifi,
  FiWifiOff,
} from "react-icons/fi";

export default function LiveFeed() {
  const [selectedCamera, setSelectedCamera] = useState<number>(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

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
                  <FiCamera className="mr-2" />
                  Cameras ({cctvLocations.length})
                </h2>
                <div className="space-y-2">
                  {cctvLocations.map((camera, index) => (
                    <button
                      key={index}
                      onClick={() => setSelectedCamera(index)}
                      className={`w-full text-left p-3 rounded-lg transition-colors ${
                        selectedCamera === index
                          ? "bg-blue-600 text-white"
                          : "bg-tile2 text-gray-300 hover:bg-gray-700"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium text-sm">
                            Camera {index + 1}
                          </p>
                          <p className="text-xs opacity-75 truncate">
                            {camera.description}
                          </p>
                        </div>
                        <div className="flex items-center">
                          {camera.streamUrl ? (
                            <FiWifi className="w-4 h-4 text-green-400" />
                          ) : (
                            <FiWifiOff className="w-4 h-4 text-red-400" />
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
                      Camera {selectedCamera + 1}
                    </h2>
                    <p className="text-gray-400 text-sm">
                      {cctvLocations[selectedCamera]?.description}
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

                <div
                  className={`relative bg-black rounded-lg overflow-hidden ${
                    isFullscreen
                      ? "fixed inset-0 z-50 rounded-none"
                      : "aspect-video"
                  }`}
                >
                  {cctvLocations[selectedCamera].streamUrl ? (
                    <HlsPlayer
                      src={cctvLocations[selectedCamera].streamUrl ?? ""}
                    />
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center">
                        <FiWifiOff className="w-16 h-16 text-gray-500 mx-auto mb-4" />
                        <p className="text-gray-400">Camera offline</p>
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

                {/* masih dummy tar sesuain aja resolusi, fps, bitrate kalo gaperlu apus aja */}
                <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-tile2 p-3 rounded-lg">
                    <p className="text-gray-400 text-xs">Status</p>
                    <p className="text-white font-medium">
                      {cctvLocations[selectedCamera]?.streamUrl
                        ? "Online"
                        : "Offline"}
                    </p>
                  </div>
                  <div className="bg-tile2 p-3 rounded-lg">
                    <p className="text-gray-400 text-xs">Resolution</p>
                    <p className="text-white font-medium">1080p</p>
                  </div>
                  <div className="bg-tile2 p-3 rounded-lg">
                    <p className="text-gray-400 text-xs">FPS</p>
                    <p className="text-white font-medium">30</p>
                  </div>
                  <div className="bg-tile2 p-3 rounded-lg">
                    <p className="text-gray-400 text-xs">Bitrate</p>
                    <p className="text-white font-medium">2.5 Mbps</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
