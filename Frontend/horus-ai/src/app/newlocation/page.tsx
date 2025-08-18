"use client";
import { useState, useRef, MouseEvent, ChangeEvent, FormEvent } from "react";
import Navigation from "@/components/Navigation";
import HlsPlayer from "@/components/HLSPlayer";
import {
  FiCamera,
  FiMap,
  FiLink,
  FiPlusCircle,
  FiXCircle,
  FiRotateCcw,
  FiUpload,
  FiLoader,
  FiVideo,
} from "react-icons/fi";

interface Point {
  x: number;
  y: number;
}

const InteractiveZoneDrawer = ({
  imageSrc,
  points,
  setPoints,
}: {
  imageSrc: string | null;
  points: Point[];
  setPoints: React.Dispatch<React.SetStateAction<Point[]>>;
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  const handleCanvasClick = (e: MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current || !imageSrc) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setPoints([...points, { x: Math.round(x), y: Math.round(y) }]);
  };

  const handleUndo = () => setPoints(points.slice(0, -1));
  const handleClear = () => setPoints([]);

  return (
    <div className="bg-primary border border-gray-700 rounded-lg p-4 flex flex-col h-full">
      <h3 className="text-white font-semibold mb-2">Draw Illegal Zone</h3>
      <p className="text-xs text-gray-500 mb-4">
        Click on the uploaded screenshot to define the polygon vertices.
      </p>
      <div
        ref={containerRef}
        onClick={handleCanvasClick}
        className="relative w-full aspect-video bg-gray-900 rounded-md overflow-hidden cursor-crosshair"
      >
        {imageSrc ? (
          <>
            <img
              src={imageSrc}
              alt="Frame screenshot preview"
              className="w-full h-full object-cover"
            />
            <svg className="absolute top-0 left-0 w-full h-full">
              {points.map((p, i) => (
                <circle
                  key={`point-${i}`}
                  cx={p.x}
                  cy={p.y}
                  r="4"
                  fill="#ef4444"
                  stroke="#ffffff"
                  strokeWidth="1"
                />
              ))}
              {points.length > 1 && (
                <polyline
                  points={points.map((p) => `${p.x},${p.y}`).join(" ")}
                  fill="rgba(239, 68, 68, 0.3)"
                  stroke="#ef4444"
                  strokeWidth="2"
                />
              )}
            </svg>
          </>
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <p className="text-gray-500">Upload a screenshot to begin</p>
          </div>
        )}
      </div>
      <div className="flex items-center justify-between mt-4">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleUndo}
            className="p-2 bg-gray-700 hover:bg-gray-600 rounded-md text-white transition"
          >
            <FiRotateCcw />
          </button>
          <button
            type="button"
            onClick={handleClear}
            className="p-2 bg-red-700 hover:bg-red-600 rounded-md text-white transition"
          >
            <FiXCircle />
          </button>
        </div>
        <p className="text-xs text-gray-400">{points.length} points selected</p>
      </div>
    </div>
  );
};

export default function AddNewLocation() {
  const [userInputUrl, setUserInputUrl] = useState("");
  const [resolvedStreamUrl, setResolvedStreamUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [previewFrame, setPreviewFrame] = useState<string | null>(null);
  const [zonePoints, setZonePoints] = useState<Point[]>([]);
  const playerContainerRef = useRef<HTMLDivElement>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState<{
    success: boolean;
    message: string;
  } | null>(null);

  const formRef = useRef<HTMLFormElement>(null);

  const handleResolveUrl = async () => {
    if (!userInputUrl) return;
    setIsLoading(true);
    setErrorMessage("");
    setResolvedStreamUrl("");
    setPreviewFrame(null);

    try {
      const backendUrl = "http://localhost:5001";
      const response = await fetch(`${backendUrl}/detector/resolve_url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: userInputUrl }),
      });
      const data = await response.json();

      if (data.ok) {
        setResolvedStreamUrl(data.stream_url);
      } else {
        setErrorMessage(data.error || "Failed to resolve URL.");
      }
    } catch (error) {
      console.error("Fetch error:", error);
      setErrorMessage("Could not connect to backend.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCaptureFrame = () => {
    const videoElement = playerContainerRef.current?.querySelector("video");
    if (!videoElement) {
      alert("Video preview not found. Please load the video first.");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
      setPreviewFrame(canvas.toDataURL("image/jpeg"));
    }
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setSubmitStatus(null);

    if (zonePoints.length < 3) {
      setSubmitStatus({
        success: false,
        message: "Please define a zone with at least 3 points.",
      });
      return;
    }

    setIsSubmitting(true);

    const formData = new FormData(formRef.current!);
    const data = Object.fromEntries(formData.entries());

    const finalData = {
      ...data,
      streamUrl: userInputUrl,
      zonePolygon: zonePoints,
    };

    try {
      const backendUrl = "http://localhost:5001";
      const response = await fetch(`${backendUrl}/cameras/add`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(finalData),
      });

      const result = await response.json();

      if (response.ok && result.ok) {
        setSubmitStatus({
          success: true,
          message:
            "Camera added successfully! You can now view it in the Live Feed.",
        });

        formRef.current?.reset();

        setUserInputUrl("");
        setResolvedStreamUrl("");
        setPreviewFrame(null);
        setZonePoints([]);
      } else {
        setSubmitStatus({
          success: false,
          message: result.error || "An unknown error occurred.",
        });
      }
    } catch (error) {
      console.error("Submission error:", error);
      setSubmitStatus({
        success: false,
        message: "Could not connect to the backend to save the camera.",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-primary">
      <Navigation />
      <main className="pt-20 p-12 mt-10">
        <div className="max-w-10xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">
              Add New Surveillance Source
            </h1>
            <p className="text-gray-400">
              Register a new camera feed to be monitored by the system.
            </p>
          </div>

          <form ref={formRef} onSubmit={handleSubmit}>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-tile1 border border-gray-700 rounded-lg p-8 space-y-6">
                <div>
                  <label
                    htmlFor="cameraName"
                    className="block text-sm font-medium text-gray-300 mb-2"
                  >
                    Camera Name
                  </label>
                  <input
                    type="text"
                    name="cameraName"
                    id="cameraName"
                    className="w-full bg-primary border border-gray-600 rounded-md text-white px-4 py-2"
                    placeholder="e.g., CCTV Dago Atas"
                    required
                  />
                </div>
                <div>
                  <label
                    htmlFor="address"
                    className="block text-sm font-medium text-gray-300 mb-2"
                  >
                    Location Address
                  </label>
                  <input
                    type="text"
                    name="address"
                    id="address"
                    className="w-full bg-primary border border-gray-600 rounded-md text-white px-4 py-2"
                    placeholder="e.g., Jl. Ir. H. Juanda No.1, Bandung"
                  />
                </div>
                <div>
                  <label
                    htmlFor="userInputUrl"
                    className="block text-sm font-medium text-gray-300 mb-2"
                  >
                    Stream URL (YouTube, etc.)
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      id="userInputUrl"
                      name="userInputUrl"
                      value={userInputUrl}
                      onChange={(e) => setUserInputUrl(e.target.value)}
                      className="flex-grow w-full bg-primary border border-gray-600 rounded-md text-white px-4 py-2"
                      placeholder="https://www.youtube.com/watch?v=..."
                      required
                    />
                    <button
                      type="button"
                      onClick={handleResolveUrl}
                      disabled={isLoading}
                      className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-md flex items-center disabled:opacity-50"
                    >
                      {isLoading ? (
                        <FiLoader className="animate-spin" />
                      ) : (
                        "Load"
                      )}
                    </button>
                  </div>
                  {errorMessage && (
                    <p className="text-red-500 text-xs mt-2">{errorMessage}</p>
                  )}
                </div>

                <div className="pt-4">
                  <button
                    type="submit"
                    disabled={isSubmitting}
                    className="w-full flex items-center justify-center bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300 disabled:opacity-60"
                  >
                    {isSubmitting ? (
                      <FiLoader className="animate-spin mr-2" />
                    ) : (
                      <FiPlusCircle className="mr-2" />
                    )}
                    {isSubmitting ? "Saving Camera..." : "Add Camera to System"}
                  </button>
                  {submitStatus && (
                    <p
                      className={`text-sm mt-4 text-center ${
                        submitStatus.success ? "text-green-400" : "text-red-400"
                      }`}
                    >
                      {submitStatus.message}
                    </p>
                  )}
                </div>
              </div>

              <div className="bg-tile1 border border-gray-700 rounded-lg p-8 flex flex-col">
                <div className="flex-grow">
                  <InteractiveZoneDrawer
                    imageSrc={previewFrame}
                    points={zonePoints}
                    setPoints={setZonePoints}
                  />
                </div>
                {resolvedStreamUrl && (
                  <div
                    ref={playerContainerRef}
                    className="mt-4 border-t border-gray-700 pt-4"
                  >
                    <p className="text-sm text-gray-400 mb-2">Live Preview:</p>
                    <HlsPlayer src={resolvedStreamUrl} />
                    <button
                      type="button"
                      onClick={handleCaptureFrame}
                      className="w-full mt-4 flex items-center justify-center bg-gray-600 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded-lg transition"
                    >
                      <FiVideo className="mr-2" /> Capture Frame for Drawing
                    </button>
                  </div>
                )}
              </div>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}
