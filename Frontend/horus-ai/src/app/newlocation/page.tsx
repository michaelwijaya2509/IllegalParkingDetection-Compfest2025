/* eslint-disable @typescript-eslint/no-unused-vars */
"use client";
import { useState, useRef, FormEvent, useEffect } from "react";
import Navigation from "@/components/Navigation";
import HlsPlayer from "@/components/HLSPlayer";
import { FiPlusCircle, FiLoader } from "react-icons/fi";
import { OSMReturn } from "@/utilities/Types";
import dynamic from "next/dynamic";

const Map = dynamic(() => import("@/components/Map"), {
  ssr: false,
});

export default function AddNewLocation() {
  const [userInputUrl, setUserInputUrl] = useState("");
  const [previewStreamUrl, setPreviewStreamUrl] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [longitude, setLongitude] = useState("");
  const [locationNotFoundError, setLocationNotFoundError] = useState("");
  const [latitude, setLatitude] = useState("");
  const [locationAddress, setLocationAddress] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState<{
    success: boolean;
    message: string;
  } | null>(null);
  const formRef = useRef<HTMLFormElement>(null);

  useEffect(() => {
    if (locationAddress.trim() === "") {
      setLongitude("");
      setLatitude("");
      return;
    }

    const handler = setTimeout(async () => {
      try {
        const response = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(locationAddress)}`
        );
        const data: OSMReturn[] = await response.json();

        if (data.length > 0) {
          setLongitude(data[0].lon);
          setLatitude(data[0].lat);
          setLocationAddress(data[0].display_name);
          setLocationNotFoundError("");
        } else {
          setLongitude("");
          setLatitude("");
          setLocationNotFoundError(
            "Cannot find location. Ensure the address is correct or set the coordinates manually."
          );
        }
      } catch (error) {
        console.error("Error fetching coordinates:", error);
      }
    }, 1000);
    return () => clearTimeout(handler);
  }, [locationAddress]);

  const handleLoadUrl = () => {
    setErrorMessage("");
    if (!userInputUrl || !userInputUrl.trim().endsWith(".m3u8")) {
      setErrorMessage("Please enter a valid HLS stream URL ending in .m3u8");
      setPreviewStreamUrl("");
      return;
    }
    setPreviewStreamUrl(userInputUrl);
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setSubmitStatus(null);
    setIsSubmitting(true);
    const formData = new FormData(formRef.current!);
    const data = Object.fromEntries(formData.entries());

    const finalData = { ...data, streamUrl: userInputUrl };
    console.log(finalData);

    try {
      const PROCESS_URL =
        process.env.NEXT_PUBLIC_PROCESS_URL;
      const response = await fetch(`${PROCESS_URL}/cameras/add`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(finalData),
      });
      const result = await response.json();
      if (response.ok && result.ok) {
        setSubmitStatus({
          success: true,
          message: "Camera added successfully!",
        });
        formRef.current?.reset();
        setUserInputUrl("");
        setPreviewStreamUrl("");
        setLongitude("");
        setLatitude("");
        setLocationAddress("");
        setLocationNotFoundError("");
      } else {
        setSubmitStatus({
          success: false,
          message: result.error || "An error occurred.",
        });
      }
    } catch (error) {
      setSubmitStatus({
        success: false,
        message: "An error occurred. Server possibly unreachable.",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-primary">
      <Navigation />
      <main className="pt-28 p-12">
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
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Column 1: The Form */}
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
                    value={locationAddress}
                    onChange={(e) => setLocationAddress(e.target.value)}
                    required
                    className="w-full bg-primary border border-gray-600 rounded-md text-white px-4 py-2"
                  />
                  {locationNotFoundError && (
                    <p className="text-red-500 text-xs mt-2">{locationNotFoundError}</p>
                  )}
                </div>
                <div className="flex flex-row w-full gap-4">
                  <div className="w-full">
                    <label
                      htmlFor="longitude"
                      className="block text-sm font-medium text-gray-300 mb-2"
                    >
                      Longitude
                    </label>
                    <input
                      type="text"
                      name="longitude"
                      id="longitude"
                      value={longitude}
                      onChange={(e) => setLongitude(e.target.value)}
                      className="w-full bg-primary border border-gray-600 rounded-md text-white px-4 py-2"
                    />
                  </div>
                  <div className="w-full">
                    <label
                      htmlFor="latitude"
                      className="block text-sm font-medium text-gray-300 mb-2"
                    >
                      Latitude
                    </label>
                    <input
                      type="text"
                      name="latitude"
                      id="latitude"
                      value={latitude}
                      onChange={(e) => setLatitude(e.target.value)}
                      className="w-full bg-primary border border-gray-600 rounded-md text-white px-4 py-2"
                    />
                  </div>
                </div>
                <div>
                  <label
                    htmlFor="userInputUrl"
                    className="block text-sm font-medium text-gray-300 mb-2"
                  >
                    HLS Stream URL (.m3u8)
                  </label>
                  <div className="flex gap-4">
                    <input
                      type="text"
                      id="userInputUrl"
                      name="streamUrl"
                      value={userInputUrl}
                      onChange={(e) => setUserInputUrl(e.target.value)}
                      className="flex-grow w-full bg-primary border border-gray-600 rounded-md text-white px-4 py-2"
                      required
                    />
                    <button
                      type="button"
                      onClick={handleLoadUrl}
                      className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-md flex items-center cursor-pointer transition"
                    >
                      Load
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
                    className="w-full flex items-center justify-center bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition disabled:opacity-60 cursor-pointer disabled:cursor-not-allowed"
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
                    
              <div className="bg-tile1 border border-gray-700 rounded-lg p-8 max-h-[600px] flex flex-col">
                <Map
                  illegalParkingLocations={[]}
                  defaultViewingCoordinates={[latitude ? parseInt(latitude) : 0, longitude ? parseInt(longitude) : 0]}
                  zoomLevel={latitude ? 18 : 14}
                  onMarkerClick={() => {}}
                  onIncidentClick={() => {}}
                />
              </div>

              <div className="bg-tile1 border border-gray-700 rounded-lg p-8 flex flex-col">
                <h3 className="text-white font-semibold mb-2">Live Preview</h3>
                <div className="relative w-full aspect-video bg-gray-900 rounded-md overflow-hidden">
                  {previewStreamUrl ? (
                    <HlsPlayer src={previewStreamUrl} />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <p className="text-gray-500">
                        Enter a URL and click Load to see a preview
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}
