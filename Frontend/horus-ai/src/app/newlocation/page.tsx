/* eslint-disable @next/next/no-img-element */
"use client";
import { useState, useRef, FormEvent } from "react";
import Navigation from "@/components/Navigation";
import HlsPlayer from "@/components/HLSPlayer";
import { FiPlusCircle, FiLoader } from "react-icons/fi";

export default function AddNewLocation() {
  const [userInputUrl, setUserInputUrl] = useState("");
  const [previewStreamUrl, setPreviewStreamUrl] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState<{
    success: boolean;
    message: string;
  } | null>(null);
  const formRef = useRef<HTMLFormElement>(null);

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

    try {
      const backendUrl =
        "https://horus-backend-395725017559.asia-southeast1.run.app/";
      const response = await fetch(`${backendUrl}/cameras/add`, {
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
      } else {
        setSubmitStatus({
          success: false,
          message: result.error || "An error occurred.",
        });
      }
    } catch (error) {
      setSubmitStatus({
        success: false,
        message: "Could not connect to the backend.",
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
                    className="w-full bg-primary border border-gray-600 rounded-md text-white px-4 py-2"
                  />
                </div>
                <div>
                  <label
                    htmlFor="userInputUrl"
                    className="block text-sm font-medium text-gray-300 mb-2"
                  >
                    HLS Stream URL (.m3u8)
                  </label>
                  <div className="flex gap-2">
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
                      className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-md flex items-center"
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
                    className="w-full flex items-center justify-center bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition disabled:opacity-60"
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
