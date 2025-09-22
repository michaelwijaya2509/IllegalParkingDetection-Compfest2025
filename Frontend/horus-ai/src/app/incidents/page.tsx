/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";
import { useState, useEffect, useContext } from "react";
import Navigation from "@/components/Navigation";
import {
  FiMapPin,
  FiClock,
  FiEye,
  FiLoader,
  FiXCircle,
  FiCheckCircle,
} from "react-icons/fi";
import Image from "next/image";
import { LuListFilter } from "react-icons/lu";
import { DataContext } from "@/context/Context";
import FilterModal from "@/components/FilterModal";

const PROCESS_URL = process.env.NEXT_PUBLIC_PROCESS_URL;

interface ApprovedIncident {
  event: {
    event_id: string;
    started_at: string;
    snapshot_url: string;
  };
  cam_id: string;
  camera: {
    lat: number;
    lon: number;
  };
  scored: {
    priority_label: "low" | "medium" | "high" | "critical";
  };
  llm_data: {
    narrative: string;
  };
  coordinates: {
    0: number;
    1: number;
  };
  address: string;
}

export default function Incidents() {
  const [incidents, setIncidents] = useState<ApprovedIncident[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedIncident, setSelectedIncident] =
    useState<ApprovedIncident | null>(null);
  const [isFilterModalOpen, setIsFilterModalOpen] = useState<boolean>(false);

  const context = useContext(DataContext);
  const { selectedCamId, setSelectedCamId } = context || { selectedCamId: null, setSelectedCamId: () => {} };

  const filterByCamId = (cameraId: string | null) => {
    setSelectedCamId(cameraId);
    setIsFilterModalOpen(false);
    console.log("Selected Camera", cameraId);
  }
  
  
  useEffect(() => {
    const fetchApprovedIncidents = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`${PROCESS_URL}/incidents/approved`);
        if (!response.ok) {
          throw new Error("Failed to fetch approved incidents.");
        }
        const data: ApprovedIncident[] = await response.json();
        setIncidents(
          data.sort(
            (a, b) =>
              new Date(b.event.started_at).getTime() -
              new Date(a.event.started_at).getTime()
          )
        );
      } catch (err: any) {
        setError(err.message || "An unknown error occurred.");
      } finally {
        setIsLoading(false);
      }
    };

    fetchApprovedIncidents();
  }, []);

  return (
    <div className="min-h-screen bg-primary">
      <Navigation />
      { 
        isFilterModalOpen &&
        <FilterModal
          isLoading={false}
          setOnBackgroundClick={() => setIsFilterModalOpen(false)}
            cameraIds={
              incidents.length > 0
                ? [...new Set(incidents.map((incident) => incident.cam_id))]
                : []
            }
          buttonLabel="Apply Filters"
          setState={(camId: string | null) => filterByCamId(camId)}
          title="Camera Location Filters"
        />
      }
      <main className="pt-28 p-12">
        <div className="max-w-10xl mx-auto">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">
                Approved Incident Reports
              </h1>
              <p className="text-gray-400">
                List of all validated illegal parking violations.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <div className="bg-tile1 rounded-lg border border-gray-700">
                <div className="flex flex-row p-4 border-b border-gray-700">
                  <h2 className="text-lg font-semibold text-white">
                    Approved Incidents ({incidents.length})
                  </h2>
                  <button className="flex gap-2 ml-auto cursor-pointer items-center justify-center" onClick={() => setIsFilterModalOpen(true)}>
                    <LuListFilter className="text-gray-400" />
                    <span className="text-gray-400 text-sm font-semibold"> {selectedCamId || "All Cameras"}</span>
                  </button>
                </div>

                {isLoading ? (
                  <div className="p-8 text-center text-gray-400 flex items-center justify-center">
                    <FiLoader className="animate-spin w-6 h-6 mr-3" />
                    Loading approved incidents...
                  </div>
                ) : error ? (
                  <div className="p-8 text-center text-red-400 flex items-center justify-center">
                    <FiXCircle className="w-6 h-6 mr-3" />
                    {error}
                  </div>
                ) : (
                  <div className="divide-y divide-gray-700 max-h-[70vh] overflow-y-auto">
                    {incidents.length > 0 ? (
                      incidents.map((incident) => (
                        selectedCamId === null || selectedCamId === incident.cam_id) && (
                        <div
                          key={incident.event.event_id}
                          className={`p-4 cursor-pointer transition-colors ${
                            selectedIncident?.event.event_id ===
                            incident.event.event_id
                              ? "bg-blue-600 bg-opacity-20"
                              : "hover:bg-tile2"
                          }`}
                          onClick={() => setSelectedIncident(incident)}
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-2">
                                <FiCheckCircle className="w-4 h-4 text-green-400" />
                                <h3 className="font-semibold text-white">
                                  {incident.address}
                                </h3>
                                <span className="px-2 py-1 text-white text-xs rounded-full bg-green-600">
                                  Approved
                                </span>
                              </div>
                              <p className="text-gray-300 text-sm mb-3">
                                {incident.llm_data.narrative}
                              </p>
                              <div className="flex items-center space-x-4 text-xs text-gray-400">
                                <div className="flex items-center space-x-1">
                                  <FiClock className="w-3 h-3" />
                                  <span>
                                    {new Date(
                                      incident.event.started_at
                                    ).toLocaleString()}
                                  </span>
                                </div>
                                <div className="flex items-center space-x-1">
                                  <FiMapPin className="w-3 h-3" />
                                  <span>
                                    {incident.coordinates[0].toFixed(4)},{" "}
                                    {incident.coordinates[1].toFixed(4)}
                                  </span>
                                </div>
                              </div>
                            </div>
                            <button className="p-2 bg-tile2 hover:bg-gray-700 rounded-lg transition-colors cursor-pointer ml-4">
                              <FiEye className="w-4 h-4 text-white" />
                            </button>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="p-8 text-center text-gray-500">
                        No approved incidents found.
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            <div className="lg:col-span-1">
              {selectedIncident ? (
                <div className="bg-tile1 rounded-lg border border-gray-700 p-4 sticky top-28">
                  <h2 className="text-lg font-semibold text-white mb-4">
                    Incident Details
                  </h2>
                  <div className="space-y-4">
                    <div>
                      <label className="text-gray-400 text-sm">Location</label>
                      <p className="text-white font-medium">
                        {selectedIncident.address}
                      </p>
                    </div>
                    <div>
                      <label className="text-gray-400 text-sm">
                        Description
                      </label>
                      <p className="text-white">
                        {selectedIncident.llm_data.narrative}
                      </p>
                    </div>
                    <div>
                      <label className="text-gray-400 text-sm">Timestamp</label>
                      <p className="text-white">
                        {new Date(
                          selectedIncident.event.started_at
                        ).toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <label className="text-gray-400 text-sm">Evidence</label>
                      <div className="mt-2 aspect-video bg-black rounded-lg overflow-hidden border border-gray-600">
                        <Image
                          src={selectedIncident.event.snapshot_url}
                          width={600}
                          height={0}
                          className="w-full h-full object-contain"
                          alt="Incident Snapshot"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-tile1 rounded-lg border border-gray-700 p-8 text-center sticky top-28">
                  <FiCheckCircle className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                  <p className="text-gray-400">
                    Select an incident to view details.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
