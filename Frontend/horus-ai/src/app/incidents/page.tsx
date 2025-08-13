"use client";
import { useState } from "react";
import Navigation from "@/components/Navigation";
import { illegalParkingLocations } from "@/utilities/DummyData";
import {
  FiAlertTriangle,
  FiMapPin,
  FiClock,
  FiEye,
  FiFilter,
} from "react-icons/fi";

export default function Incidents() {
  const [filter, setFilter] = useState("all");
  const [selectedIncident, setSelectedIncident] = useState<number | null>(null);

  const filteredIncidents = illegalParkingLocations.filter((incident) => {
    if (filter === "all") return true;
    // Add more filter logic here
    return true;
  });

  return (
    <div className="min-h-screen bg-primary">
      <Navigation />

      <main className="pt-20 p-6 mt-10">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">
                Incident Reports
              </h1>
              <p className="text-gray-400">
                Track and manage illegal parking violations
              </p>
            </div>

            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <FiFilter className="text-gray-400" />
                <select
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  className="bg-tile1 border border-gray-600 rounded-lg px-3 py-2 text-white"
                >
                  <option value="all">All Incidents</option>
                  <option value="pending">Pending</option>
                  <option value="resolved">Resolved</option>
                  <option value="urgent">Urgent</option>
                </select>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <div className="bg-tile1 rounded-lg border border-gray-700">
                <div className="p-4 border-b border-gray-700">
                  <h2 className="text-lg font-semibold text-white">
                    Recent Incidents ({filteredIncidents.length})
                  </h2>
                </div>

                <div className="divide-y divide-gray-700">
                  {filteredIncidents.map((incident, index) => (
                    <div
                      key={index}
                      className={`p-4 cursor-pointer transition-colors ${
                        selectedIncident === index
                          ? "bg-blue-600 bg-opacity-20"
                          : "hover:bg-tile2"
                      }`}
                      onClick={() => setSelectedIncident(index)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-2">
                            <FiAlertTriangle className="w-4 h-4 text-red-400" />
                            <h3 className="font-semibold text-white">
                              {incident.locationName}
                            </h3>
                            <span className="px-2 py-1 bg-red-600 text-white text-xs rounded-full">
                              Active
                            </span>
                          </div>

                          <p className="text-gray-300 text-sm mb-3">
                            {incident.narration}
                          </p>

                          <div className="flex items-center space-x-4 text-xs text-gray-400">
                            <div className="flex items-center space-x-1">
                              <FiClock className="w-3 h-3" />
                              <span>
                                {new Date(incident.timestamp).toLocaleString()}
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

                        <button className="p-2 bg-tile2 hover:bg-gray-700 rounded-lg transition-colors">
                          <FiEye className="w-4 h-4 text-white" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="lg:col-span-1">
              {selectedIncident !== null ? (
                <div className="bg-tile1 rounded-lg border border-gray-700 p-4">
                  <h2 className="text-lg font-semibold text-white mb-4">
                    Incident Details
                  </h2>

                  <div className="space-y-4">
                    <div>
                      <label className="text-gray-400 text-sm">Location</label>
                      <p className="text-white font-medium">
                        {filteredIncidents[selectedIncident].locationName}
                      </p>
                    </div>

                    <div>
                      <label className="text-gray-400 text-sm">
                        Description
                      </label>
                      <p className="text-white">
                        {filteredIncidents[selectedIncident].narration}
                      </p>
                    </div>

                    <div>
                      <label className="text-gray-400 text-sm">Timestamp</label>
                      <p className="text-white">
                        {new Date(
                          filteredIncidents[selectedIncident].timestamp
                        ).toLocaleString()}
                      </p>
                    </div>

                    <div>
                      <label className="text-gray-400 text-sm">
                        Coordinates
                      </label>
                      <p className="text-white font-mono text-sm">
                        {filteredIncidents[
                          selectedIncident
                        ].coordinates[0].toFixed(6)}
                        ,{" "}
                        {filteredIncidents[
                          selectedIncident
                        ].coordinates[1].toFixed(6)}
                      </p>
                    </div>

                    <div>
                      <label className="text-gray-400 text-sm">
                        Video Evidence
                      </label>
                      <div className="mt-2 aspect-video bg-black rounded-lg overflow-hidden">
                        <iframe
                          src={filteredIncidents[selectedIncident].videoUrl}
                          className="w-full h-full"
                          allow="autoplay"
                        />
                      </div>
                    </div>

                    <div className="flex space-x-2 pt-4">
                      <button className="flex-1 bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg transition-colors">
                        Mark Resolved
                      </button>
                      <button className="flex-1 bg-red-600 hover:bg-red-700 text-white py-2 px-4 rounded-lg transition-colors">
                        Escalate
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-tile1 rounded-lg border border-gray-700 p-8 text-center">
                  <FiAlertTriangle className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                  <p className="text-gray-400">
                    Select an incident to view details
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
