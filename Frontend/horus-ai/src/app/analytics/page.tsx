"use client";
import Navigation from "@/components/Navigation";
import { illegalParkingLocations } from "@/utilities/DummyData";
import {
  FiTrendingUp,
  FiMapPin,
  FiClock,
  FiAlertTriangle,
} from "react-icons/fi";

export default function Analytics() {
  const totalIncidents = illegalParkingLocations.length;
  const todayIncidents = illegalParkingLocations.filter((incident) => {
    const today = new Date();
    const incidentDate = new Date(incident.timestamp);
    return incidentDate.toDateString() === today.toDateString();
  }).length;

  return (
    <div className="min-h-screen bg-primary">
      <Navigation />

      <main className="pt-20 p-6 mt-10">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">
              Analytics Dashboard
            </h1>
            <p className="text-gray-400">
              Monitor surveillance system performance and trends
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-tile1 border border-gray-700 rounded-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Total Incidents</p>
                  <p className="text-2xl font-bold text-white">
                    {totalIncidents}
                  </p>
                </div>
                <div className="p-3 bg-red-600 rounded-lg">
                  <FiAlertTriangle className="w-6 h-6 text-white" />
                </div>
              </div>
            </div>

            <div className="bg-tile1 border border-gray-700 rounded-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Today's Incidents</p>
                  <p className="text-2xl font-bold text-white">
                    {todayIncidents}
                  </p>
                </div>
                <div className="p-3 bg-blue-600 rounded-lg">
                  <FiClock className="w-6 h-6 text-white" />
                </div>
              </div>
            </div>

            {/* ini masih dummy data juga */}
            <div className="bg-tile1 border border-gray-700 rounded-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Active Cameras</p>
                  <p className="text-2xl font-bold text-white">12</p>
                </div>
                <div className="p-3 bg-green-600 rounded-lg">
                  <FiMapPin className="w-6 h-6 text-white" />
                </div>
              </div>
            </div>

            <div className="bg-tile1 border border-gray-700 rounded-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Detection Rate</p>
                  <p className="text-2xl font-bold text-white">94%</p>
                </div>
                <div className="p-3 bg-purple-600 rounded-lg">
                  <FiTrendingUp className="w-6 h-6 text-white" />
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-tile1 border border-gray-700 rounded-lg p-6">
              <h2 className="text-lg font-semibold text-white mb-4">
                Incident Trends
              </h2>
              <div className="h-64 flex items-center justify-center text-gray-400">
                <p>Chart visualization here</p>
              </div>
            </div>

            <div className="bg-tile1 border border-gray-700 rounded-lg p-6">
              <h2 className="text-lg font-semibold text-white mb-4">
                Location Hotspots
              </h2>
              <div className="h-64 flex items-center justify-center text-gray-400">
                <p>Heatmap visualization here</p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
