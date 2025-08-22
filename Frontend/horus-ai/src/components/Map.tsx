/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useEffect, useRef } from "react";
import L, { Map as LeafletMap } from "leaflet";
import "leaflet/dist/leaflet.css";

export interface MapProps {
  className?: string;
  illegalParkingLocations: any[];
  defaultViewingCoordinates: [number, number];
  cctvLocations: any[];
  zoomLevel: number;
  onMarkerClick: (index: number | null) => void;
  onIncidentClick: (index: number) => void;
}

delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "/leaflet/marker-icon-2x.png",
  iconUrl: "/leaflet/marker-icon.png",
  shadowUrl: "/leaflet/marker-shadow.png",
});

const carIcon = new L.Icon({
  iconUrl: "/car-icon.png",
  iconSize: [38, 38],
  iconAnchor: [19, 38],
  popupAnchor: [0, -38],
});

const cctvIcon = new L.Icon({
  iconUrl: "/cctv-icon.png",
  iconSize: [38, 38],
  iconAnchor: [19, 38],
  popupAnchor: [0, -38],
});

export default function Map({
  className,
  illegalParkingLocations,
  defaultViewingCoordinates,
  cctvLocations,
  zoomLevel,
  onMarkerClick,
  onIncidentClick,
}: MapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<LeafletMap | null>(null);
  const illegalGroupRef = useRef<L.LayerGroup | null>(null);
  const cctvGroupRef = useRef<L.LayerGroup | null>(null);

  const areCoordinatesValid = (coords: any): coords is [number, number] => {
    return (
      Array.isArray(coords) &&
      coords.length === 2 &&
      typeof coords[0] === "number" &&
      typeof coords[1] === "number" &&
      coords[0] >= -90 &&
      coords[0] <= 90 &&
      coords[1] >= -180 &&
      coords[1] <= 180
    );
  };

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      center: defaultViewingCoordinates,
      zoom: zoomLevel,
      zoomControl: true,
    });
    mapRef.current = map;

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; HORUS AI | &copy; OpenStreetMap contributors",
      maxZoom: 19,
    }).addTo(map);

    illegalGroupRef.current = L.layerGroup().addTo(map);
    cctvGroupRef.current = L.layerGroup().addTo(map);

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!mapRef.current) return;
    mapRef.current.setView(defaultViewingCoordinates, zoomLevel, {
      animate: true,
      duration: 0.5,
    });
  }, [defaultViewingCoordinates, zoomLevel]);

  useEffect(() => {
    if (!illegalGroupRef.current) return;
    const group = illegalGroupRef.current;
    group.clearLayers();

    illegalParkingLocations?.forEach((incident, index) => {
      if (!incident || !areCoordinatesValid(incident.coordinates)) {
        console.warn("Skipping incident with invalid coordinates:", incident);
        return;
      }

      const [lat, lng] = incident.coordinates;

      const marker = L.marker([lat, lng], { icon: carIcon }).bindPopup(
        `
          <div style="min-width: 200px; font-family: sans-serif;">
            <h3 style="margin: 0 0 8px 0; color: #e74c3c; font-weight: bold;">
              🚨 Pelanggaran Parkir
            </h3>
            <p style="margin: 4px 0; font-size: 12px;">
              <strong>Lokasi:</strong> ${incident.address || "Tidak diketahui"}
            </p>
            <p style="margin: 4px 0; font-size: 12px;">
              <strong>Kamera:</strong> ${
                incident.event?.cam_id || incident.cam_id || "N/A"
              }
            </p>
            <p style="margin: 4px 0; font-size: 12px;">
              <strong>Prioritas:</strong> 
              <span style="font-weight: bold;">
                ${incident.llm_data?.priority_label?.toUpperCase() || "UNKNOWN"}
              </span>
            </p>
            <div style="margin-top: 10px; text-align: center;">
              <button onclick="window.selectIncident(${index})" 
                      style="background: #3498db; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 12px;">
                Lihat Detail
              </button>
            </div>
          </div>
        `
      );
      marker.addTo(group);
    });
  }, [illegalParkingLocations, onIncidentClick]);

  useEffect(() => {
    if (!cctvGroupRef.current) return;
    const group = cctvGroupRef.current;
    group.clearLayers();

    cctvLocations?.forEach((cctv, index) => {
      if (!cctv || !areCoordinatesValid(cctv.coordinates)) {
        console.warn("Skipping CCTV with invalid coordinates:", cctv);
        return;
      }

      const [lat, lng] = cctv.coordinates;

      const marker = L.marker([lat, lng], { icon: cctvIcon })
        .bindPopup(
          `
          <div style="min-width: 180px; font-family: sans-serif;">
            <h3 style="margin: 0 0 8px 0; color: #3498db; font-weight: bold;">
              📹 ${cctv.name || "Kamera CCTV"}
            </h3>
            <p style="margin: 4px 0; font-size: 12px;">
              <strong>Alamat:</strong> ${cctv.address || "Tidak diketahui"}
            </p>
            <div style="margin-top: 10px; text-align: center;">
              {/* This button calls the function to show the live stream */}
              <button onclick="window.selectCCTV(${index})" 
                      style="background: #2ecc71; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 12px;">
                Lihat Detail & Live Stream
              </button>
            </div>
          </div>
        `
        )
        .on("click", () => {
          if (onMarkerClick) {
            onMarkerClick(index);
          }
        });
      marker.addTo(group);
    });
  }, [cctvLocations, onMarkerClick]);

  useEffect(() => {
    (window as any).selectIncident = (index: number) => {
      if (onIncidentClick) {
        onIncidentClick(index);
      }
    };

    (window as any).selectCCTV = (index: number) => {
      if (onMarkerClick) {
        onMarkerClick(index);
      }
    };

    return () => {
      delete (window as any).selectIncident;
      delete (window as any).selectCCTV;
    };
  }, [onIncidentClick, onMarkerClick]);

  return (
    <div
      ref={containerRef}
      className={className ?? "h-screen w-full"}
      style={{ position: "relative", zIndex: 1 }}
    />
  );
}
