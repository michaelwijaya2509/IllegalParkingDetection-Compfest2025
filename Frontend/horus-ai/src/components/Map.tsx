"use client";

import { useEffect, useRef } from "react";
import L, { Map as LeafletMap } from "leaflet";
import "leaflet/dist/leaflet.css";
import { MapProps } from "@/utilities/Types";

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
}: MapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<LeafletMap | null>(null);
  const illegalGroupRef = useRef<L.LayerGroup | null>(null);
  const cctvGroupRef = useRef<L.LayerGroup | null>(null);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      center: defaultViewingCoordinates,
      zoom: zoomLevel,
    });
    mapRef.current = map;

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; HORUS AI',
      maxZoom: 19,
    }).addTo(map);

    // Prepare marker groups
    illegalGroupRef.current = L.layerGroup().addTo(map);
    cctvGroupRef.current = L.layerGroup().addTo(map);

    return () => {
      map.remove();
      mapRef.current = null;
      illegalGroupRef.current = null;
      cctvGroupRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!mapRef.current) return;
    mapRef.current.setView(defaultViewingCoordinates, zoomLevel, { animate: true });
  }, [defaultViewingCoordinates, zoomLevel]);

  useEffect(() => {
    if (!illegalGroupRef.current) return;
    const group = illegalGroupRef.current;
    group.clearLayers();

    illegalParkingLocations?.forEach((location) => {
      L.marker(location.coordinates, { icon: carIcon })
        .bindPopup(
          `<b>Illegal Parking</b><br>${location.narration}<br/><small>${location.locationName}</small>`
        )
        .addTo(group);
    });
  }, [illegalParkingLocations]);

  useEffect(() => {
    if (!cctvGroupRef.current) return;
    const group = cctvGroupRef.current;
    group.clearLayers();

    cctvLocations?.forEach((location) => {
      L.marker(location.coordinates, { icon: cctvIcon })
        .bindPopup(`<b>Surveillance Camera</b><br>${location.description}`)
        .on("click", () => {
          if (onMarkerClick) {
            onMarkerClick(cctvLocations.indexOf(location));
            mapRef.current?.setView([location.coordinates[0] + 0.001, location.coordinates[1] + 0.001], 18, { animate: true });
          }
        })
        .addTo(group);
      
    });

  }, [cctvLocations]);

  return <div ref={containerRef} className={className ?? "h-screen w-full"} />;
}
