"use client";
import HlsPlayer from '@/components/HLSPlayer';
import { cctvLocations, illegalParkingLocations } from '@/utilities/DummyData';
import dynamic from 'next/dynamic'
import { useState } from 'react';
import { BiSort } from "react-icons/bi";

// dynamic import untuk Map karena leaflet tidak support SSR
const Map = dynamic(() => import('@/components/Map'), {
  ssr: false
});

export default function Home() {
  const [previewCoordinates, setPreviewCoordinates] = useState<[number, number] | null>(null);
  const [selectedCCTVIndex, setSelectedCCTVIndex] = useState<number | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  const handleEventClick = (index: number | null) => {
    if (selectedIndex === index || index === null) {
      setSelectedIndex(null);
      setPreviewCoordinates(null);
    } else {
      setSelectedIndex(index);
      setPreviewCoordinates(illegalParkingLocations[index].coordinates);
    }
    setSelectedCCTVIndex(null);
  };

  const viewOnGoogleMaps = (coordinates: [number, number]) => {
    window.open(`https://www.google.com/maps/search/?api=1&query=${coordinates[0]},${coordinates[1]}`, '_blank');
  };

  return (
    <main className="flex flex-row justify-center w-full min-h-screen space-x-10 p-10">
      <div>
        <div className="text-3xl text-center w-[60vh] font-bold mb-10">
          Illegal Parking Cases
        </div>
        {
          illegalParkingLocations.length === 0 ? (
            <div className="text-center text-gray-500">No illegal parking cases found.</div>
          ) : (
            <div className="w-full">
              <div className="flex mb-5 items-center">
                <BiSort className="text-2xl text-gray-400 cursor-pointer" />
                <span className="ml-2 text-lg font-semibold">Sort by:</span>
                <select className="ml-2 border rounded px-2 py-1 bg-black">
                  <option value="date">Urgency</option>
                  <option value="location">Time</option>
                </select>
              </div>
              <div className="flex flex-col w-full">
                {illegalParkingLocations.map((location, index) => (
                  <button key={index} className={`w-full cursor-pointer hover:opacity-80 transition py-2 px-3 text-sm space-y-2 text-start ${index % 2 === 0 ? 'bg-tile1' : 'bg-tile2'}`} onClick={() => handleEventClick(index)}>
                    <div className="font-semibold text-md">{location.locationName}</div>
                    <div>{location.narration}</div>
                    <div className="text-gray-400 font-secondary">{new Date(location.timestamp).toLocaleString()}</div>
                  </button>
                ))}
              </div>
            </div>
          )
        }
      </div>
      <Map 
        className="h-[calc(100vh-120px)] w-full z-50" 
        illegalParkingLocations={illegalParkingLocations}
        defaultViewingCoordinates={previewCoordinates ?? [-6.921817208463581, 107.6070564264402]} 
        cctvLocations={cctvLocations} 
        zoomLevel={previewCoordinates ? 18 : 14}
        onMarkerClick={(index) => {
          if (index !== null) {
            setSelectedCCTVIndex(index);
            setSelectedIndex(null);
            setPreviewCoordinates(cctvLocations[index].coordinates);
          } else {
            setSelectedCCTVIndex(null);
            setSelectedIndex(null);
            setPreviewCoordinates(null);
          }
        }}
      />
      {
        selectedIndex !== null && (
          <div className="fixed z-100 top-20 right-30 bg-black p-4 rounded shadow-lg max-w-100 space-y-2">
            <h2 className="text-lg font-semibold">{illegalParkingLocations[selectedIndex ?? 0].locationName}</h2>
            <div className="relative w-full pb-[56.25%] h-0">
              <iframe
                src={illegalParkingLocations[selectedIndex ?? 0].videoUrl}
                className="absolute top-0 left-0 w-full h-full"
                allow="autoplay"
              />
            </div>
            <p className="text-sm text-gray-400">{illegalParkingLocations[selectedIndex ?? 0].narration}</p>
            <p className="text-sm text-gray-400">Timestamp: {new Date(illegalParkingLocations[selectedIndex ?? 0].timestamp).toLocaleString()}</p>
            <p className="font-secondary">Coordinates: {illegalParkingLocations[selectedIndex ?? 0].coordinates[0].toFixed(6)}, {illegalParkingLocations[selectedIndex ?? 0].coordinates[1].toFixed(6)}</p>
            <div className="flex flex-row space-x-2">
              <button className="mt-2 px-4 py-2 bg-blue-500 cursor-pointer text-white rounded" onClick={() => handleEventClick(selectedIndex)}>Close</button>
              <button className="mt-2 px-4 py-2 bg-red-500 cursor-pointer text-white rounded w-full" onClick={() => viewOnGoogleMaps(illegalParkingLocations[selectedIndex ?? 0].coordinates)}>Open On Google Maps</button>
            </div>
          </div>
        )
      }
      {
        selectedCCTVIndex !== null && (
          <div className="fixed z-100 top-20 right-30 bg-black p-4 rounded shadow-lg w-150 space-y-2">
            <h2 className="text-lg font-semibold">{cctvLocations[selectedCCTVIndex].description}</h2>
            <div className="relative w-full pb-[75%] h-0">
              {cctvLocations[selectedCCTVIndex].streamUrl && <HlsPlayer src={cctvLocations[selectedCCTVIndex].streamUrl} />}
            </div>
            <button className="px-4 py-2 bg-blue-500 cursor-pointer text-white rounded" onClick={() => setSelectedCCTVIndex(null)}>Close</button>
          </div>
        )
      }
      
    </main>
  );
}
