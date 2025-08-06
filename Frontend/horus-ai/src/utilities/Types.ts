export type MapProps = {
  className?: string;
  defaultViewingCoordinates: [number, number];
  illegalParkingLocations?: Array<IllegalParkingData>;
  cctvLocations?: Array<CCTVData>;
  zoomLevel: number;
  onMarkerClick?: (index: number | null) => void;
};

export type IllegalParkingData = {
  coordinates: [number, number];
  narration: string;
  videoUrl?: string; // Optional field for video URL
  timestamp: string;
  locationName: string;
  urgency: number;
};

export type CCTVData = {
  coordinates: [number, number];
  streamUrl?: string;
  description?: string;
};