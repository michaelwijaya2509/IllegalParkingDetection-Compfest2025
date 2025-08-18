export type MapProps = {
  className?: string;
  defaultViewingCoordinates: [number, number];
  illegalParkingLocations?: Array<IllegalParkingData>;
  cctvLocations?: Array<CCTVData>;
  zoomLevel: number;
  onMarkerClick?: (index: number | null) => void;
};

// Updated to match the complete ScoredEvent structure from backend
export type IllegalParkingData = {
  // Extracted and computed fields
  urgency_score: number;
  address: string;
  coordinates: [number, number];
  cam_id: string; // For backward compatibility
  timestamp: string;
  scored_at: string;

  // Original ViolationEvent nested inside
  event: {
    event_id: string;
    cam_id: string;
    duration_s: number;
    started_at: string; // ISO datetime string
    driver_left_vehicle: boolean;
    traffic_jam: boolean;
    zone_name: string;
    snapshot_url?: string | null;
    extra?: {
      track_id: string;
      [key: string]: any;
    };
  };

  // LLM Analysis Data from scored field
  llm_data: {
    priority_score: number;
    priority_label: string; // 'critical', 'high', 'medium', 'low'
    narrative: string;
    reasons: string[];
    recommended_actions: string[];
    confidence: string; // 'high', 'medium', 'low'
    category: string;
    adjustment_reason: string;
  };
};

export interface CCTVData {
  cam_id: string;
  name: string;
  address: string;
  coordinates: [number, number];
  stream_endpoint?: string | null;
  description?: string;
  is_running: boolean;
  city?: string;
  district?: string;
  zones?: string[]; // Zone file paths
}

// Backend ScoredEvent structure for reference
export interface ScoredEvent {
  event: {
    event_id: string;
    cam_id: string;
    duration_s: number;
    started_at: string;
    driver_left_vehicle: boolean;
    traffic_jam: boolean;
    zone_name: string;
    snapshot_url?: string | null;
    extra?: {
      track_id: string;
      [key: string]: any;
    };
  };
  camera: {
    cam_id: string;
    address: string;
    city: string;
    district: string;
    lat: number;
    lon: number;
  };
  scored: {
    priority_score: number;
    priority_label: string;
    narrative: string;
    reasons: string[];
    recommended_actions: string[];
    confidence: string;
    adjustment_reason: string;
    base_breakdown: Record<string, any>;
    dispatch_target?: string | null;
    category: string;
    raw_model_json: Record<string, any>;
  };
  scored_at: string;
  context_hash: string;
}
