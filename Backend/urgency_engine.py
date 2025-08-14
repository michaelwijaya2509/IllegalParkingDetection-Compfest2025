# urgency_engine.py
from __future__ import annotations

import os, json, time, queue, logging, hashlib, datetime as dt
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Iterable, Tuple

# ==== LLM client: Vertex AI via google-genai (sesuai llminference.ipynb) ====
from google import genai
from google.genai import types

PROJECT_ID = "horus-ai-468916"   # ganti jika perlu
LOCATION   = "us-central1"              # JANGAN "global" untuk Vertex AI
MODEL_NAME = "gemini-2.5-flash"         # jika belum tersedia di regionmu, ganti "gemini-2.0-flash"
ADDRESS   = "130 Quang Trung, Hải Châu, Đà Nẵng, Vietnam"
RADIUS_M  = 300
OSM_CACHE_TTLs = int(os.getenv("OSM_CACHE_TTLs", "21600"))  # 6 jam

@dataclass
class CameraMeta:
    cam_id: str
    address: str
    city: str = ""
    district: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None

@dataclass
class ViolationEvent:
    event_id: str
    cam_id: str
    duration_s: int
    started_at: str            # ISO
    driver_left_vehicle: bool
    traffic_jam: bool
    zone_name: str
    snapshot_url: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMUrgency:
    priority_score: int                 # alias dari urgency_score
    priority_label: str                 # derived
    narrative: str
    reasons: List[str]
    recommended_actions: List[str]
    confidence: str                     # "low"|"medium"|"high"
    adjustment_reason: Optional[str] = None
    base_breakdown: Optional[Dict[str, Any]] = None
    dispatch_target: Optional[str] = None
    category: Optional[str] = "Parkir_Liar"
    raw_model_json: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScoredEvent:
    event: ViolationEvent
    camera: CameraMeta
    scored: LLMUrgency
    scored_at: str
    context_hash: str

import requests
NOMINATIM = "https://nominatim.openstreetmap.org/search"
OVERPASS  = "https://overpass-api.de/api/interpreter"
HEADERS_OSM = {"User-Agent": "ParkLens-Research/1.0 (contact: you@example.com)"}

OVERPASS_QL = """
[out:json][timeout:25];
(
  node(around:{r},{lat},{lon})["amenity"~"hospital|clinic|police|fire_station|school|university|place_of_worship|bus_station"];
  node(around:{r},{lat},{lon})["public_transport"~"platform|stop_position|stop_area"];
  way(around:{r},{lat},{lon})["highway"];
  way(around:100,{lat},{lon})["highway"]["junction"="intersection"];
  way(around:{r},{lat},{lon})["access"="no"];
  way(around:{r},{lat},{lon})["busway"];
  way(around:100,{lat},{lon})["amenity"="parking"];
);
out body; >; out skel qt;
"""
#Pengambilan dan pengolahan data dari Open Street Map (itung jarak antar lokasi, tempat-tempat krusial, data lat lon)

def _haversine_m(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    R = 6371000
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2-lat1); dl = radians(lon2-lon1)
    a = sin(dphi/2)**2 + cos(p1)*cos(p2)*sin(dl/2)**2
    return 2*R*asin(sqrt(a))

def geocode_address(address: str) -> Optional[Dict[str, float]]:
    r = requests.get(NOMINATIM, params={"q": address, "format": "json", "limit": 1},
                     headers=HEADERS_OSM, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data: return None
    return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}

def fetch_overpass(lat: float, lon: float, radius_m: int) -> Dict[str, Any]:
    ql = OVERPASS_QL.format(r=radius_m, lat=lat, lon=lon)
    r = requests.post(OVERPASS, data=ql, headers=HEADERS_OSM, timeout=60)
    r.raise_for_status()
    return r.json()

def transform_overpass(raw: Dict[str, Any], lat: float, lon: float, radius_m: int) -> Dict[str, Any]:
    elements = raw.get("elements", [])
    nodes: Dict[int, Dict[str, Any]] = {}
    roads: List[Dict[str, Any]] = []
    pois:  List[Dict[str, Any]] = []
    has_parking_lt50 = False

    for e in elements:
        if e.get("type") == "node":
            nodes[e["id"]] = {"lat": e["lat"], "lon": e["lon"], "tags": e.get("tags", {})}

    for e in elements:
        tgs = e.get("tags", {})
        if e.get("type") == "way" and "highway" in tgs:
            roads.append({
                "highway": tgs.get("highway"),
                "maxspeed": tgs.get("maxspeed"),
                "lanes": tgs.get("lanes"),
            })
        if e.get("type") == "node" and (("amenity" in tgs) or ("public_transport" in tgs)):
            d = _haversine_m(lat, lon, e["lat"], e["lon"])
            pois.append({
                "kind": tgs.get("amenity") or tgs.get("public_transport"),
                "name": tgs.get("name", ""),
                "distance_m": int(d)
            })
        if e.get("type") == "way" and tgs.get("amenity") == "parking":
            n0_id = e.get("nodes", [None])[0]
            if n0_id and n0_id in nodes:
                d = _haversine_m(lat, lon, nodes[n0_id]["lat"], nodes[n0_id]["lon"])
                if d < 50: has_parking_lt50 = True

    return {
        "roads": roads[:40],
        "pois": sorted(pois, key=lambda x: x["distance_m"])[:40],
        "intersections": [],  # bisa disempurnakan
        "has_parking_lt50m": has_parking_lt50,
        "meta": {"radius_m": radius_m, "provider": "overpass"}
    }

#Bangun LLM 
def build_system_instruction() -> str: #prompt untuk llm
    return (
        "You are an urgency arbiter for illegal parking events in the ParkLens AI system.\n"
        "1) Input: an event {cam_id, lat, lon, duration_s} and OSM-derived features.\n"
        "2) Compute a deterministic base_score (0–100) using this rubric:\n"
        "   - Roads: trunk +15, primary +12, secondary +8; maxspeed ≥60 +6; lanes ≥4 +4; intersection degree ≥4 +5.\n"
        "   - POI: hospital +20 (cap 40), clinic +12, fire_station +12, police +8,\n"
        "          school/university +8, bus_station +10, government +6, place_of_worship +5.\n"
        "   - Special: access=no or busway +8; <30 m from intersection +5.\n"
        "   - Counter: official parking <50 m −10; mall-dominant without arterial −6.\n"
        "3) Adjust with event signals:\n"
        "   - duration_s > 300 +20; driver_left_vehicle True +25; traffic_jam True −20.\n"
        "Return JSON only: {priority_score, reason, recommend_action[], dispatch_target?, category?}."
        "Return JSON that matches the provided response_schema exactly. "
        "No extra fields. Keep arrays concise and factual.\n"
    )

RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "urgency_score": types.Schema(type=types.Type.INTEGER, minimum=0, maximum=100),
        "adjustment_reason": types.Schema(type=types.Type.STRING),
        "reasons": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING), max_items=5),
        "narrative": types.Schema(type=types.Type.STRING),
        "recommended_actions": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING), max_items=3),
        "confidence": types.Schema(type=types.Type.STRING, enum=["low","medium","high"]),
        "base_breakdown": types.Schema(type=types.Type.OBJECT, description="Optional scoring breakdown keyed by factor"),
    },
    required=["urgency_score","narrative","reasons","confidence"],
)

#class/threshold dari urgency
def _priority_bucket(s: int) -> str:
    if s >= 85: return "critical"
    if s >= 70: return "high"
    if s >= 50: return "medium"
    return "low"

#hashing event pelanggaran (semacam hash di github commit)
def _hash_context(ev: ViolationEvent, cam: CameraMeta, feats: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(asdict(ev), sort_keys=True).encode())
    h.update(json.dumps(asdict(cam), sort_keys=True).encode())
    h.update(json.dumps(feats, sort_keys=True).encode())
    return h.hexdigest()

# ================== Engine ==================
class UrgencyEngine:
    def __init__(self,
                 cameras: Dict[str, CameraMeta],
                 *,
                 project_id: str = PROJECT_ID,
                 location: str = LOCATION,
                 model_name: str = MODEL_NAME,
                 radius_m: int = RADIUS_M,
                 logger: Optional[logging.Logger] = None) -> None:
        self.cameras = cameras
        self.vertex = genai.Client(vertexai=True, project=project_id, location=location)
        self.model_name = model_name
        self.radius_m = radius_m
        self.log = logger or logging.getLogger("urgency_engine")
        self._in_q: "queue.Queue[ViolationEvent]" = queue.Queue()
        self._ready: List[ScoredEvent] = []
        self._osm_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}   # cam_id -> (ts, features)
        self._cache: Dict[str, ScoredEvent] = {}

    # ---- public API ----
    def ingest(self, event: ViolationEvent) -> None:
        self._in_q.put(event)

    def aggregate_and_score(self, window_s: int = 20) -> int:
        bucket: Dict[str, List[ViolationEvent]] = {}
        t0 = time.time()
        while time.time() - t0 < window_s:
            try:
                ev = self._in_q.get(timeout=0.2)
                bucket.setdefault(ev.cam_id, []).append(ev)
            except queue.Empty:
                pass

        total = 0
        for cam_id, events in bucket.items():
            picked = self._pick_per_track(events)
            scored = self.score_events(picked)
            self._ready.extend(scored)
            total += len(scored)
        return total

    def get_ready(self) -> List[ScoredEvent]:
        out, self._ready = self._ready, []
        return out

    def score_events(self, events: Iterable[ViolationEvent]) -> List[ScoredEvent]:
        out: List[ScoredEvent] = []
        for ev in events:
            cam = self._resolve_camera(ev.cam_id)
            feats = self._get_osm_features(cam)
            ctx_hash = _hash_context(ev, cam, feats)
            if ctx_hash in self._cache:
                out.append(self._cache[ctx_hash]); continue

            payload = {
                "cam_id": cam.cam_id, "lat": cam.lat, "lon": cam.lon,
                "duration_s": ev.duration_s,
                "driver_left_vehicle": ev.driver_left_vehicle,
                "traffic_jam": ev.traffic_jam,
                "zone_name": ev.zone_name
            }
            system_text = build_system_instruction()
            user_text   = json.dumps({"event": payload, "features": feats}, ensure_ascii=False)

            contents = [
                types.Content(role="user", parts=[types.Part.from_text(user_text)])
            ]
            cfg = types.GenerateContentConfig(
                system_instruction=types.Part.from_text(system_text),
                temperature=0.1,
                top_p=0.95,                     
                seed=0,                          
                max_output_tokens=1024,  
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,           
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_UNSPECIFIED", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                ],
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
            resp = self.vertex.models.generate_content(
                model=self.model_name, contents=contents, config=cfg
            )

            obj = None
            if hasattr(resp, "parsed") and resp.parsed is not None:
                obj = resp.parsed
            else:
                try:
                    obj = json.loads(resp.text)
                except Exception:
                    obj = {"_raw": getattr(resp, "text", None)}

            
            urgency_score = int(max(0, min(100, int(obj.get("urgency_score", 0)))))

            urgency = LLMUrgency(
                priority_score=urgency_score,
                priority_label=_priority_bucket(urgency_score),
                narrative=str(obj.get("narrative", ""))[:1200],
                reasons=[str(x) for x in obj.get("reasons", [])][:5],
                recommended_actions=[str(x) for x in obj.get("recommended_actions", [])][:3],
                confidence=str(obj.get("confidence", "medium")),
                adjustment_reason=obj.get("adjustment_reason"),
                base_breakdown=obj.get("base_breakdown"),
                dispatch_target=obj.get("dispatch_target"),
                category=obj.get("category", "Parkir_Liar"),
                raw_model_json=obj
            )
            scored = ScoredEvent(
                event=ev, camera=cam, scored=urgency,
                scored_at=dt.datetime.utcnow().isoformat(), context_hash=ctx_hash
            )
            self._cache[ctx_hash] = scored
            out.append(scored)
        return out

    # ---- internal helpers ----
    def _pick_per_track(self, events: List[ViolationEvent]) -> List[ViolationEvent]:
        # jika ada track_id di extra, pilih durasi terpanjang per track
        best: Dict[Any, ViolationEvent] = {}
        for e in events:
            tid = e.extra.get("track_id", e.event_id)
            cur = best.get(tid)
            if (cur is None) or (e.duration_s > cur.duration_s):
                best[tid] = e
        return list(best.values())

    def _resolve_camera(self, cam_id: str) -> CameraMeta:
        cam = self.cameras.get(cam_id)
        if not cam:
            raise KeyError(f"Unknown cam_id: {cam_id}")
        if cam.lat is None or cam.lon is None:
            geo = geocode_address(cam.address)
            if not geo:
                raise RuntimeError(f"Gagal geocode: {cam.address}")
            cam.lat, cam.lon = geo["lat"], geo["lon"]
        return cam

    def _get_osm_features(self, cam: CameraMeta) -> Dict[str, Any]:
        ts, feats = self._osm_cache.get(cam.cam_id, (0, {}))
        if time.time() - ts < OSM_CACHE_TTLs and feats:
            return feats
        raw = fetch_overpass(cam.lat, cam.lon, self.radius_m)
        feats = transform_overpass(raw, cam.lat, cam.lon, self.radius_m)
        if cam.poi:
            # gabung POI manual jika ada
            for p in cam.poi[:6]:
                feats["pois"].append({"kind": "manual", "name": p, "distance_m": 0})
        self._osm_cache[cam.cam_id] = (time.time(), feats)
        return feats