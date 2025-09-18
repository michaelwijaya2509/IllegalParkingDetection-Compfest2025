from firebase_admin import credentials, db
from typing import Any, Dict, List, Optional, Iterable, Tuple
import firebase_admin
import os
from dataclasses import dataclass, asdict, field
import datetime
from google.cloud import storage, secretmanager
import json as json_lib

PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID", "horus-ai-468916")
SECRET_ID = os.environ.get("FIREBASE_SECRET_ID", "firebase-realtimedb-credentials")

def access_secret_version(project_id, secret_id, version_id="latest"):
    """
    Mengakses payload dari secret yang ada di Secret Manager.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    
    # Akses secret version.
    response = client.access_secret_version(request={"name": name})
    
    return response.payload.data.decode("UTF-8")


try:
    credentials_json_string = access_secret_version(PROJECT_ID, SECRET_ID)
    
    credentials_info = json_lib.loads(credentials_json_string)

    # ======================== Firebase Configurations ====================== #
    cred = credentials.Certificate(credentials_info)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://horus-ai-468916-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })
    fb_db_ref = db.reference()
    # ======================== GCS Configurations ======================= #
    GCS_BUCKET_NAME =  os.environ.get("GCS_BUCKET_NAME")
    storage_client = storage.Client.from_service_account_info(credentials_info)
    gcs_bucket = storage_client.bucket(GCS_BUCKET_NAME)

except Exception as e:
    print(f"Error initializing Firebase or GCS: {e}")
    fb_db_ref = None
    gcs_bucket = None
    storage_client = None


json = {
    "camera": {
      "address": "Wonosari, Gunungkidul Regency, Special Region of Yogyakarta, Java, 55813, Indonesia",
      "cam_id": "gunungkidul1",
      "city": "",
      "district": "",
      "lat": -7.962478,
      "lon": 110.603346
    },
    "context_hash": "dab1901595146c84925c2ade727487d06b2a85c23ff5b27f38d94817113ased3",
    "event": {
      "cam_id": "gunungkidul1",
      "driver_left_vehicle": False,
      "duration_s": 50,
      "event_id": "gunungkidul-1",
      "extra": {
        "track_id": "1"
      },
      "snapshot_url": "https://i.imgur.com/SZSVZWf.png",
      "started_at": "2025-08-20T17:44:00.546865",
      "traffic_jam": False,
      "zone_name": "zona_enhanced.json"
    },
    "scored": {
      "priority_score": 28,
      "priority_label": "low",
      "narrative": "Kendaraan terdeteksi berhenti singkat di area Wonosari pada jalan kelas sekunder berdekatan dengan RSUD Wonosari. Karena pengemudi tetap berada di kendaraan dan tidak ada indikasi kemacetan serta durasi hanya ~50 detik (≤300s), urgensi penindakan rendah. Tetap direkomendasikan peringatan ringan dan pemantauan singkat untuk memastikan tidak berkembang menjadi pelanggaran lebih serius di dekat fasilitas kesehatan.",
      "reasons": [
        "Dekat rumah sakit (+20)",
        "Berada di jalan kelas sekunder (+8)"
      ],
      "recommended_actions": [
        "Sampaikan peringatan agar segera bergerak",
        "Pantau 3–5 menit; eskalasi jika durasi > 300 dtk",
        "Catat bukti (foto/log) untuk rekam insiden"
      ],
      "confidence": "medium",
      "adjustment_reason": "Tidak ada kemacetan dan pengemudi tetap di dalam kendaraan; durasi ~50 detik (≤300) sehingga tidak ada penambahan skor perilaku/waktu.",
      "base_breakdown": {
        "poi_near_hospital": 20,
        "road_secondary": 8,
        "duration_over_300s": 0,
        "driver_left_vehicle": 0,
        "traffic_jam": 0
      },
      "dispatch_target": "Dishub/Satpol PP Wonosari",
      "category": "Parkir_Liar",
      "raw_model_json": {
        "urgency_score": 28,
        "narrative": "Kendaraan terdeteksi berhenti singkat di area Wonosari pada jalan kelas sekunder berdekatan dengan RSUD Wonosari. Karena pengemudi tetap berada di kendaraan dan tidak ada indikasi kemacetan serta durasi hanya ~50 detik (≤300s), urgensi penindakan rendah.",
        "reasons": [
          "Dekat rumah sakit (+20)",
          "Berada di jalan kelas sekunder (+8)"
        ],
        "recommended_actions": [
          "Sampaikan peringatan agar segera bergerak",
          "Pantau 3–5 menit; eskalasi jika durasi > 300 dtk"
        ],
        "confidence": "medium"
    }
  },
  "scored_at": "2025-08-20T10:45:00.299077"
}



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
class CameraMeta:
    cam_id: str
    address: str
    city: str = ""
    district: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None

@dataclass
class ScoredEvent:
    event: ViolationEvent
    camera: CameraMeta
    scored: LLMUrgency
    scored_at: str
    context_hash: str

violation_event = ViolationEvent(
    event_id=json["event"]["event_id"],
    cam_id=json["event"]["cam_id"],
    duration_s=json["event"]["duration_s"],
    started_at=json["event"]["started_at"],
    driver_left_vehicle=json["event"]["driver_left_vehicle"],
    traffic_jam=json["event"]["traffic_jam"],
    zone_name=json["event"]["zone_name"],
    snapshot_url=json["event"]["snapshot_url"],
    extra={"track_id": json["event"]["extra"]["track_id"]}
)

camera_meta = CameraMeta(
    cam_id=json["camera"]["cam_id"],
    address=json["camera"]["address"],
    city=json["camera"].get("city", ""),
    district=json["camera"].get("district", ""),
    lat=json["camera"].get("lat", None),
    lon=json["camera"].get("lon", None)
)

llm_urgency = LLMUrgency(
    priority_score=json["scored"]["priority_score"],
    priority_label=json["scored"]["priority_label"],
    narrative=json["scored"]["narrative"],
    reasons=json["scored"]["reasons"],
    recommended_actions=json["scored"]["recommended_actions"],
    confidence=json["scored"]["confidence"],
    adjustment_reason=json["scored"].get("adjustment_reason", None),
    base_breakdown=json["scored"].get("base_breakdown", None),
    dispatch_target=json["scored"].get("dispatch_target", None),
    category=json["scored"].get("category", "Parkir_Liar"),
    raw_model_json=json["scored"].get("raw_model_json", {})
)



                             
fb_db_ref = db.reference()

scored = ScoredEvent(
    event=violation_event, camera=camera_meta, scored=llm_urgency,
    scored_at=datetime.datetime.now().isoformat(), context_hash=json["context_hash"]
)
scored_dict = asdict(scored)
event_id = f"{camera_meta.cam_id}-{violation_event.event_id}-{int(datetime.datetime.now().timestamp())}"

try:
  fb_db_ref.child('PendingIncidents').child(event_id).set(scored_dict)
  print(f"Pushed incident {event_id} to Firebase.")
except Exception as e:
  print(f"Failed to push incident {event_id} to Firebase: {e}")