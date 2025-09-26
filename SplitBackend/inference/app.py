# infer-api/app.py  (RELATIVE-PATH READY) — service khusus inference
from flask import Flask, jsonify, Response, request, send_from_directory
from flask_cors import CORS
import threading, queue, json, time, os, logging, subprocess, sys
import datetime as dt
from datetime import datetime, timedelta
from dataclasses import asdict
from collections import deque, Counter
from typing import Dict, List

# === Heavy deps (tetap sama) ===
import torch, torch.nn as nn
import torch.nn.functional as F
import cv2, numpy as np
from math import sqrt
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# === Project modules (tetap sama) ===
from logics.cnn_macet import check_macet_cnn
from logics.cek_supir_keluar import check_driver_exit, crop_with_margin, preprocess_frames_for_inference
from logics.urgency_engine import UrgencyEngine, ViolationEvent, CameraMeta, ScoredEvent
from models import (
    MultiHeadAttention,
    FeatureAdapter,
    CNNLSTMModel,
    CNNFeatureExtractor,
    SequenceHeadLSTM,
)

# === Cloud (tetap sama) ===
import firebase_admin
from firebase_admin import credentials, db
from google.cloud import storage, secretmanager

# --------------------------------- Common utils ---------------------------------
PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID", "horus-ai-468916")
SECRET_ID  = os.environ.get("FIREBASE_SECRET_ID", "firebase-realtimedb-credentials")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", 'horus-ai-storage')

def access_secret_version(project_id, secret_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    resp = client.access_secret_version(request={"name": name})
    return resp.payload.data.decode("UTF-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def resolve_path(p):
   if isinstance(p, int):
        return p
   if isinstance(p, str) and p.isdigit():
        return int(p)
   return p if os.path.isabs(p) else os.path.normpath(os.path.join(BASE_DIR, p))

# --------------------------------- App init ---------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("horusai.infer")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ---- Firebase & GCS via Secret Manager (tetap sama) ----
try:
    credentials_json_string = access_secret_version(PROJECT_ID, SECRET_ID)
    credentials_info = json.loads(credentials_json_string)

    cred = credentials.Certificate(credentials_info)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://horus-ai-468916-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })
    fb_db_ref = db.reference()
    log.info("Firebase initialized from Secret Manager.")

    storage_client = storage.Client.from_service_account_info(credentials_info)
    gcs_bucket = storage_client.bucket(GCS_BUCKET_NAME) if GCS_BUCKET_NAME else None
    if gcs_bucket:
        log.info(f"GCS bucket OK: {GCS_BUCKET_NAME}")
    else:
        log.warning("GCS_BUCKET_NAME not set; snapshot upload disabled.")
except Exception as e:
    log.error(f"Failed Secret Manager/Firebase/GCS init: {e}")
    fb_db_ref = None
    gcs_bucket = None
    storage_client = None

# ---- Load Behavior Model (tetap sama) ----
MODEL_EVENT_PATH = resolve_path("best_cnnlstm_stage1_fixed.pkl")
MODEL_EVENT_INFERENCE = None
try:
    log.info("Loading driver-exit model ...")
    MODEL_EVENT_INFERENCE = torch.load(MODEL_EVENT_PATH, map_location=torch.device('cpu'), weights_only=False)
    MODEL_EVENT_INFERENCE.eval()
    log.info("Driver-exit model loaded: %s", type(MODEL_EVENT_INFERENCE))
except Exception as e:
    log.exception("Failed to load driver-exit model from %s: %s", MODEL_EVENT_PATH, e)

# --------------------------------- SSE infra (tetap sama) ---------------------------------
history_events = deque(maxlen=int(os.environ.get("SSE_HISTORY_MAX", "500")))
clients: List[queue.Queue] = []
clients_lock = threading.Lock()

def broadcast(event: dict, also_store: bool = True):
    if also_store:
        history_events.append(event)
    with clients_lock:
        for q in list(clients):
            try:
                q.put_nowait(event)
            except Exception:
                pass

@app.get("/events")
def events():
    client_ip = request.remote_addr
    log.info("New SSE client: %s", client_ip)
    client_queue: queue.Queue = queue.Queue(maxsize=200)
    with clients_lock:
        clients.append(client_queue)
    def stream():
        try:
            for ev in list(history_events):
                yield f"data: {json.dumps(ev)}\n\n"
            while True:
                ev = client_queue.get()
                yield f"data: {json.dumps(ev)}\n\n"
        except GeneratorExit:
            log.info("SSE client disconnected: %s", client_ip)
            with clients_lock:
                try: clients.remove(client_queue)
                except ValueError: pass
            raise
    return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# --------------------------------- Helper funcs (tetap sama) ---------------------------------
MOVE_PX_THRESH = int(os.environ.get("MOVE_PX_THRESH", "30"))
MIN_STOP_S     = float(os.environ.get("MIN_STOP_S", "50"))
TARGET_CLASSES = set(os.environ.get("TARGET_CLASSES", "car,truck,bus").split(','))

def point_in_polygon(x: int, y: int, polygon: list[list[int]]):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), (int(x), int(y)), False) >= 0

def load_zones(paths: list[str]):
    polys: list[dict] = []
    for p in paths:
        zp = resolve_path(p)
        with open(zp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "polygon" in data:
            polys.append({"name": data.get("name", os.path.basename(zp)), "polygon": data["polygon"]})
        elif isinstance(data, list):
            polys.append({"name": os.path.basename(zp), "polygon": data})
        else:
            raise ValueError(f"Zona format tidak dikenal: {p}")
    return polys

def run_driver_exit_check(cam_id, track_id, cropped_frames, full_frames, model, track_state_obj):
    log.info(f"[THREAD] Processing {len(cropped_frames)} frames for track {track_id} ...")
    is_driver_exit = check_driver_exit(cropped_frames, model)
    if is_driver_exit:
        log.info(f"[THREAD] Driver exit detected for track {track_id}")
        track_state_obj["driver_exited"] = True
        if full_frames:
            track_state_obj["evidence_frame"] = full_frames[len(full_frames) // 2]
    else:
        log.info(f"[THREAD] No driver exit for track {track_id}")
    track_state_obj["event_check_running"] = False

# --------------------------------- Worker (LOGIC 1:1) ---------------------------------
class DetectorWorker(threading.Thread):
    def __init__(self, cam_id: str, stream_url: str, zones: list[str], model_path: str = "yolo11n.pt", device: str | None = None,
                 camera_meta: dict | None = None):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.stream_url = resolve_path(stream_url)
        self.zone_files = zones
        self.zones = load_zones(zones)
        self.stop_flag = threading.Event()
        self.device = device
        self.model_path = model_path
        self.track_state: Dict[int, dict] = {}
        self.tracker = DeepSort(max_age=200, n_init=3)
        self.model: YOLO | None = None
        self.label_map: Dict[int, str] = {}
        self.current_frame = None
        self.current_frame_lock = threading.Lock()
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.frontend_tracking_data: dict = {"tracks": [], "zones": self.zones}
        self.frontend_data_lock = threading.Lock()
        # camera meta (optional, untuk UrgencyEngine)
        self.camera_meta = camera_meta or {}

    def run(self):
        try:
            self._loop()
        except Exception as e:
            log.exception("worker_error %s: %s", self.cam_id, e)
            broadcast({"type": "worker_error", "cam_id": self.cam_id, "error": str(e)}, also_store=False)

    def _loop(self):
        dev = self.device or ("cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
        self.model = YOLO(self.model_path)
        if dev == "cuda":
            try: self.model.to("cuda")
            except Exception: pass
        self.label_map = self.model.names

        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            broadcast({"type": "stream_error", "cam_id": self.cam_id, "msg": "cannot open stream"}, also_store=False)
            return

        self.frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        COOLDOWN_TIME = 1800  # frames ~60s
        cooldown_frame_count = 0
        isMacet = False

        while not self.stop_flag.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if self.frame_width == 0 or self.frame_height == 0:
                self.frame_height, self.frame_width, _ = frame.shape

            with self.current_frame_lock:
                self.current_frame = frame.copy()

            # YOLO detect target classes only
            results = self.model(frame, verbose=False)[0]
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if self.label_map.get(cls_id, "unknown") in TARGET_CLASSES:
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], float(box.conf[0]), cls_id))

            try:
                tracks = self.tracker.update_tracks(detections, frame=frame)
            except Exception as e:
                broadcast({"type": "track_error", "cam_id": self.cam_id, "error": str(e)}, also_store=False)
                continue

            now_ts = time.time()
            current_tracks_for_frontend = []

            for tr in tracks:
                if not tr.is_confirmed(): continue

                x1, y1, x2, y2 = map(int, tr.to_ltrb())
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cls_id = tr.det_class
                cls_name = self.label_map.get(cls_id, "unknown")

                if cls_name not in TARGET_CLASSES:
                    if tr.track_id in self.track_state:
                        self.track_state.pop(tr.track_id, None)
                    continue

                zone_name_hit = None
                for z in self.zones:
                    if point_in_polygon(cx, cy, z["polygon"]):
                        zone_name_hit = z["name"]
                        break

                if zone_name_hit is None:
                    if tr.track_id in self.track_state:
                        self.track_state.pop(tr.track_id, None)
                    continue

                st = self.track_state.get(tr.track_id)
                if st is None:
                    TARGET_FRAMES_FOR_EVENT = 16
                    st = {
                        "last_pos": (cx, cy),
                        "stationary_s": 0.0,
                        "last_ts": now_ts,
                        "zone_name": zone_name_hit,
                        "frame_sequence": deque(maxlen=TARGET_FRAMES_FOR_EVENT),
                        "event_check_running": False,
                        "driver_exited": False,
                        "violation_reported": False,
                        "last_check_ts": 0,
                        "evidence_frame": None
                    }
                    self.track_state[tr.track_id] = st

                dx, dy = cx - st["last_pos"][0], cy - st["last_pos"][1]
                dist  = sqrt(dx * dx + dy * dy)
                dt_s  = now_ts - st["last_ts"]

                if dist < MOVE_PX_THRESH:
                    st["stationary_s"] += dt_s
                    cropped_frame = crop_with_margin(frame, tr.to_ltrb())
                    if cropped_frame is not None:
                        st["frame_sequence"].append((cropped_frame, frame.copy()))
                else:
                    st["stationary_s"], st["last_pos"] = 0.0, (cx, cy)
                    st["frame_sequence"].clear()
                    st["driver_exited"] = False
                    st["event_check_running"] = False
                    st["violation_reported"] = False
                    st["evience_frame"] = None
                    st["last_check_ts"] = 0

                st["last_ts"] = now_ts
                st["zone_name"] = zone_name_hit

                is_traffic_jam = isMacet
                if isMacet:
                    cooldown_frame_count += 1
                    if cooldown_frame_count >= COOLDOWN_TIME:
                        isMacet = False
                        cooldown_frame_count = 0
                    else:
                        continue

                if len(detections) >= 8 and not isMacet:
                    isMacet = check_macet_cnn(frame)
                    if isMacet:
                        is_traffic_jam = True
                        if not st.get("driver_exited", False):
                            st["last_ts"] = now_ts
                            continue

                CHECK_INTERVAL_S = 2.0
                if (len(st["frame_sequence"]) == st["frame_sequence"].maxlen and
                    not st["event_check_running"] and
                    not st["driver_exited"] and
                    (now_ts - st["last_check_ts"] > CHECK_INTERVAL_S) and
                    MODEL_EVENT_INFERENCE):
                    st["event_check_running"] = True
                    st["last_check_ts"] = now_ts
                    sequence_copy = list(st["frame_sequence"])
                    crops_for_model = [it[0] for it in sequence_copy]
                    full_frames_for_evidence = [it[1] for it in sequence_copy]
                    threading.Thread(
                        target=run_driver_exit_check,
                        args=(self.cam_id, tr.track_id, crops_for_model, full_frames_for_evidence, MODEL_EVENT_INFERENCE, st),
                        daemon=True
                    ).start()

                is_violation_by_time = st["stationary_s"] >= MIN_STOP_S

                if is_violation_by_time and not st.get("violation_reported", False):
                    st["violation_reported"] = True
                    reason = "Supir Keluar" if st.get("driver_exited", False) else "Waktu Parkir > 5 Menit"

                    snap_url = None
                    try:
                        snapshot_frame = st.get("evidence_frame") if st.get("driver_exited") else frame
                        sx1, sy1, sx2, sy2 = map(int, tr.to_ltrb())
                        crop = snapshot_frame[max(0, sy1):max(0, sy2), max(0, sx1):max(0, sx2)]
                        if crop.size > 0 and gcs_bucket:
                            ok, buffer = cv2.imencode('.jpg', crop)
                            if ok:
                                blob_name = f"snapshots/{self.cam_id}{tr.track_id}{int(now_ts)}.jpg"
                                blob = gcs_bucket.blob(blob_name)
                                blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')
                                snap_url = blob.public_url
                                log.info("Snapshot uploaded: %s", snap_url)
                            else:
                                log.error("Failed to encode snapshot JPEG")
                    except Exception as e:
                        log.exception("Failed to upload snapshot: %s", e)

                    event_id = f"{self.cam_id}-{tr.track_id}-{int(now_ts)}"

                    # pastikan meta kamera tersedia untuk UrgencyEngine
                    if self.cam_id not in camera_metas:
                        camera_metas[self.cam_id] = CameraMeta(
                            cam_id=self.cam_id,
                            address=self.camera_meta.get("address","Alamat tidak diketahui"),
                            city=self.camera_meta.get("city",""),
                            district=self.camera_meta.get("district",""),
                        )

                    violation_event = ViolationEvent(
                        event_id=event_id,
                        cam_id=self.cam_id,
                        duration_s=int(st["stationary_s"]),
                        started_at=dt.datetime.fromtimestamp(now_ts - st["stationary_s"]).isoformat(),
                        driver_left_vehicle=st.get("driver_exited", False),
                        traffic_jam=is_traffic_jam,
                        zone_name=st["zone_name"],
                        snapshot_url=snap_url,
                        extra={"track_id": tr.track_id}
                    )

                    scored_events = urgency_engine.score_events([(violation_event, is_traffic_jam)])
                    if scored_events:
                        scored = scored_events[0]
                        scored_dict = asdict(scored)
                        try:
                            if fb_db_ref:
                                fb_db_ref.child('PendingIncidents').child(event_id).set(scored_dict)
                                log.info("Pushed incident %s to Firebase", event_id)
                        except Exception as e:
                            log.error("Failed push incident %s: %s", event_id, e)

                        broadcast({"type": "violation_event", "data": scored_dict}, also_store=True)

                is_driver_locked = st.get("driver_exited", False)
                current_tracks_for_frontend.append({
                    "track_id": tr.track_id,
                    "bbox": [x1, y1, x2, y2],
                    "class_name": cls_name,
                    "stationary_s": int(st["stationary_s"]),
                    "is_close_to_violation": st["stationary_s"] >= MIN_STOP_S * 0.8,
                    "is_violation": st.get("violation_reported", False),
                    "is_driver_locked": is_driver_locked,
                    "is_paused_by_traffic": is_traffic_jam and not is_driver_locked
                })

            with self.frontend_data_lock:
                self.frontend_tracking_data["tracks"] = current_tracks_for_frontend

        cap.release()
        log.info("worker %s stopped", self.cam_id)

        try:
            broadcast({"type": "stream_eof", "cam_id": self.cam_id}, also_store=True)
        except Exception:
            pass

# Hapus diri dari registry jika masih tercatat sebagai worker aktif
        try:
            if workers.get(self.cam_id) is self:
                del workers[self.cam_id]
        except Exception:
            pass    

    def stop(self):
        self.stop_flag.set()

    def get_frame(self):
        with self.current_frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def get_tracking_data(self):
        with self.frontend_data_lock:
            return self.frontend_tracking_data.copy()

# --------------------------------- Globals for service ---------------------------------
workers: Dict[str, DetectorWorker] = {}

#dictionary baru untuk menympan paramter terakhir video untuk autostart ketika direfresh
last_start_params: Dict[str, dict] = {}

# UrgencyEngine (kamera akan ditambahkan dinamis saat /worker/start jika meta diberikan)
camera_metas: Dict[str, CameraMeta] = {}
urgency_engine = UrgencyEngine(cameras=camera_metas)

SNAP_DIR = resolve_path(os.environ.get("SNAP_DIR", "snaps"))
os.makedirs(SNAP_DIR, exist_ok=True)

# --------------------------------- MJPEG streamer (tetap sama) ---------------------------------
def generate_frames(cam_id: str):
    idle = 0
    while cam_id in workers:
        frame = workers[cam_id].get_frame() if cam_id in workers else None
        if frame is None:
            idle += 1
            if idle > 250:  # ~7 detik
                break
            time.sleep(0.03)
            continue
        idle = 0
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.get('/video/<cam_id>')
def video_feed(cam_id):
    autostart = request.args.get("autostart") == "1"
    if cam_id not in workers and autostart and cam_id in last_start_params:
        p = last_start_params[cam_id]
        w = DetectorWorker(
            cam_id=cam_id,
            stream_url=p["stream_url"],
            zones=p["zones"],
            model_path=p["model"],
            camera_meta=p["camera_meta"]
        )
        w.start()
        workers[cam_id] = w

    if cam_id not in workers:
        return jsonify({"error": "Camera not running"}), 404

    return Response(generate_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

# --------------------------------- Worker management API (baru) ---------------------------------
@app.post("/worker/start")
def worker_start():
    """
    Body JSON:
    {
      "cam_id": "gunungkidul1",
      "stream_url": "rtsp://...",
      "zones": ["config/zona1.json", ...],
      "model": "yolo11n.pt",         # optional
      "camera_meta": { "address":"...", "city":"", "district":"" }  # optional
    }
    """
    p = request.get_json(force=True)
    cam_id = p["cam_id"]
    if cam_id in workers:
        return jsonify({"ok": False, "msg": "already running"}), 400
    stream_url = p["stream_url"]
    zones = p["zones"]
    model = p.get("model", "yolo11n.pt")
    camera_meta = p.get("camera_meta", {})

    last_start_params[cam_id] = {
        "stream_url": stream_url,
        "zones": zones,
        "model": model,
        "camera_meta": camera_meta
    }

    w = DetectorWorker(cam_id=cam_id, stream_url=stream_url, zones=zones,
                       model_path=model, camera_meta=camera_meta)
    w.start()
    workers[cam_id] = w

    # siapkan meta untuk UrgencyEngine (jika belum ada)
    if cam_id not in camera_metas:
        camera_metas[cam_id] = CameraMeta(
            cam_id=cam_id,
            address=camera_meta.get("address","Alamat tidak diketahui"),
            city=camera_meta.get("city",""),
            district=camera_meta.get("district",""),
        )

    return jsonify({"ok": True, "cam_id": cam_id, "stream_url": stream_url, "zones": zones})

@app.post("/worker/stop")
def worker_stop():
    p = request.get_json(force=True)
    cam_id = p["cam_id"]
    w = workers.get(cam_id)
    if not w:
        return jsonify({"ok": False, "msg": "not running"}), 404
    w.stop()
    del workers[cam_id]
    return jsonify({"ok": True})

@app.get("/worker/status")
def worker_status():
    return jsonify({
        "running_cameras": list(workers.keys()),
        "sse_clients": len(clients),
    })

@app.get("/detector/tracking_data/<cam_id>")
def get_tracking_data(cam_id):
    """Get current tracking data for frontend display"""
    worker = workers.get(cam_id)
    if not worker:
        return jsonify({"error": "Camera not running"}), 404

    data = worker.get_tracking_data()
    data["timestamp"] = time.time()
    data["video_width"] = worker.frame_width
    data["video_height"] = worker.frame_height
    return jsonify(data)

# --------------------------------- Misc ---------------------------------
@app.get("/snaps/<path:name>")
def snaps(name):
    return send_from_directory(SNAP_DIR, name)

@app.get("/health")
def health():
    return {"status": "ok", "workers": list(workers.keys()), "clients": len(clients)}

# --------------------------------- Main ---------------------------------
if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
