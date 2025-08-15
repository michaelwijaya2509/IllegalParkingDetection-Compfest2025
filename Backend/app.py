# app.py (MODIFIED)

from flask import Flask, jsonify, Response, request, send_from_directory
from flask_cors import CORS
import threading
import queue
import json
import time
import os
import logging
from collections import deque

import cv2
import numpy as np
from math import sqrt

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ------------------------ Flask & CORS ------------------------
app = Flask(__name__)
CORS(app) # Izinkan semua origin untuk semua route
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("parklenz.app")

# ------------------------ SSE infra (Tidak diubah) --------------------------
history_events = deque(maxlen=int(os.environ.get("SSE_HISTORY_MAX", "500")))
clients: list[queue.Queue] = []
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

@app.route("/events")
def events():
    client_ip = request.remote_addr
    log.info(f"🟩 New SSE client: %s", client_ip)
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
            log.info("🔴 SSE client disconnected: %s", client_ip)
            with clients_lock:
                try:
                    clients.remove(client_queue)
                except ValueError:
                    pass
            raise
    return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ------------------------ Rule-based detector -----------------
workers: dict[str, "DetectorWorker"] = {}
SNAP_DIR = os.environ.get("SNAP_DIR", "snaps")
os.makedirs(SNAP_DIR, exist_ok=True)
MOVE_PX_THRESH = int(os.environ.get("MOVE_PX_THRESH", "10"))
MIN_STOP_S = float(os.environ.get("MIN_STOP_S", "300"))
TARGET_CLASSES = set(os.environ.get("TARGET_CLASSES", "car,truck,bus").split(','))

def point_in_polygon(x: int, y: int, polygon: list[list[int]]):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), (int(x), int(y)), False) >= 0

def load_zones(paths: list[str]):
    polys: list[dict] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "polygon" in data:
            polys.append({"name": data.get("name", os.path.basename(p)), "polygon": data["polygon"]})
        elif isinstance(data, list):
            polys.append({"name": os.path.basename(p), "polygon": data})
        else:
            raise ValueError(f"Zona format tidak dikenal: {p}")
    return polys

class DetectorWorker(threading.Thread):
    def __init__(self, cam_id: str, stream_url: str, zone_files: list[str], model_path: str = "yolo11n.pt", device: str | None = None):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.stream_url = stream_url
        self.zone_files = zone_files
        self.zones = load_zones(zone_files)
        self.stop_flag = threading.Event()
        self.device = device
        self.model_path = model_path
        self.track_state: dict[int, dict] = {}
        self.tracker = DeepSort(max_age=60, n_init=3)
        self.model: YOLO | None = None
        self.label_map: dict[int, str] = {}
        self.current_frame = None
        self.current_frame_lock = threading.Lock()
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.current_frame = None
        self.frontend_tracking_data: dict = {"tracks": [], "zones": self.zones}
        self.frontend_data_lock = threading.Lock()

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
        
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while not self.stop_flag.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            with self.current_frame_lock:
                self.current_frame = frame.copy()

            # 1) Deteksi
            results = self.model(frame, verbose=False)[0]
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if self.label_map.get(cls_id, "unknown") in TARGET_CLASSES:
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], float(box.conf[0]), cls_id))
            
            # 2) Tracking
            try: tracks = self.tracker.update_tracks(detections, frame=frame)
            except Exception as e:
                broadcast({"type": "track_error", "cam_id": self.cam_id, "error": str(e)}, also_store=False)
                continue
            
            now_ts = time.time()
            
            current_tracks_for_frontend = []

            # 3) Rule
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
                    st = {"last_pos": (cx, cy), "stationary_s": 0.0, "last_ts": now_ts, "zone_name": zone_name_hit}
                    self.track_state[tr.track_id] = st

                dx, dy = cx - st["last_pos"][0], cy - st["last_pos"][1]
                dist = sqrt(dx * dx + dy * dy)
                dt_s = now_ts - st["last_ts"]
                if dist < MOVE_PX_THRESH: st["stationary_s"] += dt_s
                else: st["stationary_s"], st["last_pos"] = 0.0, (cx, cy)
                st["last_ts"] = now_ts
                st["zone_name"] = zone_name_hit
                
                is_close_to_violation = st["stationary_s"] >= MIN_STOP_S * 0.8
                is_violation = st["stationary_s"] >= MIN_STOP_S
                
                current_tracks_for_frontend.append({
                    "track_id": tr.track_id,
                    "bbox": [x1, y1, x2, y2],
                    "class_name": cls_name,
                    "stationary_s": int(st["stationary_s"]),
                    "is_close_to_violation": is_close_to_violation,
                    "is_violation": is_violation,
                })
                
                # 4) Trigger pelanggaran (logic SSE tetap sama)
                if is_violation:
                    snap_path = os.path.join(SNAP_DIR, f"{self.cam_id}_{tr.track_id}_{int(now_ts)}.jpg")
                    snap_url = None
                    try:
                        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                        if crop.size > 0:
                            cv2.imwrite(snap_path, crop)
                            snap_url = f"/snaps/{os.path.basename(snap_path)}"
                    except Exception: pass
                    
                    event_payload = { "type": "violation", "cam_id": self.cam_id, "track_id": tr.track_id, "class": cls_name, "duration_s": int(st["stationary_s"]), "zone_name": st["zone_name"], "ts": int(now_ts), "snapshot_path": snap_path, "snapshot_url": snap_url }
                    broadcast(event_payload)
                    st["stationary_s"] = 0.0
            
            with self.frontend_data_lock:
                self.frontend_tracking_data["tracks"] = current_tracks_for_frontend
                
        cap.release()
        log.info("worker %s stopped", self.cam_id)

    def stop(self):
        self.stop_flag.set()

    def get_frame(self):
        with self.current_frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def get_tracking_data(self):
        with self.frontend_data_lock:
            return self.frontend_tracking_data.copy()

# ------------------------ Video Streaming Routes (Tidak diubah) ------------------------
def generate_frames(cam_id: str):
    worker = workers.get(cam_id)
    if not worker: return
    while cam_id in workers:
        frame = worker.get_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/video/<cam_id>')
def video_feed(cam_id):
    if cam_id not in workers: return jsonify({"error": "Camera not running"}), 404
    return Response(generate_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------------ Camera config (Perbaiki Path) ----------------------

CAMCFG_PATH = "Backend/config/cameras.json"
try:
    with open(CAMCFG_PATH, "r", encoding="utf-8") as f:
        CAMCFG = {c["cam_id"]: c for c in json.load(f)}
except FileNotFoundError:
    CAMCFG = {}
    log.warning("cameras.json not found at %s", CAMCFG_PATH)

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

# ------------------------ API: detectors ---------------------
@app.post("/detector/start_by_id")
def start_detector_by_id():
    p = request.get_json(force=True)
    cam_id = p["cam_id"]
    if cam_id in workers: return jsonify({"ok": False, "msg": "already running"}), 400
    cfg = CAMCFG.get(cam_id)
    if not cfg: return jsonify({"ok": False, "msg": f"cam_id {cam_id} not found in config"}), 404
    stream_url, zones = cfg["stream_url"], cfg["zones"]
    model = p.get("model", "yolo11n.pt")
    w = DetectorWorker(cam_id=cam_id, stream_url=stream_url, zone_files=zones, model_path=model)
    w.start()
    workers[cam_id] = w
    return jsonify({"ok": True, "cam_id": cam_id, "stream_url": stream_url, "zones": zones})

@app.post("/detector/stop")
def stop_detector():
    p = request.get_json(force=True)
    cam_id = p["cam_id"]
    w = workers.get(cam_id)
    if not w: return jsonify({"ok": False, "msg": "not running"}), 404
    w.stop()
    del workers[cam_id]
    return jsonify({"ok": True})

@app.get("/detector/status")
def detector_status():
    running_cams = {}
    for cam_id, worker in workers.items():
        cfg = CAMCFG.get(cam_id, {})
        running_cams[cam_id] = {
            "name": cfg.get("name"),
            "is_running": True,
            "stream_url": f"/video/{cam_id}"
        }
    return jsonify({
        "running_cameras": running_cams,
        "sse_clients": len(clients),
    })


@app.get("/cameras")
def cameras():
    camera_list = []
    for cam_id, cfg in CAMCFG.items():
        is_running = cam_id in workers
        camera_list.append({
            **cfg,
            "is_running": is_running,
            "stream_endpoint": f"/video/{cam_id}" if is_running else None
        })
    return jsonify(camera_list)
    
@app.post("/cameras/reload")
def cameras_reload():
    global CAMCFG
    try:
        with open(CAMCFG_PATH, "r", encoding="utf-8") as f:
            CAMCFG = {c["cam_id"]: c for c in json.load(f)}
        return jsonify({"ok": True, "count": len(CAMCFG)})
    except Exception as e: return jsonify({"ok": False, "error": str(e)}), 500

# ------------------------ API: misc --------------------------
@app.get("/snaps/<path:name>")
def snaps(name):
    return send_from_directory(SNAP_DIR, name)

@app.get("/health")
def health():
    return {"status": "ok", "workers": list(workers.keys()), "clients": len(clients)}

# ------------------------ Main -------------------------------
if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5001"))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host=host, port=port, debug=debug)