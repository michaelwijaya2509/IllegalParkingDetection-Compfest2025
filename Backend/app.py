"""
curl -X POST http://localhost:5000/detector/start_by_id \
  -H "Content-Type: application/json" \
  -d '{"cam_id":"viet1"}'

"""

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
CORS(app, resources={r"/events": {"origins": "*"}})
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("parklenz.app")

# ------------------------ SSE infra --------------------------
# Catatan: untuk MVP, kita simpan history di memori. Frontend yang connect
# pertama kali akan menerima replay dari history ini, lalu event baru via SSE.
history_events = deque(maxlen=int(os.environ.get("SSE_HISTORY_MAX", "500")))
clients: list[queue.Queue] = []
clients_lock = threading.Lock()


def broadcast(event: dict, also_store: bool = True):
    """Siarkan satu event ke semua klien SSE aktif."""
    if also_store:
        history_events.append(event)
    # broadcast non-blocking ke semua klien
    with clients_lock:
        for q in list(clients):
            try:
                q.put_nowait(event)
            except Exception:
                # queue penuh/klien bermasalah → abaikan agar tidak ganggu klien lain
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
            # Replay event lama
            for ev in list(history_events):
                yield f"data: {json.dumps(ev)}\n\n"
            # Kirim event baru sampai klien putus
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

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------ Rule-based detector -----------------
# 1 kamera = 1 worker thread yang memproses stream/video dan mengirim event "violation" via SSE.
workers: dict[str, "DetectorWorker"] = {}

SNAP_DIR = os.environ.get("SNAP_DIR", "snaps")
os.makedirs(SNAP_DIR, exist_ok=True)

MOVE_PX_THRESH = int(os.environ.get("MOVE_PX_THRESH", "10"))   # ambang gerak (px)
MIN_STOP_S = float(os.environ.get("MIN_STOP_S", "300"))        # durasi diam (detik) → default 5 menit
TARGET_CLASSES = set(os.environ.get("TARGET_CLASSES", "car,truck,bus").split(','))


def point_in_polygon(x: int, y: int, polygon: list[list[int]]):
    return (
        cv2.pointPolygonTest(np.array(polygon, np.int32), (int(x), int(y)), False) >= 0
    )


def load_zones(paths: list[str]):
    polys: list[dict] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # dukung dua format: list koordinat atau object dengan {name, polygon}
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

        # state per track
        self.track_state: dict[int, dict] = {}  # track_id -> {last_pos, stationary_s, last_ts, zone_name}
        self.tracker = DeepSort(max_age=60, n_init=3)
        self.model: YOLO | None = None
        self.label_map: dict[int, str] = {}

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
            try:
                self.model.to("cuda")
            except Exception:
                pass

        self.label_map = self.model.names

        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            broadcast({"type": "stream_error", "cam_id": self.cam_id, "msg": "cannot open stream"}, also_store=False)
            return

        colors = {
            "car": (255, 0, 0),
            "truck": (255, 0, 0),
            "bus": (255, 0, 0),
            "motorcycle": (0, 255, 0),
            "bicycle": (0, 255, 0),
            "person": (0, 0, 255),
        }

        while not self.stop_flag.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                # Jika file lokal selesai, akhiri thread. Jika RTSP drop, break juga untuk sederhana.
                break

            # 1) Deteksi
            results = self.model(frame, verbose=False)[0]
            detections = []
            for box in results.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.label_map.get(cls_id, "unknown")
                if label not in colors:  # filter kelas non-target dari visualisasi
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, cls_id))

            # 2) Tracking
            try:
                tracks = self.tracker.update_tracks(detections, frame=frame)
            except Exception as e:
                broadcast({"type": "track_error", "cam_id": self.cam_id, "error": str(e)}, also_store=False)
                continue

            now_ts = time.time()

            # 3) Rule: di red-zone & diam berapa lama
            for tr in tracks:
                if not tr.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, tr.to_ltrb())
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                cls_id = tr.det_class
                cls_name = self.label_map.get(cls_id, "unknown")

                # hanya kendaraan target
                if cls_name not in TARGET_CLASSES:
                    # bersihkan state jika ada
                    if tr.track_id in self.track_state:
                        self.track_state.pop(tr.track_id, None)
                    continue

                # cek red-zone
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
                    st = {
                        "last_pos": (cx, cy),
                        "stationary_s": 0.0,
                        "last_ts": now_ts,
                        "zone_name": zone_name_hit,
                    }
                    self.track_state[tr.track_id] = st

                # update waktu diam berbasis jarak
                dx = cx - st["last_pos"][0]
                dy = cy - st["last_pos"][1]
                dist = sqrt(dx * dx + dy * dy)

                dt_s = now_ts - st["last_ts"]
                if dist < MOVE_PX_THRESH:
                    st["stationary_s"] += dt_s
                else:
                    st["stationary_s"] = 0.0
                    st["last_pos"] = (cx, cy)
                st["last_ts"] = now_ts
                st["zone_name"] = zone_name_hit

                # 4) Trigger pelanggaran (diam >= ambang)
                if st["stationary_s"] >= MIN_STOP_S:
                    # snapshot opsional
                    snap_path = os.path.join(
                        SNAP_DIR, f"{self.cam_id}_{tr.track_id}_{int(now_ts)}.jpg"
                    )
                    snap_url = None
                    try:
                        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                        if crop.size > 0:
                            cv2.imwrite(snap_path, crop)
                            snap_url = f"/snaps/{os.path.basename(snap_path)}"
                    except Exception:
                        pass

                    event_payload = {
                        "type": "violation",
                        "cam_id": self.cam_id,
                        "track_id": tr.track_id,
                        "class": cls_name,
                        "duration_s": int(st["stationary_s"]),
                        "zone_name": st["zone_name"],
                        "ts": int(now_ts),
                        "snapshot_path": snap_path,
                        "snapshot_url": snap_url,
                    }
                    broadcast(event_payload)

                    # reset agar tidak spam tiap frame
                    st["stationary_s"] = 0.0

        cap.release()
        log.info("worker %s stopped", self.cam_id)

    def stop(self):
        self.stop_flag.set()


# ------------------------ Camera config ----------------------
# config/cameras.json skema contoh:
# [
#   {
#     "cam_id": "CCTV-BDG-01",
#     "name": "Dago - Halte Utama",
#     "address": "Jl. Ir. H. Juanda No. 1, Bandung",
#     "stream_url": "data/videos/vietnam2.mp4",
#     "zones": ["data/zones/CCTV-BDG-01_zone1.json"]
#   }
# ]

CAMCFG_PATH = os.environ.get("CAMERAS_JSON", "config/cameras.json")
try:
    with open(CAMCFG_PATH, "r", encoding="utf-8") as f:
        CAMCFG = {c["cam_id"]: c for c in json.load(f)}
except FileNotFoundError:
    CAMCFG = {}
    log.warning("cameras.json not found at %s — you can still start detectors via /detector/start", CAMCFG_PATH)


# ------------------------ API: detectors ---------------------
@app.post("/detector/start")
def start_detector():
    """Start detektor dengan payload eksplisit (stream_url & zones diberikan)."""
    p = request.get_json(force=True)
    cam_id = p["cam_id"]
    if cam_id in workers:
        return jsonify({"ok": False, "msg": "already running"}), 400
    stream_url = p["stream_url"]
    zones = p["zones"]
    model = p.get("model", "yolo11n.pt")

    w = DetectorWorker(cam_id=cam_id, stream_url=stream_url, zone_files=zones, model_path=model)
    w.start()
    workers[cam_id] = w
    return jsonify({"ok": True, "cam_id": cam_id, "stream_url": stream_url, "zones": zones})


@app.post("/detector/start_by_id")
def start_detector_by_id():
    """Start detektor hanya dengan cam_id yang ada di config/cameras.json."""
    p = request.get_json(force=True)
    cam_id = p["cam_id"]
    if cam_id in workers:
        return jsonify({"ok": False, "msg": "already running"}), 400
    cfg = CAMCFG.get(cam_id)
    if not cfg:
        return jsonify({"ok": False, "msg": f"cam_id {cam_id} not found in config"}), 404

    stream_url = cfg["stream_url"]
    zones = cfg["zones"]
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
    if not w:
        return jsonify({"ok": False, "msg": "not running"}), 404
    w.stop()
    del workers[cam_id]
    return jsonify({"ok": True})


@app.get("/detector/status")
def detector_status():
    return jsonify({
        "running": list(workers.keys()),
        "clients": len(clients),
        "history_events": len(history_events),
    })


# ------------------------ API: cameras cfg -------------------
@app.get("/cameras")
def cameras():
    return jsonify(list(CAMCFG.values()))


@app.post("/cameras/reload")
def cameras_reload():
    global CAMCFG
    try:
        with open(CAMCFG_PATH, "r", encoding="utf-8") as f:
            CAMCFG = {c["cam_id"]: c for c in json.load(f)}
        return jsonify({"ok": True, "count": len(CAMCFG)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ------------------------ API: misc --------------------------
@app.post("/test-event")
def test_event():
    data = request.get_json(force=True)
    broadcast(data)
    return jsonify({"status": "ok"}), 200


@app.get("/snaps/<path:name>")
def snaps(name):
    return send_from_directory(SNAP_DIR, name)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "workers": list(workers.keys()),
        "clients": len(clients),
        "history_events": len(history_events),
    }


# ------------------------ Main -------------------------------
if __name__ == "__main__":
    # Terkadang OpenMP crash pada beberapa environment Windows
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host=host, port=port, debug=debug)
