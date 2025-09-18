# control-api/app.py  — backend ringan (routing, config, incidents, analytics)
from flask import Flask, jsonify, request, redirect
from flask_cors import CORS
import subprocess
import sys
import os, json, time, logging
from datetime import datetime, timedelta
from collections import Counter
from slugify import slugify

import requests
from google.oauth2 import id_token
from google.auth.transport.requests import Request

import firebase_admin
from firebase_admin import credentials, db
from google.cloud import secretmanager

# ---------------- Base & Helpers ----------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("horusai.control")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def resolve_path(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(BASE_DIR, p))

def access_secret_version(project_id, secret_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    resp = client.access_secret_version(request={"name": name})
    return resp.payload.data.decode("UTF-8")

# ---------------- Env ----------------
PROJECT_ID       = os.environ.get("GOOGLE_PROJECT_ID", "horus-ai-468916")
SECRET_ID        = os.environ.get("FIREBASE_SECRET_ID", "firebase-realtimedb-credentials")
CAMCFG_PATH      = resolve_path(os.environ.get("CAMCFG_PATH", "config/cameras.json"))
INFER_BASE_URL   = "http://localhost:8080"             # e.g. https://infer-api-xxxxx.run.app
INFER_AUDIENCE   = os.environ.get("INFER_AUDIENCE", INFER_BASE_URL)  # Cloud Run URL as audience
INFER_UNAUTH     = os.environ.get("INFER_ALLOW_UNAUTH", "false").lower() == "true"

log.info(f"INFER_BASE_URL={INFER_BASE_URL}  INFER_AUDIENCE={INFER_AUDIENCE}  INFER_UNAUTH={INFER_UNAUTH}")


# ---------------- Firebase init (ringan) ----------------
try:
    cred_json = json.loads(access_secret_version(PROJECT_ID, SECRET_ID))
    cred = credentials.Certificate(cred_json)
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://horus-ai-468916-default-rtdb.asia-southeast1.firebasedatabase.app/"
    })
    fb_db_ref = db.reference()
    log.info("Firebase initialized in control-api.")
except Exception as e:
    log.exception("Failed Firebase init in control-api: %s", e)
    fb_db_ref = None

# ---------------- Cameras config ----------------
try:
    with open(CAMCFG_PATH, "r", encoding="utf-8") as f:
        CAMCFG = {c["cam_id"]: c for c in json.load(f)}
except FileNotFoundError:
    CAMCFG = {}
    log.warning("cameras.json not found at %s", CAMCFG_PATH)

# ---------------- Infer caller (Cloud Run auth) ----------------
def call_infer(path, method="GET", json_data=None, timeout=60):
    url = INFER_BASE_URL.rstrip("/") + path
    headers = {}

    # --- log buat verifikasi ---
    log.info(f"call_infer url={url} INFER_UNAUTH={INFER_UNAUTH}")

    # Jika unauth ATAU target localhost -> JANGAN ambil token
    need_auth = (not INFER_UNAUTH) and not (
        url.startswith("http://localhost") or url.startswith("http://127.0.0.1")
    )

    if need_auth:
        try:
            token = id_token.fetch_id_token(Request(), INFER_AUDIENCE)
            headers["Authorization"] = f"Bearer {token}"
        except Exception as e:
            log.warning("Token fetch failed (%s). Proceeding without Authorization header.", e)

    if method == "POST":
        r = requests.post(url, json=json_data, headers=headers, timeout=timeout)
    else:
        r = requests.get(url, headers=headers, timeout=timeout)

    # Log kalau non-2xx biar kebaca di terminal
    if not r.ok:
        log.error("Inference call %s %s failed: %s %s", method, url, r.status_code, r.text[:400])

    r.raise_for_status()
    return r.json()


# ---------------- Detectors (forward ke infer-api) ----------------
@app.post("/detector/start_by_id")
def start_detector_by_id():
    p = request.get_json(force=True)
    cam_id = p["cam_id"]
    cfg = CAMCFG.get(cam_id)
    if not cfg:
        return jsonify({"ok": False, "msg": f"cam_id {cam_id} not found in config"}), 404
    payload = {
        "cam_id": cam_id,
        "stream_url": cfg["stream_url"],
        "zones": cfg["zones"],
        "model": p.get("model", "yolo11n.pt"),
        "camera_meta": {
            "address": cfg.get("address", "Alamat tidak diketahui"),
            "city": cfg.get("city", ""),
            "district": cfg.get("district", "")
        }
    }
    try:
        resp = call_infer("/worker/start", "POST", payload)
        return jsonify(resp)
    except requests.HTTPError as e:
        body = e.response.text if e.response is not None else str(e)
        log.error("call_infer /worker/start HTTPError: %s", body)
        return jsonify({"ok": False, "error": "upstream_error", "detail": body}), 500
    except Exception as e:
        log.exception("call_infer /worker/start failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/detector/stop")
def stop_detector():
    p = request.get_json(force=True)
    cam_id = p["cam_id"]
    try:
        resp = call_infer("/worker/stop", "POST", {"cam_id": cam_id})
        return jsonify(resp)
    except requests.HTTPError as e:
        return jsonify({"ok": False, "error": str(e), "detail": e.response.text}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/detector/status")
def detector_status():
    try:
        s = call_infer("/worker/status", "GET")
        # peta ke format lama agar FE kamu tetap cocok
        running = {}
        for cam_id in s.get("running_cameras", []):
            cfg = CAMCFG.get(cam_id, {})
            running[cam_id] = {
                "name": cfg.get("name"),
                "is_running": True,
                "stream_url": f"/video/{cam_id}"  # kita redirect ke infer-api di route /video
            }
        return jsonify({"running_cameras": running, "sse_clients": s.get("sse_clients", 0)})
    except Exception as e:
        return jsonify({"running_cameras": {}, "sse_clients": 0, "error": str(e)}), 500

# ---------------- Video redirect (optional – satu domain untuk FE) -------------
@app.get("/video/<cam_id>")
def video_proxy(cam_id):
    # paling simple: redirect 302 ke infer-api MJPEG
    target = INFER_BASE_URL.rstrip("/") + f"/video/{cam_id}"
    return redirect(target, code=302)

# ---------------- Cameras CRUD (tetap di control-api) ----------------
@app.get("/cameras")
def cameras():
    out = []
    try:
        stat = call_infer("/worker/status", "GET")
        running_set = set(stat.get("running_cameras", []))
    except Exception:
        running_set = set()
    for cam_id, cfg in CAMCFG.items():
        out.append({
            **cfg,
            "is_running": cam_id in running_set,
            "stream_endpoint": f"/video/{cam_id}" if cam_id in running_set else None
        })
    return jsonify(out)

@app.post("/cameras/add")
def add_camera():
    global CAMCFG
    data = request.get_json(force=True)
    cam_id = f"{slugify(data['cameraName'])}-{int(time.time())}"
    new_cfg = {
        "cam_id": cam_id,
        "name": data["cameraName"],
        "address": data["address"],
        "stream_url": data["streamUrl"],
        "zones": data.get("zones", [])
    }
    with open(CAMCFG_PATH, "r+", encoding="utf-8") as f:
        try:
            all_cams = json.load(f)
        except json.JSONDecodeError:
            all_cams = []
        all_cams.append(new_cfg)
        f.seek(0); json.dump(all_cams, f, indent=2); f.truncate()
    CAMCFG = {c["cam_id"]: c for c in all_cams}
    return jsonify({"ok": True, "cam_id": cam_id})

@app.post("/cameras/delete")
def delete_camera():
    global CAMCFG
    cam_id = request.get_json(force=True).get("cam_id")
    if not cam_id: return jsonify({"ok": False, "error": "cam_id required"}), 400
    with open(CAMCFG_PATH, "r", encoding="utf-8") as f:
        all_cams = json.load(f)
    new_list = [c for c in all_cams if c.get("cam_id") != cam_id]
    with open(CAMCFG_PATH, "w", encoding="utf-8") as f:
        json.dump(new_list, f, indent=2); f.truncate()
    CAMCFG = {c["cam_id"]: c for c in new_list}
    return jsonify({"ok": True})

@app.post("/cameras/reload")
def cameras_reload():
    global CAMCFG
    with open(CAMCFG_PATH, "r", encoding="utf-8") as f:
        CAMCFG = {c["cam_id"]: c for c in json.load(f)}
    return jsonify({"ok": True, "count": len(CAMCFG)})

# ---------------- Incidents & Analytics (tetap di control-api) ----------------
@app.get("/incidents/pending")
def get_pending_incidents():
    try:
        incidents = fb_db_ref.child('PendingIncidents').get()
        return jsonify([] if not incidents else list(incidents.values()))
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.get("/incidents/approved")
def get_approved_incidents():
    try:
        incidents = fb_db_ref.child('ApprovedIncidents').get()
        return jsonify([] if not incidents else list(incidents.values()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/incident/accept")
def accept_incident():
    payload = request.get_json(force=True)
    inc = payload.get("incident_data")
    if not inc or "event" not in inc or "event_id" not in inc["event"]:
        return jsonify({"ok": False, "error": "Invalid incident data"}), 400
    iid = inc["event"]["event_id"]
    try:
        fb_db_ref.child('ApprovedIncidents').child(iid).set(inc)
        fb_db_ref.child('PendingIncidents').child(iid).delete()
        return jsonify({"ok": True, "incident_id": iid})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/incident/decline")
def decline_incident():
    payload = request.get_json(force=True)
    inc = payload.get("incident_data")
    if not inc or "event" not in inc or "event_id" not in inc["event"]:
        return jsonify({"ok": False, "error": "Invalid incident data"}), 400
    iid = inc["event"]["event_id"]
    try:
        fb_db_ref.child('DeclinedIncidents').child(iid).set(inc)
        fb_db_ref.child('PendingIncidents').child(iid).delete()
        return jsonify({"ok": True, "incident_id": iid})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/analytics/summary")
def analytics_summary():
    try:
        all_incidents_dict = fb_db_ref.child('ApprovedIncidents').get()
        if not all_incidents_dict:
            # hit infer for active count
            try:
                status = call_infer("/worker/status")
                active = len(status.get("running_cameras", []))
            except Exception:
                active = 0
            return jsonify({
                "totalApprovedIncidents": 0,
                "todayApprovedIncidents": 0,
                "activeCameras": active,
                "incidentTrends": [], "locationHotspots": [],
                "violationTypes": [], "commonReasons": []
            })
        all_incidents = list(all_incidents_dict.values())
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        total = len(all_incidents)
        today = 0
        dates, locs, cats, reasons = [], [], [], []
        for inc in all_incidents:
            try:
                t = datetime.fromisoformat(inc["event"]["started_at"])
                if t >= today_start: today += 1
                dates.append(t.date())
                locs.append(inc.get("address", ""))
                ld = inc.get("llm_data", {})
                if "category" in ld: cats.append(ld["category"])
                if "reasons" in ld: reasons.extend(ld["reasons"])
            except Exception:
                pass
        from collections import Counter
        dc = Counter(dates)
        trends = [{"date": (now - timedelta(days=i)).date().isoformat(),
                   "count": dc.get((now - timedelta(days=i)).date(), 0)} for i in range(6, -1, -1)]
        hotspots = [{"location": l, "count": c} for l, c in Counter(locs).most_common(5)]
        vtypes   = [{"category": c, "count": n} for c, n in Counter(cats).most_common(5)]
        common   = [{"reason": r, "count": n} for r, n in Counter(reasons).most_common(5)]
        try:
            status = call_infer("/worker/status")
            active = len(status.get("running_cameras", []))
        except Exception:
            active = 0
        return jsonify({
            "totalApprovedIncidents": total,
            "todayApprovedIncidents": today,
            "activeCameras": active,
            "incidentTrends": trends,
            "locationHotspots": hotspots,
            "violationTypes": vtypes,
            "commonReasons": common
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================= MCP : Finding Events API ========================
@app.post("/events/refresh_cache")
def refresh_upcoming_events_cache():

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        return jsonify({"error": "GEMINI_API_KEY variable not set."}), 500

    try:
        log.info("Starting expensive MCP event fetch to refresh cache...")
        python_executable = sys.executable
        client_script_path = resolve_path("utils/mcp_events_client.py")

        process = subprocess.run(
            [python_executable, client_script_path, "--gemini-api-key", gemini_api_key],
            capture_output=True, text=True, check=True, timeout=300
        )

        events_data = json.loads(process.stdout)
        if "error" in events_data:
             raise Exception(events_data["error"])

        # Hapus cache lama dan simpan data baru ke Firebase
        log.info("MCP fetch successful. Clearing old cache and saving new data to Firebase...")
        fb_db_ref.child('UpcomingEvents').delete()
        fb_db_ref.child('UpcomingEvents').set(events_data)
        
        # Simpan timestamp update
        timestamp = datetime.now().isoformat()
        fb_db_ref.child('UpcomingEvents_meta').child('last_updated').set(timestamp)
        
        log.info("Firebase cache updated successfully.")
        return jsonify({"ok": True, "message": "Event cache refreshed successfully.", "updated_at": timestamp})

    except Exception as e:
        log.exception("Failed to refresh event cache: %s", e)
        return jsonify({"error": "Failed to refresh event cache.", "details": str(e)}), 500

@app.get("/events/upcoming")
def get_upcoming_events_from_cache():
  
    try:
        events_data = fb_db_ref.child('UpcomingEvents').get()
        if not events_data:
            return jsonify({"all_events": [], "jakarta_events": [], "error": "Cache is empty."})
        return jsonify(events_data)
    except Exception as e:
        log.exception("Failed to get events from cache: %s", e)
        return jsonify({"error": "Could not retrieve events from Firebase.", "details": str(e)}), 500

@app.get("/events/cache_status")
def get_events_cache_status():
  
    try:
        timestamp = fb_db_ref.child('UpcomingEvents_meta').child('last_updated').get()
        return jsonify({"last_updated": timestamp})
    except Exception as e:
        return jsonify({"error": "Could not retrieve cache status.", "details": str(e)}), 500


@app.get("/health")
def health():
    try:
        s = call_infer("/worker/status", "GET")
        active = len(s.get("running_cameras", []))
    except Exception:
        active = 0
    return {"status": "ok", "active_workers": active}

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST","0.0.0.0"),
            port=int(os.environ.get("PORT","5001")),
            debug=os.environ.get("FLASK_DEBUG","true").lower()=="true")
