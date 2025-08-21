# app.py (RELATIVE-PATH READY)

from flask import Flask, jsonify, Response, request, send_from_directory
from flask_cors import CORS
import threading
import queue
import json
import time
import datetime as dt
import subprocess
import os
import logging
import torch
import torch.nn as nn
from collections import deque, Counter
import yt_dlp
import requests
import math
from slugify import slugify 
import sys
import pickle

from urllib.parse import urlparse, urljoin
from logics.cnn_macet import check_macet_cnn
from logics.cek_supir_keluar import check_driver_exit, crop_with_margin, preprocess_frames_for_inference
from logics.urgency_engine import UrgencyEngine, ViolationEvent, CameraMeta, ScoredEvent

from datetime import datetime, timedelta
from dataclasses import asdict
import cv2
import numpy as np
from math import sqrt
import torch.nn.functional as F


from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import firebase_admin
from firebase_admin import credentials, db
from google.cloud import storage, secretmanager
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights 


class MultiHeadAttention(nn.Module):

    """ Multi-Head Attention buat temporal modeling """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # x: (B, T, d_model)
        batch_size, seq_len = x.size(0), x.size(1)
        residual = x
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Outputnya projection dan residual
        output = self.w_o(attention_output)
        output = self.dropout(output)
        return self.layer_norm(output + residual)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V)

class FeatureAdapter(nn.Module):

    """ Adapter layer buat mapping CNN features ke sequence modeling """

    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x):
        # x: (B, T, input_dim) atau (B*T, input_dim)
        original_shape = x.shape
        if len(original_shape) == 3:
            B, T, D = original_shape
            x = x.view(B*T, D)
            adapted = self.adapter(x) + self.residual(x)
            return adapted.view(B, T, -1)
        else:
            return self.adapter(x) + self.residual(x)
        
class SequenceHeadLSTM(nn.Module):

    """
    Enhanced Head temporal LSTM dengan Multi-Head Attention, Bidirectional LSTM, dan Residual connections.
    Input :
      feats   : (B, T, D)  -> fitur dari CNN per frame
      lengths : (B,)       -> panjang asli tiap sequence (<= T)
    Output:
      logits  : (B, num_classes)
    """

    def __init__(self, feat_dim: int = 1536, adapter_dim: int = 512, hidden: int = 512, 
                 layers: int = 1, num_classes: int = 2, bidirectional: bool = True, 
                 dropout: float = 0.3, num_attention_heads: int = 8, use_residual: bool = True):
        super().__init__()
        
        # Feature adapter
        self.feature_adapter = FeatureAdapter(feat_dim, adapter_dim, dropout)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            d_model=adapter_dim, 
            num_heads=num_attention_heads, 
            dropout=dropout
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=adapter_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=(dropout if layers > 1 else 0.0),
        )
        
        # Residual connection params
        self.use_residual = use_residual
        if use_residual:
            lstm_out_dim = hidden * (2 if bidirectional else 1)
            self.residual_proj = nn.Linear(adapter_dim, lstm_out_dim)
        
        out_dim = hidden * (2 if bidirectional else 1)
        
        # Enhanced classifier dengan lebih banyak layer
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def create_attention_mask(self, lengths, max_len):

        """ attention mask untuk variable length sequences """

        batch_size = len(lengths)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=lengths.device)
        for i, length in enumerate(lengths):
            mask[i, :length] = True
        return mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T) untuk broadcasting

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # feats: (B, T, D), lengths harus di CPU untuk pack
        B, T, D = feats.shape
        
        # Feature adaptation
        adapted_feats = self.feature_adapter(feats)  # (B, T, adapter_dim)
        
        # Multi-head attention dengan mask
        attention_mask = self.create_attention_mask(lengths, T)
        attended_feats = self.attention(adapted_feats, mask=attention_mask)  # (B, T, adapter_dim)
        
        # LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            attended_feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, (h_n, _) = self.lstm(packed)  # h_n: (num_layers * num_dirs, B, H)
        
        # Unpack untuk residual connection
        if self.use_residual:
            unpacked_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            seq_mask = torch.arange(T, device=feats.device).unsqueeze(0) < lengths.unsqueeze(1)
            seq_mask = seq_mask.unsqueeze(-1).float()  # (B, T, 1)
            
            pooled_attended = (attended_feats * seq_mask).sum(dim=1) / lengths.unsqueeze(1).float()
            pooled_attended = self.residual_proj(pooled_attended)  # (B, lstm_out_dim)

        # final hidden state
        if self.lstm.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2H)
        else:
            h_last = h_n[-1]                               # (B, H)

        # Add residual connection
        if self.use_residual:
            h_last = h_last + pooled_attended

        logits = self.classifier(h_last)                   # (B, num_classes)
        return logits
    

class CNNFeatureExtractor(nn.Module):

    """
    Ekstraktor fitur frame berbasis EfficientNet-B7 (tanpa classifier).
    - Jika input (B, C, H, W) -> output (B, D)
    - Jika input (B, T, C, H, W) -> output (B, T, D)
    D = cnn_feature_size (Eff-B7 = 2560).
    """
    
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()  # keluarkan vektor fitur akhir (B, 1536)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:  # (B, C, H, W)
            return self.backbone(x)  # (B, D)

        elif x.dim() == 5:  # (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            feats = self.backbone(x_flat)         # (B*T, D)
            D = feats.shape[1]
            return feats.view(B, T, D)            # (B, T, D)

        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
class CNNLSTMModel(nn.Module):

    """
    Enhanced CNN+LSTM model dengan Multi-Head Attention dan Feature Adapter.
    Input : x (B,T,C,H,W), lengths (B,)
    Output: logits (B,num_classes)
    """

    def __init__(self, num_classes=2, cnn_feature_size=1536, adapter_dim=512,
                 lstm_hidden=512, lstm_layers=1, bidirectional=True, dropout=0.3,
                 num_attention_heads=8, use_residual=True, freeze_backbone=True):
        super().__init__()
        self.extractor = CNNFeatureExtractor(freeze_backbone=freeze_backbone)
        self.head = SequenceHeadLSTM(
            feat_dim=cnn_feature_size,
            adapter_dim=adapter_dim,
            hidden=lstm_hidden,
            layers=lstm_layers,
            num_classes=num_classes,
            bidirectional=bidirectional,
            dropout=dropout,
            num_attention_heads=num_attention_heads,
            use_residual=use_residual
        )

    def forward(self, x_btchw: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        feats = self.extractor(x_btchw)            # (B, T, D)
        logits = self.head(feats, lengths)         # (B, num_classes)
        return logits
    



# ======================== Load Credentials from Secret Manager ================= #
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
    
    credentials_info = json.loads(credentials_json_string)

    # ======================== Firebase Configurations ====================== #
    cred = credentials.Certificate(credentials_info)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://horus-ai-468916-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })
    fb_db_ref = db.reference()
    logging.info("Successfully initialized Firebase from Secret Manager.")

    # ======================== GCS Configurations ======================= #
    GCS_BUCKET_NAME =  os.environ.get("GCS_BUCKET_NAME")
    storage_client = storage.Client.from_service_account_info(credentials_info)
    gcs_bucket = storage_client.bucket(GCS_BUCKET_NAME)
    logging.info(f"Successfully connected to GCS bucket from Secret Manager: {GCS_BUCKET_NAME}")

except Exception as e:
    logging.error(f"Failed to load credentials from Secret Manager: {e}")
    fb_db_ref = None
    gcs_bucket = None
    storage_client = None


# ======================== Path Helpers ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(p):
    """
    Return absolute path anchored at this app.py location (unless already absolute).
    Also accepts webcam index passed as int or digit-string ("0", "1").
    """
    if isinstance(p, int):
        return p
    if isinstance(p, str) and p.isdigit():
        return int(p)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(BASE_DIR, p))

# ------------------------ Flask & CORS ------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("horusai.app")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Izinkan semua origin untuk semua route

MODEL_EVENT_PATH = resolve_path("best_cnnlstm_stage1.pkl")
MODEL_EVENT_INFERENCE = None

try:
    log.info("==============================================")
    log.info(">>> ATTEMPTING TO LOAD DRIVER EXIT MODEL... <<<")
    
     # Gunakan torch.load dan map_location
    MODEL_EVENT_INFERENCE = torch.load(
    MODEL_EVENT_PATH, 
        map_location=torch.device('cpu'), 
        weights_only=False  # <-- TAMBAHKAN ARGUMEN INI
    )
    
    # Best practice: setel model ke mode evaluasi setelah dimuat
    MODEL_EVENT_INFERENCE.eval()
    
    log.info(">>> MODEL LOADED SUCCESSFULLY! Object type: %s", type(MODEL_EVENT_INFERENCE))
    log.info("==============================================")

except Exception as e:
    log.error("==============================================")
    log.error(">>> FAILED TO LOAD DRIVER EXIT MODEL <<<")
    log.error("Path: %s", MODEL_EVENT_PATH)
    log.error("Error Details: %s", e, exc_info=True) # exc_info=True akan memberikan traceback lengkap
    log.error("==============================================")


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
# ini list of workers yang berbentuk dictionary dan setiap worker adalah object dari DetectorWorker class
workers: dict[str, "DetectorWorker"] = {}
SNAP_DIR = resolve_path(os.environ.get("SNAP_DIR", "snaps"))
os.makedirs(SNAP_DIR, exist_ok=True)
MOVE_PX_THRESH = int(os.environ.get("MOVE_PX_THRESH", "30"))
MIN_STOP_S = float(os.environ.get("MIN_STOP_S", "10000"))
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
    log.info(f"[THREAD] Memproses {len(cropped_frames)} frames untuk track ID {track_id}...")
    
    is_driver_exit = check_driver_exit(cropped_frames, model)
    
    if is_driver_exit:
        log.info(f"[THREAD] Terdeteksi supir keluar untuk track ID: {track_id}")
        track_state_obj["driver_exited"] = True
        if full_frames:
            track_state_obj["evidence_frame"] = full_frames[len(full_frames) // 2]
    else:
        log.info(f"[THREAD] Tidak terdeteksi supir keluar untuk track ID: {track_id}")

    track_state_obj["event_check_running"] = False


# Setiap worker adalah object dari DetectorWorker. Setiap worker akan menangani satu kamera

class DetectorWorker(threading.Thread):
    def __init__(self, cam_id: str, stream_url: str, zone_files: list[str], model_path: str = "yolo11n.pt", device: str | None = None):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.stream_url = resolve_path(stream_url)
        self.zone_files = zone_files
        self.zones = load_zones(zone_files)
        self.stop_flag = threading.Event()
        self.device = device
        self.model_path = model_path
        self.track_state: dict[int, dict] = {}
        self.tracker = DeepSort(max_age=200, n_init=3)
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
            try:
                self.model.to("cuda")
            except Exception:
                pass
        self.label_map = self.model.names
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            broadcast({"type": "stream_error", "cam_id": self.cam_id, "msg": "cannot open stream"}, also_store=False)
            return
    
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        COOLDOWN_TIME = 1800 # 1800 frames, 60 seconds
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
                if not tr.is_confirmed():
                    continue

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
                dist = sqrt(dx * dx + dy * dy)
                dt_s = now_ts - st["last_ts"]

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
                            print(f"Macet terdeteksi, timer untuk track ID {tr.track_id} dijeda.")
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
                    
                    log.info(f"Mulai pengecekan supir keluar untuk track id: {tr.track_id}")
                    sequence_copy = list(st["frame_sequence"])
                    crops_for_model = [item[0] for item in sequence_copy]
                    full_frames_for_evidence = [item[1] for item in sequence_copy]
                    checker_thread = threading.Thread(
                        target=run_driver_exit_check,
                        args=(self.cam_id, tr.track_id, crops_for_model, full_frames_for_evidence,  MODEL_EVENT_INFERENCE, st)
                    )
                    checker_thread.start()

                is_violation_by_time = st["stationary_s"] >= MIN_STOP_S

                if is_violation_by_time and not st.get("violation_reported", False):
                    st["violation_reported"] = True

                    reason = "Supir Keluar" if st.get("driver_exited", False) else "Waktu Parkir > 5 Menit"
                    print(f"Terdeteksi ILLEGAL PARKING (Alasan: {reason})")

                    snap_url = None
                    try:

                        snapshot_frame = st.get("evidence_frame") if st.get("driver_exited") else frame

                        snap_x1, snap_y1, snap_x2, snap_y2 = map(int, tr.to_ltrb())

                        crop = snapshot_frame[max(0, snap_y1):max(0, snap_y2), max(0, snap_x1):max(0, snap_x2)]
                        if crop.size > 0 and gcs_bucket:
                            success, buffer = cv2.imencode('.jpg', crop)
                            if success:
                                blob_name = f"snapshots/{self.cam_id}{tr.track_id}{int(now_ts)}.jpg"
                                blob = gcs_bucket.blob(blob_name)
                                blob.upload_from_string(
                                    buffer.tobytes(),
                                    content_type='image/jpeg'
                                )
                                snap_url = blob.public_url
                                log.info(f"Snapshot uploaded to GCS: {snap_url}")
                            else:
                                log.error("Failed to encode snapshot to JPEG format.")
                    except Exception as e:
                        log.exception(f"❌ Failed to upload snapshot to GCS: {e}")
                        pass

                    event_id = f"{self.cam_id}-{tr.track_id}-{int(now_ts)}"
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
                            fb_db_ref.child('PendingIncidents').child(event_id).set(scored_dict)
                            log.info(f"Successfully pushed incident {event_id} to Firebase. ")
                        except Exception as e:
                            log.error(f"❌ Failed to push incident {event_id} to Firebase: {e}")

                        broadcast({"type": "violation_event", "data": scored_dict}, also_store=True)

                        print("LLM Output: ", scored)
                
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
    if not worker:
        return
    while cam_id in workers:
        frame = worker.get_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/video/<cam_id>')
def video_feed(cam_id):
    if cam_id not in workers:
        return jsonify({"error": "Camera not running"}), 404
    return Response(generate_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------------ Camera config (Perbaiki Path) ----------------------
CAMCFG_PATH = resolve_path(os.environ.get("CAMCFG_PATH", "config/cameras.json"))
log.info("Loading camera config from: %s", CAMCFG_PATH)
try:
    with open(CAMCFG_PATH, "r", encoding="utf-8") as f:
        CAMCFG = {c["cam_id"]: c for c in json.load(f)}
except FileNotFoundError:
    CAMCFG = {}
    log.warning("cameras.json not found at %s", CAMCFG_PATH)

# metadata kamera buat UrgencyEngine
camera_metas = {
    cam_id: CameraMeta(
        cam_id=cam_id,
        address=cfg.get("address", "Alamat tidak diketahui"),
        city=cfg.get("city", ""),
        district=cfg.get("district", "")
    ) for cam_id, cfg in CAMCFG.items()
}

urgency_engine = UrgencyEngine(cameras=camera_metas)

# @app.get("/detector/tracking_data/<cam_id>")
# def get_tracking_data(cam_id):
#     """Get current tracking data for frontend display"""
#     worker = workers.get(cam_id)
#     if not worker:
#         return jsonify({"error": "Camera not running"}), 404

#     data = worker.get_tracking_data()
#     data["timestamp"] = time.time()
#     data["video_width"] = worker.frame_width
#     data["video_height"] = worker.frame_height
#     return jsonify(data)

# ------------------------ API: detectors ---------------------
@app.post("/detector/start_by_id")
def start_detector_by_id():
    p = request.get_json(force=True)
    cam_id = p["cam_id"]
    if cam_id in workers:
        return jsonify({"ok": False, "msg": "already running"}), 400
    cfg = CAMCFG.get(cam_id)
    if not cfg:
        return jsonify({"ok": False, "msg": f"cam_id {cam_id} not found in config"}), 404
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
    if not w:
        return jsonify({"ok": False, "msg": "not running"}), 404
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

# Route buat return list of cameras ke frontend berdasarkan camera di cameras_json
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



@app.post("/cameras/add")
def add_camera():
    global CAMCFG
    try:
        data = request.get_json(force=True)

        cam_id = f"{slugify(data['cameraName'])}-{int(time.time())}"
        
        zones_dir = resolve_path("data/zones")
        os.makedirs(zones_dir, exist_ok=True)
        zone_filename = f"{cam_id}-zone.json"
        zone_filepath = os.path.join(zones_dir, zone_filename)
        
        polygon_data = [[p['x'], p['y']] for p in data['zonePolygon']]
        
        zone_content = {
            "name": f"{data['cameraName']} Zone",
            "polygon": polygon_data
        }
        with open(zone_filepath, "w", encoding="utf-8") as f:
            json.dump(zone_content, f, indent=2)
        log.info("Saved new zone file to: %s", zone_filepath)


        new_camera_config = {
            "cam_id": cam_id,
            "name": data["cameraName"],
            "address": data["address"],
            "stream_url": data["streamUrl"],
            "zones": [f"data/zones/{zone_filename}"] 
        }

        with open(CAMCFG_PATH, "r+", encoding="utf-8") as f:
            # Read all cameras
            all_cameras = json.load(f)
            all_cameras.append(new_camera_config)

            f.seek(0)

            json.dump(all_cameras, f, indent=2)
  
            f.truncate()
        
        CAMCFG = {c["cam_id"]: c for c in all_cameras}
        log.info("Successfully added new camera '%s' and reloaded config.", cam_id)
        
        return jsonify({"ok": True, "cam_id": cam_id, "message": "Camera added successfully."})

    except Exception as e:
        log.exception("Failed to add new camera: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500
    



@app.post("/cameras/reload")
def cameras_reload():
    global CAMCFG
    try:
        with open(CAMCFG_PATH, "r", encoding="utf-8") as f:
            CAMCFG = {c["cam_id"]: c for c in json.load(f)}
        return jsonify({"ok": True, "count": len(CAMCFG)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ------------------------ Incidents API --------------------- 

@app.get("/incidents/pending")
def get_pending_incidents():
    try:
        incidents = fb_db_ref.child('PendingIncidents').get()
        if not incidents:
            return jsonify([])
        return jsonify(list(incidents.values()))
    except Exception as e:
        log.error(f"Error fetching pending incidents from Firebase: {e}")
        return jsonify({"error": str(e)}), 500

@app.get("/incidents/approved")
def get_approved_incidents():
    try:
        incidents = fb_db_ref.child('ApprovedIncidents').get()
        if not incidents:
            return jsonify([])
        return jsonify(list(incidents.values()))
    except Exception as e:
        log.error(f"Error fetching approved incidents from Firebase: {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/incident/accept")
def accept_incident():
    payload = request.get_json(force=True)
    incident_data = payload.get("incident_data")
    if not incident_data or "event" not in incident_data or "event_id" not in incident_data["event"]:
        return jsonify({"ok": False, "error": "Invalid incident data"}), 400

    incident_id = incident_data["event"]["event_id"]
    try:
        fb_db_ref.child('ApprovedIncidents').child(incident_id).set(incident_data)
        fb_db_ref.child('PendingIncidents').child(incident_id).delete()
        return jsonify({"ok": True, "incident_id": incident_id})
    except Exception as e:
        log.error(f"Error accepting incident {incident_id}: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/incident/decline")
def decline_incident():
    payload = request.get_json(force=True)
    incident_data = payload.get("incident_data")
    if not incident_data or "event" not in incident_data or "event_id" not in incident_data["event"]:
        return jsonify({"ok": False, "error": "Invalid incident data"}), 400
        
    incident_id = incident_data["event"]["event_id"]
    try:
        fb_db_ref.child('DeclinedIncidents').child(incident_id).set(incident_data)
        fb_db_ref.child('PendingIncidents').child(incident_id).delete()
        return jsonify({"ok": True, "incident_id": incident_id})
    except Exception as e:
        log.error(f"Error declining incident {incident_id}: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500
    

# ================== START: New Analytics Endpoint ==================

@app.get("/analytics/summary")
def get_analytics_summary():
    try:

        all_incidents_dict = fb_db_ref.child('ApprovedIncidents').get()
        if not all_incidents_dict:
            return jsonify({
                "totalApprovedIncidents": 0, "todayApprovedIncidents": 0, "activeCameras": len(workers),
                "incidentTrends": [], "locationHotspots": [], "violationTypes": [], "commonReasons": []
            })

        all_incidents = list(all_incidents_dict.values())
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

      
        total_approved_incidents = len(all_incidents)
        today_approved_incidents = 0
        
      
        incident_dates = []
        locations = []
        categories = []
        reasons = []
        
        for incident in all_incidents:
            try:
                # Cek insiden hari ini
                incident_time = datetime.fromisoformat(incident["event"]["started_at"])
                if incident_time >= today_start:
                    today_approved_incidents += 1
                
                incident_dates.append(incident_time.date())
                
                locations.append(incident["address"])
                
                if "llm_data" in incident and "category" in incident["llm_data"]:
                    categories.append(incident["llm_data"]["category"])
                
                if "llm_data" in incident and "reasons" in incident["llm_data"]:
                    reasons.extend(incident["llm_data"]["reasons"])
            except (KeyError, TypeError, ValueError) as e:
                log.warning(f"Skipping malformed incident in analytics: {e}")


        # 7 hari terakhir
        date_counts = Counter(incident_dates)
        trends = []
        for i in range(7):
            date = (now - timedelta(days=i)).date()
            trends.append({"date": date.isoformat(), "count": date_counts.get(date, 0)})
        trends.reverse()

        # 5 Lokasi Hotspot Teratas
        location_counts = Counter(locations)
        hotspots = [{"location": loc, "count": count} for loc, count in location_counts.most_common(5)]
        
        # Tipe Pelanggaran Teratas
        category_counts = Counter(categories)
        violation_types = [{"category": cat, "count": count} for cat, count in category_counts.most_common(5)]
        
        # Alasan Pelanggaran Teratas
        reason_counts = Counter(reasons)
        common_reasons = [{"reason": r, "count": count} for r, count in reason_counts.most_common(5)]
        
        summary = {
            "totalApprovedIncidents": total_approved_incidents,
            "todayApprovedIncidents": today_approved_incidents,
            "activeCameras": len(workers),
            "incidentTrends": trends,
            "locationHotspots": hotspots,
            "violationTypes": violation_types,
            "commonReasons": common_reasons
        }
        
        return jsonify(summary)
        
    except Exception as e:
        log.exception("Error generating analytics summary: %s", e)
        return jsonify({"error": str(e)}), 500

# ========================= MCP : Finding Events API ========================
@app.post("/events/refresh_cache")
def refresh_upcoming_events_cache():

    gemini_api_key = os.environ.get("GEMIINI_API_KEY")
    if not gemini_api_key:
        return jsonify({"error": "GEMINI_API_KEY variable not set."}), 500

    try:
        log.info("Starting expensive MCP event fetch to refresh cache...")
        python_executable = sys.executable
        client_script_path = resolve_path("mcp_events_client.py")

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

# ------------------------ API: misc --------------------------
@app.get("/snaps/<path:name>")
def snaps(name):
    return send_from_directory(SNAP_DIR, name)

@app.get("/health")
def health():
    return {"status": "ok", "workers": list(workers.keys()), "clients": len(clients)}



# ------------------------ API: YouTube URL Handler  --------------------------
@app.post("/detector/resolve_url")
def resolve_youtube_url():
    
    # Endpoint mengubah URL halaman YouTube menjadi URL stream langsung (.m3u8).
    
    p = request.get_json(force=True)
    page_url = p.get("url")

    if not page_url:
        return jsonify({"ok": False, "error": "URL parameter is missing"}), 400

    log.info(f"Resolving URL: {page_url}")
    
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'force_generic_extractor': False,
        'format': 'best[protocol=m3u8_native]', 
        'simulate': True,
        'forceurl': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(page_url, download=False)
            stream_url = info['url']
        
        log.info(f"Resolved to: {stream_url}")
        return jsonify({"ok": True, "stream_url": stream_url})

    except Exception as e:
        log.error(f"Failed to resolve URL {page_url}: {e}")
        return jsonify({"ok": False, "error": f"Failed to resolve URL: {e}"}), 500
    


@app.route("/stream_proxy/<path:subpath>")
@app.route("/stream_proxy/")
def stream_proxy(subpath=None):

    base_url = request.args.get('url')
    if not base_url:
        return "URL parameter is missing", 400

    if subpath:
        full_url = urljoin(base_url, subpath)
    else:
        full_url = base_url

    try:
        response = requests.get(full_url, stream=True)
        response.raise_for_status() 

        content_type = response.headers.get('Content-Type', '')

        if 'mpegurl' in content_type:
            content = response.text
            lines = content.split('\n')
            new_lines = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    new_lines.append(f"/stream_proxy/{line}?url={base_url}")
                else:
                    new_lines.append(line)
            
            new_content = '\n'.join(new_lines)
            return Response(new_content, content_type=content_type)
        
        else:
            return Response(response.iter_content(chunk_size=1024), content_type=content_type)

    except requests.exceptions.RequestException as e:
        log.error(f"Failed to proxy stream from {full_url}: {e}")
        return f"Failed to proxy stream: {str(e)}", 500

# ------------------------ Main -------------------------------

if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5001"))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host=host, port=port, debug=debug)
