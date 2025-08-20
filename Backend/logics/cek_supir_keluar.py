import time
import pickle
import json
import numpy as np
import cv2
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

def load_zones(zone, video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return []

    h, w = frame.shape[:2]
    polygons = []
    for path in zone:
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                polygons.append(data)
            else:
                poly = data["polygon"]
                meta = data.get("video_metadata", {})
                ow, oh = meta.get("width", w), meta.get("height", h)
                scale_x, scale_y = w / ow, h / oh
                scaled = [[int(x * scale_x), int(y * scale_y)] for x, y in poly]
                polygons.append(scaled)
    return polygons

def point_in_polygon(x, y, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), (int(x), int(y)), False) >= 0

def crop_with_margin(frame, box, margin_ratio=0.3, out_size=(224, 224)):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    # pastikan bbox valid dulu
    if x2 <= x1 or y2 <= y1:
        return None

    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    x1c = max(0, x1 - mx)
    y1c = max(0, y1 - my)
    x2c = min(W, x2 + mx)
    y2c = min(H, y2 + my)

    # validasi lagi setelah clamp
    if x2c <= x1c or y2c <= y1c:
        return None

    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return None

    return cv2.resize(crop, out_size)



def extract_frames_from_video(video_path, zone, model_detection, tracker, target_frames=16, frame_interval=16, margin_ratio=0.3, min_sequence_length=16):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video {video_path}")
        return {}
    
    frame_count = 0
    vehicle_snapshots = defaultdict(list)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        #YOLO untuk deteksi kendaraan
        results = model_detection(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if conf > 0.5 and model_detection.names[cls_id] in ["car", "truck", "bus"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, cls_id))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = list(map(int, track.to_ltrb()))
            cx = (ltrb[0] + ltrb[2]) // 2
            cy = (ltrb[1] + ltrb[3]) // 2

            # Cek kendaraan ada di zona illegal atau bukan
            in_red_zone = any(point_in_polygon(cx, cy, polygon) for polygon in zone)

            if in_red_zone and frame_count % frame_interval == 0:
                if len(vehicle_snapshots[track_id]) < target_frames:
                    cropped_frame = crop_with_margin(frame, ltrb, margin_ratio)
                    vehicle_snapshots[track_id].append(cropped_frame)

    cap.release()
    
    filtered_snapshots = {}
    for vehicle_id, frames in vehicle_snapshots.items():
        if len(frames) >= min_sequence_length:
            if len(frames) >= target_frames:
                filtered_snapshots[vehicle_id] = frames[:target_frames]
            else:
                filtered_snapshots[vehicle_id] = frames
    
    print(f"Found {len(filtered_snapshots)} vehicle sequences.")

    for vehicle_id, frames in filtered_snapshots.items():
        print(f"Vehicle {vehicle_id}: {len(frames)} frames")
    
    return filtered_snapshots




def preprocess_frames_for_inference(frames):

    if not frames:
        return None
    
    mean = np.array([0.485, 0.456, 0.406])  
    std = np.array([0.229, 0.224, 0.225])   
    
    processed_frames = []
    
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        frame_resized = cv2.resize(frame_rgb, (256, 256))
        
        h, w = frame_resized.shape[:2]
        top = (h - 224) // 2
        left = (w - 224) // 2
        frame_cropped = frame_resized[top:top+224, left:left+224]
        
        frame_tensor = frame_cropped.astype(np.float32) / 255.0
        
        frame_normalized = (frame_tensor - mean) / std
        
        frame_final = np.transpose(frame_normalized, (2, 0, 1)) 
        
        processed_frames.append(frame_final)
    
    processed_frames = np.array(processed_frames)
    
    processed_frames = np.expand_dims(processed_frames, axis=0)
    
    return processed_frames



def check_driver_exit(vehicle_frames, model_inference):
    try:
        processed_data = preprocess_frames_for_inference(vehicle_frames)
        
        if processed_data is None:
            return False
        input_tensor = torch.from_numpy(processed_data).float()
        

        num_frames = input_tensor.shape[1] 
        lengths_tensor = torch.tensor([num_frames]) 
        
        if callable(model_inference):
            with torch.no_grad(): 
                logits = model_inference(input_tensor, lengths_tensor)
                probabilities = torch.softmax(logits, dim=1)
                prediction = probabilities[0, 1].item() #
        else:
            print("Model doesn't have any inference method")
            return False
        
        driver_exit = prediction > 0.4
        
        print(f"Driver exit prediction probability: {prediction:.4f} -> {driver_exit}")
        return driver_exit
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return False
    

def process_video_for_driver_exit_detection(video_path, zone_jsons, model_inference, model_detection, tracker):
    
    red_zone_polygons = load_zones(zone_jsons, video_path)
    
    if not red_zone_polygons:
        print("No zones loaded.")
        return {}
    
    vehicle_frames = extract_frames_from_video(
        video_path=video_path,
        zone=red_zone_polygons,
        model_detection=model_detection,
        tracker=tracker,
        target_frames=16,
        frame_interval=16,
        min_sequence_length=16
    )
    
    if not vehicle_frames:
        print("No vehicle sequences found.")
        return {}
    
    results = {}
    
    for vehicle_id, frames in vehicle_frames.items():
        driver_exit = check_driver_exit(frames, model_inference)
        results[vehicle_id] = driver_exit
    
    return results


if __name__ == "__main__":

    video_path = "illegal_2.mp4"  
    zone_jsons = ["zona_enhanced.json", "zona_vietnam1.json"]
    
    model_inference_path = "car_open_detection.pkl"  

    try:
        with open(model_inference_path, 'rb') as f:
            model_inference = pickle.load(f)
    except Exception as e:
        print(f"Failed to load inference model: {e}")
    
    model_detection = YOLO('yolo11n.pt')
    tracker = DeepSort(max_age=60, n_init=3)
    
    red_zone_polygons = load_zones(zone_jsons, video_path)
    
    results = process_video_for_driver_exit_detection(
        video_path=video_path,
        zone_jsons=zone_jsons,
        model_inference=model_inference,
        model_detection=model_detection,
        tracker=tracker
    )
    
    if results:
        for vehicle_id, driver_exit in results.items():
            status = "Supir Keluar" if driver_exit else "Supir Tidak Keluar"
            print(f"Vehicle {vehicle_id}: {status}")
    else:
        print("Tidak ada kendaraan yang terdeteksi di red zone.")
