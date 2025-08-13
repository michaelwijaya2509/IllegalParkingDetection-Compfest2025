import time
from cnn_macet import detect_macet
from cek_supir_keluar import check_driver_exit
from urgency_engine import calculate_urgency
from datetime import datetime

def is_stationary_over_time(vehicle_id, duration, ignore_jam=False):
    if (duration < 5 * 30): # 30 fps
        return False

    return True

def start_detection_loop(event_queue):
    """
    Ini snippet placeholder doang, nanti buah aja sama code di file integrate.py
    """
    
    while True:
        detected_car = {
            "id": "vehicle_123",
            "coordinate": [-6.696969, 69.696969],
            "in_red_zone": True,
            "timer": 0
        }

        # Rule-based checks
        if detected_car["in_red_zone"]:
            if is_stationary_over_time(detected_car["id"], detected_car["timer"]):
                urgency = calculate_urgency(detected_car["coordinate"], reason="Stationary in red zone > 5 min")
                push_event(event_queue, detected_car, urgency, "Stationary > 5 min")

            elif detect_macet("frame"):
                pass

            elif check_driver_exit(detected_car["id"]):
                if is_stationary_over_time(detected_car["id"], detected_car["timer"], ignore_jam=True):
                    urgency = calculate_urgency(detected_car["coordinate"], reason="Driver exited, > 5 min")
                    push_event(event_queue, detected_car, urgency, "Driver exited > 5 min")

def push_event(event_queue, vehicle, urgency, reason):
    event_queue.put({
        "urgency": urgency,
        "coordinate": vehicle["coordinate"],
        "time": datetime.utcnow().isoformat(),
        "videoClipUrl": f"https://example.com/video/{vehicle['id']}.mp4", # masukin url cctv
        "reason": reason
    })
