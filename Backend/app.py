from flask import Flask, jsonify, Response, request
from flask_cors import CORS # prevents CORS error in FE
import threading
import queue
import json
from inference import start_detection_loop

app = Flask(__name__)
CORS(app, resources={r"/events": {"origins": "*"}})

# Queue for real-time event streaming
event_queue = queue.Queue()

@app.route("/events")
def events():
    """Stream illegal parking events via SSE."""
    def stream():
        while True:
            event = event_queue.get()
            yield f"data: {json.dumps(event)}\n\n"
    return Response(stream(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"  # For nginx, if used
    })


@app.route("/test-event", methods=["POST"])
def test_event():
    """Manually push an event (for debugging)."""
    data = request.get_json()
    event_queue.put(data)
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    # Start detection logic in a separate thread
    detection_thread = threading.Thread(
        target=start_detection_loop,
        args=(event_queue,),
        daemon=True
    )
    detection_thread.start()

    app.run(host="0.0.0.0", port=5000, debug=True)

@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Content-Type"] = "text/event-stream"
    return response
