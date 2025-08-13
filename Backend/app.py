from flask import Flask, jsonify, Response, request
from flask_cors import CORS
import threading
import queue
import json

app = Flask(__name__)
CORS(app, resources={r"/events": {"origins": "*"}})

history_events = []
clients = []

@app.route("/events")
def events():
    client_ip = request.remote_addr
    print(f"🟩 New client connected: {client_ip}")

    client_queue = queue.Queue()
    clients.append(client_queue)

    def stream():
        try:
            # Broadcast history events ke client ketika baru terhubung
            for event in history_events:
                yield f"data: {json.dumps(event)}\n\n"

            # Broadcast event baru ke client (client listens)
            while True:
                event = client_queue.get()
                yield f"data: {json.dumps(event)}\n\n"
        except GeneratorExit:
            print(f"🔴 Client {client_ip} disconnected")
            clients.remove(client_queue)
            raise

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

# Untuk testing saja (bisa pakai curl)

# Copy comment di bawah ke terminal (uncomment dulu)

# curl -X POST http://localhost:5000/test-event `
# -H "Content-Type: application/json" `
# -d '{"urgency": 0.9, "locationName": "Jl. Braga", "coordinate": [-6.92145482747903, 107.6096481364423], "time": "2025-08-13T17:36:00.000Z", "videoClipUrl": "", "reason": "Mobil terparkir di samping trotoar, menghalangi pejalan kaki dan akses ke rumah sakit"}'

@app.route("/test-event", methods=["POST"])
def test_event():
    data = request.get_json()
    history_events.append(data)

    # Broadcast to all client queues
    for q in clients:
        q.put(data)

    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run("0.0.0.0", 5000, debug=True)
