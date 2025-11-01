from flask import Flask, request, jsonify, send_file
import os
import uuid
import sys
from flask_socketio import SocketIO, emit
import base64
import io
from PIL import Image
import pandas as pd
from flask_cors import CORS
import serial
import serial.tools.list_ports


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ShelfSpaceOptimization.shelf_problem import FixedShelfPacker3D
from RouteOptimization.path_finding import run_pathfinding_animation_dynamic
from InboundOutboundForecast.inbound_outbound_forecast import predict_forecast_for_a_category
from FireDetection.shelf_detection import process_image
from InboundOutboundForecast.employee_perf import predict_performance
from ShelfSpaceOptimization.shelf_problem_new import FixedShelfPacker3DIncremental
from ShelfSpaceOptimization.shelf_compare import compare_packers
from ShelfSpaceOptimization.shelf_probblem_2 import pack_from_request_payload

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# ---------- Arduino Serial Config ----------
SERIAL_BAUD = 9600
# Let users override; default to a likely port
# SERIAL_PORT = os.getenv("ARDUINO_PORT", "COM8" if os.name == "nt" else "/dev/ttyUSB0")
SERIAL_PORT = os.getenv("ARDUINO_PORT", "COM4 ")
arduino_ser = None

# def init_serial():
#     global arduino_ser
#     try:
#         arduino_ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.5)
#         print(f"[Serial] Connected to {SERIAL_PORT} @ {SERIAL_BAUD}")
#     except Exception as e:
#         arduino_ser = None
#         print(f"[Serial] Could not open {SERIAL_PORT} ({e}). Set ARDUINO_PORT env var to the correct port.")

def init_serial():
    """
    Try the configured port first; if it fails, auto-detect a likely Arduino/CH340.
    Includes brief retries to avoid transient 'Access is denied' on Windows.
    """
    import time
    import serial.tools.list_ports

    global arduino_ser
    port = SERIAL_PORT

    def try_open(p):
        try:
            s = serial.Serial(p, SERIAL_BAUD, timeout=0.5)
            print(f"[Serial] Connected to {p} @ {SERIAL_BAUD}")
            return s
        except PermissionError as e:
            # another app (or the reloader) holds the port — brief retry helps
            print(f"[Serial] PermissionError for {p}: {e}")
            return None
        except Exception as e:
            print(f"[Serial] Could not open {p}: {e}")
            return None

    # 1) Try requested/default port first with a couple of retries
    for attempt in range(3):
        arduino_ser = try_open(port)
        if arduino_ser:
            return
        time.sleep(0.8)

    # 2) Auto-detect: look for Arduino/CH340/USB Serial devices
    print("[Serial] Auto-detecting serial ports…")
    candidates = []
    for p in serial.tools.list_ports.comports():
        text = f"{p.description} {p.manufacturer} {p.hwid}".lower()
        score = 0
        if "arduino" in text: score += 5
        if "ch340" in text or "wch" in text: score += 4
        if "cp210" in text or "silicon labs" in text: score += 3
        if "usb serial" in text: score += 2
        candidates.append((score, p.device, p.description))

    if candidates:
        print("[Serial] Available ports:")
        for s, dev, desc in sorted(candidates, reverse=True):
            print(f"  - {dev} | {desc} | score={s}")

        best = sorted(candidates, reverse=True)[0][1]
        for attempt in range(3):
            arduino_ser = try_open(best)
            if arduino_ser:
                return
            time.sleep(0.8)

    # 3) Give up (non-fatal; app still runs without Arduino)
    arduino_ser = None
    print("[Serial] No usable serial port. Set ARDUINO_PORT=COMx or plug the board.")

# def init_serial():
#     global arduino_ser
#     try:
#         port = SERIAL_PORT
#         # If a URL like loop:// is provided, use serial_for_url; otherwise normal Serial
#         if port and "://" in port:
#             arduino_ser = serial.serial_for_url(port, SERIAL_BAUD, timeout=0.5)
#         else:
#             arduino_ser = serial.Serial(port, SERIAL_BAUD, timeout=0.5)
#         print(f"[Serial] Connected to {port} @ {SERIAL_BAUD}")
#     except Exception as e:
#         arduino_ser = None
#         print(f"[Serial] Could not open {SERIAL_PORT} ({e}). Set ARDUINO_PORT to the correct port.")

def send_to_arduino(line: str):
    """
    Sends a single line (e.g., '1,0,0,0\\n') to Arduino, if serial is available.
    """
    global arduino_ser
    if not arduino_ser or not arduino_ser.is_open:
        return
    try:
        if not line.endswith("\n"):
            line += "\n"
        arduino_ser.write(line.encode("utf-8"))
    except Exception as e:
        print(f"[Serial] Write error: {e}")

# def send_to_arduino(line: str):
#     global arduino_ser
#     if not arduino_ser or not arduino_ser.is_open:
#         return
#     try:
#         if not line.endswith("\n"):
#             line += "\n"
#         arduino_ser.write(line.encode("utf-8"))
#
#         # Only for loop:// testing — read back what we just wrote
#         echoed = arduino_ser.readline().decode("utf-8", errors="ignore").strip()
#         if echoed:
#             print(f"[Serial][echo] {echoed}")
#     except Exception as e:
#         print(f"[Serial] Write error: {e}")

def encode_direction(full_direction: str | None) -> str:
    """
    Maps your directions to the requested 4-bit output:
      0,0,0,0 = nofire
      1,0,0,0 = up-right
      0,1,0,0 = up-left
      0,0,1,0 = down-right
      0,0,0,1 = down-left
      1,1,1,1 = Stationary
    """
    if full_direction is None:
        return "0,0,0,0"

    d = full_direction.lower()
    if d == "up-right":
        return "1,0,0,0"
    if d == "up-left":
        return "0,1,0,0"
    if d == "down-right":
        return "0,0,1,0"
    if d == "down-left":
        return "0,0,0,1"
    if d == "stationary":
        return "1,1,1,1"
    # Any other direction (e.g., just "right"/"left"/"up"/"down") -> treat as nofire
    return "0,0,0,0"

def pick_top_fire_direction(result: dict) -> str:
    """
    Chooses the most confident fire (if any) and returns the encoded 4-bit signal.
    Falls back to nofire when there are no fires.
    """
    fires = result.get("fires", []) or []
    if not fires:
        return encode_direction(None)

    # Prefer highest confidence
    top = max(fires, key=lambda f: f.get("conf", 0.0))
    full_dir = top.get("full_direction")  # your process_image already sets this
    return encode_direction(full_dir)

# @socketio.on('detect_fire_from_frame')
# def handle_detect_fire_from_frame(data):
#     try:
#         image_base64 = data.get("image")
#         if not image_base64:
#             emit('fire_detection_result', {"error": "No image received"})
#             return True  # ack
#         image_bytes = base64.b64decode(image_base64.split(",")[-1])
#         result = process_image(io.BytesIO(image_bytes))
#         emit('fire_detection_result', result)
#     except Exception as e:
#         emit('fire_detection_result', {"error": str(e)})
#     finally:
#         return True  # ack

@socketio.on('detect_fire_from_frame')
def handle_detect_fire_from_frame(data):
    try:
        image_base64 = data.get("image")
        if not image_base64:
            emit('fire_detection_result', {"error": "No image received"})
            # send "nofire" when nothing received (optional)
            send_to_arduino("0,0,0,0")
            return True  # ack

        image_bytes = base64.b64decode(image_base64.split(",")[-1])
        result = process_image(io.BytesIO(image_bytes))

        # --- NEW: send signal to Arduino
        signal = pick_top_fire_direction(result)
        send_to_arduino(signal)
        print(f"[FireDetection] Sent signal to Arduino: {signal}")

        emit('fire_detection_result', result)
    except Exception as e:
        emit('fire_detection_result', {"error": str(e)})
        # optional: send "nofire" on error
        send_to_arduino("0,0,0,0")
    finally:
        return True  # ack

@app.route('/generate2', methods=['POST'])
def generate_v2():
    data = request.get_json()
    result = pack_from_request_payload(data, save_gif=True)
    # If you want to save to Mongo here, you can use:
    # db.packing_runs.insert_one(result["run"])
    return jsonify(result)

@app.route('/generate-incremental', methods=['POST'])
def generate_incremental():
    """
    Incremental packing:
    - Respects prior placements (nothing moves)
    - Inserts only NEW items into remaining free spaces
    - Returns updated state you can persist back to MongoDB
    Request JSON:
    {
        "shelf_width": 100,
        "shelf_height": 50,
        "shelf_depth": 60,
        "shelf_count": 10,
        "selected_shelf_id": null,
        "compatibility_rules": { ... },
        "items": [ [w,h,d,"type","color"], ... ],           # NEW items only
        "existing_state": { ... }                            # OPTIONAL: result from previous run
    }
    """
    try:
        data = request.get_json()

        shelf_width = data.get('shelf_width')
        shelf_height = data.get('shelf_height')
        shelf_depth = data.get('shelf_depth')
        shelf_count = data.get('shelf_count')
        selected_shelf_id = data.get('selected_shelf_id')
        compatibility_rules = data.get('compatibility_rules')
        items_to_pack = data.get('items', [])
        existing_state = data.get('existing_state')  # pass what you loaded from MongoDB

        if not all([shelf_width, shelf_height, shelf_depth, shelf_count, compatibility_rules]) \
           or items_to_pack is None:
            return jsonify({"error": "Missing required parameters"}), 400

        packer = FixedShelfPacker3DIncremental(
            shelf_width=shelf_width,
            shelf_height=shelf_height,
            shelf_depth=shelf_depth,
            shelf_count=shelf_count,
            compatibility_rules=compatibility_rules,
            selected_shelf_id=selected_shelf_id,
            existing_state=existing_state
        )

        for item in items_to_pack:
            packer.add_item(*item)

        # Place only the NEW items; existing placements stay fixed
        packer.place_all_new_items()

        # Save to unique file
        filename = f"static/shelf_inc_{uuid.uuid4().hex}.gif"
        packer.animate(save_path=filename)

        # Updated state
        packing_result = packer.get_packing_result_json()

        return jsonify({
            "video_url": f"/{filename}",
            "result": packing_result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/compare-shelf-packers", methods=["POST"])
def compare_shelf_packers():
    try:
        data = request.get_json()
        required = ["shelf_width", "shelf_height", "shelf_depth", "shelf_count", "compatibility_rules"]
        if not all(k in data for k in required):
            return jsonify({"error": f"Missing required parameters: {required}"}), 400
        result = compare_packers(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()

    # Required inputs
    shelf_width = data.get('shelf_width')
    shelf_height = data.get('shelf_height')
    shelf_depth = data.get('shelf_depth')
    shelf_count = data.get('shelf_count')
    selected_shelf_id = data.get('selected_shelf_id')
    compatibility_rules = data.get('compatibility_rules')
    items_to_pack = data.get('items')

    if not all([shelf_width, shelf_height, shelf_depth, shelf_count, compatibility_rules, items_to_pack]):
        return jsonify({"error": "Missing required parameters"}), 400

    packer = FixedShelfPacker3D(
        shelf_width=shelf_width,
        shelf_height=shelf_height,
        shelf_depth=shelf_depth,
        shelf_count=shelf_count,
        compatibility_rules=compatibility_rules,
        selected_shelf_id=selected_shelf_id
    )

    for item in items_to_pack:
        packer.add_item(*item)

    # Save to unique file
    filename = f"static/shelf_{uuid.uuid4().hex}.gif"
    packer.animate(save_path=filename)

    # Get the packing result JSON
    packing_result = packer.get_packing_result_json()

    return jsonify({
        "video_url": f"/{filename}",
        "result": packing_result
    })

@app.route('/pathfinding', methods=['POST'])
def generate_pathfinding_video():
    data = request.get_json()

    # Extract parameters from request
    shelf_height = data.get('shelf_height')
    shelf_count = data.get('shelf_count')
    shelf_interval = data.get('shelf_interval')
    picking_locations = data.get('picking_locations')
    workers = data.get('workers')
    force_aisles = data.get('force_aisles', True)  # optional; defaults to True

    if not all([shelf_height, shelf_count, shelf_interval, picking_locations, workers]):
        return jsonify({"error": "Missing picking_locations or workers : shelf_height, shelf_count, shelf_interval, picking_locations, workers"}), 400

    filename = f"static/path_{uuid.uuid4().hex}.gif"
    run_pathfinding_animation_dynamic(
        shelf_height=shelf_height,
        shelf_count=shelf_count,
        shelf_interval=shelf_interval,
        picking_locations=picking_locations,
        obstacles=workers,
        save_path=filename,
        optimize_order=True,     # shortest total
        lock_picked=False,       # do NOT wall off 1-cell aisles
        force_aisles=force_aisles,
        pause_pick_frames=8,     # <-- pause at each pick
    )

    return jsonify({"video_url": f"/{filename}"})

# @app.route('/pathfinding', methods=['POST'])
# def generate_pathfinding_video():
#     data = request.get_json()
#
#     # Extract parameters from request
#     shelf_height = data.get('shelf_height')
#     shelf_count = data.get('shelf_count')
#     shelf_interval = data.get('shelf_interval')
#     picking_locations = data.get('picking_locations')
#     workers = data.get('workers')
#
#     if not all([shelf_height, shelf_count, shelf_interval, picking_locations, workers]):
#         return jsonify({"error": "Missing picking_locations or workers : shelf_height, shelf_count, shelf_interval, picking_locations, workers"}), 400
#
#     filename = f"static/path_{uuid.uuid4().hex}.gif"
#     run_pathfinding_animation_dynamic(
#         shelf_height=shelf_height,
#         shelf_count=shelf_count,
#         shelf_interval=shelf_interval,
#         picking_locations=picking_locations,
#         obstacles=workers,
#         save_path=filename
#     )
#
#     return jsonify({"video_url": f"/{filename}"})

# @app.route('/detect-fire', methods=['POST'])
# def detect_route():
#     if 'image' not in request.files:
#         return {'error': 'No image uploaded'}, 400
#
#     file = request.files['image']
#     result = process_image(file)
#     return jsonify(result)

@app.route('/detect-fire', methods=['POST'])
def detect_route():
    if 'image' not in request.files:
        # optional: send "nofire" when bad request
        send_to_arduino("0,0,0,0")
        return {'error': 'No image uploaded'}, 400

    file = request.files['image']
    result = process_image(file)

    # --- NEW: send signal to Arduino
    signal = pick_top_fire_direction(result)
    send_to_arduino(signal)

    return jsonify(result)


@app.route('/predict-performance', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        result = predict_performance(data)
        return jsonify({"predicted_performance": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/forecast-image", methods=["POST"])
def get_forecast_plot():
    try:
        data = request.get_json()
        category = data["category"]
        start_month = data.get("start_month")
        end_month = data.get("end_month")
        start_week = data.get("start_week")
        end_week = data.get("end_week")
        year = data.get("year")
        time_frame = data.get("time_frame")
        threshold = data.get("threshold")

        if not all([category, start_month, end_month, time_frame, year]):
            return jsonify({"error": "Missing required parameters : category, start_month, end_month, time_frame, year"}), 400

        forecast_df, image_path = predict_forecast_for_a_category(
            category,
            start_month,
            end_month,
            start_week,
            end_week,
            year,
            time_frame,
            threshold
        )

        return send_file(image_path, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/forecast-data", methods=["POST"])
def get_forecast_data():
    try:
        data = request.get_json()
        category = data.get("category")
        start_month = data.get("start_month")
        end_month = data.get("end_month")
        start_week = data.get("start_week", 1)
        end_week = data.get("end_week", 5)
        year = data.get("year", pd.Timestamp.now().year)
        time_frame = data.get("time_frame", 60)
        threshold = data.get("threshold", 0.1)

        if not all([category, start_month, end_month, time_frame, year]):
            return jsonify({"error": "Missing required parameters : category, start_month, end_month, time_frame, year"}), 400

        # Call your function
        forecast_df, _ = predict_forecast_for_a_category(
            category,
            start_month,
            end_month,
            start_week,
            end_week,
            year,
            time_frame,
            threshold
        )

        # For better charting, convert any timestamps to string (ISO format)
        forecast_df['ds'] = forecast_df['ds'].astype(str)

        # Optionally: also return real data if you want
        # comparison["ds"] = comparison["ds"].astype(str)
        # response = {
        #     "forecast": forecast_df.to_dict(orient="records"),
        #     "comparison": comparison.to_dict(orient="records"),
        # }

        response = {
            "forecast": forecast_df.to_dict(orient="records")
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    init_serial()
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, use_reloader=False)