import cv2
from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import base64
import os
import io

# --- optional if you keep yolov5 local for torch.hub ---
import sys
sys.path.append('yolov5')

# ---------- Depth preprocessing ----------
transform = Compose([
    Resize((384, 384)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------- Geometry / thresholds ----------
PROXIMITY_THRESHOLD = 20  # meters in your scaled top-down space (tuned)
KNOWN_OBJECT_DISTANCE_M = 2.0
KNOWN_MIDAS_DEPTH_VALUE = 279.66
PIXELS_PER_METER_AT_2M = 100
X_SCALE = 1 / PIXELS_PER_METER_AT_2M
DEPTH_SCALE = KNOWN_OBJECT_DISTANCE_M / KNOWN_MIDAS_DEPTH_VALUE

# ---------- Trackers ----------
prev_fire_boxes = []
fire_histories = {}
initial_fire_positions = {}
next_fire_id = 0
MAX_HISTORY = 5

# ---------- Robust decode ----------
def _bytes_to_cv2_image(image_bytes: bytes):
    npbuf = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None

# ---------- IOU ----------
def iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    return interArea / float(box1Area + box2Area - interArea + 1e-6)

# ---------- Depth ----------
def process_depth(img_rgb):
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        depth = midas(img_tensor)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = depth.squeeze().cpu().numpy()
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return depth, depth_normalized

# ---------- Models ----------
BASE_DIR = os.path.dirname(__file__)
shelf_model = YOLO(os.path.join(BASE_DIR, 'shelf.pt'))
fire_model = torch.hub.load(os.path.join(BASE_DIR, 'yolov5'),
                            'custom',
                            path=os.path.join(BASE_DIR, 'fire_detection_best_2.pt'),
                            source='local')

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# ---------- Main pipeline ----------
def process_image(file):
    """
    Returns a dict with:
      - fire_detected: bool
      - confidence: max fire confidence
      - fires: [{id, bbox, size, direction, full_direction, distance_m, conf}]
      - shelves: [{id, bbox, distance_m}]
      - distances: [{fire_id, shelf_id, distance_m, direction, f_direction}]
      - image_base64: annotated frame (JPG, base64)
    """
    global prev_fire_boxes, fire_histories, initial_fire_positions, next_fire_id

    # --- decode ---
    image_bytes = file.read() if hasattr(file, "read") else file
    image = _bytes_to_cv2_image(image_bytes)
    if image is None:
        return {"error": "Decode failed. Send JPEG/PNG or enable WEBP support on server."}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_map, _ = process_depth(image_rgb)

    h, w = image.shape[:2]
    center_x = w // 2

    fires_payload = []
    shelves_payload = []
    distances_payload = []

    # --- shelves ---
    shelf_results = shelf_model.predict(image_rgb, imgsz=640, conf=0.3)
    shelf_detections = shelf_results[0].boxes.xyxy.cpu().numpy()
    shelf_scores = shelf_results[0].boxes.conf.cpu().numpy()

    shelf_id = 0
    for box, score in zip(shelf_detections, shelf_scores):
        x1, y1, x2, y2 = map(int, box)
        x_center = int((x1 + x2) / 2)
        # draw
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'Shelf: {score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # distance (approx)
        bbox_depth = depth_map[y1:y2, x1:x2]
        avg_depth = float(np.mean(bbox_depth)) if bbox_depth.size > 0 else 0.0
        approx_distance = avg_depth * DEPTH_SCALE

        shelves_payload.append({
            "id": shelf_id,
            "bbox": [x1, y1, x2, y2],
            "distance_m": round(approx_distance, 2)
        })
        shelf_id += 1

    # --- fires ---
    fire_results = fire_model(image_rgb)
    fire_dets = fire_results.xyxy[0].cpu().numpy()
    new_fire_boxes = []

    for det in fire_dets:
        if len(det) != 6:
            continue
        x1, y1, x2, y2 = map(int, det[:4])
        conf = float(det[4])
        curr_box = [x1, y1, x2, y2]
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2

        # ID match
        matched_id = None
        for prev_id, prev_box in prev_fire_boxes:
            if iou(prev_box, curr_box) > 0.3:
                matched_id = prev_id
                break
        if matched_id is None:
            matched_id = next_fire_id
            next_fire_id += 1
        new_fire_boxes.append((matched_id, curr_box))

        # history
        fire_histories.setdefault(matched_id, []).append((x_center, y_center))
        if len(fire_histories[matched_id]) > MAX_HISTORY:
            fire_histories[matched_id].pop(0)
        initial_fire_positions.setdefault(matched_id, (x_center, y_center))

        # directions
        avg_x = int(np.mean([p[0] for p in fire_histories[matched_id]]))
        avg_y = int(np.mean([p[1] for p in fire_histories[matched_id]]))
        dx = x_center - avg_x
        dy = y_center - avg_y

        if abs(dx) < 2 and abs(dy) < 2:
            direction = "Stationary"
        elif abs(dx) > abs(dy):
            direction = "Right" if dx > 0 else "Left"
        else:
            direction = "Down" if dy > 0 else "Up"

        start_x, start_y = initial_fire_positions[matched_id]
        dx_total = x_center - start_x
        dy_total = y_center - start_y

        if abs(dx_total) < 2 and abs(dy_total) < 2:
            full_direction = "Stationary"
        elif abs(dx_total) > abs(dy_total):
            full_direction = "Right" if dx_total > 0 else "Left"
        else:
            full_direction = "Down" if dy_total > 0 else "Up"

        if abs(dx_total) >= 2 and abs(dy_total) >= 2:
            if dx_total > 0 and dy_total > 0:
                full_direction = "Down-Right"
            elif dx_total < 0 and dy_total > 0:
                full_direction = "Down-Left"
            elif dx_total > 0 and dy_total < 0:
                full_direction = "Up-Right"
            elif dx_total < 0 and dy_total < 0:
                full_direction = "Up-Left"

        # distance + size estimate
        bbox_area = (x2 - x1) * (y2 - y1)
        bbox_depth = depth_map[y1:y2, x1:x2]
        avg_depth = float(np.mean(bbox_depth)) if bbox_depth.size > 0 else 0.0
        approx_distance = avg_depth * DEPTH_SCALE
        volume_estimate = bbox_area * (approx_distance ** 2)

        if volume_estimate < 1e5:
            fire_size = "Small Fire"
        elif volume_estimate < 3e5:
            fire_size = "Medium Fire"
        else:
            fire_size = "Large Fire"

        # draw
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, f'{fire_size}: {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, f"{direction} / {full_direction}", (x_center, y_center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.arrowedLine(image, (start_x, start_y), (x_center, y_center),
                        (255, 0, 255), 2, tipLength=0.3)

        fires_payload.append({
            "id": matched_id,
            "bbox": [x1, y1, x2, y2],
            "size": fire_size,
            "direction": direction,
            "full_direction": full_direction,
            "distance_m": round(float(approx_distance), 2),
            "conf": round(conf, 4),
        })

    # update trackers
    prev_fire_boxes = new_fire_boxes

    # --- fire ↔ shelf distances ---
    for f in fires_payload:
        fx = ( (f["bbox"][0] + f["bbox"][2]) // 2 - center_x ) * X_SCALE
        fy = f["distance_m"]
        for s in shelves_payload:
            sx = ( (s["bbox"][0] + s["bbox"][2]) // 2 - center_x ) * X_SCALE
            sy = s["distance_m"]
            dist = float(np.sqrt((fx - sx) ** 2 + (fy - sy) ** 2))
            distances_payload.append({
                "fire_id": f["id"],
                "shelf_id": s["id"],
                "distance_m": round(dist, 2),
                "direction": f["direction"],
                "f_direction": f["full_direction"]
            })

    # --- final encoding ---
    ok, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8') if ok else None

    max_conf = max((f["conf"] for f in fires_payload), default=0.0)
    return {
        "fire_detected": len(fires_payload) > 0,
        "confidence": float(max_conf),
        "fires": fires_payload,
        "shelves": shelves_payload,
        "distances": distances_payload,
        "image_base64": encoded_image
    }
