# ShelfSpaceOptimization/shelf_compare.py

import os
import uuid
from typing import Dict, Any, List, Tuple

import matplotlib
matplotlib.use("Agg")

from .shelf_problem import FixedShelfPacker3D
from .shelf_problem_new import FixedShelfPacker3DIncremental


def _ensure_static_dir():
    os.makedirs("static", exist_ok=True)


def _items_from_existing_state(existing_state: Dict[str, Any]) -> List[Tuple[float, float, float, str, str]]:
    """
    Convert placed items in existing_state -> (w, h, d, item_type, color) tuples
    ignoring their coordinates, so the 'full' packer can repack everything.
    """
    items = []
    for shelf in existing_state.get("shelves", []):
        for pi in shelf.get("placed_items", []):
            items.append((
                float(pi["width"]),
                float(pi["height"]),
                float(pi["depth"]),
                str(pi.get("item_type")),
                pi.get("color")
            ))
    return items


def _volume_sum_free_spaces(shelves_json: List[Dict[str, Any]]) -> float:
    vol = 0.0
    for s in shelves_json:
        for fs in s.get("free_spaces", []):
            vol += float(fs["width"]) * float(fs["height"]) * float(fs["depth"])
    return vol


def _total_capacity(shelf_width: float, shelf_height: float, shelf_depth: float, shelf_count: int) -> float:
    # Capacity is per-shelf volume times count
    return float(shelf_width) * float(shelf_height) * float(shelf_depth) * int(shelf_count)


def compare_packers(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs BOTH:
      1) Full repack: FixedShelfPacker3D with (existing placed + new items) all at once
      2) Incremental: FixedShelfPacker3DIncremental honoring existing placements, adding NEW items only

    Saves two GIFs and returns metrics for free space, utilization, unplaced counts,
    and which method leaves the MOST free space.

    Expected keys in `payload`:
      - shelf_width, shelf_height, shelf_depth, shelf_count
      - compatibility_rules
      - items  (list of [w,h,d,type,color])   -> "NEW items"
      - existing_state (optional)             -> prior state for incremental

    Returns:
      {
        "full": {"video_url": "...", "result": {...}, "free_volume": ..., "utilization": ..., "unplaced_count": ...},
        "incremental": {...},
        "better_method": "full" | "incremental" | "tie"
      }
    """
    _ensure_static_dir()

    # Common inputs
    sw = payload["shelf_width"]
    sh = payload["shelf_height"]
    sd = payload["shelf_depth"]
    sc = payload["shelf_count"]
    rules = payload["compatibility_rules"]
    selected_shelf_id = payload.get("selected_shelf_id")
    new_items = payload.get("items", []) or []
    existing_state = payload.get("existing_state")

    # ---------- 1) FULL REPACK ----------
    # Combine existing placed items (if any) + new items, then repack everything from scratch
    full_items = []
    if existing_state:
        full_items.extend(_items_from_existing_state(existing_state))
    full_items.extend([tuple(i) for i in new_items])

    packer_full = FixedShelfPacker3D(
        shelf_width=sw,
        shelf_height=sh,
        shelf_depth=sd,
        shelf_count=sc,
        compatibility_rules=rules,
        selected_shelf_id=selected_shelf_id
    )
    for w, h, d, t, c in full_items:
        packer_full.add_item(w, h, d, t, c)

    full_gif = f"static/shelf_full_{uuid.uuid4().hex}.gif"
    packer_full.animate(save_path=full_gif)
    full_res = packer_full.get_packing_result_json()

    # ---------- 2) INCREMENTAL ----------
    # Respect existing placements; place ONLY new items
    packer_inc = FixedShelfPacker3DIncremental(
        shelf_width=sw,
        shelf_height=sh,
        shelf_depth=sd,
        shelf_count=sc,
        compatibility_rules=rules,
        selected_shelf_id=selected_shelf_id,
        existing_state=existing_state
    )
    for w, h, d, t, c in new_items:
        packer_inc.add_item(w, h, d, t, c)
    packer_inc.place_all_new_items()

    inc_gif = f"static/shelf_inc_{uuid.uuid4().hex}.gif"
    packer_inc.animate(save_path=inc_gif)
    inc_res = packer_inc.get_packing_result_json()

    # ---------- Metrics ----------
    capacity = _total_capacity(sw, sh, sd, sc)

    full_free = _volume_sum_free_spaces(full_res["shelves"])
    inc_free = _volume_sum_free_spaces(inc_res["shelves"])

    full_used = capacity - full_free
    inc_used = capacity - inc_free

    # Avoid division by zero (degenerate shelves)
    full_util = (full_used / capacity) * 100.0 if capacity > 0 else 0.0
    inc_util = (inc_used / capacity) * 100.0 if capacity > 0 else 0.0

    full_unplaced = len(full_res.get("unplaced_items", []))
    inc_unplaced = len(inc_res.get("unplaced_items", []))

    if abs(full_free - inc_free) < 1e-9:
        better = "tie"
    else:
        better = "full" if full_free > inc_free else "incremental"

    return {
        "full": {
            "video_url": f"/{full_gif}",
            "result": full_res,
            "free_volume": full_free,
            "utilization_pct": full_util,
            "unplaced_count": full_unplaced,
        },
        "incremental": {
            "video_url": f"/{inc_gif}",
            "result": inc_res,
            "free_volume": inc_free,
            "utilization_pct": inc_util,
            "unplaced_count": inc_unplaced,
        },
        "capacity_volume": capacity,
        "better_method": better
    }
