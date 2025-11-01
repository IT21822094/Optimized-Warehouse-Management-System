# ShelfSpaceOptimization/shelf_problem_new.py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches


class FixedShelfPacker3DIncremental:
    """
    Incremental 3D shelf packer (product-aware).
    - Honors existing placements and shelf compatibilities.
    - Uses persisted free_spaces to place only NEW items without moving old ones.
    - If no space is available for a new item, it is added to unplaced_items with a reason.
    - State is fully serializable via get_packing_result_json() and restorable via from_state().

    Persisted state (supports product_id + created_by):
    {
      "shelves": [
        {
          "id": int,
          "width": number,
          "height": number,
          "depth": number,
          "placed_items": [
            {
              "x":..,"y":..,"z":..,
              "width":..,"height":..,"depth":..,
              "item_type":"..","color":"..",
              "product_id":"<mongo-id, optional>",
              "is_new": bool
            }
          ],
          "free_spaces": [
            {
              "x":..,"y":..,"z":..,
              "width":..,"height":..,"depth":..,
              "created_by": { "product_id": "<mongo-id>", "is_new": bool }   # optional
            }
          ]
        },
        ...
      ],
      "unplaced_items":[
        {
          "width":..,"height":..,"depth":..,
          "item_type":"..","color":"..",
          "product_id":"<mongo-id, optional>",
          "is_new": bool,
          "reason": "why the item could not be placed"
        }
      ]
    }
    """

    def __init__(
        self,
        shelf_width,
        shelf_height,
        shelf_depth,
        shelf_count,
        compatibility_rules,
        selected_shelf_id=None,
        existing_state=None
    ):
        self.shelf_width = shelf_width
        self.shelf_height = shelf_height
        self.shelf_depth = shelf_depth
        self.shelf_count = shelf_count
        self.compatibility_rules = {k: set(v) for k, v in compatibility_rules.items()}
        self.selected_shelf_id = selected_shelf_id

        # items queue holds (w,h,d,type,color,product_id)
        self.items = []
        # unplaced holds (w,h,d,type,color,product_id,is_new,reason)
        self.unplaced_items = []

        if existing_state:
            self._load_from_state(existing_state)
        else:
            self._init_empty_state()

        # ensure we have at least shelf_count shelves (append empty/open shelves)
        self._ensure_shelf_count()

        # derive/ensure per-shelf compatibility from existing items if not set
        for shelf in self.shelves:
            if not shelf.get("compatibility"):
                if shelf["placed_items"]:
                    # item_type from placed tuple (index 6)
                    t0 = shelf["placed_items"][0][6]
                    shelf["compatibility"] = self.compatibility_rules.get(t0, {t0}).copy()
                else:
                    shelf["compatibility"] = set()

        # figure & axes for optional animation
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._animation_step_index = 0

    # ---------- State helpers ----------

    def _init_empty_state(self):
        """
        Internal tuple shapes:
          placed_items: (x,y,z,w,h,d,item_type,color,product_id,is_new)
          free_spaces : (x,y,z,w,h,d,created_by_product_id,created_by_is_new)
        """
        self.shelves = [
            {
                "id": i,
                "width": self.shelf_width,
                "height": self.shelf_height,
                "depth": self.shelf_depth,
                "compatibility": set(),
                "placed_items": [],
                "free_spaces": [
                    (0, 0, 0, self.shelf_width, self.shelf_height, self.shelf_depth, None, False)
                ],
            }
            for i in range(self.shelf_count)
        ]

    def _load_from_state(self, state_dict):
        shelves = state_dict.get("shelves", [])
        self.shelves = []
        for s in shelves:
            # read placed_items (support optional product_id); mark as existing (is_new=False)
            placed_raw = s.get("placed_items", [])
            placed = [
                (
                    pi["x"], pi["y"], pi["z"],
                    pi["width"], pi["height"], pi["depth"],
                    pi.get("item_type"), pi.get("color"),
                    pi.get("product_id"),  # product_id
                    False                   # is_new
                )
                for pi in placed_raw
            ]

            # read free_spaces; support optional created_by.product_id
            free = []
            if s.get("free_spaces"):
                for fs in s["free_spaces"]:
                    created_by_pid = None
                    cb = fs.get("created_by")
                    if isinstance(cb, dict):
                        created_by_pid = cb.get("product_id")
                    free.append((
                        fs["x"], fs["y"], fs["z"],
                        fs["width"], fs["height"], fs["depth"],
                        created_by_pid,  # created_by_product_id
                        False            # created_by_is_new (existing)
                    ))
            else:
                # If free spaces are not provided:
                # If no items: whole shelf free; otherwise we cannot reconstruct exactly—start with no free space.
                if not placed:
                    free = [(0, 0, 0, s["width"], s["height"], s["depth"], None, False)]
                else:
                    free = []

            self.shelves.append({
                "id": s.get("id"),
                "width": s.get("width", self.shelf_width),
                "height": s.get("height", self.shelf_height),
                "depth": s.get("depth", self.shelf_depth),
                "compatibility": set(),         # will set later
                "placed_items": placed,         # tuples
                "free_spaces": free,            # tuples
            })

        # carry forward previous unplaced (optional) — mark them as existing (is_new=False)
        prev_unplaced = state_dict.get("unplaced_items", [])
        for ui in prev_unplaced:
            self.unplaced_items.append((
                ui["width"], ui["height"], ui["depth"],
                ui.get("item_type"), ui.get("color"),
                ui.get("product_id"),
                False,  # is_new
                ui.get("reason")  # may be None if older state had no reasons
            ))

    def _ensure_shelf_count(self):
        """
        If the loaded state has fewer shelves than requested shelf_count,
        append empty/open shelves so new item types (e.g., flammable) can be placed.
        """
        current = len(self.shelves)
        if current >= self.shelf_count:
            return

        for i in range(current, self.shelf_count):
            self.shelves.append({
                "id": i,
                "width": self.shelf_width,
                "height": self.shelf_height,
                "depth": self.shelf_depth,
                "compatibility": set(),  # open until first placement
                "placed_items": [],
                "free_spaces": [
                    (0, 0, 0, self.shelf_width, self.shelf_height, self.shelf_depth, None, False)
                ],
            })

    @classmethod
    def from_state(
        cls,
        compatibility_rules,
        state_dict,
        *,
        shelf_count=None,
        shelf_width=None,
        shelf_height=None,
        shelf_depth=None,
        selected_shelf_id=None,
    ):
        """
        Build from an existing_state but allow overrides:
        - shelf_count: desired total shelves after loading (we will append empty shelves)
        - shelf_width/height/depth: global dims if you want to enforce them
        - selected_shelf_id: passthrough
        """
        if not state_dict.get("shelves"):
            raise ValueError("State dict has no shelves")

        s0 = state_dict["shelves"][0]
        width  = shelf_width  if shelf_width  is not None else s0["width"]
        height = shelf_height if shelf_height is not None else s0["height"]
        depth  = shelf_depth  if shelf_depth  is not None else s0["depth"]
        count  = shelf_count  if shelf_count  is not None else len(state_dict["shelves"])

        return cls(
            shelf_width=width,
            shelf_height=height,
            shelf_depth=depth,
            shelf_count=count,
            compatibility_rules=compatibility_rules,
            selected_shelf_id=selected_shelf_id,
            existing_state=state_dict
        )

    # ---------- Public API ----------

    def add_item(self, width, height, depth, item_type, color=None, product_id=None):
        """Queue a NEW item with optional product_id."""
        self.items.append((width, height, depth, item_type, color, product_id))

    def place_all_new_items(self):
        """Place only the items in the NEW items queue."""
        for _ in range(len(self.items)):
            self._place_next_item()

    # ---------- Packing internals ----------

    def _candidate_shelves_for_item(self, item_type):
        """Return shelves considered for placement (respect selected_shelf_id)."""
        if self.selected_shelf_id is not None:
            return [s for s in self.shelves if s["id"] == self.selected_shelf_id]
        return self.shelves

    def _fits_any_rotation(self, w, h, d, slot_w, slot_h, slot_d):
        """Return (fits:boolean, best_rot: (rw,rh,rd) or None)."""
        best = None
        fits = False
        for rw, rh, rd in [
            (w, h, d),
            (h, w, d),
            (d, w, h),
        ]:
            if rw <= slot_w and rh <= slot_h and rd <= slot_d:
                fits = True
                # choose the rotation with minimal leftover volume in this slot
                waste = (slot_w - rw) * (slot_h - rh) * (slot_d - rd)
                if best is None or waste < best[1]:
                    best = ((rw, rh, rd), waste)
        return fits, (best[0] if best else None)

    def _diagnose_unplaced(self, item_type, width, height, depth):
        """
        Build a human-readable reason for why an item couldn't be placed.
        We examine (in order):
          1) Whether the selected shelf id exists
          2) Compatibility constraints
          3) Geometric fit (largest free block & a close rotation)
        """
        candidate_shelves = self._candidate_shelves_for_item(item_type)

        # 1) No candidate shelf (bad selected_shelf_id)
        if self.selected_shelf_id is not None and not candidate_shelves:
            return f"Selected shelf #{self.selected_shelf_id} not found."

        # 2) Compatibility filtering
        compatible = []
        incompatible_ids = []
        for s in candidate_shelves:
            comp = s["compatibility"]
            if (not comp) or (item_type in comp):
                compatible.append(s)
            else:
                incompatible_ids.append(s["id"])

        if candidate_shelves and not compatible:
            # strictly restricted to selected shelf and that shelf is incompatible
            if self.selected_shelf_id is not None:
                expected = sorted(list(next(iter(candidate_shelves))["compatibility"]))
                return (
                    f"Selected shelf #{self.selected_shelf_id} incompatible with item_type '{item_type}'. "
                    f"Allowed types: {expected if expected else '[open]'}."
                )
            # multi-shelf case but none compatible
            return f"No compatible shelf allows item_type '{item_type}'."

        # 3) Geometry: show largest available block and closest rotation miss
        # Gather the largest free block among compatible shelves
        largest_block = None  # (vol, (w,h,d), shelf_id, (x,y,z))
        for s in compatible:
            for (x, y, z, w, h, d, _cbpid, _cbnew) in s["free_spaces"]:
                vol = w * h * d
                if largest_block is None or vol > largest_block[0]:
                    largest_block = (vol, (w, h, d), s["id"], (x, y, z))

        if largest_block is None:
            return "No free spaces are available on the selected/compatible shelf."

        (slot_w, slot_h, slot_d) = largest_block[1]
        shelf_id = largest_block[2]
        # check if *any* rotation fits any incompatible details (we already know no fit existed)
        fits, best_rot = self._fits_any_rotation(width, height, depth, slot_w, slot_h, slot_d)
        # fits will be False here, since we call this only after failing to place
        # Provide dimensional hint
        rotations = [(width, height, depth), (height, width, depth), (depth, width, height)]
        # find the rotation that is "closest" to fitting: minimize sum of positive overflows
        def overflow_sum(rw, rh, rd, W, H, D):
            return max(rw - W, 0) + max(rh - H, 0) + max(rd - D, 0)
        closest = min(rotations, key=lambda r: overflow_sum(r[0], r[1], r[2], slot_w, slot_h, slot_d))
        ow = max(closest[0] - slot_w, 0)
        oh = max(closest[1] - slot_h, 0)
        od = max(closest[2] - slot_d, 0)

        return (
            "No free space large enough on the selected/compatible shelf. "
            f"Largest free block on shelf #{shelf_id} is {slot_w}×{slot_h}×{slot_d} (WxHxD). "
            f"Closest rotation of the item is {closest[0]}×{closest[1]}×{closest[2]}, "
            f"which exceeds by ΔW={ow}, ΔH={oh}, ΔD={od}."
        )

    def _find_best_slot(self, item_type, width, height, depth):
        best = None
        min_waste = float("inf")

        # Restrict to selected shelf when provided (strict rule).
        candidate_shelves = self._candidate_shelves_for_item(item_type)

        for shelf in candidate_shelves:
            # Shelf compatibility: empty set => open; otherwise item_type must be allowed.
            if shelf["compatibility"] and item_type not in shelf["compatibility"]:
                continue

            for i, (x, y, z, w, h, d, _cbpid, _cbnew) in enumerate(shelf["free_spaces"]):
                # axis-aligned rotations
                for rw, rh, rd in [
                    (width,  height, depth),
                    (height, width,  depth),
                    (depth,  width,  height),
                ]:
                    if rw <= w and rh <= h and rd <= d:
                        waste = (w - rw) * (h - rh) * (d - rd)
                        if waste < min_waste:
                            min_waste = waste
                            best = (shelf, i, x, y, z, rw, rh, rd)

        return best

    def _place_next_item(self):
        if not self.items:
            return

        width, height, depth, item_type, color, product_id = self.items.pop(0)
        fit = self._find_best_slot(item_type, width, height, depth)

        if not fit:
            # could not place — leave items fixed, record as unplaced (is_new=True) with reason
            reason = self._diagnose_unplaced(item_type, width, height, depth)
            self.unplaced_items.append((width, height, depth, item_type, color, product_id, True, reason))
            return

        shelf, idx, x, y, z, w, h, d = fit

        # Initialize shelf compatibility if it was open
        if not shelf["compatibility"]:
            shelf["compatibility"] = self.compatibility_rules.get(item_type, {item_type}).copy()

        # Place the item (store product_id, mark as new)
        shelf["placed_items"].append((x, y, z, w, h, d, item_type, color, product_id, True))

        # Correct guillotine split: compute leftovers inside the *old* free-space bounds
        old_x, old_y, old_z, old_w, old_h, old_d, _cbpid, _cbnew = shelf["free_spaces"][idx]
        del shelf["free_spaces"][idx]

        # +X within old space
        rx = (old_x + old_w) - (x + w)
        if rx > 0:
            shelf["free_spaces"].append((x + w, y, z, rx, h, d, product_id, True))

        # +Y within old space
        ry = (old_y + old_h) - (y + h)
        if ry > 0:
            shelf["free_spaces"].append((x, y + h, z, w, ry, d, product_id, True))

        # +Z within old space
        rz = (old_z + old_d) - (z + d)
        if rz > 0:
            shelf["free_spaces"].append((x, y, z + d, w, h, rz, product_id, True))

        # Optional: keep degenerate out
        shelf["free_spaces"] = [
            fs for fs in shelf["free_spaces"] if fs[3] > 0 and fs[4] > 0 and fs[5] > 0
        ]

    # ---------- Visualization & Serialization ----------

    def _draw_frame(self):
        self.ax.clear()
        self.ax.set_title("3D Shelf Packing (Incremental — fixed items stay put)", fontsize=14)
        self.ax.set_xlabel("Width (X)")
        self.ax.set_ylabel("Height (Y)")
        self.ax.set_zlabel("Depth (Z)")

        self.ax.set_xlim(0, self.shelf_width)
        self.ax.set_ylim(0, self.shelf_height)
        self.ax.set_zlim(0, self.shelf_depth * self.shelf_count + 20)
        self.ax.view_init(elev=25, azim=-60)

        color_map = {
            "toxic": "red",
            "acid": "blue",
            "flammable": "orange",
            "biohazard": "purple",
            "explosive": "yellow",
            "corrosive": "brown",
            "normal": "green",
        }

        # Draw shelves & placed items (only selected shelf if provided)
        for shelf in self.shelves:
            shelf_id = shelf["id"]
            if self.selected_shelf_id is not None and shelf_id != self.selected_shelf_id:
                continue

            x0, y0, z0 = 0, 0, shelf_id * self.shelf_depth
            x1 = self.shelf_width
            y1 = self.shelf_height
            z1 = z0 + self.shelf_depth

            edges = [
                [(x0, x1), (y0, y0), (z0, z0)],
                [(x0, x1), (y1, y1), (z0, z0)],
                [(x0, x1), (y0, y0), (z1, z1)],
                [(x0, x1), (y1, y1), (z1, z1)],
                [(x0, x0), (y0, y1), (z0, z0)],
                [(x1, x1), (y0, y1), (z0, z0)],
                [(x0, x0), (y0, y1), (z1, z1)],
                [(x1, x1), (y0, y1), (z1, z1)],
                [(x0, x0), (y0, y0), (z0, z1)],
                [(x1, x1), (y0, y0), (z0, z1)],
                [(x0, x0), (y1, y1), (z0, z1)],
                [(x1, x1), (y1, y1), (z0, z1)],
            ]
            for xe, ye, ze in edges:
                self.ax.plot3D(xe, ye, ze, color='blue', linewidth=1.0, alpha=0.3)

            for x, y, z, w, h, d, item_type, item_color, _pid, _is_new in shelf["placed_items"]:
                z_offset = shelf_id * self.shelf_depth
                color = item_color if item_color else color_map.get(item_type, "gray")
                self.ax.bar3d(x, y, z_offset + z, w, h, d, color=color, alpha=0.7, edgecolor="black")

        # legend
        seen = {}
        for shelf in self.shelves:
            for _, _, _, _, _, _, item_type, item_color, _pid, _is_new in shelf["placed_items"]:
                if item_type not in seen:
                    seen[item_type] = item_color if item_color else color_map.get(item_type, "gray")
        legend_patches = [mpatches.Patch(color=c, label=t) for t, c in seen.items()]
        if legend_patches:
            self.ax.legend(handles=legend_patches, loc='upper left', fontsize=7)

    def animate(self, save_path="static/shelf_incremental.gif"):
        """Render final state to a single-frame GIF."""
        def _frame(_):
            self._draw_frame()
            return []
        anim = animation.FuncAnimation(
            self.fig, _frame, frames=1, interval=500, repeat=False, blit=False
        )
        anim.save(save_path, writer='pillow')
        plt.close(self.fig)

    def get_packing_result_json(self):
        shelves_data = []
        for shelf in self.shelves:
            shelves_data.append({
                "id": shelf["id"],
                "width": shelf["width"],
                "height": shelf["height"],
                "depth": shelf["depth"],
                "placed_items": [
                    {
                        "x": x, "y": y, "z": z,
                        "width": w, "height": h, "depth": d,
                        "item_type": item_type,
                        "color": color,
                        "product_id": product_id,
                        "is_new": is_new
                    }
                    for (x, y, z, w, h, d, item_type, color, product_id, is_new) in shelf["placed_items"]
                ],
                "free_spaces": [
                    {
                        "x": x, "y": y, "z": z,
                        "width": w, "height": h, "depth": d,
                        **({"created_by": {"product_id": cbpid, "is_new": cbnew}}
                           if cbpid is not None else {})
                    }
                    for (x, y, z, w, h, d, cbpid, cbnew) in shelf["free_spaces"]
                ],
            })

        result = {
            "shelves": shelves_data,
            "unplaced_items": [
                {
                    "width": width, "height": height, "depth": depth,
                    "item_type": item_type, "color": color,
                    "product_id": product_id,
                    "is_new": is_new,
                    **({"reason": reason} if reason else {})
                }
                for (width, height, depth, item_type, color, product_id, is_new, reason) in self.unplaced_items
            ]
        }
        return result
