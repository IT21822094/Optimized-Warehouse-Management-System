# ShelfSpaceOptimization/shelf_probblem_2.py

import os
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simple lightweight data containers
class Position:
    def __init__(self, x,y,z): self.x,self.y,self.z = x,y,z

class Dimensions:
    def __init__(self, w,h,d): self.width,self.height,self.depth = w,h,d

class PlacedItem:
    def __init__(self, position, size, item_type, color, product_id=None, is_new=True):
        self.position = position
        self.size = size
        self.item_type = item_type
        self.color = color
        self.product_id = product_id
        self.is_new = is_new

class FreeSpace:
    def __init__(self, position, size, created_by=None):
        self.position = position
        self.size = size
        self.created_by = created_by or {}  # {"product_id":..., "is_new":bool}

class Shelf:
    def __init__(self, shelf_id, dims: Dimensions):
        self.shelf_id = shelf_id
        self.dimensions = dims
        self.compatibility = set()
        self.placed_items = []   # [PlacedItem,...]
        self.free_spaces = [FreeSpace(Position(0,0,0), Dimensions(dims.width,dims.height,dims.depth))]

class FixedShelfPacker3DMongo:
    """
    Packer that keeps product_id and produces Mongo-ready JSON.
    """
    def __init__(self, shelf_width, shelf_height, shelf_depth, shelf_count, compatibility_rules, selected_shelf_id=None):
        self.dims = Dimensions(shelf_width, shelf_height, shelf_depth)
        self.shelves = [Shelf(i, self.dims) for i in range(shelf_count)]
        self.compatibility_rules = {k: set(v) for k,v in compatibility_rules.items()}
        self.selected_shelf_id = selected_shelf_id
        self.items = []  # (w,h,d,type,color,product_id)

        self.fig = plt.figure(figsize=(12,8))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def add_item(self, w,h,d, item_type, color=None, product_id=None):
        self.items.append((w,h,d,item_type,color,product_id))

    # ---------- packing ----------

    def _candidate_shelves(self):
        if self.selected_shelf_id is not None:
            return [s for s in self.shelves if s.shelf_id == self.selected_shelf_id]
        return self.shelves

    def _try_place_on_shelf(self, shelf: Shelf, w,h,d, item_type, color, product_id):
        best = None
        min_waste = float("inf")

        for idx, fs in enumerate(shelf.free_spaces):
            x,y,z = fs.position.x, fs.position.y, fs.position.z
            W,H,D = fs.size.width, fs.size.height, fs.size.depth

            for rw,rh,rd in [(w,h,d),(h,w,d),(d,w,h)]:
                if rw<=W and rh<=H and rd<=D:
                    waste = (W-rw)*(H-rh)*(D-rd)
                    if waste < min_waste:
                        min_waste = waste
                        best = (idx, x,y,z, rw,rh,rd)

        if best is None:
            return False

        idx, x,y,z, rw,rh,rd = best

        # init shelf compatibility if open
        if not shelf.compatibility:
            shelf.compatibility = self.compatibility_rules.get(item_type, {item_type}).copy()
        else:
            if item_type not in shelf.compatibility:
                return False  # respect compatibility

        # place
        shelf.placed_items.append(
            PlacedItem(Position(x,y,z), Dimensions(rw,rh,rd), item_type, color, product_id, is_new=True)
        )

        # correct split inside consumed free-space
        old = shelf.free_spaces[idx]
        old_x, old_y, old_z = old.position.x, old.position.y, old.position.z
        old_W, old_H, old_D = old.size.width, old.size.height, old.size.depth
        del shelf.free_spaces[idx]

        created_by = {"product_id": product_id, "is_new": True}

        # +X
        rx = (old_x + old_W) - (x + rw)
        if rx > 0:
            shelf.free_spaces.append(
                FreeSpace(Position(x+rw, y, z), Dimensions(rx, rh, rd), created_by=created_by)
            )
        # +Y
        ry = (old_y + old_H) - (y + rh)
        if ry > 0:
            shelf.free_spaces.append(
                FreeSpace(Position(x, y+rh, z), Dimensions(rw, ry, rd), created_by=created_by)
            )
        # +Z
        rz = (old_z + old_D) - (z + rd)
        if rz > 0:
            shelf.free_spaces.append(
                FreeSpace(Position(x, y, z+rd), Dimensions(rw, rh, rz), created_by=created_by)
            )

        # prune degenerates
        shelf.free_spaces = [
            fs for fs in shelf.free_spaces
            if fs.size.width>0 and fs.size.height>0 and fs.size.depth>0
        ]
        return True

    def place_all(self):
        for (w,h,d,t,c,pid) in self.items:
            placed = False
            # Strict: only selected shelf if provided
            for shelf in self._candidate_shelves():
                if self._try_place_on_shelf(shelf, w,h,d, t,c,pid):
                    placed = True
                    break
            if not placed:
                # leave unplaced (export step will include them if you want)
                pass

    # ---------- output ----------

    def export_embedded_document(self):
        return {
            "shelves": [
                {
                    "id": sh.shelf_id,
                    "width": self.dims.width,
                    "height": self.dims.height,
                    "depth": self.dims.depth,
                    "placed_items": [
                        {
                            "x": pi.position.x, "y": pi.position.y, "z": pi.position.z,
                            "width": pi.size.width, "height": pi.size.height, "depth": pi.size.depth,
                            "item_type": pi.item_type, "color": pi.color,
                            "product_id": pi.product_id, "is_new": pi.is_new
                        }
                        for pi in sh.placed_items
                    ],
                    "free_spaces": [
                        {
                            "x": fs.position.x, "y": fs.position.y, "z": fs.position.z,
                            "width": fs.size.width, "height": fs.size.height, "depth": fs.size.depth,
                            **({"created_by": fs.created_by} if fs.created_by else {})
                        }
                        for fs in sh.free_spaces
                    ],
                }
                for sh in self.shelves
            ]
        }

    def animate(self, save_path):
        self.ax.clear()
        self.ax.set_title("3D Shelf Packing (Mongo export)", fontsize=14)
        self.ax.set_xlabel("Width (X)")
        self.ax.set_ylabel("Height (Y)")
        self.ax.set_zlabel("Depth (Z)")
        self.ax.set_xlim(0, self.dims.width)
        self.ax.set_ylim(0, self.dims.height)
        self.ax.set_zlim(0, self.dims.depth * len(self.shelves) + 20)
        self.ax.view_init(elev=25, azim=-60)

        for sh in self.shelves:
            if self.selected_shelf_id is not None and sh.shelf_id != self.selected_shelf_id:
                continue
            zoff = sh.shelf_id * self.dims.depth
            for pi in sh.placed_items:
                self.ax.bar3d(
                    pi.position.x, pi.position.y, zoff + pi.position.z,
                    pi.size.width, pi.size.height, pi.size.depth,
                    alpha=0.7, edgecolor="black"
                )

        anim = animation.FuncAnimation(self.fig, lambda _ : [], frames=1, interval=400, repeat=False)
        anim.save(save_path, writer='pillow')
        plt.close(self.fig)


# ---------- public entrypoint used by main.py ----------

def pack_from_request_payload(data: dict, save_gif: bool = True):
    """
    Request JSON:
    {
      "shelf_width": number,
      "shelf_height": number,
      "shelf_depth": number,
      "shelf_count": number,
      "selected_shelf_id": int | null,
      "compatibility_rules": { "flammable": ["flammable", ...], ... },
      "items": [ [w,h,d,"type","color","product_id"], ... ]
    }
    """
    shelf_width  = data["shelf_width"]
    shelf_height = data["shelf_height"]
    shelf_depth  = data["shelf_depth"]
    shelf_count  = data["shelf_count"]
    compat       = data["compatibility_rules"]
    selected     = data.get("selected_shelf_id")

    packer = FixedShelfPacker3DMongo(
        shelf_width, shelf_height, shelf_depth, shelf_count,
        compatibility_rules=compat,
        selected_shelf_id=selected
    )

    for item in data.get("items", []):
        w,h,d,t,c = item[0], item[1], item[2], item[3], item[4]
        pid = item[5] if len(item) > 5 else None
        packer.add_item(w,h,d,t,c,pid)

    packer.place_all()

    # render
    out_path = f"static/shelf_{uuid.uuid4().hex}.gif"
    if save_gif:
        os.makedirs("static", exist_ok=True)
        packer.animate(out_path)

    run_doc = packer.export_embedded_document()
    return {"run": run_doc, "video_url": f"/{out_path}" if save_gif else None}
