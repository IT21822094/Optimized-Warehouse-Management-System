import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)
import matplotlib.patches as mpatches
import matplotlib

matplotlib.use('Agg')


class FixedShelfPacker3D:
    def __init__(self, shelf_width, shelf_height, shelf_depth, shelf_count, compatibility_rules, selected_shelf_id=None):
        self.shelf_width = shelf_width
        self.shelf_height = shelf_height
        self.shelf_depth = shelf_depth
        self.shelf_count = shelf_count
        self.compatibility_rules = {k: set(v) for k, v in compatibility_rules.items()}
        self.selected_shelf_id = selected_shelf_id

        self.items = []            # list of (w, h, d, item_type, color)
        self.unplaced_items = []   # list of (w, h, d, item_type, color)

        self.shelves = [
            {
                "id": i,
                "width": shelf_width,
                "height": shelf_height,
                "depth": shelf_depth,
                "compatibility": set(),
                "placed_items": [],  # list of (x, y, z, w, h, d, item_type, color)
                "free_spaces": [(0, 0, 0, shelf_width, shelf_height, shelf_depth)],
            }
            for i in range(shelf_count)
        ]

        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.current_item_index = 0

    def add_item(self, width, height, depth, item_type, color=None):
        """Append an item; color can be named ('red') or hex ('#FF0000')."""
        self.items.append((width, height, depth, item_type, color))

    def find_best_shelf(self, item_type, width, height, depth):
        best_shelf = None
        min_waste = float("inf")

        for shelf in self.shelves:
            if not shelf["compatibility"] or item_type in shelf["compatibility"]:
                for i, (x, y, z, w, h, d) in enumerate(shelf["free_spaces"]):
                    # allow three axis-aligned rotations
                    for rw, rh, rd in [(width, height, depth), (height, width, depth), (depth, width, height)]:
                        if rw <= w and rh <= h and rd <= d:
                            waste = (w - rw) * (h - rh) * (d - rd)
                            if waste < min_waste:
                                min_waste = waste
                                best_shelf = (shelf, i, x, y, z, rw, rh, rd)

        return best_shelf

    def place_item(self):
        if self.current_item_index >= len(self.items):
            return

        width, height, depth, item_type, color = self.items[self.current_item_index]
        best_fit = self.find_best_shelf(item_type, width, height, depth)

        if best_fit:
            shelf, i, x, y, z, w, h, d = best_fit

            # initialize shelf compatibility when first item is placed
            if not shelf["compatibility"]:
                shelf["compatibility"] = self.compatibility_rules.get(item_type, {item_type})

            # place the item and split free space (simple guillotine split along +x, +y, +z)
            shelf["placed_items"].append((x, y, z, w, h, d, item_type, color))
            del shelf["free_spaces"][i]

            shelf["free_spaces"].append((x + w, y, z, shelf["width"] - (x + w), h, d))
            shelf["free_spaces"].append((x, y + h, z, w, shelf["height"] - (y + h), d))
            shelf["free_spaces"].append((x, y, z + d, w, h, shelf["depth"] - (z + d)))
        else:
            self.unplaced_items.append((width, height, depth, item_type, color))

        self.current_item_index += 1

    def update_animation(self, frame):
        self.ax.clear()
        self.place_item()

        self.ax.set_title("3D Shelf Packing (Selected Shelf Only)", fontsize=14)
        self.ax.set_xlabel("Width (X)")
        self.ax.set_ylabel("Height (Y)")
        self.ax.set_zlabel("Depth/Shelf (Z)")

        self.ax.set_xlim(0, self.shelf_width)
        self.ax.set_ylim(0, self.shelf_height)
        self.ax.set_zlim(0, self.shelf_depth * self.shelf_count + 20)
        self.ax.view_init(elev=25, azim=-60)

        # Fallback colors if an item doesn't provide one
        color_map = {
            "toxic": "red",
            "acid": "blue",
            "flammable": "orange",
            "biohazard": "purple",
            "explosive": "yellow",
            "corrosive": "brown",
            "normal": "green",
        }

        # Draw each shelf frame and items
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
            for x_edge, y_edge, z_edge in edges:
                self.ax.plot3D(x_edge, y_edge, z_edge, color='blue', linewidth=1.0, alpha=0.3)

            # Draw all items for the selected shelf
            for x, y, z, w, h, d, item_type, item_color in shelf["placed_items"]:
                z_offset = shelf_id * self.shelf_depth
                color = item_color if item_color else color_map.get(item_type, "gray")
                self.ax.bar3d(x, y, z_offset + z, w, h, d, color=color, alpha=0.7, edgecolor="black")

        # Info panel (right side)
        info_lines = []
        for shelf in self.shelves:
            if self.selected_shelf_id is not None and shelf["id"] != self.selected_shelf_id:
                continue
            for idx, (x, y, z, w, h, d, item_type, item_color) in enumerate(shelf["placed_items"]):
                info_lines.append(
                    f"[#{idx + 1}] {item_type} ({w}x{h}x{d}) → (x={x}, y={y}, z={z})"
                )

        if info_lines:
            self.ax.text2D(
                1.05, 0.95,
                "\n".join(info_lines),
                transform=self.ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.7)
            )

        # Build a legend from actually seen types (first seen color per type)
        seen = {}
        for shelf in self.shelves:
            for _, _, _, _, _, _, item_type, item_color in shelf["placed_items"]:
                if item_type not in seen:
                    seen[item_type] = item_color if item_color else color_map.get(item_type, "gray")

        legend_patches = [mpatches.Patch(color=c, label=t) for t, c in seen.items()]
        if legend_patches:
            self.ax.legend(handles=legend_patches, loc='upper left', fontsize=7)

    def animate(self, save_path="static/shelf_animation.mp4"):
        anim = animation.FuncAnimation(self.fig, self.update_animation, frames=len(self.items) + 2, interval=500, repeat=False, blit=False)
        anim.save(save_path, writer='pillow')  # .gif supported by pillow
        plt.close(self.fig)

        if self.unplaced_items:
            print("Unplaced Items:")
            for w, h, d, t, c in self.unplaced_items:
                print(f"- {t} ({w}x{h}x{d}) color={c}")

    def get_packing_result_json(self):
        shelves_data = []

        for shelf in self.shelves:
            shelf_dict = {
                "id": shelf["id"],
                "width": shelf["width"],
                "height": shelf["height"],
                "depth": shelf["depth"],
                "placed_items": [
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "width": w,
                        "height": h,
                        "depth": d,
                        "item_type": item_type,
                        "color": color,
                    }
                    for (x, y, z, w, h, d, item_type, color) in shelf["placed_items"]
                ],
                "free_spaces": [
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "width": w,
                        "height": h,
                        "depth": d
                    }
                    for (x, y, z, w, h, d) in shelf["free_spaces"]
                ]
            }
            shelves_data.append(shelf_dict)

        result = {
            "shelves": shelves_data,
            "unplaced_items": [
                {
                    "width": width,
                    "height": height,
                    "depth": depth,
                    "item_type": item_type,
                    "color": color,
                }
                for (width, height, depth, item_type, color) in self.unplaced_items
            ]
        }

        return result
