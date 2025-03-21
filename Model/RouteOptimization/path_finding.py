import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
from itertools import permutations
import matplotlib
matplotlib.use('Agg')

# --------------------------- Cost model ---------------------------
PREFERRED_ROWS_DEFAULT = set()
PREFERRED_COLS_DEFAULT = set()
LANE_COST = 1
NORMAL_COST = 1

def _step_cost(cell, preferred_rows, preferred_cols):
    r, c = cell
    return LANE_COST if (r in preferred_rows or c in preferred_cols) else NORMAL_COST


# --------------------------- Labels / helpers ---------------------------
def shelf_label_for_column(col, shelf_interval):
    """
    Shelves are placed on columns: x = shelf_interval*(i+1) - 1
    So shelf_no = (col + 1) // shelf_interval
    """
    shelf_no = max(1, int((col + 1) // shelf_interval))
    return f"Shelf {shelf_no:02d}"


def snap_to_aisle(grid, cell):
    """If a target lands on a blocked cell, move it to a free neighbor or nearest aisle."""
    r, c = map(int, cell)
    H, W = grid.shape
    if 0 <= r < H and 0 <= c < W and grid[r, c] == 0:
        return (r, c)
    for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W and grid[rr, cc] == 0:
            return (rr, cc)
    # fallback: BFS to nearest aisle cell
    from collections import deque
    q, seen = deque([(r, c)]), {(r, c)}
    while q:
        rr, cc = q.popleft()
        if 0 <= rr < H and 0 <= cc < W and grid[rr, cc] == 0:
            return (rr, cc)
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = rr+dr, cc+dc
            if (0 <= nr < H and 0 <= nc < W) and (nr, nc) not in seen:
                seen.add((nr, nc)); q.append((nr, nc))
    return (r, c)


# --------------------------- Grids ---------------------------
def create_warehouse(shelf_height=15, shelf_count=0, shelf_interval=2, obstacles=None):
    """
    Warehouse grid for drawing: 0 = free, 1 = shelf/obstacle.
    Shelves are single blocked columns at x = shelf_interval*(i+1) - 1 for row=1..H-2.
    Row 0 (top) and row H-1 (bottom) stay free for visuals/routes if enabled in aisle grid.
    """
    width = 1 + shelf_interval * shelf_count
    H, W = shelf_height, width
    grid = np.zeros((H, W), dtype=np.uint8)

    # shelves (block middle rows so the bar is visible, leave top/bottom free)
    for i in range(shelf_count):
        x = shelf_interval * i + shelf_interval - 1
        for h in range(1, H - 1):
            grid[h, x] = 1

    if obstacles:
        for x, y in obstacles:
            if 0 <= x < H and 0 <= y < W:
                grid[x, y] = 1
    return grid


def build_aisle_grid(
    shelf_height: int,
    shelf_count: int,
    shelf_interval: int,
    keep_perimeter_aisle: bool = False,
    cross_aisle_rows: set | None = None,
):
    """
    Returns a grid where AISLES are 0 (free) and everything else 1 (blocked).
    - For each shelf column, we open the LEFT lane (col = shelf_col-1, else RIGHT lane).
    - Optionally open top/bottom perimeter rows.
    - Optionally open specific cross-aisle rows (e.g., bottom row only).
    """
    H = int(shelf_height)
    W = 1 + shelf_interval * shelf_count
    g = np.ones((H, W), dtype=np.uint8)  # start blocked

    # Perimeter aisles
    if keep_perimeter_aisle and H >= 2:
        g[0, :] = 0
        g[H - 1, :] = 0

    # User-specified cross-aisles (safe way to allow horizontal travel)
    if cross_aisle_rows:
        for r in cross_aisle_rows:
            if 0 <= r < H:
                g[r, :] = 0

    # Aisles next to each shelf column
    for i in range(shelf_count):
        shelf_col = shelf_interval * (i + 1) - 1
        left_aisle = shelf_col - 1
        right_aisle = shelf_col + 1
        if 0 <= left_aisle < W:
            g[:, left_aisle] = 0
        elif 0 <= right_aisle < W:
            g[:, right_aisle] = 0

    return g


# --------------------------- A* ---------------------------
def shortest_path(warehouse, start, goal,
                  preferred_rows=PREFERRED_ROWS_DEFAULT,
                  preferred_cols=PREFERRED_COLS_DEFAULT):
    s = tuple(start); t = tuple(goal)
    pq, seen, dist, rank, prev = [], set(), {s: 0}, {s: np.linalg.norm(np.array(s)-np.array(t))}, {}
    heapq.heappush(pq, (rank[s], s))

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == t:
            path = []
            while cur in prev:
                path.append(cur)
                cur = prev[cur]
            path.append(s)
            path.reverse()
            return path
        if cur in seen:
            continue
        seen.add(cur)

        for dx, dy in ((0,1), (1,0), (0,-1), (-1,0)):
            nxt = (cur[0]+dx, cur[1]+dy)
            if 0 <= nxt[0] < warehouse.shape[0] and 0 <= nxt[1] < warehouse.shape[1]:
                if warehouse[nxt[0], nxt[1]] == 1:
                    continue
                if nxt in seen:
                    continue

                step_cost = dist[cur] + _step_cost(nxt, preferred_rows, preferred_cols)
                if nxt not in dist or step_cost < dist[nxt]:
                    prev[nxt] = cur
                    dist[nxt] = step_cost
                    rank[nxt] = step_cost + np.linalg.norm(np.array(nxt) - np.array(t))
                    heapq.heappush(pq, (rank[nxt], nxt))
    return []


# --------------------------- Stop ordering (shortest total) ---------------------------
def _seg_len(grid, a, b):
    p = shortest_path(grid, a, b)
    return (len(p) - 1) if p else np.inf


def order_stops(grid, picks, optimize=True):
    """Small exact TSP (<=8) else greedy nearest-neighbor."""
    if not picks:
        return []

    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for p in map(tuple, picks):
        if p not in seen:
            uniq.append(p); seen.add(p)

    start = uniq[0]
    rest = uniq[1:]

    if optimize and len(rest) <= 8:
        best = None
        best_cost = np.inf
        for mid_perm in permutations(rest):
            seq = [start] + list(mid_perm)
            cost = 0.0
            ok = True
            for i in range(len(seq) - 1):
                c = _seg_len(grid, seq[i], seq[i+1])
                if not np.isfinite(c):
                    ok = False; break
                cost += c
            if ok and cost < best_cost:
                best = seq; best_cost = cost
        return best if best else [start] + rest

    # greedy nearest-neighbor
    seq = [start]
    remaining = rest.copy()
    while remaining:
        cur = seq[-1]
        nxt = min(remaining, key=lambda p: _seg_len(grid, cur, p))
        seq.append(nxt); remaining.remove(nxt)
    return seq


# --------------------------- Animation helpers ---------------------------
def animate_dynamic_step_by_step(warehouse,
                                 picking_locations,
                                 obstacles=None,
                                 optimize_order=True,
                                 lock_picked=False,
                                 shelf_interval=2,
                                 pause_pick_frames=8):
    from matplotlib.patches import FancyArrowPatch

    base_grid = warehouse.copy()
    obs = list(obstacles or [])

    fig, ax = plt.subplots(figsize=(6, 7), facecolor="#f8f9fb")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False); ax.invert_yaxis()
    ax.set_title("Warehouse Route", fontsize=14, fontweight="semibold", pad=12)
    ax.set_facecolor("#ffffff")

    route = order_stops(base_grid, picking_locations, optimize=optimize_order)

    path = []
    full_path = []
    current_step = [0]
    next_target = [1]
    done_flag = [False]
    picked = set([tuple(route[0])])

    # Static layers
    ax.imshow(base_grid, cmap='Blues', interpolation='nearest', alpha=0.95)

    # Plot picks + labels
    for loc in route:
        r, c = loc
        ax.plot(c, r, marker='o', markersize=6, markerfacecolor="#ffd54f",
                markeredgecolor="#8d6e63", lw=0.0)
        label = shelf_label_for_column(c, shelf_interval)
        ax.text(c + 0.2, r - 0.2, label, fontsize=8, color="#37474f",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cfd8dc", alpha=0.9))

    # Obstacles
    for ox, oy in obs:
        ax.plot(oy, ox, 's', markersize=8, markerfacecolor="#90caf9",
                markeredgecolor="#1565c0")

    # Dynamic layers
    line, = ax.plot([], [], '-', lw=2.5, color="#2e7d32", alpha=0.9)
    arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle='-|>', mutation_scale=14,
                            linewidth=2.0, color="#c62828", alpha=0.95)
    ax.add_patch(arrow); arrow.set_visible(False)

    # “Picking…” text (only visible during pauses)
    pick_text = ax.text(0, 0, "", fontsize=10, color="#1b5e20",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#a5d6a7", alpha=0.95))
    pick_text.set_visible(False)

    def set_arrow(a, p0, p1):
        a.set_positions((p0[1], p0[0]), (p1[1], p1[0]))
        a.set_visible(True)

    def pause_at(r, c):
        pick_text.set_text("Picking…")
        pick_text.set_position((c + 0.4, r - 0.4))
        pick_text.set_visible(True)
        fig.canvas.draw()  # refresh
        for _ in range(pause_pick_frames):
            yield  # let the writer grab a few frames
        pick_text.set_visible(False)

    def update(_frame):
        nonlocal path, full_path
        if not path:
            if next_target[0] >= len(route):
                if not done_flag[0]:
                    done_flag[0] = True
                    ani.event_source.stop()
                return line, arrow

            start = tuple(route[current_step[0]])
            goal = tuple(route[next_target[0]])

            grid = base_grid.copy()
            if lock_picked:
                for (px, py) in picked:
                    if (px, py) != goal:
                        grid[px, py] = 1

            start = snap_to_aisle(grid, start)
            goal  = snap_to_aisle(grid, goal)

            path = shortest_path(grid, start, goal)
            if not path:
                current_step[0] = next_target[0]
                next_target[0] += 1
                return line, arrow

        step = path.pop(0)
        full_path.append(step)

        x_vals = [p[1] for p in full_path]
        y_vals = [p[0] for p in full_path]
        line.set_data(x_vals, y_vals)

        if len(full_path) >= 2:
            p0, p1 = full_path[-2], full_path[-1]
            set_arrow(arrow, p0, p1)
        else:
            arrow.set_visible(False)

        if not path:
            # pause at the pick
            for _ in pause_at(*route[next_target[0]]):
                pass
            picked.add(tuple(route[next_target[0]]))
            if next_target[0] >= len(route) - 1:
                if not done_flag[0]:
                    done_flag[0] = True
                    ani.event_source.stop()
            else:
                current_step[0] = next_target[0]
                next_target[0] += 1

        return line, arrow

    ani = animation.FuncAnimation(fig, update, interval=250,
                                  blit=False, cache_frame_data=False)
    return fig, ani


def run_pathfinding_animation_dynamic(
    shelf_height,
    shelf_count,
    shelf_interval,
    picking_locations,
    obstacles=None,
    save_path="static/path.gif",
    optimize_order=True,
    lock_picked=False,
    force_aisles=True,
    pause_pick_frames=8,
):
    """
    If force_aisles is True:
      - Build an aisle grid and route only on those cells.
      - Use ONE bottom cross-aisle so we can switch columns without using the far top edge.
      - Picks are snapped to the aisle network before solving.
    """
    from matplotlib.animation import PillowWriter, FFMpegWriter
    from matplotlib.patches import FancyArrowPatch
    import os

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    warehouse = create_warehouse(shelf_height, shelf_count, shelf_interval, obstacles)

    # Working grid to solve on
    if force_aisles:
        # bottom row as main cross-aisle; no top perimeter to avoid big detours
        cross_rows = {int(shelf_height) - 1}
        working = build_aisle_grid(
            shelf_height, shelf_count, shelf_interval,
            keep_perimeter_aisle=False,
            cross_aisle_rows=cross_rows
        )
        if obstacles:
            for ox, oy in obstacles:
                if 0 <= ox < working.shape[0] and 0 <= oy < working.shape[1]:
                    working[ox, oy] = 1
        picking_locations = [snap_to_aisle(working, p) for p in picking_locations]
    else:
        working = warehouse

    # Order stops (shortest total)
    route = order_stops(working, picking_locations, optimize=optimize_order)

    # ---- Plot
    fig, ax = plt.subplots(figsize=(6, 7), facecolor="#f8f9fb")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.invert_yaxis()
    ax.set_title("Warehouse Route", fontsize=14, fontweight="semibold", pad=12)
    ax.set_facecolor("#ffffff")
    ax.imshow(warehouse, cmap='Blues', interpolation='nearest', alpha=0.95)

    for loc in route:
        r, c = loc
        ax.plot(c, r, marker='o', markersize=6, markerfacecolor="#ffd54f",
                markeredgecolor="#8d6e63", lw=0.0)
        ax.text(c + 0.2, r - 0.2, shelf_label_for_column(c, shelf_interval),
                fontsize=8, color="#37474f",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cfd8dc", alpha=0.9))

    if obstacles:
        for ox, oy in obstacles:
            ax.plot(oy, ox, 's', markersize=8, markerfacecolor="#90caf9",
                    markeredgecolor="#1565c0")

    line, = ax.plot([], [], '-', lw=2.5, color="#2e7d32", alpha=0.9)
    arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle='-|>', mutation_scale=14,
                            linewidth=2.0, color="#c62828", alpha=0.95)
    ax.add_patch(arrow); arrow.set_visible(False)

    # “Picking…” text for saved frames
    pick_text = ax.text(0, 0, "", fontsize=10, color="#1b5e20",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#a5d6a7", alpha=0.95))
    pick_text.set_visible(False)

    def set_arrow(a, p0, p1):
        a.set_positions((p0[1], p0[0]), (p1[1], p1[0]))
        a.set_visible(True)

    full_path = []
    picked = set([tuple(route[0])])

    # Writer selection
    ext = os.path.splitext(save_path)[1].lower()
    writer = PillowWriter(fps=3) if ext == ".gif" else FFMpegWriter(fps=3, bitrate=1200)

    try:
        with writer.saving(fig, save_path, dpi=70):
            for i in range(len(route) - 1):
                start = tuple(route[i])
                goal = tuple(route[i + 1])

                grid = working.copy()
                if lock_picked:
                    for px, py in picked:
                        if (px, py) != goal:
                            grid[px, py] = 1

                start = snap_to_aisle(grid, start)
                goal  = snap_to_aisle(grid, goal)

                path = shortest_path(grid, start, goal,
                                     preferred_rows=set(),
                                     preferred_cols=set())
                if not path:
                    continue

                for step in path:
                    full_path.append(step)
                    x_vals = [p[1] for p in full_path]
                    y_vals = [p[0] for p in full_path]
                    line.set_data(x_vals, y_vals)

                    if len(full_path) >= 2:
                        p0, p1 = full_path[-2], full_path[-1]
                        set_arrow(arrow, p0, p1)
                    else:
                        arrow.set_visible(False)

                    writer.grab_frame()

                # pause frames at the pick
                pr, pc = goal
                pick_text.set_text("Picking…")
                pick_text.set_position((pc + 0.4, pr - 0.4))
                pick_text.set_visible(True)
                for _ in range(pause_pick_frames):
                    writer.grab_frame()
                pick_text.set_visible(False)

                picked.add(goal)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Failed to write animation. If you're saving to MP4 you need ffmpeg installed and in PATH. "
            "Either install ffmpeg or save as .gif to use PillowWriter."
        ) from e
    finally:
        plt.close(fig)

    print("All paths done.")
