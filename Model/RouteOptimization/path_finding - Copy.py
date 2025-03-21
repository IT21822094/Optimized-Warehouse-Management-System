import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
import os
import sys
from itertools import permutations
import matplotlib
matplotlib.use('Agg')

def shortest_path(warehouse, start, goal):
    s = tuple(start)
    t = tuple(goal)

    pq = []
    seen = set()
    dist = {s: 0}
    rank = {s: np.linalg.norm(np.array(s) - np.array(t))}
    prev = {}

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

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nxt = (cur[0] + dx, cur[1] + dy)

            if 0 <= nxt[0] < warehouse.shape[0] and 0 <= nxt[1] < warehouse.shape[1]:
                if warehouse[nxt[0], nxt[1]] == 1 and nxt != t:
                    continue
                if nxt in seen:
                    continue

                step_cost = dist[cur] + 1
                if nxt not in dist or step_cost < dist[nxt]:
                    prev[nxt] = cur
                    dist[nxt] = step_cost
                    rank[nxt] = step_cost + np.linalg.norm(np.array(nxt) - np.array(t))
                    heapq.heappush(pq, (rank[nxt], nxt))

    return []


def create_warehouse(shelf_height=15, shelf_count=0, shelf_interval=2, obstacles=None):
    width = 1 + shelf_interval * shelf_count
    grid = np.zeros((shelf_height, width), dtype=np.uint8)

    for i in range(shelf_count):
        x = shelf_interval * i + shelf_interval - 1
        for h in range(1, shelf_height - 1):
            grid[h, x] = 1

    if obstacles:
        for x, y in obstacles:
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                grid[x, y] = 1
    return grid

def _seg_len(grid, a, b):
    p = shortest_path(grid, a, b)
    return (len(p) - 1) if p else np.inf

def order_stops(grid, picks, optimize=True):
    if not picks:
        return []

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

    seq = [start]
    remaining = rest.copy()
    while remaining:
        cur = seq[-1]
        nxt = min(remaining, key=lambda p: _seg_len(grid, cur, p))
        seq.append(nxt); remaining.remove(nxt)
    return seq


def animate_dynamic_step_by_step(warehouse,
                                 picking_locations,
                                 obstacles=None,
                                 optimize_order=True,
                                 lock_picked=True):

    base_grid = warehouse.copy()
    obs = list(obstacles or [])

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.invert_yaxis()
    ax.set_title("Warehouse Path")

    route = order_stops(base_grid, picking_locations, optimize=optimize_order)

    path = []
    full_path = []
    current_step = [0]
    next_target = [1]
    done_flag = [False]
    picked = set([tuple(route[0])])

    line, = ax.plot([], [], 'g-', lw=2)
    point, = ax.plot([], [], 'ro')

    ax.imshow(base_grid, cmap='Blues', interpolation='nearest')
    for loc in route:
        ax.plot(loc[1], loc[0], 'yo', markersize=6)
    for ox, oy in obs:
        ax.plot(oy, ox, 'bs', markersize=8)

    def update(_frame):
        nonlocal path, full_path
        if not path:
            if next_target[0] >= len(route):
                if not done_flag[0]:
                    print("All paths done.")
                    done_flag[0] = True
                    ani.event_source.stop()
                return line, point

            start = tuple(route[current_step[0]])
            goal = tuple(route[next_target[0]])

            grid = base_grid.copy()
            if lock_picked:
                for (px, py) in picked:
                    if (px, py) != goal:
                        grid[px, py] = 1

            path = shortest_path(grid, start, goal)
            if not path:
                print(f"No path from {start} to {goal}")
                current_step[0] = next_target[0]
                next_target[0] += 1
                return line, point

        step = path.pop(0)
        full_path.append(step)

        x_vals = [p[1] for p in full_path]
        y_vals = [p[0] for p in full_path]
        line.set_data(x_vals, y_vals)
        point.set_data([step[1]], [step[0]])

        if not path:
            picked.add(tuple(route[next_target[0]]))
            if next_target[0] >= len(route) - 1:
                if not done_flag[0]:
                    print("All paths done.")
                    done_flag[0] = True
                    ani.event_source.stop()
            else:
                current_step[0] = next_target[0]
                next_target[0] += 1

        return line, point

    ani = animation.FuncAnimation(
        fig, update, interval=300, blit=False, cache_frame_data=False
    )
    return fig, ani


def run_pathfinding_animation_dynamic(
    shelf_height,
    shelf_count,
    shelf_interval,
    picking_locations,
    obstacles=None,
    save_path="static/path.gif",
    optimize_order=True,
    lock_picked=True,
):
    from matplotlib.animation import PillowWriter, FFMpegWriter  # import both here

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    warehouse = create_warehouse(shelf_height, shelf_count, shelf_interval, obstacles)
    route = order_stops(warehouse, picking_locations, optimize=optimize_order)

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.invert_yaxis()
    ax.set_title("Warehouse Path")

    ax.imshow(warehouse, cmap='Blues', interpolation='nearest')
    for loc in route:
        ax.plot(loc[1], loc[0], 'yo', markersize=6)
    if obstacles:
        for ox, oy in obstacles:
            ax.plot(oy, ox, 'bs', markersize=8)

    line, = ax.plot([], [], 'g-', lw=2)
    point, = ax.plot([], [], 'ro')

    full_path = []
    picked = set([tuple(route[0])])

    # choose writer by extension
    ext = os.path.splitext(save_path)[1].lower()
    if ext == ".gif":
        writer = PillowWriter(fps=2)
    else:
        # fall back to ffmpeg for mp4/mov, but this requires ffmpeg installed
        writer = FFMpegWriter(fps=2, bitrate=800)

    try:
        with writer.saving(fig, save_path, dpi=60):
            for i in range(len(route) - 1):
                start = tuple(route[i])
                goal = tuple(route[i + 1])

                grid = warehouse.copy()
                if lock_picked:
                    for px, py in picked:
                        if (px, py) != goal:
                            grid[px, py] = 1

                path = shortest_path(grid, start, goal)
                if not path:
                    print(f"No path from {start} to {goal}")
                    continue

                for step in path:
                    full_path.append(step)
                    x_vals = [p[1] for p in full_path]
                    y_vals = [p[0] for p in full_path]
                    line.set_data(x_vals, y_vals)
                    point.set_data([step[1]], [step[0]])
                    writer.grab_frame()

                picked.add(goal)
    except FileNotFoundError as e:
        # This will happen if ext != .gif and ffmpeg isn't available
        raise RuntimeError(
            "Failed to write animation. "
            "If you're saving to MP4 you need ffmpeg installed and in PATH. "
            "Either install ffmpeg or save as .gif to use PillowWriter."
        ) from e
    finally:
        plt.close(fig)

    print("All paths done.")


# if __name__ == "__main__":
#     # your example
#     shelf_height = 15
#     shelf_count = 5
#     shelf_interval = 2
#
#     picking_locations = [[1, 0], [3, 2], [5, 4]]
#     workers = [[2, 2], [14, 0]]  # treated as obstacles
#
#     # Finds the shortest route (start fixed at [1,0]) and stops at its last pick
#     run_pathfinding_animation_dynamic(
#         shelf_height=shelf_height,
#         shelf_count=shelf_count,
#         shelf_interval=shelf_interval,
#         picking_locations=picking_locations,
#         obstacles=workers,
#         save_path="output/pathfinding.mp4",
#         optimize_order=True,   # exact for small sets
#         lock_picked=True       # never step onto a previously picked cell
#     )
