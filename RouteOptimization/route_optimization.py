import numpy as np
import matplotlib.pyplot as plt
import heapq
import matplotlib.animation as animation
from optimal_path import calculate_the_routing_sequence


def a_star(warehouse, start, goal):
    open_list = []
    closed_list = set()
    g_scores = {start: 0}
    f_scores = {start: np.linalg.norm(np.array(start) - np.array(goal))}
    came_from = {}

    heapq.heappush(open_list, (f_scores[start], start))

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        closed_list.add(current)

        # Check the neighbors (up, down, left, right) allowed only orthogonally moves
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < warehouse.shape[0] and 0 <= neighbor[1] < warehouse.shape[1]:
                if warehouse[neighbor[0], neighbor[1]] == 1:  # Skip shelves (blocked paths)
                    continue
                if neighbor in closed_list:
                    continue

                tentative_g_score = g_scores.get(current, float('inf')) + 1
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g_score
                    f_scores[neighbor] = g_scores[neighbor] + np.linalg.norm(np.array(neighbor) - np.array(goal))
                    heapq.heappush(open_list, (f_scores[neighbor], neighbor))

    return None


# dynamically add shelves to the grid
def add_shelves(warehouse, shelf_positions):
    for (x, y) in shelf_positions:
        if 0 <= x < warehouse.shape[0] and 0 <= y < warehouse.shape[1]:
            warehouse[x, y] = 1  # Mark this location as a shelf (blocked)


# animate the full route
def animate_full_route(paths, warehouse, grid_width, shelf_height):
    fig, ax = plt.subplots(figsize=(5, 7))

    ax.set_xticks(np.arange(0, grid_width, 1))
    ax.set_yticks(np.arange(0, shelf_height, 1))
    ax.set_xticklabels(range(grid_width))
    ax.set_yticklabels(range(shelf_height))
    ax.invert_yaxis()
    ax.grid(which='both')

    # Draw warehouse grid
    ax.imshow(warehouse, cmap='Blues', interpolation='nearest')

    # Plot all picking locations
    for path in paths:
        ax.plot(path[0][1], path[0][0], 'go')  # Green for start
        ax.plot(path[-1][1], path[-1][0], 'ro')  # Red for end

    # Line object for animation
    line, = ax.plot([], [], 'g-', lw=2)

    # Coordinate label
    coord_label = ax.text(0, 0, "", fontsize=10, color="black", fontweight="bold",
                          bbox=dict(facecolor='white', edgecolor='black'))

    # Combine all paths into a single list for animation
    full_path = [point for subpath in paths for point in subpath]

    # animation
    def update(frame):
        if frame < len(full_path):
            x_vals = [p[1] for p in full_path[:frame + 1]]
            y_vals = [p[0] for p in full_path[:frame + 1]]
            line.set_data(x_vals, y_vals)
            coord_label.set_text(f"({full_path[frame][1]}, {full_path[frame][0]})")
            coord_label.set_position((full_path[frame][1], full_path[frame][0]))

        return line, coord_label

    ani = animation.FuncAnimation(fig, update, frames=len(full_path), interval=500, blit=True)
    plt.title("Warehouse Pathfinding Animation")
    plt.show()


# shelf positions
def shelf_positions(shelf_count):
    shelf_x_pos = shelf_interval - 1
    shelf_arr = []

    for c_shelf in range(0, shelf_count):
        for c_height in range(1, shelf_height - 1):
            shelf_arr.append((c_height, shelf_x_pos))

        shelf_x_pos += shelf_interval

    return shelf_arr


# sort route array
def sort_the_route_array(seq_arr, picking_locations):
    for i in range(1, len(seq_arr)):
        key = seq_arr[i]
        key_loc = picking_locations[i]
        j = i - 1

        while j >= 0 and key < seq_arr[j]:
            seq_arr[j + 1] = seq_arr[j]
            picking_locations[j + 1] = picking_locations[j]
            j -= 1
        seq_arr[j + 1] = key
        picking_locations[j + 1] = key_loc


if __name__ == "__main__":

    # grid dimensions
    grid_width = 1
    shelf_height = 15
    shelf_interval = 2
    shelf_count = 5

    # Adjust grid width
    for count in range(0, shelf_count):
        grid_width += shelf_interval

    warehouse = np.zeros((shelf_height, grid_width))

    # Add shelves
    shelves_to_add = shelf_positions(shelf_count)
    add_shelves(warehouse, shelves_to_add)

    # Picking locations
    picking_locations = [(0, 0), (0, 9), (5, 2), (7, 6), (12, 8),(6, 2)]

    # Calculate routing sequence
    route, total_distance = calculate_the_routing_sequence(picking_locations)
    route.pop()

    # Sort route
    sort_the_route_array(route, picking_locations)

    # Compute paths for all picking locations
    started_point = picking_locations[0]
    all_paths = []  # Store all paths sequentially

    # Ensure path is computed for every step without skipping
    for index in range(len(picking_locations) - 1):
        path = a_star(warehouse, picking_locations[index], picking_locations[index + 1])
        if path:
            all_paths.append(path)

    # Also add the path returning to the start (if needed)
    return_path = a_star(warehouse, picking_locations[-1], picking_locations[0])
    if return_path:
        all_paths.append(return_path)

    # Animate all paths together
    animate_full_route(all_paths, warehouse, grid_width, shelf_height)

    print("Optimal path:", all_paths)
