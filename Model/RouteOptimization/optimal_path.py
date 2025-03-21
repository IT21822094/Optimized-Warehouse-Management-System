from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import matplotlib.pyplot as plt

def create_distance_matrix(locations, distance_type='manhattan'):
    n = len(locations)
    distance_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                x1, y1 = locations[i]
                x2, y2 = locations[j]
                if distance_type == 'manhattan':
                    dist = abs(x1 - x2) + abs(y1 - y2)
                else:
                    # Euclidean distance
                    dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                distance_matrix[i][j] = dist
    return distance_matrix


def solve_tsp(locations, start_location_index=0, distance_type='manhattan', return_to_start=False):
    # Build distance matrix
    distance_matrix = create_distance_matrix(locations, distance_type)
    n = len(locations)

    # Create the routing index manager
    if return_to_start:
        # Force the route to start and end at the same location
        manager = pywrapcp.RoutingIndexManager(n, 1, start_location_index, start_location_index)
    else:
        manager = pywrapcp.RoutingIndexManager(n, 1, start_location_index)

    # Create the Routing Model
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        # Convert from routing variable Index to our location index
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Set parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the TSP travel salesmen problem
    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        print("No solution found!")
        return None, None

    # Extract the solution route in terms of location indices
    route = []
    route_distance = 0
    index = routing.Start(0)

    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        next_index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(index, next_index, 0)
        index = next_index
    route.append(manager.IndexToNode(index))  # the last node (end)

    return route, route_distance


def calculate_the_routing_sequence(picking_locations):

    shelf_locations = picking_locations

    # Solve the TSP
    route, total_dist = solve_tsp(
        shelf_locations,
        start_location_index=0,
        distance_type='manhattan',
        return_to_start=False
    )

    # Print and plot results
    if route is not None:
        print("Optimal Route (list of indices in shelf_locations):", route)
        print("Total Distance:", total_dist)

    return route,total_dist


