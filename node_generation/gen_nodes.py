import random
import math
import json
from collections import deque
import sys
import ast
import os

def generate_connected_points(num, max_retries=100):
    for retry in range(max_retries):
        # Maximum number of neighbors allowed per point
        if(num == 5):
            max_neighbors = 4
        elif(num == 10):
            max_neighbors = 4 #4
        elif(num == 15):
            max_neighbors = 5 #5
        elif(num == 20):
            max_neighbors = 5 #5
        
        # SINK NODE
        points = [(50, 50)]
        
        # Try to add points until we have num+1 total
        attempts = 0
        max_attempts = num * 100  # Prevent infinite loop
        
        while len(points) < num + 1 and attempts < max_attempts:
            # Generate a random point within reasonable bounds, 500 for now
            x = random.randint(0, 500)
            y = random.randint(0, 500)
            new_point = (x, y)
            
            # Check if this point is acceptabe
            if is_valid_point(new_point, points, max_neighbors):
                points.append(new_point)
                attempts = 0
            else:
                attempts += 1
        
        if len(points) == num + 1:
            return points
        else:
            if retry < max_retries - 1:
                print(f"Attempt {retry+1}: Could only generate {len(points)} points out of {num+1} requested. Restarting...")
    
    # Bad params
    raise ValueError(f"Failed to generate {num+1} points after {max_retries} attempts. Consider adjusting parameters.")

def is_valid_point(new_point, existing_points, max_neighbors):
    # count neighbors
    neighbors = []
    
    for point in existing_points:
        if distance(new_point, point) <= 50:
            neighbors.append(point)
    
    # Check if it has too many neighbors
    if len(neighbors) > max_neighbors:
        return False
    
    # Check if it has at least one neighbor, so it can reach sink
    if len(neighbors) == 0:
        return False
    
    # For all existing points, make sure max neighbours is satisfied
    for point in existing_points:
        if distance(new_point, point) <= 50:
            neighbor_count = sum(1 for p in existing_points if distance(p, point) <= 50)
            if neighbor_count + 1 > max_neighbors:
                return False
    # add new point temporarily to check connectivity
    temp_points = existing_points + [new_point]
    
    # graph remains connected
    if not is_connected(temp_points):
        return False
    
    return True

def is_connected(points):
    if not points:
        return True
    
    #adjacency list
    graph = {i: [] for i in range(len(points))}
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j and distance(points[i], points[j]) <= 50:
                graph[i].append(j)
    
    #BFS from the first point (50, 50)
    visited = set()
    queue = deque([0])
    visited.add(0)
    
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    # check if all points are reachable
    return len(visited) == len(points)

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def save_points_to_file(filename, num_nodes, num_samples=20):
    all_points = []
    for _ in range(num_samples):
        points = generate_connected_points(num_nodes)
        all_points.append(points)
    
    with open(filename, 'w') as f:
        json.dump(all_points, f, indent=4)
    print(f"Saved {num_samples} sets of points to {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 program.py 5,10,15")
        sys.exit(1)

    try:
        node_list = [int(n.strip()) for n in sys.argv[1].split(",") if n.strip()]
        if not node_list:
            raise ValueError
    except ValueError:
        print("Error: Argument must be a comma-separated list of integers. Example: 5,10,15")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    for num_nodes in node_list:
        save_points_to_file(os.path.join(output_dir, f"generated_points-lr-{num_nodes}.json"), num_nodes)
        save_points_to_file(os.path.join(output_dir, f"generated_points-wp-{num_nodes}.json"), num_nodes)
        save_points_to_file(os.path.join(output_dir, f"generated_points-bh-{num_nodes}.json"), num_nodes)
        save_points_to_file(os.path.join(output_dir, f"generated_points-df-{num_nodes}.json"), num_nodes)
        save_points_to_file(os.path.join(output_dir, f"generated_points-fn-{num_nodes}.json"), num_nodes)
