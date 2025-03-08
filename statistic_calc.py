import time
import tkinter as tk

from maze_app import Graph, MazeApp


def solve_graph(maze_app, graph, algorithm):
    maze_app.graph = graph
    
    if algorithm == "BFS":
        start_time = time.perf_counter()
        maze_app.bfs()
        end_time = time.perf_counter()
    elif algorithm == "Dijkstra":
        start_time = time.perf_counter()
        maze_app.dijkstra()
        end_time = time.perf_counter()
    elif algorithm == "A*":
        start_time = time.perf_counter()
        maze_app.a_star()
        end_time = time.perf_counter()
    else:
        raise ValueError("Unknown algorithm")
    
    return (end_time - start_time) * 1000  # Return time in milliseconds

def collect_statistics(num_graphs, maze_size):
    root = tk.Tk()
    maze_app = MazeApp(root)
    root.withdraw()  # Hide the Tkinter window

    algorithms = ["BFS", "Dijkstra", "A*"]
    total_times = {algo: 0.0 for algo in algorithms}  # Store cumulative times
    
    for i in range(num_graphs):
        # Generate a random graph
        graph = Graph(maze_size)
        graph.generateSquareGraph()
        
        # Solve with each algorithm and accumulate time
        for algo in algorithms:
            time_taken = solve_graph(maze_app, graph, algo)
            total_times[algo] += time_taken
    
    # Calculate averages
    avg_times = {algo: total_times[algo] / num_graphs for algo in algorithms}
    
    # Print results
    print(f"\nStatistics for {num_graphs} mazes of size {maze_size}x{maze_size}:")
    for algo, avg_time in avg_times.items():
        print(f"{algo}: Average Time = {avg_time:.4f} ms")

    # Clean up Tkinter instance
    root.destroy()

if __name__ == "__main__":
    num_graphs = 100 
    
    maze_size = 9 
    print(f"Running analysis for {num_graphs} mazes of size {maze_size}x{maze_size}...")
    collect_statistics(num_graphs, maze_size)

    maze_size = 16
    print(f"\n\nRunning analysis for {num_graphs} mazes of size {maze_size}x{maze_size}...")
    collect_statistics(num_graphs, maze_size)

    maze_size = 21
    print(f"\n\nRunning analysis for {num_graphs} mazes of size {maze_size}x{maze_size}...")
    collect_statistics(num_graphs, maze_size)