import heapq
import random
import time
import tkinter as tk
from collections import deque
from tkinter import ttk


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.visited = False
        self.parent = None

class Graph:
    def __init__(self, size):
        self.size = size
        self.nodes = [[Node(i, j) for j in range(size)] for i in range(size)]
        self.edges = set()

    def generateSquareGraph(self):
        def get_unvisited_neighbors(x, y):
            neighbors = []
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.size and 0 <= new_y < self.size and 
                    not self.nodes[new_x][new_y].visited):
                    neighbors.append((new_x, new_y))
            return neighbors

        stack = [(0, 0)]
        self.nodes[0][0].visited = True
        
        while stack:
            current_x, current_y = stack[-1]
            neighbors = get_unvisited_neighbors(current_x, current_y)
            
            if neighbors:
                next_x, next_y = random.choice(neighbors)
                self.edges.add(((current_x, current_y), (next_x, next_y)))
                self.edges.add(((next_x, next_y), (current_x, current_y)))
                self.nodes[next_x][next_y].visited = True
                stack.append((next_x, next_y))
            else:
                stack.pop()

    def getAdjacencyMatrix(self):
        matrix = [[0] * self.size * self.size for _ in range(self.size * self.size)]
        for (x1, y1), (x2, y2) in self.edges:
            i1 = x1 * self.size + y1
            i2 = x2 * self.size + y2
            matrix[i1][i2] = 1
            matrix[i2][i1] = 1
        return matrix

class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Generator and Solver")

        # Input frame
        input_frame = ttk.Frame(root, padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(input_frame, text="Maze Size:").grid(row=0, column=0)
        self.size_var = tk.IntVar(value=10)
        ttk.Entry(input_frame, textvariable=self.size_var).grid(row=0, column=1)
        
        ttk.Label(input_frame, text="Algorithm:").grid(row=1, column=0)
        self.algo_var = tk.StringVar(value="BFS")
        algo_combo = ttk.Combobox(input_frame, textvariable=self.algo_var, 
                                values=["BFS", "Dijkstra", "A*"])
        algo_combo.grid(row=1, column=1)

        ttk.Label(input_frame, text="Mode:").grid(row=2, column=0)
        self.mode_var = tk.StringVar(value="Instant")
        mode_combo = ttk.Combobox(input_frame, textvariable=self.mode_var, 
                                values=["Instant", "Step"])
        mode_combo.grid(row=2, column=1)

        ttk.Button(input_frame, text="Generate Maze", command=self.generate_maze).grid(row=3, column=0)
        ttk.Button(input_frame, text="Solve Maze", command=self.solve_maze).grid(row=3, column=1)
        self.step_button = ttk.Button(input_frame, text="Next Step", command=self.next_step, state="disabled")
        self.step_button.grid(row=4, column=0, columnspan=2)

        self.time_label = ttk.Label(input_frame, text="Time: ")
        self.time_label.grid(row=5, column=0, columnspan=2)

        # Canvas for maze
        self.cell_size = 30
        self.canvas = tk.Canvas(root, width=300, height=300)
        self.canvas.grid(row=1, column=0)

        self.graph = None
        self.step_data = None
        self.step_index = 0

    def generate_maze(self):
        size = self.size_var.get()
        self.canvas.delete("all")
        self.canvas.config(width=size*self.cell_size, height=size*self.cell_size)
        
        self.graph = Graph(size)
        self.graph.generateSquareGraph()
        self.draw_maze()
        self.step_button.config(state="disabled")
        self.step_data = None
        self.step_index = 0

    def draw_maze(self):
        size = self.graph.size
        for i in range(size):
            for j in range(size):
                x1, y1 = j * self.cell_size, i * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                
                if ((i, j), (i, j+1)) not in self.graph.edges and j+1 < size:
                    self.canvas.create_line(x2, y1, x2, y2, fill="black")
                if ((i, j), (i+1, j)) not in self.graph.edges and i+1 < size:
                    self.canvas.create_line(x1, y2, x2, y2, fill="black")
                if j == 0 or ((i, j), (i, j-1)) not in self.graph.edges:
                    self.canvas.create_line(x1, y1, x1, y2, fill="black")
                if i == 0 or ((i, j), (i-1, j)) not in self.graph.edges:
                    self.canvas.create_line(x1, y1, x2, y1, fill="black")

        self.canvas.create_oval(5, 5, self.cell_size-5, self.cell_size-5, fill="green")
        end_x = (size-1)*self.cell_size
        self.canvas.create_oval(end_x+5, end_x+5, end_x+self.cell_size-5, 
                              end_x+self.cell_size-5, fill="red")

    def solve_maze(self):
        if not self.graph:
            return

        self.canvas.delete("path", "highlight")
        start_time = time.time()
        algorithm = self.algo_var.get()
        mode = self.mode_var.get()

        if algorithm == "BFS":
            path, steps = self.bfs()
        elif algorithm == "Dijkstra":
            path, steps = self.dijkstra()
        else:  # A*
            path, steps = self.a_star()

        end_time = time.time()
        self.time_label.config(text=f"Time: {(end_time - start_time)*1000:.2f} ms")

        if mode == "Instant":
            self.draw_path(path)
        else:  # Step mode
            self.step_data = steps
            self.step_index = 0
            self.step_button.config(state="normal")
            self.next_step()

    def next_step(self):
        if self.step_data and self.step_index < len(self.step_data):
            self.canvas.delete("highlight")
            # Updated to unpack 3 values instead of 2
            current, considered, parent = self.step_data[self.step_index]
            
            # Highlight current node
            x, y = current
            self.canvas.create_rectangle(
                y*self.cell_size, x*self.cell_size,
                (y+1)*self.cell_size, (x+1)*self.cell_size,
                fill="yellow", outline="", tags="highlight"
            )

            # Highlight considered nodes
            for nx, ny in considered:
                self.canvas.create_rectangle(
                    ny*self.cell_size, nx*self.cell_size,
                    (ny+1)*self.cell_size, (nx+1)*self.cell_size,
                    fill="orange", outline="", tags="highlight"
                )

            self.step_index += 1
            if self.step_index == len(self.step_data):
                self.step_button.config(state="disabled")
                # Updated path reconstruction to use the parent from last step
                final_parent_dict = {n: p for n, _, p in self.step_data if p is not None}
                final_parent_dict[(0, 0)] = None  # Ensure start node is included
                path = self.reconstruct_path(final_parent_dict, (self.graph.size-1, self.graph.size-1))
                self.draw_path(path)

    def bfs(self):
        size = self.graph.size
        queue = deque([(0, 0)])
        visited = {(0, 0)}
        parent = {(0, 0): None}
        steps = []

        while queue:
            x, y = queue.popleft()
            considered = []
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < size and 0 <= new_y < size and 
                    ((x, y), (new_x, new_y)) in self.graph.edges and 
                    (new_x, new_y) not in visited):
                    queue.append((new_x, new_y))
                    visited.add((new_x, new_y))
                    parent[(new_x, new_y)] = (x, y)
                    considered.append((new_x, new_y))
            
            steps.append(((x, y), considered, parent[(x, y)]))
            if x == size-1 and y == size-1:
                break

        return self.reconstruct_path(parent, (size-1, size-1)), steps

    def dijkstra(self):
        size = self.graph.size
        pq = [(0, (0, 0))]
        distances = {(0, 0): 0}
        parent = {(0, 0): None}
        steps = []

        while pq:
            dist, (x, y) = heapq.heappop(pq)
            considered = []
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < size and 0 <= new_y < size and 
                    ((x, y), (new_x, new_y)) in self.graph.edges):
                    new_dist = dist + 1
                    if (new_x, new_y) not in distances or new_dist < distances[(new_x, new_y)]:
                        distances[(new_x, new_y)] = new_dist
                        parent[(new_x, new_y)] = (x, y)
                        heapq.heappush(pq, (new_dist, (new_x, new_y)))
                        considered.append((new_x, new_y))
            
            steps.append(((x, y), considered, parent[(x, y)]))
            if x == size-1 and y == size-1:
                break

        return self.reconstruct_path(parent, (size-1, size-1)), steps

    def a_star(self):
        def heuristic(x, y):
            return abs(size-1-x) + abs(size-1-y)

        size = self.graph.size
        pq = [(heuristic(0, 0), 0, (0, 0))]
        costs = {(0, 0): 0}
        parent = {(0, 0): None}
        steps = []

        while pq:
            _, cost, (x, y) = heapq.heappop(pq)
            considered = []
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < size and 0 <= new_y < size and 
                    ((x, y), (new_x, new_y)) in self.graph.edges):
                    new_cost = cost + 1
                    if (new_x, new_y) not in costs or new_cost < costs[(new_x, new_y)]:
                        costs[(new_x, new_y)] = new_cost
                        priority = new_cost + heuristic(new_x, new_y)
                        parent[(new_x, new_y)] = (x, y)
                        heapq.heappush(pq, (priority, new_cost, (new_x, new_y)))
                        considered.append((new_x, new_y))
            
            steps.append(((x, y), considered, parent[(x, y)]))
            if x == size-1 and y == size-1:
                break

        return self.reconstruct_path(parent, (size-1, size-1)), steps

    def reconstruct_path(self, parent, end):
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = parent[current]
        return path[::-1]

    def draw_path(self, path):
        if not path:
            return
        
        self.canvas.delete("path")
        for i in range(len(path)-1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            self.canvas.create_line(
                y1*self.cell_size + self.cell_size/2,
                x1*self.cell_size + self.cell_size/2,
                y2*self.cell_size + self.cell_size/2,
                x2*self.cell_size + self.cell_size/2,
                fill="blue", width=2, tags="path"
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()