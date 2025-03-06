import heapq
import time
import tkinter as tk
from collections import deque  # Added this import
from tkinter import filedialog, ttk

import cv2

from maze_classes import Graph
from vision_handler import detect_lines  # Added this import


class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Generator and Solver")

        # Input frame
        input_frame = ttk.Frame(root, padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        ttk.Label(input_frame, text="Maze Size:").grid(row=0, column=0, pady=2)
        self.size_var = tk.IntVar(value=10)
        ttk.Entry(input_frame, textvariable=self.size_var).grid(row=0, column=1, pady=2)
        
        ttk.Label(input_frame, text="Algorithm:").grid(row=1, column=0, pady=2)
        self.algo_var = tk.StringVar(value="BFS")
        algo_combo = ttk.Combobox(input_frame, textvariable=self.algo_var, 
                                values=["BFS", "Dijkstra", "A*"])
        algo_combo.grid(row=1, column=1, pady=2)

        ttk.Label(input_frame, text="Speed (ms):").grid(row=2, column=0, pady=2)
        self.speed_var = tk.IntVar(value=100)
        ttk.Entry(input_frame, textvariable=self.speed_var).grid(row=2, column=1, pady=2)

        ttk.Button(input_frame, text="Generate Maze", command=self.generate_maze).grid(row=3, column=0, pady=2)
        ttk.Button(input_frame, text="Solve Maze", command=self.solve_maze).grid(row=3, column=1, pady=2)
        ttk.Button(input_frame, text="Load Image", command=self.load_image).grid(row=3, column=2, pady=2)

        self.time_label = ttk.Label(input_frame, text="Total Time: 0.0000 ms")
        self.time_label.grid(row=4, column=0, columnspan=3, pady=2)

        # Canvas for maze
        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self.cell_size = 30
        self.offset = 10
        self.graph = None
        self.step_data = None
        self.step_index = 0
        self.is_solving = False
        self.start = (0, 0)
        self.end = (0, 0)

    def generate_maze(self):
        size = self.size_var.get()
        self.canvas.delete("all")
        
        available_width = self.canvas_width - 2 * self.offset
        available_height = self.canvas_height - 2 * self.offset
        self.cell_size = min(available_width // size, available_height // size)
        self.cell_size = max(self.cell_size, 10)
        
        self.graph = Graph(size)
        self.graph.generateSquareGraph()
        self.start = (0, 0)
        self.end = (size-1, size-1)
        self.draw_maze()
        self.step_data = None
        self.step_index = 0
        self.is_solving = False
        self.time_label.config(text="Total Time: 0.0000 ms")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        image = cv2.imread(file_path)
        if file_path:
            self.graph =  detect_lines(image)
            self.start, self.end = (0, 0), (self.graph.size-1, self.graph.size-1)

            available_width = self.canvas_width - 2 * self.offset
            available_height = self.canvas_height - 2 * self.offset
            self.cell_size = min(available_width // self.graph.size, available_height // self.graph.size)
            self.cell_size = max(self.cell_size, 10)
            
            self.canvas.delete("all")
            self.draw_maze()
            self.step_data = None
            self.step_index = 0
            self.is_solving = False
            self.time_label.config(text="Total Time: 0.0000 ms")

    def draw_maze(self):
        size = self.graph.size
                
        for i in range(size):
            for j in range(size):
                x1 = j * self.cell_size + self.offset
                y1 = i * self.cell_size + self.offset
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                line_width = 2
                
                if ((i, j), (i, j+1)) not in self.graph.edges and j+1 < size:
                    self.canvas.create_line(x2, y1, x2, y2, fill="black", width=line_width)
                if ((i, j), (i+1, j)) not in self.graph.edges and i+1 < size:
                    self.canvas.create_line(x1, y2, x2, y2, fill="black", width=line_width)
                if j == 0 or ((i, j), (i, j-1)) not in self.graph.edges:
                    self.canvas.create_line(x1, y1, x1, y2, fill="black", width=line_width)
                if i == 0 or ((i, j), (i-1, j)) not in self.graph.edges:
                    self.canvas.create_line(x1, y1, x2, y1, fill="black", width=line_width)

        border_width = 4
        self.canvas.create_rectangle(
            self.offset, self.offset, 
            size * self.cell_size + self.offset, size * self.cell_size + self.offset,
            outline="black", width=border_width, tags="border"
        )

        start_end_size = max(self.cell_size // 6, 5)
        sx, sy = self.start
        self.canvas.create_oval(
            sy * self.cell_size + self.offset + start_end_size,
            sx * self.cell_size + self.offset + start_end_size,
            sy * self.cell_size + self.offset + self.cell_size - start_end_size,
            sx * self.cell_size + self.offset + self.cell_size - start_end_size,
            fill="green"
        )
        ex, ey = self.end
        self.canvas.create_oval(
            ey * self.cell_size + self.offset + start_end_size,
            ex * self.cell_size + self.offset + start_end_size,
            ey * self.cell_size + self.offset + self.cell_size - start_end_size,
            ex * self.cell_size + self.offset + self.cell_size - start_end_size,
            fill="red"
        )

    def solve_maze(self):
        if not self.graph or self.is_solving:
            return

        self.canvas.delete("path", "highlight", "temp_path")
        self.step_data = None
        self.step_index = 0
        self.is_solving = True
        
        start_time = time.perf_counter()
        algorithm = self.algo_var.get()

        if algorithm == "BFS":
            path, steps = self.bfs()
        elif algorithm == "Dijkstra":
            path, steps = self.dijkstra()
        else:  # A*
            path, steps = self.a_star()

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000
        self.time_label.config(text=f"Total Time: {total_time:.4f} ms")
        
        self.step_data = steps
        self.animate_steps()

    def animate_steps(self):
        if not self.step_data or self.step_index >= len(self.step_data):
            self.is_solving = False
            return

        self.canvas.delete("highlight", "temp_path")
        current, considered, parent = self.step_data[self.step_index]
        
        x, y = current
        self.canvas.create_rectangle(
            y * self.cell_size + self.offset, x * self.cell_size + self.offset,
            (y + 1) * self.cell_size + self.offset, (x + 1) * self.cell_size + self.offset,
            fill="yellow", outline="", tags="highlight"
        )

        for nx, ny in considered:
            self.canvas.create_rectangle(
                ny * self.cell_size + self.offset, nx * self.cell_size + self.offset,
                (ny + 1) * self.cell_size + self.offset, (nx + 1) * self.cell_size + self.offset,
                fill="orange", outline="", tags="highlight"
            )

        temp_parent_dict = {n: p for n, _, p in self.step_data[:self.step_index + 1] if p is not None}
        temp_parent_dict[self.start] = None
        temp_path = self.reconstruct_path(temp_parent_dict, current)
        self.draw_temp_path(temp_path)

        self.step_index += 1
        if self.step_index == len(self.step_data):
            final_parent_dict = {n: p for n, _, p in self.step_data if p is not None}
            final_parent_dict[self.start] = None
            final_path = self.reconstruct_path(final_parent_dict, self.end)
            self.draw_path(final_path)
            self.is_solving = False
        else:
            self.root.after(self.speed_var.get(), self.animate_steps)

    def bfs(self):
        size = self.graph.size
        queue = deque([self.start])  # Now deque is defined
        visited = {self.start}
        parent = {self.start: None}
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
            if (x, y) == self.end:
                break

        return self.reconstruct_path(parent, self.end), steps

    def dijkstra(self):
        size = self.graph.size
        pq = [(0, self.start)]
        distances = {self.start: 0}
        parent = {self.start: None}
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
            if (x, y) == self.end:
                break

        return self.reconstruct_path(parent, self.end), steps

    def a_star(self):
        def heuristic(x, y):
            ex, ey = self.end
            return abs(ex - x) + abs(ey - y)

        size = self.graph.size
        pq = [(heuristic(*self.start), 0, self.start)]
        costs = {self.start: 0}
        parent = {self.start: None}
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
            if (x, y) == self.end:
                break

        return self.reconstruct_path(parent, self.end), steps

    def reconstruct_path(self, parent, end):
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = parent.get(current)
        return path[::-1]

    def draw_path(self, path):
        if not path:
            return
        
        self.canvas.delete("path")
        for i in range(len(path)-1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            self.canvas.create_line(
                y1 * self.cell_size + self.offset + self.cell_size/2,
                x1 * self.cell_size + self.offset + self.cell_size/2,
                y2 * self.cell_size + self.offset + self.cell_size/2,
                x2 * self.cell_size + self.offset + self.cell_size/2,
                fill="blue", width=2, tags="path"
            )

    def draw_temp_path(self, path):
        if not path:
            return
        
        self.canvas.delete("temp_path")
        for i in range(len(path)-1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            self.canvas.create_line(
                y1 * self.cell_size + self.offset + self.cell_size/2,
                x1 * self.cell_size + self.offset + self.cell_size/2,
                y2 * self.cell_size + self.offset + self.cell_size/2,
                x2 * self.cell_size + self.offset + self.cell_size/2,
                fill="red", width=2, tags="temp_path"
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()