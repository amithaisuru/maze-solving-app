import heapq
import math
import random
import time
import tkinter as tk
from collections import deque
from tkinter import ttk


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.neighbors = []
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}
    
    def __lt__(self, other):
        return False



class MazeApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Maze Generator and Solver")
        
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        self.graph = None
        self.canvas = None
        self.base_cell_size = 20
        self.cell_size = self.base_cell_size
        self.solving = False
        self.grid = None
        self.generation_running = False
        self.solving_running = False
        self.editing_mode = False
        self.last_solve_time = 0
        self.last_path_length = 0
        
        self.setup_ui()
        self.root.bind('<Configure>', self.on_window_resize)
    
    def setup_ui(self):
        # Input frame
        input_frame = ttk.Frame(self.root, padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        input_frame.grid_columnconfigure(tuple(range(10)), weight=1)
        
        # Size input
        ttk.Label(input_frame, text="Maze Size:").grid(row=0, column=0, padx=5)
        self.size_var = tk.StringVar(value="15")
        ttk.Entry(input_frame, textvariable=self.size_var, width=10).grid(row=0, column=1, padx=5)
        
        # Algorithm selection
        ttk.Label(input_frame, text="Algorithm:").grid(row=0, column=2, padx=5)
        self.algorithm_var = tk.StringVar(value="BFS")
        algorithms = ttk.Combobox(input_frame, textvariable=self.algorithm_var, values=["BFS", "Dijkstra", "A*"])
        algorithms.grid(row=0, column=3, padx=5)
        
        # Multiple solutions checkbox
        self.multiple_solutions_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(input_frame, text="Multiple Solutions", 
                       variable=self.multiple_solutions_var).grid(row=0, column=4, padx=5)
        
        # Animation speed control
        ttk.Label(input_frame, text="Delay (ms):").grid(row=0, column=5, padx=5)
        self.speed_var = tk.StringVar(value="50")
        speed_spinbox = ttk.Spinbox(input_frame, from_=1, to=1000, textvariable=self.speed_var, width=8)
        speed_spinbox.grid(row=0, column=6, padx=5)
        
        # Edit mode toggle
        self.edit_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(input_frame, text="Edit Mode", 
                       variable=self.edit_mode_var,
                       command=self.toggle_edit_mode).grid(row=0, column=7, padx=5)
        
        # Buttons
        ttk.Button(input_frame, text="Generate Maze", command=self.generate_maze).grid(row=0, column=8, padx=5)
        ttk.Button(input_frame, text="Solve Maze", command=self.solve_maze).grid(row=0, column=9, padx=5)
        
        # Status frame for solve time and path length
        status_frame = ttk.Frame(self.root, padding="5")
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        self.solve_time_var = tk.StringVar(value="Solve Time: -")
        self.path_length_var = tk.StringVar(value="Path Length: -")
        
        ttk.Label(status_frame, textvariable=self.solve_time_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(status_frame, textvariable=self.path_length_var).pack(side=tk.LEFT, padx=5)
        
        # Canvas frame with scrollbars
        canvas_frame = ttk.Frame(self.root, padding="10")
        canvas_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        self.h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        self.v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        self.h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.canvas = tk.Canvas(canvas_frame, bg='white',
                              xscrollcommand=self.h_scrollbar.set,
                              yscrollcommand=self.v_scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.h_scrollbar.config(command=self.canvas.xview)
        self.v_scrollbar.config(command=self.canvas.yview)
        
        # Bind canvas click event
        self.canvas.bind('<Button-1>', self.on_canvas_click)

    def toggle_edit_mode(self):
        self.editing_mode = self.edit_mode_var.get()
        if self.editing_mode:
            self.canvas.config(cursor="crosshair")
        else:
            self.canvas.config(cursor="")

    def on_canvas_click(self, event):
        if not self.editing_mode or not self.grid:
            return
        
        # Convert canvas coordinates to grid coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        grid_x = int(canvas_x // self.cell_size)
        grid_y = int(canvas_y // self.cell_size)
        
        # Toggle wall/path
        if 0 <= grid_x < len(self.grid[0]) and 0 <= grid_y < len(self.grid):
            # Don't allow editing start and end points
            if (grid_x, grid_y) in [(1, 1), (len(self.grid)-2, len(self.grid)-2)]:
                return
                
            current_value = self.grid[grid_y][grid_x]
            new_value = 0 if current_value == 1 else 1
            self.grid[grid_y][grid_x] = new_value
            
            # Update the graph structure if clicking on a wall
            if grid_x % 2 == 1 and grid_y % 2 == 1:  # Cell
                pass
            elif grid_x % 2 == 0 and grid_y % 2 == 1:  # Vertical wall
                cell1_x = (grid_x - 1) // 2
                cell2_x = cell1_x + 1
                cell_y = grid_y // 2
                if (cell1_x, cell_y) in self.graph.nodes and (cell2_x, cell_y) in self.graph.nodes:
                    self.graph.nodes[(cell1_x, cell_y)].walls['E'] = new_value == 1
                    self.graph.nodes[(cell2_x, cell_y)].walls['W'] = new_value == 1
            elif grid_x % 2 == 1 and grid_y % 2 == 0:  # Horizontal wall
                cell_x = grid_x // 2
                cell1_y = (grid_y - 1) // 2
                cell2_y = cell1_y + 1
                if (cell_x, cell1_y) in self.graph.nodes and (cell_x, cell2_y) in self.graph.nodes:
                    self.graph.nodes[(cell_x, cell1_y)].walls['S'] = new_value == 1
                    self.graph.nodes[(cell_x, cell2_y)].walls['N'] = new_value == 1
            
            self.redraw_maze()

    def recursive_backtrack_animated(self, current, visited):
        if not self.generation_running:
            return
            
        visited.add((current.x, current.y))
        grid_y = 2 * current.y + 1
        grid_x = 2 * current.x + 1
        self.update_cell(grid_x, grid_y, 0, 'lightblue')
        
        directions = [(0, -1, 'N', 'S'), (0, 1, 'S', 'N'), 
                     (1, 0, 'E', 'W'), (-1, 0, 'W', 'E')]
        random.shuffle(directions)
        
        delay = int(self.speed_var.get())
        self.canvas.after(delay)
        
        for dx, dy, d1, d2 in directions:
            next_x, next_y = current.x + dx, current.y + dy
            
            if (next_x, next_y) in self.graph.nodes and (next_x, next_y) not in visited:
                # For multiple solutions, randomly skip some walls
                if self.multiple_solutions_var.get() and random.random() < 0.3:
                    continue
                    
                wall_x = grid_x + dx
                wall_y = grid_y + dy
                self.update_cell(wall_x, wall_y, 0, 'white')
                
                current.walls[d1] = False
                next_node = self.graph.nodes[(next_x, next_y)]
                next_node.walls[d2] = False
                
                self.recursive_backtrack_animated(next_node, visited)
        
        self.update_cell(grid_x, grid_y, 0, 'white')
        
        if (current.x, current.y) == (0, 0):
            self.generation_running = False

    def animate_solution(self, path):
        delay = int(self.speed_var.get())
        self.last_path_length = len(path)
        self.path_length_var.set(f"Path Length: {self.last_path_length}")
        
        def draw_path_segment(index):
            if index >= len(path):
                return
            
            pos = path[index]
            grid_x = 2 * pos[0] + 1
            grid_y = 2 * pos[1] + 1
            self.update_cell(grid_x, grid_y, 3, 'red')
            
            if index + 1 < len(path):
                next_pos = path[index + 1]
                wall_x = grid_x + (next_pos[0] - pos[0])
                wall_y = grid_y + (next_pos[1] - pos[1])
                self.update_cell(wall_x, wall_y, 3, 'red')
            
            self.canvas.after(delay, lambda: draw_path_segment(index + 1))
        
        draw_path_segment(0)

    def solve_maze(self):
        if not self.graph or self.solving_running:
            return
            
        self.solving_running = True
        start = (0, 0)
        end = (self.graph.size - 1, self.graph.size - 1)
        
        # Clear previous solution
        for y in range(len(self.grid)):
            for x in range(len(self.grid)):
                if self.grid[y][x] in [2, 3]:
                    self.update_cell(x, y, 0, 'white')
        
        start_time = time.time()
        algorithm = self.algorithm_var.get()
        
        if algorithm == "BFS":
            self.bfs_animated(start, end)
        elif algorithm == "Dijkstra":
            self.dijkstra_animated(start, end)
        else:  # A*
            self.astar_animated(start, end)
            
        self.last_solve_time = time.time() - start_time
        self.solve_time_var.set(f"Solve Time: {self.last_solve_time:.4f}s")

    def on_window_resize(self, event):
        if self.grid:
            # Only respond to main window resizes, not internal widget resizes
            if event.widget == self.root:
                self.adjust_cell_size()
                self.redraw_maze()

    def adjust_cell_size(self):
        # Get available space
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if self.grid and canvas_width > 0 and canvas_height > 0:
            # Calculate new cell size based on available space and grid size
            grid_size = len(self.grid)
            width_cell_size = (canvas_width - 20) // grid_size
            height_cell_size = (canvas_height - 20) // grid_size
            
            # Use the smaller of the two to maintain square cells
            self.cell_size = max(min(width_cell_size, height_cell_size, self.base_cell_size), 5)

    def redraw_maze(self):
        if self.grid:
            self.canvas.delete("all")
            self.draw_grid()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def initialize_grid(self, size):
        grid_size = 2 * size + 1
        self.grid = [[1 for _ in range(grid_size)] for _ in range(grid_size)]
        # Clear cells (not walls)
        for y in range(1, grid_size, 2):
            for x in range(1, grid_size, 2):
                self.grid[y][x] = 0
        return grid_size

    def draw_grid(self):
        for y in range(len(self.grid)):
            for x in range(len(self.grid)):
                x1 = x * self.cell_size
                y1 = y * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                if self.grid[y][x] == 1:  # Wall
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='black', outline='gray')
                elif self.grid[y][x] == 2:  # Visited during solving
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='pink', outline='gray')
                elif self.grid[y][x] == 3:  # Final path
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='red', outline='gray')
                else:  # Empty cell
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='gray')

    def update_cell(self, x, y, value, color='white'):
        if 0 <= y < len(self.grid) and 0 <= x < len(self.grid[0]):
            self.grid[y][x] = value
            x1 = x * self.cell_size
            y1 = y * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='gray')
            self.canvas.update_idletasks()

    def generate_maze(self):
        if self.generation_running:
            return
            
        self.generation_running = True
        size = int(self.size_var.get())
        grid_size = self.initialize_grid(size)
        self.graph = Graph(size)
        
        # Clear canvas and draw initial grid
        self.canvas.delete("all")
        self.adjust_cell_size()
        self.draw_grid()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Start generation from top-left cell
        start_node = self.graph.nodes[(0, 0)]
        visited = set()
        self.recursive_backtrack_animated(start_node, visited)

    def bfs_animated(self, start, end):
        queue = deque([(start, [start])])
        visited = {start}
        delay = int(self.speed_var.get())
        
        def process_next():
            if not queue or not self.solving_running:
                self.solving_running = False
                return
                
            current, path = queue.popleft()
            
            if current == end:
                self.animate_solution(path)
                self.solving_running = False
                return
            
            # Show current cell being visited
            grid_y = 2 * current[1] + 1
            grid_x = 2 * current[0] + 1
            self.update_cell(grid_x, grid_y, 2, 'pink')
            
            for neighbor in self.graph.get_neighbors(*current):
                next_pos = (neighbor.x, neighbor.y)
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))
            
            self.canvas.after(delay, process_next)
        
        process_next()

    def dijkstra_animated(self, start, end):
        distances = {pos: float('infinity') for pos in self.graph.nodes}
        distances[start] = 0
        pq = [(0, start)]
        previous = {start: None}
        delay = int(self.speed_var.get())
        
        def process_next():
            if not pq or not self.solving_running:
                self.solving_running = False
                return
                
            current_dist, current = heapq.heappop(pq)
            
            # Show current cell being visited
            grid_y = 2 * current[1] + 1
            grid_x = 2 * current[0] + 1
            self.update_cell(grid_x, grid_y, 2, 'pink')
            
            if current == end:
                path = []
                while current:
                    path.append(current)
                    current = previous[current]
                self.animate_solution(path[::-1])
                self.solving_running = False
                return
            
            for neighbor in self.graph.get_neighbors(*current):
                next_pos = (neighbor.x, neighbor.y)
                distance = current_dist + 1
                
                if distance < distances[next_pos]:
                    distances[next_pos] = distance
                    previous[next_pos] = current
                    heapq.heappush(pq, (distance, next_pos))
            
            self.canvas.after(delay, process_next)
        
        process_next()

    def astar_animated(self, start, end):
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        delay = int(self.speed_var.get())
        
        def process_next():
            if not frontier or not self.solving_running:
                self.solving_running = False
                return
                
            current = heapq.heappop(frontier)[1]
            
            # Show current cell being visited
            grid_y = 2 * current[1] + 1
            grid_x = 2 * current[0] + 1
            self.update_cell(grid_x, grid_y, 2, 'pink')
            
            if current == end:
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                self.animate_solution(path[::-1])
                self.solving_running = False
                return
            
            for neighbor in self.graph.get_neighbors(*current):
                next_pos = (neighbor.x, neighbor.y)
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(next_pos, end)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
            
            self.canvas.after(delay, process_next)
        
        process_next()

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def stop_animation(self):
        self.generation_running = False
        self.solving_running = False

    def reset_maze(self):
        if self.canvas:
            self.canvas.delete("all")
        self.grid = None
        self.graph = None
        self.generation_running = False
        self.solving_running = False

if __name__ == "__main__":
    root = tk.Tk()
    
    # Set minimum window size
    root.minsize(600, 400)
    
    # Configure style
    style = ttk.Style()
    style.configure('TButton', padding=5)
    style.configure('TFrame', padding=5)
    
    app = MazeApp(root)
    
    # Add menu bar
    menubar = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Reset", command=app.reset_maze)
    file_menu.add_command(label="Stop Animation", command=app.stop_animation)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    menubar.add_cascade(label="File", menu=file_menu)
    root.config(menu=menubar)
    
    root.mainloop()