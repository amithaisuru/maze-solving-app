import heapq
import time
import tkinter as tk
from collections import deque
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk

from maze_classes import Graph
from vision_handler import detect_lines


class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MazeMaster")
        
        # Theme state
        self.dark_mode = tk.BooleanVar(value=False)
        
        # Apply initial theme
        self.apply_theme()
        
        # Main container frame
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Header with title and theme toggle
        header_frame = ttk.Frame(self.main_frame)
        header_frame.grid(row=0, column=0, columnspan=3, pady=(0, 20), sticky=(tk.W, tk.E))
        header_frame.columnconfigure(1, weight=1)
        
        # Title
        self.title_label = ttk.Label(header_frame, text="Maze Solver", font=("Segoe UI", 18, "bold"))
        self.title_label.grid(row=0, column=0, sticky=tk.W)
        
        # Theme toggle
        theme_frame = ttk.Frame(header_frame)
        theme_frame.grid(row=0, column=1, sticky=tk.E)
        
        self.theme_label = ttk.Label(theme_frame, text="Dark Mode:", font=("Segoe UI", 10))
        self.theme_label.pack(side=tk.LEFT, padx=(0, 5))
        
        theme_toggle = ttk.Checkbutton(
            theme_frame, 
            variable=self.dark_mode, 
            command=self.toggle_theme,
            style="Switch.TCheckbutton")
        theme_toggle.pack(side=tk.LEFT)
        
        # Left panel for controls
        self.control_frame = ttk.Frame(self.main_frame, padding="10")
        self.control_frame.grid(row=1, column=0, sticky=(tk.N, tk.W), padx=(0, 20))
        
        # Controls section header
        self.controls_header = ttk.Label(self.control_frame, text="CONTROLS", font=("Segoe UI", 12, "bold"))
        self.controls_header.grid(row=0, column=0, columnspan=2, pady=(0, 15), sticky=tk.W)
        
        # Input frame with modern spacing and grouping
        self.input_frame = ttk.Frame(self.control_frame)
        self.input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Size input
        self.size_label = ttk.Label(self.input_frame, text="Maze Size:")
        self.size_label.grid(row=0, column=0, pady=10, sticky=tk.W)
        self.size_var = tk.IntVar(value=10)
        size_entry = ttk.Entry(self.input_frame, textvariable=self.size_var, width=10)
        size_entry.grid(row=0, column=1, pady=10, padx=(10, 0), sticky=tk.W)
        
        # Algorithm selection
        self.algo_label = ttk.Label(self.input_frame, text="Algorithm:")
        self.algo_label.grid(row=1, column=0, pady=10, sticky=tk.W)
        self.algo_var = tk.StringVar(value="BFS")
        algo_combo = ttk.Combobox(self.input_frame, textvariable=self.algo_var, 
                                values=["BFS", "Dijkstra", "A*"], width=10, state="readonly")
        algo_combo.grid(row=1, column=1, pady=10, padx=(10, 0), sticky=tk.W)
        
        # Speed input
        self.speed_label = ttk.Label(self.input_frame, text="Speed (ms):")
        self.speed_label.grid(row=2, column=0, pady=10, sticky=tk.W)
        self.speed_var = tk.IntVar(value=100)
        speed_entry = ttk.Entry(self.input_frame, textvariable=self.speed_var, width=10)
        speed_entry.grid(row=2, column=1, pady=10, padx=(10, 0), sticky=tk.W)
        
        # Buttons section
        self.button_frame = ttk.Frame(self.control_frame)
        self.button_frame.grid(row=2, column=0, pady=20, sticky=tk.W)
        
        self.generate_btn = ttk.Button(self.button_frame, text="Generate Maze", command=self.generate_maze, style="Generate.TButton")
        self.generate_btn.grid(row=0, column=0, pady=5, sticky=tk.W)
        
        self.solve_btn = ttk.Button(self.button_frame, text="Solve Maze", command=self.solve_maze, style="Solve.TButton")
        self.solve_btn.grid(row=1, column=0, pady=5, sticky=tk.W)
        
        self.load_btn = ttk.Button(self.button_frame, text="Load Image", command=self.load_image, style="Load.TButton")
        self.load_btn.grid(row=2, column=0, pady=5, sticky=tk.W)
        
        # Stats section
        self.stats_frame = ttk.Frame(self.control_frame, padding=(0, 20, 0, 0))
        self.stats_frame.grid(row=3, column=0, sticky=tk.W)
        
        self.stats_header = ttk.Label(self.stats_frame, text="STATISTICS", font=("Segoe UI", 12, "bold"))
        self.stats_header.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        self.time_label = ttk.Label(self.stats_frame, text="Total Time: 0.0000 ms", font=("Segoe UI", 10))
        self.time_label.grid(row=1, column=0, pady=5, sticky=tk.W)
        
        # Canvas for maze
        self.canvas_frame = ttk.Frame(self.main_frame, padding=2, relief="solid", borderwidth=1)
        self.canvas_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.canvas_width = 500
        self.canvas_height = 500
        self.canvas = tk.Canvas(self.canvas_frame, width=self.canvas_width, height=self.canvas_height, 
                               highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Frame for source image
        self.image_frame = ttk.Frame(self.main_frame, padding=2, relief="solid", borderwidth=1)
        self.image_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(20, 0))
        
        self.image_canvas = tk.Canvas(self.image_frame, width=self.canvas_width, height=self.canvas_height,
                                     highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Image label
        self.image_label = ttk.Label(self.image_frame, text="Source Image", font=("Segoe UI", 12, "bold"),
                                    anchor="center")
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Make the UI responsive
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.columnconfigure(2, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        # Set minimum window size
        self.root.minsize(1200, 700)
        
        self.cell_size = 30
        self.offset = 20
        self.graph = None
        self.step_data = None
        self.step_index = 0
        self.is_solving = False
        self.start = (0, 0)
        self.end = (0, 0)
        self.source_image = None
        self.photo_image = None

        self.selection_mode = False
        self.clicks_remaining = 0    

        self.canvas.bind("<Button-1>", self.canvas_click)

    def canvas_click(self, event):
        """Handle canvas clicks for selecting start and end points"""
        if not self.selection_mode or not self.graph:
            return
        
        # Convert canvas coordinates to maze coordinates
        x = (event.x - self.offset) // self.cell_size
        y = (event.y - self.offset) // self.cell_size
        
        # Check if click is within maze bounds
        if (0 <= x < self.graph.size and 0 <= y < self.graph.size):
            if self.clicks_remaining == 2:
                # Set start point
                self.start = (y, x)
                self.clicks_remaining -= 1
                self.redraw_maze_with_points()
                self.root.title("MazeMaster - Click end point")
            elif self.clicks_remaining == 1:
                # Set end point and exit selection mode
                self.end = (y, x)
                self.clicks_remaining -= 1
                self.selection_mode = False
                self.redraw_maze_with_points()
                self.root.title("MazeMaster")

    def redraw_maze_with_points(self):
        """Redraw the maze with updated start and end points"""
        self.canvas.delete("all")
        self.draw_maze()

    def apply_theme(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        # Define colors based on theme
        if self.dark_mode.get():
            # Dark mode colors
            bg_color = "#1e1e1e"
            text_color = "#ffffff"
            accent_color = "#2d2d2d"
            canvas_bg = "#2d2d2d"
            grid_color = "#3a3a3a"
            wall_color = "#cccccc"
            border_color = "#888888"
            highlight_color = "#3d4e70"
            considered_color = "#5f4e3d"
            path_glow = "#004080"
            path_color = "#0078d7"
            temp_path_color = "#d04030"
            start_color = "#2e8b57"
            end_color = "#b22222"
        else:
            # Light mode colors
            bg_color = "#f5f5f7"
            text_color = "#000000"
            accent_color = "#ffffff"
            canvas_bg = "#ffffff"
            grid_color = "#f0f0f0"
            wall_color = "#333333"
            border_color = "#111111"
            highlight_color = "#FFF59D"
            considered_color = "#FFCC80"
            path_glow = "#90CAF9"
            path_color = "#2196F3"
            temp_path_color = "#FF5722"
            start_color = "#4CAF50"
            end_color = "#F44336"
            
        # Store colors for later use
        self.theme_colors = {
            "canvas_bg": canvas_bg,
            "grid": grid_color,
            "wall": wall_color,
            "border": border_color,
            "highlight": highlight_color,
            "considered": considered_color,
            "path_glow": path_glow,
            "path": path_color,
            "temp_path": temp_path_color,
            "start": start_color,
            "end": end_color
        }
        
        # Configure the style for all widgets
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=text_color)
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("TEntry", fieldbackground=accent_color, foreground=text_color)
        style.configure("TCombobox", fieldbackground=accent_color, foreground=text_color)
        
        # Configure button styles
        style.configure("Generate.TButton", background="#4CAF50")
        style.configure("Solve.TButton", background="#2196F3")
        style.configure("Load.TButton", background="#FF9800")
        
        # Configure switch style for the theme toggle
        style.configure("Switch.TCheckbutton", background=bg_color)
        
        # Update root and canvas backgrounds
        self.root.configure(background=bg_color)
        if hasattr(self, 'canvas'):
            self.canvas.configure(background=canvas_bg)
            
            # If we have a maze, redraw it with the new colors
            if self.graph:
                self.canvas.delete("all")
                self.draw_maze()
                
                # Redraw path if we are in the middle of solving
                if self.step_data and self.step_index > 0:
                    if self.step_index < len(self.step_data):
                        # Redraw current state
                        temp_parent_dict = {n: p for n, _, p in self.step_data[:self.step_index] if p is not None}
                        temp_parent_dict[self.start] = None
                        current, _, _ = self.step_data[self.step_index-1]
                        temp_path = self.reconstruct_path(temp_parent_dict, current)
                        self.draw_temp_path(temp_path)
                    else:
                        # Redraw final path
                        final_parent_dict = {n: p for n, _, p in self.step_data if p is not None}
                        final_parent_dict[self.start] = None
                        final_path = self.reconstruct_path(final_parent_dict, self.end)
                        self.draw_path(final_path)
        
        if hasattr(self, 'image_canvas'):
            self.image_canvas.configure(background=canvas_bg)
            
            # Display the source image if exists
            if self.source_image is not None:
                self.display_source_image(self.source_image)

    def toggle_theme(self):
        self.apply_theme()

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
        
        # Clear the source image
        self.clear_source_image()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.source_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Generate maze from image
                self.graph, _ = detect_lines(image)
                
                self.selection_mode = True
                self.clicks_remaining = 2  
                self.root.title("MazeMaster - Click start point")

                available_width = self.canvas_width - 2 * self.offset
                available_height = self.canvas_height - 2 * self.offset
                self.cell_size = min(available_width // self.graph.size, available_height // self.graph.size)
                self.cell_size = max(self.cell_size, 10)
                
                # Clear and draw maze without start,end points initially
                self.canvas.delete("all")
                self.draw_maze()
                self.step_data = None
                self.step_index = 0
                self.is_solving = False
                self.time_label.config(text="Total Time: 0.0000 ms")
                
                self.display_source_image(self.source_image)
                self.image_label.config(text="")

    def display_source_image(self, image):
        if image is None:
            return
        
        # Clear the image canvas
        self.image_canvas.delete("all")
        
        # Resize image to fit the canvas while maintaining aspect ratio
        img_height, img_width = image.shape[:2]
        canvas_width = self.image_canvas.winfo_width() or self.canvas_width
        canvas_height = self.image_canvas.winfo_height() or self.canvas_height
        
        # Calculate the scaling factor
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to PIL format and then to PhotoImage
        pil_image = Image.fromarray(resized_image)
        self.photo_image = ImageTk.PhotoImage(pil_image)
        
        # Calculate position to center the image
        x_pos = (canvas_width - new_width) // 2
        y_pos = (canvas_height - new_height) // 2
        
        # Create image on canvas
        self.image_canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.photo_image)

    def clear_source_image(self):
        self.image_canvas.delete("all")
        self.source_image = None
        self.photo_image = None
        self.image_label.config(text="Source Image")

    def draw_maze(self):
        size = self.graph.size
        
        # Draw grid background with theme-appropriate color
        for i in range(size + 1):
            x = i * self.cell_size + self.offset
            y = i * self.cell_size + self.offset
            if i < size:
                self.canvas.create_line(self.offset, y, size * self.cell_size + self.offset, y, 
                                      fill=self.theme_colors["grid"], width=1, tags="grid")
                self.canvas.create_line(x, self.offset, x, size * self.cell_size + self.offset, 
                                      fill=self.theme_colors["grid"], width=1, tags="grid")
        
        # Draw walls
        for i in range(size):
            for j in range(size):
                x1 = j * self.cell_size + self.offset
                y1 = i * self.cell_size + self.offset
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                line_width = 2
                
                if ((i, j), (i, j+1)) not in self.graph.edges and j+1 < size:
                    self.canvas.create_line(x2, y1, x2, y2, fill=self.theme_colors["wall"], width=line_width)
                if ((i, j), (i+1, j)) not in self.graph.edges and i+1 < size:
                    self.canvas.create_line(x1, y2, x2, y2, fill=self.theme_colors["wall"], width=line_width)
                if j == 0 or ((i, j), (i, j-1)) not in self.graph.edges:
                    self.canvas.create_line(x1, y1, x1, y2, fill=self.theme_colors["wall"], width=line_width)
                if i == 0 or ((i, j), (i-1, j)) not in self.graph.edges:
                    self.canvas.create_line(x1, y1, x2, y1, fill=self.theme_colors["wall"], width=line_width)

        # Draw border
        border_width = 3
        self.canvas.create_rectangle(
            self.offset, self.offset, 
            size * self.cell_size + self.offset, size * self.cell_size + self.offset,
            outline=self.theme_colors["border"], width=border_width, tags="border"
        )

        # draw start and end points if not in selection mode
        if not self.selection_mode or self.clicks_remaining < 2:
            start_end_size = max(self.cell_size // 5, 6)
            sx, sy = self.start
            self.canvas.create_oval(
                sy * self.cell_size + self.offset + start_end_size,
                sx * self.cell_size + self.offset + start_end_size,
                sy * self.cell_size + self.offset + self.cell_size - start_end_size,
                sx * self.cell_size + self.offset + self.cell_size - start_end_size,
                fill=self.theme_colors["start"], outline=self.theme_colors["start"], width=2
            )
        if not self.selection_mode or self.clicks_remaining == 0:
            ex, ey = self.end
            self.canvas.create_oval(
                ey * self.cell_size + self.offset + start_end_size,
                ex * self.cell_size + self.offset + start_end_size,
                ey * self.cell_size + self.offset + self.cell_size - start_end_size,
                ex * self.cell_size + self.offset + self.cell_size - start_end_size,
                fill=self.theme_colors["end"], outline=self.theme_colors["end"], width=2
            )

    def solve_maze(self):
        if not self.graph or self.is_solving or self.selection_mode:
            return

        self.canvas.delete("path", "highlight", "temp_path")
        self.step_data = None
        self.step_index = 0
        self.is_solving = True
        
        start_time = time.perf_counter()
        algorithm = self.algo_var.get()

        if algorithm == "BFS":
            _, steps = self.bfs()
        elif algorithm == "Dijkstra":
            _, steps = self.dijkstra()
        else:  # A*
            _, steps = self.a_star()

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
        current, considered, _ = self.step_data[self.step_index]
        
        x, y = current
        
        self.canvas.create_rectangle(
            y * self.cell_size + self.offset + 1, x * self.cell_size + self.offset + 1,
            (y + 1) * self.cell_size + self.offset - 1, (x + 1) * self.cell_size + self.offset - 1,
            fill=self.theme_colors["highlight"], outline="", tags="highlight"
        )

        for nx, ny in considered:
            self.canvas.create_rectangle(
                ny * self.cell_size + self.offset + 1, nx * self.cell_size + self.offset + 1,
                (ny + 1) * self.cell_size + self.offset - 1, (nx + 1) * self.cell_size + self.offset - 1,
                fill=self.theme_colors["considered"], outline="", tags="highlight"
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
        queue = deque([self.start])
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
        # Initialize open and closed sets
        open_set = {}
        closed_set = set()
        
        # Track g-scores (cost from start) and f-scores (g + heuristic)
        g_scores = {self.start: 0}
        f_scores = {self.start: heuristic(*self.start)}
        
        # Priority queue with f-score as priority
        pq = [(f_scores[self.start], 0, self.start)]  # (f_score, tie_breaker, node)
        parent = {self.start: None}
        steps = []
        counter = 0  # Tie-breaker for equal f-scores

        while pq:
            _, _, current = heapq.heappop(pq)
            x, y = current
            
            # Skip if already processed
            if current in closed_set:
                continue
                
            # Add to closed set
            closed_set.add(current)
            considered = []
            
            # If we've reached the end, break
            if current == self.end:
                steps.append((current, considered, parent[current]))
                break

            # Check neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (x + dx, y + dy)
                new_x, new_y = neighbor
                
                # Skip if out of bounds or wall
                if not (0 <= new_x < size and 0 <= new_y < size and 
                        ((x, y), (new_x, new_y)) in self.graph.edges):
                    continue
                    
                # Skip if already processed
                if neighbor in closed_set:
                    continue
                    
                # Calculate new g-score
                tentative_g = g_scores[current] + 1
                
                # If new path is better or node not yet discovered
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    # Update parent
                    parent[neighbor] = current
                    # Update scores
                    g_scores[neighbor] = tentative_g
                    f_scores[neighbor] = tentative_g + heuristic(*neighbor)
                    considered.append(neighbor)
                    
                    # Add to priority queue with tie-breaker
                    counter += 1
                    heapq.heappush(pq, (f_scores[neighbor], counter, neighbor))
            
            steps.append((current, considered, parent[current]))

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
        
        # Draw a glowing path
        for i in range(len(path)-1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            
            # Draw shadow
            self.canvas.create_line(
                y1 * self.cell_size + self.offset + self.cell_size/2,
                x1 * self.cell_size + self.offset + self.cell_size/2,
                y2 * self.cell_size + self.offset + self.cell_size/2,
                x2 * self.cell_size + self.offset + self.cell_size/2,
                fill=self.theme_colors["path_glow"], width=5, tags="path"
            )
            
            # Draw the main path
            self.canvas.create_line(
                y1 * self.cell_size + self.offset + self.cell_size/2,
                x1 * self.cell_size + self.offset + self.cell_size/2,
                y2 * self.cell_size + self.offset + self.cell_size/2,
                x2 * self.cell_size + self.offset + self.cell_size/2,
                fill=self.theme_colors["path"], width=3, tags="path"
            )

    def draw_temp_path(self, path):
        if not path:
            return
        
        self.canvas.delete("temp_path")
        for i in range(len(path)-1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            
            # Draw the temporary path
            self.canvas.create_line(
                y1 * self.cell_size + self.offset + self.cell_size/2,
                x1 * self.cell_size + self.offset + self.cell_size/2,
                y2 * self.cell_size + self.offset + self.cell_size/2,
                x2 * self.cell_size + self.offset + self.cell_size/2,
                fill=self.theme_colors["temp_path"], width=2, tags="temp_path"
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()