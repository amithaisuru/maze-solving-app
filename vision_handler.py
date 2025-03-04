import cv2
import numpy as np

from maze_classes import Graph, Node


class VisionHandler:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        self.height, self.width = self.image.shape[:2]
        self.binary = None
        self.grid_size = None
        self.cell_size = None

    def preprocess_image(self):
        """Convert to grayscale and threshold to binary."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        self.binary = binary
        return binary

    def detect_cell_size(self):
        """Identify the nearest two vertical lines to determine cell width/height."""
        binary = self.binary
        
        # Use a horizontal profile (middle row) to detect vertical lines
        mid_row = binary[self.height // 2, :]
        transitions = []
        
        # Find transitions from white (255) to black (0) or vice versa
        for x in range(1, len(mid_row)):
            if mid_row[x-1] != mid_row[x]:
                transitions.append(x)
        
        # Find the smallest distance between two consecutive transitions (cell width)
        if len(transitions) < 2:
            raise ValueError("Could not detect enough vertical lines in the image")
        
        distances = [transitions[i+1] - transitions[i] for i in range(len(transitions)-1)]
        self.cell_size = min(distances)  # Assume smallest consistent gap is cell size
        
        # Estimate grid size based on image dimensions
        self.grid_size = min(self.width // self.cell_size, self.height // self.cell_size)
        return self.cell_size, self.grid_size

    def overlay_grid(self):
        """Overlay a grid on the binary image for visualization and analysis."""
        binary_with_grid = cv2.cvtColor(self.binary, cv2.COLOR_GRAY2BGR)
        
        # Draw vertical lines
        for j in range(self.grid_size + 1):
            x = j * self.cell_size
            cv2.line(binary_with_grid, (x, 0), (x, self.height), (0, 255, 0), 1)
        
        # Draw horizontal lines
        for i in range(self.grid_size + 1):
            y = i * self.cell_size
            cv2.line(binary_with_grid, (0, y), (self.width, y), (0, 255, 0), 1)
        
        # Optional: Save or display the grid overlay for debugging
        # cv2.imwrite("grid_overlay.png", binary_with_grid)
        return binary_with_grid

    def analyze_grid(self):
        """Analyze the binary image with the overlaid grid to detect walls."""
        edges = set()
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                # Check right border
                if j + 1 < self.grid_size:
                    right_region = self.binary[y1:y2, x2-2:x2+2]
                    if np.mean(right_region) > 128:  # Wall present (black in inverted binary)
                        pass
                    else:
                        edges.add(((i, j), (i, j+1)))
                        edges.add(((i, j+1), (i, j)))
                
                # Check bottom border
                if i + 1 < self.grid_size:
                    bottom_region = self.binary[y2-2:y2+2, x1:x2]
                    if np.mean(bottom_region) > 128:  # Wall present
                        pass
                    else:
                        edges.add(((i, j), (i+1, j)))
                        edges.add(((i+1, j), (i, j)))
        
        return edges

    def detect_start_end(self):
        """Detect green (start) and red (end) points."""
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        green_lower = np.array([35, 100, 100])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_y, green_x = np.where(green_mask > 0)
        start = (green_y[0] // self.cell_size, green_x[0] // self.cell_size) if green_y.size > 0 else (0, 0)
        
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
        red_y, red_x = np.where(red_mask > 0)
        end = (red_y[0] // self.cell_size, red_x[0] // self.cell_size) if red_y.size > 0 else (self.grid_size-1, self.grid_size-1)
        
        return start, end

    def image_to_graph(self):
        """Convert image to Graph object."""
        self.preprocess_image()
        self.detect_cell_size()
        
        self.graph = Graph(self.grid_size)
        self.graph.edges = self.analyze_grid()
        
        # Optional: Overlay grid for debugging (not needed for graph generation)
        # self.overlay_grid()
        
        start, end = self.detect_start_end()
        return self.graph, start, end

if __name__ == "__main__":
    handler = VisionHandler("sample_maze.png")
    graph, start, end = handler.image_to_graph()
    print(f"Grid Size: {graph.size}x{graph.size}")
    print(f"Edges: {len(graph.edges)}")
    print(f"Start: {start}, End: {end}")