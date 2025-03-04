import cv2
import numpy as np

from maze_classes import Graph, Node  # Import from new file


class VisionHandler:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        self.height, self.width = self.image.shape[:2]

    def preprocess_image(self):
        """Convert to grayscale and threshold to binary."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        #display image
        cv2.imshow("Binary Image", binary)
        return binary

    def detect_grid_size(self, binary):
        """Estimate grid size by detecting cell boundaries."""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        outer_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(outer_contour)
        
        horizontal_lines = 0
        vertical_lines = 0
        sample_row = binary[y + h // 2, x:x+w]
        sample_col = binary[y:y+h, x + w // 2]
        
        for i in range(1, len(sample_row)):
            if sample_row[i-1] == 0 and sample_row[i] == 255:
                horizontal_lines += 1
        for i in range(1, len(sample_col)):
            if sample_col[i-1] == 0 and sample_col[i] == 255:
                vertical_lines += 1
        
        grid_size = min(horizontal_lines + 1, vertical_lines + 1)
        return grid_size

    def detect_borders(self, binary, grid_size):
        """Detect walls and create edge set."""
        cell_size = min(self.width // grid_size, self.height // grid_size)
        edges = set()
        
        for i in range(grid_size):
            for j in range(grid_size):
                x1 = j * cell_size
                y1 = i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                if j + 1 < grid_size:
                    right_region = binary[y1:y2, x2-2:x2+2]
                    if np.mean(right_region) > 128:
                        pass
                    else:
                        edges.add(((i, j), (i, j+1)))
                        edges.add(((i, j+1), (i, j)))
                
                if i + 1 < grid_size:
                    bottom_region = binary[y2-2:y2+2, x1:x2]
                    if np.mean(bottom_region) > 128:
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
        start = (green_y[0] // (self.height // self.graph.size), 
                 green_x[0] // (self.width // self.graph.size)) if green_y.size > 0 else (0, 0)
        
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
        red_y, red_x = np.where(red_mask > 0)
        end = (red_y[0] // (self.height // self.graph.size), 
               red_x[0] // (self.width // self.graph.size)) if red_y.size > 0 else (self.graph.size-1, self.graph.size-1)
        
        return start, end

    def image_to_graph(self):
        """Convert image to Graph object."""
        binary = self.preprocess_image()
        grid_size = self.detect_grid_size(binary)
        
        self.graph = Graph(grid_size)
        self.graph.edges = self.detect_borders(binary, grid_size)
        
        start, end = self.detect_start_end()
        return self.graph, start, end

if __name__ == "__main__":
    handler = VisionHandler("sample_maze.png")
    graph, start, end = handler.image_to_graph()
    print(f"Grid Size: {graph.size}x{graph.size}")
    print(f"Edges: {len(graph.edges)}")
    print(f"Start: {start}, End: {end}")