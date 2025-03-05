import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize


class MazeSolver:
    def __init__(self, image_path):
        """
        Initialize the MazeSolver with the maze image
        
        :param image_path: Path to the maze image file
        """
        # Read the image
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Validate image loading
        if self.original_image is None:
            raise ValueError("Could not read the image file")
        
        # Image preprocessing attributes
        self.binary_image = None
        self.skeleton = None
        self.maze_graph = nx.Graph()
    
    def preprocess_image(self):
        """
        Preprocess the maze image:
        1. Apply Otsu's thresholding
        2. Skeletonize the image
        """
        # Apply Otsu's thresholding
        _, self.binary_image = cv2.threshold(
            self.original_image, 
            0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Skeletonize the image using scikit-image
        self.skeleton = skeletonize(self.binary_image // 255)
        
        return self.skeleton
    
    def create_maze_graph(self):
        """
        Convert maze to graph representation using skeletonization
        """
        height, width = self.skeleton.shape
        
        # Find junction and endpoint pixels
        def get_neighbor_count(x, y):
            """Count white neighbors in 8-connectivity"""
            count = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < width and 0 <= ny < height and 
                        self.skeleton[ny, nx] and 
                        not (dx == 0 and dy == 0)):
                        count += 1
            return count
        
        # Find junction and endpoint nodes
        nodes = []
        for y in range(height):
            for x in range(width):
                if self.skeleton[y, x]:
                    neighbor_count = get_neighbor_count(x, y)
                    # Junctions have 3 or more neighbors, endpoints have 1
                    if neighbor_count != 2:
                        nodes.append((x, y))
        
        # Add nodes to graph
        self.maze_graph.add_nodes_from(nodes)
        
        # Connect nodes
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if self._trace_path(node1, node2):
                    # Calculate path length
                    path_length = self._calculate_path_length(node1, node2)
                    self.maze_graph.add_edge(node1, node2, weight=path_length)
        
        return self.maze_graph
    
    def _trace_path(self, start, end):
        """
        Trace a path between two points using breadth-first search on skeleton
        """
        if start == end:
            return False
        
        height, width = self.skeleton.shape
        visited = np.zeros_like(self.skeleton, dtype=bool)
        queue = [(start[0], start[1])]
        visited[start[1], start[0]] = True
        
        # Possible 8-connectivity movements
        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        # Limit search to prevent infinite loops
        max_steps = 500
        steps = 0
        
        while queue and steps < max_steps:
            x, y = queue.pop(0)
            steps += 1
            
            # Check if reached end point
            if (x, y) == end:
                return True
            
            # Explore neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Check bounds and unvisited skeleton pixels
                if (0 <= nx < width and 0 <= ny < height and 
                    self.skeleton[ny, nx] and 
                    not visited[ny, nx]):
                    queue.append((nx, ny))
                    visited[ny, nx] = True
        
        return False
    
    def _calculate_path_length(self, start, end):
        """
        Calculate Manhattan distance between two points
        """
        return abs(start[0] - end[0]) + abs(start[1] - end[1])
    
    def visualize_graph(self):
        """
        Visualize the maze graph
        """
        plt.figure(figsize=(15, 15))
        
        # Plot original skeleton
        plt.subplot(121)
        plt.imshow(self.skeleton, cmap='binary')
        plt.title("Maze Skeleton")
        
        # Plot graph
        plt.subplot(122)
        pos = {node: node for node in self.maze_graph.nodes()}
        nx.draw(
            self.maze_graph, 
            pos, 
            with_labels=True, 
            node_color='red', 
            node_size=50, 
            font_size=8
        )
        plt.title("Maze Graph Representation")
        
        plt.tight_layout()
        plt.show()
        
        # Print graph information
        print(f"Number of nodes: {self.maze_graph.number_of_nodes()}")
        print(f"Number of edges: {self.maze_graph.number_of_edges()}")

# Example usage
def main():
    maze_solver = MazeSolver('maze2.jpg')
    maze_solver.preprocess_image()
    maze_solver.create_maze_graph()
    maze_solver.visualize_graph()

if __name__ == "__main__":
    main()