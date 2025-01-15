from node import Node


class Graph:
    def __init__(self, size):
        self.size = size
        self.nodes = {}
        self.initialize_grid()
    
    def initialize_grid(self):
        for y in range(self.size):
            for x in range(self.size):
                self.nodes[(x, y)] = Node(x, y)
    
    def get_neighbors(self, x, y):
        neighbors = []
        directions = [(0, -1, 'N', 'S'), (0, 1, 'S', 'N'), 
                     (1, 0, 'E', 'W'), (-1, 0, 'W', 'E')]
        
        for dx, dy, d1, d2 in directions:
            new_x, new_y = x + dx, y + dy
            if (new_x, new_y) in self.nodes:
                if not self.nodes[(x, y)].walls[d1]:
                    neighbors.append(self.nodes[(new_x, new_y)])
        return neighbors