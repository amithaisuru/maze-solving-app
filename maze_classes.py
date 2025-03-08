import random
from collections import deque


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

    def printEdges(self):
        for edge in self.edges:
            print(f"Edge: {edge[0]} -> {edge[1]}")
    
    def addEdge(self, x1, y1, x2, y2):
        # Check if coordinates are valid
        if (0 <= x1 < self.size and 0 <= y1 < self.size and 
            0 <= x2 < self.size and 0 <= y2 < self.size):
            # Add edges in both directions for an undirected graph
            self.edges.add(((x1, y1), (x2, y2)))
            self.edges.add(((x2, y2), (x1, y1)))
            self.nodes[x2][y2].visited = True

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
                self.addEdge(current_x, current_y, next_x, next_y)
                self.nodes[next_x][next_y].visited = True
                stack.append((next_x, next_y))
            else:
                stack.pop()
    
    def printNodes(self):
        for i in range(self.size):
            for j in range(self.size):
                print(f"Node ({i}, {j}):")