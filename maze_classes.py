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