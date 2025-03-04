class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.neighbors = []
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}
    
    def __lt__(self, other):
        return False