import cv2
import numpy as np

from maze_classes import Graph


def seperate_lines(lines):
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            vertical_lines.append(line)
        elif y1 == y2:
            horizontal_lines.append(line)
    
    #select one line from set of lines within range 5 pixels
    horizontal_lines = sorted(horizontal_lines, key=lambda x: x[0][1])
    vertical_lines = sorted(vertical_lines, key=lambda x: x[0][0])

    new_horizontal_lines = []
    new_vertical_lines = []

    for line in horizontal_lines:
        x1, y1, x2, y2 = line[0]
        if len(new_horizontal_lines) == 0:
            new_horizontal_lines.append(line)
        else:
            x1_, y1_, x2_, y2_ = new_horizontal_lines[-1][0]
            if abs(y1 - y1_) > 5:
                new_horizontal_lines.append(line)
    
    for line in vertical_lines:
        x1, y1, x2, y2 = line[0]
        if len(new_vertical_lines) == 0:
            new_vertical_lines.append(line)
        else:
            x1_, y1_, x2_, y2_ = new_vertical_lines[-1][0]
            if abs(x1 - x1_) > 5:
                new_vertical_lines.append(line)
    
    #extend lines through the image
    extended_horizontal_lines = []
    for line in new_horizontal_lines:
        x1, y1, x2, y2 = line[0]
        x1 = 0
        x2 = 800
        line[0] = (x1, y1, x2, y2)
        extended_horizontal_lines.append(line)
    
    extended_vertical_lines = []
    for line in new_vertical_lines:
        x1, y1, x2, y2 = line[0]
        y1 = 0
        y2 = 800
        line[0] = (x1, y1, x2, y2)
        extended_vertical_lines.append(line)

    return extended_horizontal_lines, extended_vertical_lines

def get_top_left_point(horizontal_lines, vertical_lines):
    y = horizontal_lines[0][0][1]
    x = vertical_lines[0][0][0]
    return x,y

def get_cell_size(horizontal_lines, vertical_lines):
    cell_size = (horizontal_lines[1][0][1] - horizontal_lines[0][0][1], vertical_lines[1][0][0] - vertical_lines[0][0][0])
    return cell_size

def get_grid_size(horizontal_lines, vertical_lines):
    grid_size = (len(horizontal_lines) - 1, len(vertical_lines) - 1)
    return grid_size

def get_x_values_of_vertical_lines(vertical_lines):
    x_values = []
    for line in vertical_lines:
        x1, y1, x2, y2 = line[0]
        x_values.append(x1)
    return x_values

def get_y_values_of_horizontal_lines(horizontal_lines):
    y_values = []
    for line in horizontal_lines:
        x1, y1, x2, y2 = line[0]
        y_values.append(y1)
    return y_values

def detect_lines(src_image):
    #resize image
    src_image = cv2.resize(src_image, (800, 800))

    #gaussian blur
    src_image = cv2.GaussianBlur(src_image, (5, 5), 0)

    # Convert the image to grayscale
    gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)

    #coverting to binary image
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    #morphological closing 
    kernel = np.ones((5,5),np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    #get center lines
    skeleton = cv2.ximgproc.thinning(closed)
    _, skeleton = cv2.threshold(skeleton, 150, 255, cv2.THRESH_BINARY)

    # Apply edge detection method on the image
    edges = cv2.Canny(skeleton, 100, 150)

    # Apply Hough Line Transform on the image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)

    horizontal_lines, vertical_lines = seperate_lines(lines)
    return generate_graph(binary, horizontal_lines, vertical_lines), src_image

def generate_graph(binary_image, horizontal_lines, vertical_lines):

    graph_size = get_grid_size(horizontal_lines, vertical_lines)[0]
    cell_size = get_cell_size(horizontal_lines, vertical_lines)[0]
    x_values = get_x_values_of_vertical_lines(vertical_lines)
    y_values = get_y_values_of_horizontal_lines(horizontal_lines)
    #remove first value from x_values and y_values
    x_values = x_values[1:]
    y_values = y_values[1:]
    
    n = graph_size  # Single integer for square grid
    graph = Graph(graph_size)  # Assuming Graph(n) initializes an n x n grid of Nodes

    #horizontal edges
    for i in range(n):
        for j in range(n-1):
            x = x_values[j]
            y = y_values[i] - cell_size//2
            if binary_image[y, x] == 0:
                graph.addEdge(i, j, i, j+1)
    
    #vertical edges
    for i in range(n-1):
        for j in range(n):
            x = x_values[j] - cell_size//2
            y = y_values[i]
            if binary_image[y, x] == 0:
                graph.addEdge(i, j, i+1, j)
    
    return graph