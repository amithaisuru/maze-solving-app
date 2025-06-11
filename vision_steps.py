"""
Computer Vision and Image Processing Module
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
            x1_, y1_, _, _ = new_horizontal_lines[-1][0]
            if abs(y1 - y1_) > 5:
                new_horizontal_lines.append(line)
    
    for line in vertical_lines:
        x1, y1, x2, y2 = line[0]
        if len(new_vertical_lines) == 0:
            new_vertical_lines.append(line)
        else:
            x1_, y1_, _, _ = new_vertical_lines[-1][0]
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
        x1, _, _, _ = line[0]
        x_values.append(x1)
    return x_values

def get_y_values_of_horizontal_lines(horizontal_lines):
    y_values = []
    for line in horizontal_lines:
        _, y1, _, _ = line[0]
        y_values.append(y1)
    return y_values

def draw_detected_lines(image, horizontal_lines, vertical_lines):
    """Draw detected lines on an image"""
    line_image = image.copy()
    
    # Draw horizontal lines in blue
    for line in horizontal_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Draw vertical lines in green
    for line in vertical_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return line_image

def add_border_to_image(image, border_size=5, border_color=(0, 0, 0)):
    """Add a border to an image"""
    return cv2.copyMakeBorder(
        image, 
        border_size, border_size, border_size, border_size, 
        cv2.BORDER_CONSTANT, 
        value=border_color
    )

def detect_lines(src_image, show_intermediate=False):
    # Store intermediate images in a dictionary
    intermediate_images = {}
    
    # Original resized image
    src_image = cv2.resize(src_image, (800, 800))
    intermediate_images['1. Resized'] = src_image.copy()

    # Gaussian blur
    blurred = cv2.GaussianBlur(src_image, (5, 5), 0)
    intermediate_images['2. Blurred'] = blurred.copy()

    # Convert the image to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    intermediate_images['3. Grayscale'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Converting to binary image
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    intermediate_images['4. Binary'] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Morphological closing 
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    intermediate_images['5. Closed'] = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)

    # Get center lines using skeletonization
    skeleton = cv2.ximgproc.thinning(closed)
    _, skeleton = cv2.threshold(skeleton, 150, 255, cv2.THRESH_BINARY)
    intermediate_images['6. Skeleton'] = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

    # Apply edge detection method on the image
    edges = cv2.Canny(skeleton, 100, 150)

    # Apply Hough Line Transform on the image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)

    # Separate horizontal and vertical lines
    horizontal_lines, vertical_lines = seperate_lines(lines)
    
    # Draw detected lines on binary image
    line_image = draw_detected_lines(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), horizontal_lines, vertical_lines)
    intermediate_images['7. Detected Lines'] = line_image

    # Display intermediate images if requested
    if show_intermediate:
        display_intermediate_images(intermediate_images)

    return generate_graph(binary, horizontal_lines, vertical_lines), src_image, intermediate_images

def display_intermediate_images(images):
    """Display all intermediate images in a grid with borders"""
    n_images = len(images)
    
    # Calculate grid dimensions
    if n_images <= 4:
        rows, cols = 1, n_images
    elif n_images <= 8:
        rows, cols = 2, 4
    else:
        rows, cols = 3, 4
    
    # Create figure with custom grid layout
    fig = plt.figure(figsize=(12, 4 * rows))
    gs = GridSpec(rows, cols, figure=fig, wspace=0.1, hspace=0.3)
    
    for i, (title, img) in enumerate(images.items()):
        row, col = i // cols, i % cols
        ax = fig.add_subplot(gs[row, col])
        
        # Convert BGR to RGB for matplotlib
        if img.shape[-1] == 3:  # Check if image has 3 channels
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        # Add a border to the image
        img_with_border = add_border_to_image(img_rgb, border_size=5, border_color=(0, 0, 0))
            
        ax.imshow(img_with_border)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a border around the subplot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.show()

def generate_graph(binary_image, horizontal_lines, vertical_lines):
    graph_size = get_grid_size(horizontal_lines, vertical_lines)[0]
    cell_size = get_cell_size(horizontal_lines, vertical_lines)[0]
    x_values = get_x_values_of_vertical_lines(vertical_lines)
    y_values = get_y_values_of_horizontal_lines(horizontal_lines)
    
    # Remove first value from x_values and y_values
    x_values = x_values[1:]
    y_values = y_values[1:]
    
    n = graph_size  # Single integer for square grid
    graph = Graph(graph_size)

    # Horizontal edges
    for i in range(n):
        for j in range(n-1):
            x = x_values[j]
            y = y_values[i] - cell_size//2
            if binary_image[y, x] == 0:
                graph.addEdge(i, j, i, j+1)
    
    # Vertical edges
    for i in range(n-1):
        for j in range(n):
            x = x_values[j] - cell_size//2
            y = y_values[i]
            if binary_image[y, x] == 0:
                graph.addEdge(i, j, i+1, j)
    
    return graph

# Example usage:
input_image = cv2.imread('test_images/maze1.jpg')
graph, src_image, intermediate_images = detect_lines(input_image, show_intermediate=True)