import math
import os

import cv2
import numpy as np


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

def detect_lines(src_image):
    #resize image
    src_image = cv2.resize(src_image, (800, 800))

    #gaussian blur
    src_image = cv2.GaussianBlur(src_image, (5, 5), 0)

    # Convert the image to grayscale
    gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)

    #coverting to binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    #morphological closing 
    kernel = np.ones((5,5),np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imshow('closed', closed)
    cv2.waitKey(0)

    #get center lines
    skeleton = cv2.ximgproc.thinning(closed)
    _, skeleton = cv2.threshold(skeleton, 150, 255, cv2.THRESH_BINARY)

    cv2.imshow('skeleton', skeleton)
    cv2.waitKey(0)

    # Apply edge detection method on the image
    edges = cv2.Canny(skeleton, 100, 150)

    # Apply Hough Line Transform on the image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)

    horizontal_lines, vertical_lines = seperate_lines(lines)

    # Draw line1 on the image
    new_lines = horizontal_lines + vertical_lines
    for line in new_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(src_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Display the image
    cv2.imshow('image', src_image)
    cv2.waitKey(0) 

image = cv2.imread('maze1.jpg')
detect_lines(image)