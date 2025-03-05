import math
import os

import cv2
import numpy as np


def seperate_lines(lines):
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        rho, theta = line[0]
        if (theta < (np.pi / 4. )) or (theta > (3. * np.pi / 4.0)):
            vertical_lines.append(line)
        else:
            horizontal_lines.append(line)
    return horizontal_lines, vertical_lines

def detect_lines(src_image):
    #gaussian blur
    src_image = cv2.GaussianBlur(src_image, (5, 5), 0)

    # Convert the image to grayscale
    gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)

    #coverting to binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Apply edge detection method on the image
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Apply Hough Line Transform on the image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    horizontal_lines, vertical_lines = seperate_lines(lines)

    # Draw line1 on the image
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(src_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the image
    cv2.imshow('image', src_image)
    cv2.waitKey(0) 



image = cv2.imread('maze1.jpg')
detect_lines(image)