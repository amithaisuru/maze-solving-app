# MazeMaster: Maze Generation and Solving Application

## About the Project

MazeMaster is a Python-based desktop application designed for generating, visualizing, and solving square mazes using various pathfinding algorithms. It provides an intuitive graphical user interface (GUI) built with Tkinter, allowing users to generate random mazes, solve them with different algorithms, and visualize the solving process. Additionally, the application supports uploading maze images, processing them using computer vision techniques, and solving the detected maze. The project also includes a script to analyze the performance of different solving algorithms across multiple maze sizes.

## Features

With MazeMaster, users can:
- **Generate Random Mazes**: Create square mazes of customizable sizes using a depth-first search (DFS) algorithm with randomized neighbor selection.
- **Solve Mazes**: Solve generated or uploaded mazes using one of three pathfinding algorithms: Breadth-First Search (BFS), Dijkstra’s Algorithm, or A* (A-star) Algorithm.
- **Visualize Solutions**: Watch an animated step-by-step visualization of the solving process, with customizable animation speed.
- **Upload and Solve Maze Images**: Upload maze images in PNG or JPEG format, which are processed using computer vision to extract the maze structure for solving.
- **Select Start and End Points**: Interactively choose start and end points for image-based mazes by clicking on the canvas.
- **Toggle Themes**: Switch between light and dark themes for a better user experience.
- **View Statistics**: Display the total time taken to solve a maze.
- **Performance Analysis**: Run statistical analysis to compare the performance of BFS, Dijkstra’s, and A* algorithms across multiple maze sizes.

## Technologies Used

- **Python 3**: Core programming language for the application.
- **Tkinter**: Used for creating the GUI, including the maze visualization and control panel.
- **OpenCV (opencv-contrib-python)**: Handles image processing for maze detection from uploaded images.
- **NumPy**: Supports numerical operations, particularly in image processing.
- **Pillow (PIL)**: Manages image handling for displaying source images in the GUI.
- **Matplotlib**: Used for displaying intermediate images during the computer vision process (for debugging).
- **Batch Script (Windows)**: Simplifies dependency installation and application execution on Windows.

## Algorithms

### Maze Generation
- **Depth-First Search (DFS)**: The maze is generated using a randomized DFS algorithm. Starting from a single cell, the algorithm explores unvisited neighbors randomly, adding edges to create paths and backtracking when no unvisited neighbors remain. This ensures a perfect maze (a single solution path with no loops).

### Maze Solving
- **Breadth-First Search (BFS)**: Explores the maze level by level, ensuring the shortest path is found in an unweighted graph. It uses a queue to track nodes to visit.
- **Dijkstra’s Algorithm**: Finds the shortest path by maintaining a priority queue of nodes based on their distance from the start. In this unweighted graph, it behaves similarly to BFS but is implemented with a priority queue.
- **A* (A-star) Algorithm**: An informed search algorithm that uses a heuristic (Manhattan distance) to prioritize nodes likely to lead to the goal, optimizing the search process.

## Image Processing for Maze Detection

The application uses computer vision techniques (via OpenCV) to detect and extract maze structures from uploaded images. The process involves the following steps:
1. **Image Resizing**: The input image is resized to a fixed 800x800 resolution to standardize processing.
2. **Gaussian Blur**: A Gaussian blur is applied to reduce noise and smooth the image.
3. **Grayscale Conversion**: The image is converted to grayscale to simplify further processing.
4. **Binary Thresholding**: The grayscale image is thresholded to create a binary image, where maze walls are typically black (0) and paths are white (255).
5. **Morphological Closing**: A morphological closing operation fills small gaps in the maze walls using a 5x5 kernel.
6. **Skeletonization**: The binary image is thinned to produce a skeleton, reducing walls to single-pixel lines for easier line detection.
7. **Edge Detection**: Canny edge detection is applied to the skeletonized image to identify edges.
8. **Hough Line Transform**: Detects horizontal and vertical lines in the maze, which represent the grid structure.
9. **Line Filtering**: Lines are separated into horizontal and vertical groups, and duplicates within 5 pixels are removed to avoid redundant lines.
10. **Line Extension**: Detected lines are extended to span the entire 800x800 image, ensuring a complete grid.
11. **Graph Construction**: The grid is used to create a graph where nodes represent cells, and edges are added where paths exist (based on pixel values in the binary image). The graph is then used for solving.
12. **Intermediate Visualization (Optional)**: Intermediate images (e.g., blurred, binary, skeletonized) can be displayed for debugging purposes.

This process allows the application to convert a maze image into a graph representation that can be solved using the implemented pathfinding algorithms.

## How to Run the Project

### Prerequisites
- **Python 3.6+**: Ensure Python is installed and added to your system PATH.
- **pip**: Python package manager (usually included with Python).
- A Windows system (for the provided batch script) or manual installation on other operating systems.

### Installation and Execution
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   - On **Windows**, run the provided batch script to automatically check for Python, pip, and install dependencies:
     ```bash
     run_app.bat
     ```
     This script checks for Python and pip, installs dependencies listed in `requirements.txt`, and runs `maze_app.py`.
   - On **other operating systems**, manually install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Application**:
   - If using the batch script on Windows, it automatically runs `maze_app.py`.
   - Otherwise, execute:
     ```bash
     python maze_app.py
     ```

4. **Run Statistical Analysis** (Optional):
   To analyze the performance of BFS, Dijkstra’s, and A* algorithms for different maze sizes:
   ```bash
   python statistic_calc.py
   ```
   This generates performance statistics for 100 mazes of sizes 9x9, 16x16, and 21x21.

### Usage
- **Generate a Maze**: Enter a maze size (e.g., 10), select an algorithm (BFS, Dijkstra’s, or A*), and click "Generate Maze" to create a random maze.
- **Solve a Maze**: Click "Solve Maze" to animate the solving process. Adjust the speed (in milliseconds) for the animation.
- **Load an Image**: Click "Load Image" to upload a maze image (PNG/JPEG). After loading, click on the maze canvas to select the start point, then the end point. Solve the maze as above.
- **Toggle Theme**: Use the "Dark Mode" checkbox to switch between light and dark themes.
- **View Statistics**: The total solving time is displayed in the statistics section after solving a maze.

## Project Structure
- `maze_app.py`: Main application with the Tkinter GUI and pathfinding algorithms.
- `maze_classes.py`: Defines `Node` and `Graph` classes for maze representation and generation.
- `vision_handler.py`: Handles computer vision for maze detection from images (simplified version).
- `vision_steps.py`: Extended vision processing with intermediate image visualization for debugging.
- `statistic_calc.py`: Analyzes performance of solving algorithms.
- `run_app.bat`: Windows batch script for dependency installation and running the app.
- `requirements.txt`: Lists required Python packages.

## Notes
- Ensure maze images are clear, with distinct walls and paths, for accurate detection.
- The application assumes square mazes for both generated and image-based mazes.
- For image-based mazes, select start and end points within the maze grid after loading the image.

## License
This project is open-source and available under the [MIT License](LICENSE).