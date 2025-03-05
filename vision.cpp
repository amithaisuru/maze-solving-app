#include <opencv2/opencv.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <iostream>
#include <vector>
#include <queue>

class MazeSolver {
public:
    // Graph type definition
    using Graph = boost::adjacency_list<
        boost::vecS,       // Edge list type
        boost::vecS,       // Vertex list type
        boost::undirectedS, // Graph type
        cv::Point,         // Vertex property (coordinates)
        boost::property<boost::edge_weight_t, double> // Edge property (path length)
    >;
    
    using Vertex = Graph::vertex_descriptor;
    using Edge = Graph::edge_descriptor;

    // Constructor
    MazeSolver(const std::string& imagePath) {
        // Read image in grayscale
        image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        
        if (image.empty()) {
            throw std::runtime_error("Could not read the image file");
        }
    }

    // Preprocess image (thresholding and skeletonization)
    void preprocessImage() {
        // Apply Otsu's thresholding
        cv::threshold(image, binaryImage, 0, 255, 
            cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
        
        // Skeletonize the image
        skeletonize(binaryImage, skeleton);
    }

    // Create graph representation of the maze
    void createMazeGraph() {
        // Find junction and endpoint pixels
        std::vector<cv::Point> nodes = findNodes();

        // Add nodes to graph
        for (const auto& node : nodes) {
            boost::add_vertex(node, mazeGraph);
        }

        // Connect nodes
        connectNodes(nodes);
    }

    // Visualize the graph
    void visualizeGraph() {
        // Print graph info
        std::cout << "Number of nodes: " << boost::num_vertices(mazeGraph) << std::endl;
        std::cout << "Number of edges: " << boost::num_edges(mazeGraph) << std::endl;

        // Optional: You might want to use a visualization library like Graphviz
        // or output graph data for external visualization
    }

private:
    // Skeletonization function
    void skeletonize(const cv::Mat& input, cv::Mat& output) {
        cv::Mat img = input.clone();
        output = cv::Mat::zeros(img.size(), CV_8UC1);

        // Repeated thinning
        cv::Mat prev, diff;
        do {
            cv::Mat temp;
            cv::ximgproc::thinning(img, temp);
            diff = temp != prev;
            prev = temp;
            img = temp;
        } while (cv::countNonZero(diff) > 0);

        output = img;
    }

    // Find junction and endpoint nodes
    std::vector<cv::Point> findNodes() {
        std::vector<cv::Point> nodes;
        
        for (int y = 1; y < skeleton.rows - 1; ++y) {
            for (int x = 1; x < skeleton.cols - 1; ++x) {
                if (skeleton.at<uchar>(y, x) == 255) {
                    int neighborCount = countNeighbors(x, y);
                    
                    // Junction or endpoint
                    if (neighborCount != 2) {
                        nodes.emplace_back(x, y);
                    }
                }
            }
        }

        return nodes;
    }

    // Count white neighbors
    int countNeighbors(int x, int y) {
        int count = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                
                int nx = x + dx, ny = y + dy;
                if (nx >= 0 && nx < skeleton.cols && 
                    ny >= 0 && ny < skeleton.rows && 
                    skeleton.at<uchar>(ny, nx) == 255) {
                    count++;
                }
            }
        }
        return count;
    }

    // Connect nodes that have a valid path between them
    void connectNodes(const std::vector<cv::Point>& nodes) {
        for (size_t i = 0; i < nodes.size(); ++i) {
            for (size_t j = i + 1; j < nodes.size(); ++j) {
                if (tracePath(nodes[i], nodes[j])) {
                    // Calculate path length
                    double pathLength = calculatePathLength(nodes[i], nodes[j]);
                    
                    // Add edge to graph
                    boost::add_edge(
                        boost::vertex(i, mazeGraph), 
                        boost::vertex(j, mazeGraph), 
                        pathLength, 
                        mazeGraph
                    );
                }
            }
        }
    }

    // Trace path between two points
    bool tracePath(const cv::Point& start, const cv::Point& end) {
        if (start == end) return false;

        const int MAX_STEPS = 500;
        std::vector<std::vector<bool>> visited(
            skeleton.rows, 
            std::vector<bool>(skeleton.cols, false)
        );
        
        std::queue<cv::Point> queue;
        queue.push(start);
        visited[start.y][start.x] = true;

        const int dx[] = {0, 0, 1, -1, 1, 1, -1, -1};
        const int dy[] = {1, -1, 0, 0, 1, -1, 1, -1};

        int steps = 0;
        while (!queue.empty() && steps < MAX_STEPS) {
            cv::Point current = queue.front();
            queue.pop();
            steps++;

            if (current == end) return true;

            for (int i = 0; i < 8; ++i) {
                int nx = current.x + dx[i];
                int ny = current.y + dy[i];

                // Check bounds and unvisited skeleton pixels
                if (nx >= 0 && nx < skeleton.cols && 
                    ny >= 0 && ny < skeleton.rows && 
                    skeleton.at<uchar>(ny, nx) == 255 && 
                    !visited[ny][nx]) {
                    queue.emplace(nx, ny);
                    visited[ny][nx] = true;
                }
            }
        }

        return false;
    }

    // Calculate Manhattan distance
    double calculatePathLength(const cv::Point& start, const cv::Point& end) {
        return std::abs(start.x - end.x) + std::abs(start.y - end.y);
    }

    // Member variables
    cv::Mat image;           // Original image
    cv::Mat binaryImage;     // Binary thresholded image
    cv::Mat skeleton;        // Skeletonized image
    Graph mazeGraph;         // Boost graph representation
};

int main() {
    try {
        // Create MazeSolver instance
        MazeSolver solver("maze_image.png");

        // Preprocess the image
        solver.preprocessImage();

        // Create maze graph
        solver.createMazeGraph();

        // Visualize graph info
        solver.visualizeGraph();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}