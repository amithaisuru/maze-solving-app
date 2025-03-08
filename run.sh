#!/bin/bash

# Script to install dependencies for the Python maze project

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Check if Python 3 is installed
echo "Checking for Python 3..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed!${NC}"
    echo "Please install Python 3 first. On Debian/Ubuntu: 'sudo apt-get install python3'"
    echo "On Fedora: 'sudo dnf install python3'"
    exit 1
else
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}Found $PYTHON_VERSION${NC}"
fi

# Check if pip is installed
echo "Checking for pip..."
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}pip3 is not installed! Installing now...${NC}"
    # Install pip
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
    rm get-pip.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install pip!${NC}"
        exit 1
    fi
    echo -e "${GREEN}pip installed successfully${NC}"
else
    PIP_VERSION=$(pip3 --version)
    echo -e "${GREEN}Found $PIP_VERSION${NC}"
fi

# Check for tkinter (optional, as it might not be bundled)
echo "Checking for tkinter..."
if ! python3 -c "import tkinter" 2>/dev/null; then
    echo -e "${RED}tkinter is not installed! Attempting to install...${NC}"
    # Install tkinter based on the system
    if [ -f /etc/debian_version ]; then
        sudo apt-get update
        sudo apt-get install -y python3-tk
    elif [ -f /etc/redhat-release ]; then
        sudo dnf install -y python3-tkinter
    else
        echo -e "${RED}Unknown system, please install tkinter manually${NC}"
        exit 1
    fi
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install tkinter!${NC}"
        exit 1
    fi
    echo -e "${GREEN}tkinter installed successfully${NC}"
else
    echo -e "${GREEN}tkinter is already available${NC}"
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}requirements.txt not found in the current directory!${NC}"
    exit 1
fi

# Install dependencies from requirements.txt
echo "Installing project dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install dependencies!${NC}"
    exit 1
fi

echo -e "${GREEN}All dependencies installed successfully!${NC}"
echo "You can now run the project with: python3 main.py"