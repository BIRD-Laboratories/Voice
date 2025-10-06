#!/bin/bash

# Chatbot Setup Script for Debian
# This script sets up a Python virtual environment and installs all dependencies

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_NAME="chatbot_venv"
VENV_PATH="$SCRIPT_DIR/$VENV_NAME"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on Debian
check_debian() {
    if [ ! -f /etc/debian_version ]; then
        print_error "This script is designed for Debian-based systems only."
        exit 1
    fi
}

# Check if Python 3 is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Installing..."
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-venv
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_status "Found Python $PYTHON_VERSION"
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    sudo apt-get update
    sudo apt-get install -y \
        portaudio19-dev \
        python3-dev \
        libasound2-dev \
        libasound2 \
        alsa-utils \
        alsa-base \
        build-essential \
        git \
        wget
    
    print_status "System dependencies installed successfully."
}

# Create or activate virtual environment
setup_venv() {
    if [ -d "$VENV_PATH" ]; then
        print_status "Virtual environment '$VENV_NAME' already exists."
        print_status "Activating existing virtual environment..."
        source "$VENV_PATH/bin/activate"
    else
        print_status "Creating new virtual environment '$VENV_NAME'..."
        python3 -m venv "$VENV_PATH"
        source "$VENV_PATH/bin/activate"
        print_status "Virtual environment created and activated."
    fi
}

# Check for requirements.txt
check_requirements() {
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        print_error "requirements.txt not found in $SCRIPT_DIR"
        print_status "Creating requirements.txt..."
        cat > "$REQUIREMENTS_FILE" << 'EOF'
torch>=2.0.0
transformers>=4.35.0
pyaudio>=0.2.11
numpy>=1.21.0
scipy>=1.7.0
llama-cpp-python>=0.2.0
sentencepiece>=0.1.99
protobuf>=3.20.0
soundfile>=0.12.0
EOF
        print_status "Created requirements.txt"
    fi
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip first
    pip install --upgrade pip
    
    # Install PyAudio with specific flags for Debian
    print_status "Installing PyAudio..."
    pip install pyaudio
    
    # Install other dependencies
    print_status "Installing packages from requirements.txt..."
    pip install -r "$REQUIREMENTS_FILE"
    
    print_status "All Python dependencies installed successfully."
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    # Test PyAudio
    if python3 -c "import pyaudio; print('PyAudio: OK')" 2>/dev/null; then
        print_status "PyAudio installation verified"
    else
        print_error "PyAudio installation failed"
        exit 1
    fi
    
    # Test torch
    if python3 -c "import torch; print('PyTorch: OK')" 2>/dev/null; then
        print_status "PyTorch installation verified"
    else
        print_error "PyTorch installation failed"
        exit 1
    fi
    
    print_status "All tests passed!"
}

# Main execution
main() {
    print_status "Starting chatbot setup for Debian..."
    
    check_debian
    check_python
    install_system_deps
    setup_venv
    check_requirements
    install_python_deps
    test_installation
    
    print_status "Setup completed successfully!"
    print_status "Virtual environment is active at: $VENV_PATH"
    print_status "You can now run your chatbot script."
    print_status ""
    print_status "To run your chatbot in the future:"
    print_status "  source $VENV_PATH/bin/activate"
    print_status "  python3 script.py"
    print_status ""
    print_status "Or run directly:"
    print_status "  $VENV_PATH/bin/python3 script.py"
}

# Run main function
main "$@"
