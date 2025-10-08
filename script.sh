#!/bin/bash

# Chatbot Setup Script for Debian and Arch Linux
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_NAME="chatbot_venv"
VENV_PATH="$SCRIPT_DIR/$VENV_NAME"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

# Default distro
DISTRO="debian"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    echo "Usage: $0 [--distro debian|arch] [-d debian|arch]"
    echo "  Default distro: debian"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --distro|-d)
            if [[ "$2" == "debian" || "$2" == "arch" ]]; then
                DISTRO="$2"
                shift 2
            else
                print_error "Invalid distro: $2 (must be 'debian' or 'arch')"
                usage
            fi
            ;;
        --help|-h)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

print_status "Selected distribution: $DISTRO"

# Check if system matches selected distro (optional but helpful)
check_distro_match() {
    case "$DISTRO" in
        debian)
            if [ ! -f /etc/debian_version ]; then
                print_warning "Selected 'debian', but system doesn't appear to be Debian-based."
                # Still proceed — user might know what they're doing
            fi
            ;;
        arch)
            if [ ! -f /etc/arch-release ]; then
                print_warning "Selected 'arch', but system doesn't appear to be Arch Linux."
            fi
            ;;
    esac
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed."
        case "$DISTRO" in
            debian)
                sudo apt-get update
                sudo apt-get install -y python3 python3-pip python3-venv
                ;;
            arch)
                sudo pacman -Sy --noconfirm python python-pip
                ;;
        esac
    fi
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_status "Found Python $PYTHON_VERSION"
}

install_system_deps() {
    print_status "Installing system dependencies for $DISTRO..."

    case "$DISTRO" in
        debian)
            sudo apt-get update
            sudo apt-get install -y \
                portaudio19-dev \
                libasound2-dev \
                python3-dev \
                build-essential \
                git \
                wget
            ;;
        arch)
            # Arch uses different package names
            sudo pacman -Sy --noconfirm \
                portaudio \
                alsa-lib \
                base-devel \
                git \
                wget
            # Note: python is assumed installed; python-pip handled above
            ;;
        *)
            print_error "Unsupported distro: $DISTRO"
            exit 1
            ;;
    esac

    print_status "System dependencies installed."
}

setup_venv() {
    if [ ! -d "$VENV_PATH" ]; then
        print_status "Creating virtual environment '$VENV_NAME'..."
        python3 -m venv "$VENV_PATH"
    else
        print_status "Virtual environment already exists."
    fi
}

check_requirements() {
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        print_status "Creating requirements.txt..."
        cat > "$REQUIREMENTS_FILE" << 'EOF'
torch
transformers
pyaudio
numpy
scipy
llama-cpp-python
sentencepiece
protobuf
soundfile
gradio
EOF
    fi
}

install_python_deps() {
    print_status "Installing Python dependencies..."
    "$VENV_PATH/bin/pip" install --upgrade pip
    "$VENV_PATH/bin/pip" install -r "$REQUIREMENTS_FILE"
    print_status "Python dependencies installed."
}

test_installation() {
    print_status "Testing installation..."

    if "$VENV_PATH/bin/python" -c "import pyaudio; print('PyAudio: OK')" 2>/dev/null; then
        print_status "PyAudio: OK"
    else
        print_error "PyAudio test failed"
        exit 1
    fi

    if "$VENV_PATH/bin/python" -c "import torch; print('PyTorch: OK')" 2>/dev/null; then
        print_status "PyTorch: OK"
    else
        print_error "PyTorch test failed"
        exit 1
    fi

    print_status "All tests passed!"
}

main() {
    print_status "Starting chatbot setup for $DISTRO..."

    check_distro_match
    check_python
    install_system_deps
    setup_venv
    check_requirements
    install_python_deps
    test_installation

    print_status "✅ Setup completed successfully!"
    print_status "Virtual environment: $VENV_PATH"
    print_status ""
    print_status "To use it, run:"
    print_status "  source $VENV_PATH/bin/activate"
    print_status "  python script.py"
    print_status ""
    print_status "Or run directly:"
    print_status "  $VENV_PATH/bin/python script.py"
}

main "$@"
