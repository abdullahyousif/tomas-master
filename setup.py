#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"    {text}")
    print("="*60)

def print_step(step, text):
    """Print a step with its number"""
    print(f"\n[{step}] {text}")

def run_command(command, error_message="Command failed"):
    """Run a shell command and handle errors"""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding='utf-8')
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {error_message}")
        print(f"Command: {command}")
        print(f"Error output: {e.stderr}")
        return None

def check_requirements():
    """Check if all required tools are installed"""
    print_step(1, "Checking requirements")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("ERROR: Python 3.8 or higher is required")
        return False
    
    # Check ADB
    print("Checking ADB...")
    adb_output = run_command("adb --version", "ADB not found. Please install Android Debug Bridge")
    if not adb_output:
        print("You need to install ADB (Android Debug Bridge)")
        print("Visit https://developer.android.com/tools/releases/platform-tools")
        return False
    
    print("ADB found: " + adb_output.split("\n")[0])
    
    # Check connected devices
    print("Checking connected devices...")
    devices_output = run_command("adb devices", "Failed to check connected devices")
    if not devices_output or "List of devices attached" not in devices_output:
        print("Failed to get device list")
        return False
    
    lines = devices_output.split("\n")
    devices = [line.split("\t")[0] for line in lines[1:] if line.strip() and "\t" in line]
    
    if not devices:
        print("WARNING: No devices connected. Connect a device before running the bot.")
    else:
        print(f"Found {len(devices)} connected device(s):")
        for i, device in enumerate(devices):
            print(f"  {i+1}. {device}")
    
    return True

def setup_directories():
    """Create the necessary directory structure"""
    print_step(2, "Setting up directories")
    
    directories = ["models", "logs", "config", "data", "elements"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")
    
    # Create placeholder for model file
    model_readme_path = os.path.join("models", "README.md")
    if not os.path.exists(model_readme_path):
        with open(model_readme_path, 'w') as f:
            f.write("# YOLOv11/YOLOv5 Model Directory\n\n")
            f.write("Place your model file (`my_model.pt`) in this directory.\n\n")
            f.write("Make sure the model is trained to detect the following game elements:\n\n")
            f.write("- attack_aiming_icon\n")
            f.write("- attack_symbol\n")
            f.write("- close_pannels\n")
            f.write("- autospin_button\n")
            f.write("- let_me_rest\n")
            f.write("- ok_button\n")
            f.write("- power_boost_x1\n")
            f.write("- power_boost_x15\n")
            f.write("- power_boost_x1500\n")
            f.write("- power_boost_x2\n")
            f.write("- power_boost_x20000\n")
            f.write("- power_boost_x3\n")
            f.write("- power_boost_x400\n")
            f.write("- power_boost_x50\n")
            f.write("- power_boost_x6000\n")
            f.write("- raid_hole_icon\n")
            f.write("- raid_symbol\n")
            f.write("- raid_x_icon\n")
            f.write("- spin_button\n")
        print(f"Created file: {model_readme_path}")
    
    # Create default config file
    config_path = os.path.join("config", "settings.json")
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            f.write('''{
  "device_id": null,
  "model_path": "models/my_model.pt",
  "detection_confidence": 0.5,
  "action_delay": 0.5,
  "power_boost_sequence": [
    {"level": "X1", "attacks": 8},
    {"level": "X15", "attacks": 3},
    {"level": "X50", "attacks": 4},
    {"level": "X400", "attacks": 3},
    {"level": "X1500", "attacks": 1},
    {"level": "X6000", "attacks": 1},
    {"level": "X20000", "attacks": 1}
  ],
  "log_level": "INFO",
  "debug_mode": false
}''')
        print(f"Created default config file: {config_path}")
    else:
        print(f"Config file already exists: {config_path}")
    
    return True

def install_dependencies():
    """Install required Python packages"""
    print_step(3, "Installing dependencies")
    
    # Write consolidated requirements to file
    requirements = [
        "numpy>=1.20.0",
        "opencv-python>=4.5.5",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "PyYAML>=6.0",
        "tqdm>=4.64.0",
        "matplotlib>=3.5.0",
        "psutil>=5.9.0",
        "ultralytics>=8.0.0",
        "adbutils>=0.7.2",
        "pillow>=9.0.0"
    ]
    
    with open("requirements.txt", 'w') as f:
        f.write("\n".join(requirements))
    
    print("Installing required packages...")
    result = run_command("pip install -r requirements.txt", "Failed to install dependencies")
    
    if result is not None:
        print("Dependencies installed successfully")
        return True
    return False

def main():
    """Main setup function"""
    print_header("COIN MASTER BOT SETUP")
    
    # Check requirements
    if not check_requirements():
        print("\nSetup failed: missing requirements")
        return 1
    
    # Setup directories
    if not setup_directories():
        print("\nSetup failed: couldn't create directories")
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("\nSetup failed: couldn't install dependencies")
        return 1
    
    # Setup complete
    print_header("SETUP COMPLETE")
    print("\nYou're all set to use the Coin Master Bot!")
    print("\nImportant:")
    print("1. Make sure to place your trained YOLOv11/YOLOv5 model file at 'models/my_model.pt'")
    print("2. Connect your Android device via USB and enable USB debugging")
    print("3. Run the bot with: python simple_controller.py")
    print("\nHappy automating!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())