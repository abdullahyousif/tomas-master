import os
import sys
import time
import argparse
import threading
import logging
import signal
import traceback
from typing import Dict, Any, Optional
import tkinter as tk

from adb_controller import ADBController
from detector import YOLODetector
from game_logic import GameController, GameState
from ui_handler import CoinMasterBotGUI
from config_handler import ConfigHandler
from game_logic_integrator import GameLogicIntegrator
from logger import setup_logger, add_ui_handler

# Global variables
running = True
game_controller = None
ui_handler = None


def signal_handler(sig, frame):
    """
    Handle interrupt signals (Ctrl+C).
    """
    global running, game_controller
    logging.info("Interrupt received, shutting down...")
    running = False
    if game_controller:
        game_controller.stop()


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Coin Master Automation Bot")
    
    parser.add_argument(
        "--config", 
        type=str,
        default="config/settings.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        help="ADB device ID to use"
    )
    
    parser.add_argument(
        "--console",
        action="store_true",
        help="Use console UI instead of graphical UI"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with visualization"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLOv11 model file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Logging level"
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point for the Coin Master Bot.
    """
    global running, game_controller, ui_handler
    
    # Register signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        print("Starting Coin Master Bot...")
        
        # Load configuration
        print("Loading configuration...")
        config_handler = None
        try:
            config_handler = ConfigHandler(args.config)
            config = config_handler.config
            print(f"Configuration loaded from {args.config}")
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            print("Using default configuration")
            config = {}
        
        # Override config with command line arguments
        if args.device:
            config["device_id"] = args.device
        
        if args.debug:
            config["debug_mode"] = True
        
        if args.model:
            config["model_path"] = args.model
        
        if args.log_level:
            config["log_level"] = args.log_level
        
        # Setup logging
        print("Setting up logging...")
        logger = setup_logger(config.get("log_level", "INFO"))
        
        # Create required directories if they don't exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("config", exist_ok=True)
        
        # Log startup info
        logger.info("Coin Master Bot starting up...")
        logger.info(f"Using configuration from {args.config}")
        
        # Initialize ADB controller
        logger.info("Initializing ADB controller...")
        print("Checking ADB connection...")
        adb = ADBController(config.get("device_id"))
        
        if not adb.connected:
            logger.error("Failed to connect to ADB device. Make sure a device is connected and ADB is installed.")
            print("ERROR: Failed to connect to ADB device. Make sure a device is connected and ADB is installed.")
            print("Run 'adb devices' in a terminal to verify the connection.")
            return 1
        
        # Initialize YOLOv11 detector
        logger.info("Initializing YOLOv11 detector...")
        model_path = config.get("model_path", "models/my_model.pt")
        print(f"Loading model from: {model_path}")
        
        # Ensure model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            print(f"ERROR: Model file not found at {model_path}")
            print(f"Make sure to place your YOLOv11/YOLOv5 model at {model_path}")
            print("You can run setup.py to create the necessary directory structure.")
            return 1
        
        try:
            detector = YOLODetector(
                model_path=model_path,
                conf_threshold=config.get("detection_confidence", 0.5)
            )
            print("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load the detection model: {str(e)}")
            print(f"ERROR: Failed to load the detection model: {str(e)}")
            print("Make sure your model is in the correct format and all dependencies are installed.")
            print("Check the logs for more details.")
            return 1
        
        # Initialize game controller
        logger.info("Initializing game controller...")
        print("Initializing game controller...")
        game_controller = GameController(adb, detector, config)
        
        # Initialize and start GUI
        logger.info("Initializing GUI...")
        print("Initializing GUI...")
        root = tk.Tk()
        gui = CoinMasterBotGUI(root)
        
        # Inject dependencies into GUI
        gui.adb = adb
        gui.detector = detector
        gui.game_controller = game_controller
        gui.config = config
        
        # Configure initial GUI state
        if args.device:
            gui.device_id.set(args.device)
        if args.model:
            gui.model_path.set(args.model)
            
        # Start GUI main loop
        root.mainloop()
        
        # Add UI handler to logger
        add_ui_handler(logger, ui_handler)
        
        # Start UI handler
        logger.info("Starting UI handler...")
        print("Starting UI handler...")
        ui_handler.start()
        
        # Wait for user to start the bot
        logger.info("Ready! Press 's' to start the bot, 'q' to quit.")
        print("\nReady! Press 's' to start the bot, 'q' to quit.\n")
        
        # Main loop
        while running:
            time.sleep(0.1)
        
        # Clean shutdown
        logger.info("Shutting down...")
        print("Shutting down...")
        if game_controller:
            game_controller.stop()
        
        if ui_handler:
            ui_handler.stop()
        
        logger.info("Coin Master Bot stopped.")
        print("Coin Master Bot stopped.")
        return 0
        
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"CRITICAL ERROR: {str(e)}")
        print("Stack trace:")
        print(traceback_str)
        if 'logger' in locals():
            logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        else:
            print(f"Error before logger initialization: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    print(f"Exiting with code: {exit_code}")
    sys.exit(exit_code)