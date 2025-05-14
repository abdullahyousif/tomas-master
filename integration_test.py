#!/usr/bin/env python3

import os
import sys
import time
import logging
import argparse

# Import required modules
from adb_controller import ADBController
from detector import YOLODetector
from fixed_ui_detector import FixedUIDetector
from integration_example import HybridDetector
from game_logic_integrator import GameLogicIntegrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("integration_test.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Coin Master Bot Integration Test")
    
    parser.add_argument(
        "--device",
        type=str,
        help="ADB device ID to use (optional, uses first device if not specified)"
    )
    
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="models/my_model.pt",
        help="Path to YOLOv11 model file"
    )
    
    parser.add_argument(
        "--fixed-ui-config",
        type=str,
        default="fixed_ui_elements.json",
        help="Path to fixed UI configuration file"
    )
    
    parser.add_argument(
        "--no-fixed-ui",
        action="store_true",
        help="Disable fixed UI detection"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (saves detection results)"
    )
    
    parser.add_argument(
        "--test-screenshot",
        type=str,
        help="Path to test screenshot (if specified, only runs detection without starting the bot)"
    )
    
    return parser.parse_args()

def test_detection_only(args):
    """Test detection on a static screenshot."""
    import cv2
    import numpy as np
    
    logger.info("Testing detection only mode")
    
    # Check if test screenshot exists
    if not os.path.exists(args.test_screenshot):
        logger.error(f"Test screenshot not found at: {args.test_screenshot}")
        return False
    
    # Load screenshot
    image = cv2.imread(args.test_screenshot)
    if image is None:
        logger.error(f"Failed to load image: {args.test_screenshot}")
        return False
    
    logger.info(f"Successfully loaded test image: {args.test_screenshot}")
    
    # Initialize detectors
    try:
        # Initialize YOLO detector
        logger.info(f"Initializing YOLO detector with model: {args.yolo_model}")
        yolo_detector = YOLODetector(
            model_path=args.yolo_model,
            conf_threshold=0.5
        )
        
        # Initialize fixed UI detector if enabled
        fixed_ui_detector = None
        if not args.no_fixed_ui:
            logger.info(f"Initializing fixed UI detector with config: {args.fixed_ui_config}")
            fixed_ui_detector = FixedUIDetector(config_file=args.fixed_ui_config)
            
            # Initialize hybrid detector
            logger.info("Initializing hybrid detector")
            hybrid_detector = HybridDetector(
                yolo_model_path=args.yolo_model,
                fixed_ui_config=args.fixed_ui_config,
                yolo_conf_threshold=0.5,
                template_threshold=0.8
            )
        
        # Run detection
        logger.info("Running detection on test image")
        
        # YOLO detection
        yolo_results = yolo_detector.detect(image)
        yolo_tap_targets = yolo_detector.get_tap_targets(yolo_results)
        
        logger.info(f"YOLO detection: found {len(yolo_results)} objects")
        logger.info(f"YOLO tap targets: {len(yolo_tap_targets)}")
        
        # Draw YOLO results
        yolo_result_image = yolo_detector.draw_detections(image, yolo_results)
        
        # Fixed UI detection (if enabled)
        if fixed_ui_detector:
            fixed_ui_results = fixed_ui_detector.detect(image)
            fixed_ui_tap_targets = fixed_ui_detector.get_tap_targets(fixed_ui_results)
            
            logger.info(f"Fixed UI detection: found {sum(1 for r in fixed_ui_results.values() if r['detected'])} elements")
            logger.info(f"Fixed UI tap targets: {len(fixed_ui_tap_targets)}")
            
            # Draw fixed UI results
            fixed_ui_result_image = fixed_ui_detector.draw_detections(image, fixed_ui_results)
            
            # Hybrid detection
            hybrid_results = hybrid_detector.detect(image)
            hybrid_tap_targets = hybrid_detector.get_all_tap_targets(hybrid_results)
            
            logger.info(f"Hybrid detection: found {len(hybrid_results['dynamic_objects'])} dynamic objects and " +
                        f"{sum(1 for r in hybrid_results['fixed_ui'].values() if r['detected'])} fixed UI elements")
            logger.info(f"Hybrid tap targets: {len(hybrid_tap_targets)}")
            
            # Draw hybrid results
            hybrid_result_image = hybrid_detector.draw_results(image, hybrid_results)
        
        # Save detection results if debug mode enabled
        if args.debug:
            output_dir = "test_results"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save YOLO result
            cv2.imwrite(f"{output_dir}/yolo_detection.png", yolo_result_image)
            logger.info(f"Saved YOLO detection result to: {output_dir}/yolo_detection.png")
            
            # Save fixed UI result if enabled
            if fixed_ui_detector:
                cv2.imwrite(f"{output_dir}/fixed_ui_detection.png", fixed_ui_result_image)
                logger.info(f"Saved fixed UI detection result to: {output_dir}/fixed_ui_detection.png")
                
                # Save hybrid result
                cv2.imwrite(f"{output_dir}/hybrid_detection.png", hybrid_result_image)
                logger.info(f"Saved hybrid detection result to: {output_dir}/hybrid_detection.png")
        
        logger.info("Detection test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in detection test: {str(e)}", exc_info=True)
        return False

def main():
    """Main function."""
    args = parse_args()
    
    # If test screenshot is provided, only run detection test
    if args.test_screenshot:
        success = test_detection_only(args)
        return 0 if success else 1
    
    try:
        # Initialize ADB controller
        logger.info("Initializing ADB controller")
        adb = ADBController(device_id=args.device)
        
        if not adb.connected:
            logger.error("Failed to connect to device. Make sure a device is connected and ADB is installed.")
            return 1
        
        logger.info(f"Connected to device: {adb.device_id}")
        logger.info(f"Screen resolution: {adb.screen_resolution}")
        
        # Initialize YOLO detector
        logger.info(f"Initializing YOLO detector with model: {args.yolo_model}")
        yolo_detector = YOLODetector(
            model_path=args.yolo_model,
            conf_threshold=0.5
        )
        
        # Initialize fixed UI detector if enabled
        fixed_ui_detector = None
        if not args.no_fixed_ui:
            logger.info(f"Initializing fixed UI detector with config: {args.fixed_ui_config}")
            fixed_ui_detector = FixedUIDetector(config_file=args.fixed_ui_config)
        
        # Create basic configuration
        config = {
            "device_id": adb.device_id,
            "model_path": args.yolo_model,
            "fixed_ui_config": args.fixed_ui_config if not args.no_fixed_ui else None,
            "detection_confidence": 0.5,
            "action_delay": 0.5,
            "debug_mode": args.debug
        }
        
        # Initialize game controller
        logger.info("Initializing game controller")
        game_controller = GameLogicIntegrator(
            adb=adb,
            detector=yolo_detector,
            fixed_ui_detector=fixed_ui_detector,
            config=config
        )
        
        # Test screenshot capture
        logger.info("Taking test screenshot")
        screenshot = adb.capture_screen()
        
        if screenshot is None:
            logger.error("Failed to capture screen")
            return 1
        
        # Save test screenshot if debug mode enabled
        if args.debug:
            os.makedirs("test_results", exist_ok=True)
            import cv2
            cv2.imwrite("test_results/test_screenshot.png", screenshot)
            logger.info("Saved test screenshot to: test_results/test_screenshot.png")
        
        # Test detection
        logger.info("Running detection test on screenshot")
        
        if game_controller.using_hybrid_detection:
            # Run hybrid detection
            results = game_controller.hybrid_detector.detect(screenshot)
            tap_targets = game_controller.hybrid_detector.get_all_tap_targets(results)
            
            logger.info(f"Hybrid detection: found {len(results['dynamic_objects'])} dynamic objects and " +
                        f"{sum(1 for r in results['fixed_ui'].values() if r['detected'])} fixed UI elements")
            logger.info(f"Tap targets: {len(tap_targets)}")
            
            # Save detection result if debug mode enabled
            if args.debug:
                result_image = game_controller.hybrid_detector.draw_results(screenshot, results)
                cv2.imwrite("test_results/hybrid_detection.png", result_image)
                logger.info("Saved hybrid detection result to: test_results/hybrid_detection.png")
        else:
            # Run YOLO detection only
            detections = game_controller.yolo_detector.detect(screenshot)
            tap_targets = game_controller.yolo_detector.get_tap_targets(detections)
            
            logger.info(f"YOLO detection: found {len(detections)} objects")
            logger.info(f"Tap targets: {len(tap_targets)}")
            
            # Save detection result if debug mode enabled
            if args.debug:
                result_image = game_controller.yolo_detector.draw_detections(screenshot, detections)
                cv2.imwrite("test_results/yolo_detection.png", result_image)
                logger.info("Saved YOLO detection result to: test_results/yolo_detection.png")
        
        # Start game controller
        logger.info("Starting game controller")
        game_controller.start()
        
        # Run for a while
        logger.info("Running for 30 seconds...")
        
        for i in range(30):
            # Print progress
            sys.stdout.write(f"\rRunning... {i+1}/30s")
            sys.stdout.flush()
            
            # Print stats every 5 seconds
            if (i + 1) % 5 == 0:
                stats = game_controller.get_stats()
                logger.info(f"Current state: {stats['state']}")
                logger.info(f"Power boost: {stats['current_power_boost']}")
                logger.info(f"Attacks: {stats['attacks_in_current_level']}/{stats['target_attacks_for_level']} (Total: {stats['attacks_completed']})")
                logger.info(f"Raids completed: {stats['raids_completed']}")
            
            time.sleep(1)
        
        # Stop game controller
        logger.info("\nStopping game controller")
        game_controller.stop()
        
        # Print final stats
        stats = game_controller.get_stats()
        logger.info(f"Final state: {stats['state']}")
        logger.info(f"Power boost: {stats['current_power_boost']}")
        logger.info(f"Attacks: {stats['attacks_in_current_level']}/{stats['target_attacks_for_level']} (Total: {stats['attacks_completed']})")
        logger.info(f"Raids completed: {stats['raids_completed']}")
        
        logger.info("Integration test completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in integration test: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
