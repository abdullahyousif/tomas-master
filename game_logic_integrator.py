import json
import os
import time
import logging
import threading
from enum import Enum
from tkinter import messagebox
from typing import List, Dict, Tuple, Optional, Any

import cv2

# Import required modules
from adb_controller import ADBController
from detector import YOLODetector, Detection
from fixed_ui_detector import FixedUIDetector
from integration_example import HybridDetector
from config_handler import ConfigHandler
from power_boost_manager import PowerBoostManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GameState(Enum):
    """Game state enumeration"""
    INITIALIZING = "Initializing"
    WAITING = "Waiting for game screen"
    SPINNING = "Spinning"
    ATTACKING = "Attacking"
    RAIDING = "Raiding"
    IDLE = "Idle"
    ERROR = "Error"
    STOPPED = "Stopped"
    TAPPING_FIXED_UI = "Tapping Fixed UI"

class PowerBoostLevel(Enum):
    """Power boost level enumeration"""
    X1 = "X1" #default power boost level
    X2 = "X2" #one tap from default power boost level
    X3 = "X3" #two taps from default power boost level
    X15 = "X15" #three taps from default power boost level
    X50 = "X50" #four taps from default power boost level
    X400 = "X400" #five taps from default power boost level
    X1500 = "X1500" #six taps from default power boost level
    X6000 = "X6000" #seven taps from default power boost level
    X20000 = "X20000" #eight taps from default power boost level
    
    @staticmethod
    def from_string(value: str) -> 'PowerBoostLevel':
        """Convert string to enum value"""
        try:
            return PowerBoostLevel[value]
        except KeyError:
            # Try with X prefix if not found
            if not value.startswith('X'):
                return PowerBoostLevel.from_string('X' + value)
            raise ValueError(f"Invalid power boost level: {value}")

class GameLogicIntegrator:
    """
    Enhanced Game Controller that integrates both YOLO detection and Fixed UI detection
    for more robust game automation
    """
    
    def __init__(self, 
                adb: ADBController, 
                detector: YOLODetector,
                fixed_ui_detector: Optional[FixedUIDetector] = None,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the game controller.
        
        Args:
            adb: ADB controller instance
            detector: YOLO detector instance
            fixed_ui_detector: Fixed UI detector instance (optional)
            config: Configuration dictionary (optional)
        """
        self.adb = adb
        
        # Create hybrid detector if fixed_ui_detector is provided, otherwise use plain YOLO detector
        if fixed_ui_detector:
            logger.info("Initializing hybrid detector with both YOLO and fixed UI detection")
            
            # Store references to the individual detectors
            self.yolo_detector = detector
            self.fixed_ui_detector = fixed_ui_detector
            self.using_hybrid_detection = True
            
            # Create hybrid detector from the individual detectors if needed, 
            # or just use the separate detectors to avoid redundancy
            yolo_model_path = detector.model_path if hasattr(detector, 'model_path') else "models/my_model.pt"
            fixed_ui_config = fixed_ui_detector.config_file if hasattr(fixed_ui_detector, 'config_file') else "fixed_ui_elements.json"
            yolo_conf_threshold = detector.conf_threshold if hasattr(detector, 'conf_threshold') else 0.5
            template_threshold = fixed_ui_detector.detection_threshold if hasattr(fixed_ui_detector, 'detection_threshold') else 0.8
            
            # Create a hybrid detector for visualization purposes only
            self.hybrid_detector = HybridDetector(
                yolo_model_path=yolo_model_path,
                fixed_ui_config=fixed_ui_config,
                yolo_conf_threshold=yolo_conf_threshold,
                template_threshold=template_threshold
            )
        else:
            logger.info("Using only YOLO detector (no fixed UI detection)")
            self.yolo_detector = detector
            self.fixed_ui_detector = None
            self.hybrid_detector = None
            self.using_hybrid_detection = False
        
        # Initialize the PowerBoostManager with the correct templates path
        # Get the power boost templates path from config or use default
        power_boost_templates_path = config.get('power_boost_templates_path', "power_boost_templates.json")
        
        # Log the path being used for transparency
        logger.info(f"Initializing PowerBoostManager with templates from: {power_boost_templates_path}")
        
        self.power_boost_manager = PowerBoostManager(
            config_file=power_boost_templates_path,
            adb_controller=adb
        )

        # Current power boost level (will be detected)
        self.current_power_boost_level = "X1"

        # Last power boost detection time
        self.last_power_boost_detection_time = 0

        # Power boost detection interval (in seconds)
        self.power_boost_detection_interval = 5

        # Load configuration
        if config is None:
            config_handler = ConfigHandler()
            self.config = config_handler.config
        else:
            self.config = config
        
        # Game state
        self.state = GameState.INITIALIZING
        self.running = False
        
        # Add this line to fix the error
        self.action_thread = None
        
        # Sequence configuration
        self.power_boost_sequence = self.config.get('power_boost_sequence', [
            {"level": "X1", "attacks": 8},
            {"level": "X15", "attacks": 3},
            {"level": "X50", "attacks": 4},
            {"level": "X400", "attacks": 3},
            {"level": "X1500", "attacks": 1},
            {"level": "X6000", "attacks": 1},
            {"level": "X20000", "attacks": 1}
        ])
        
        # Current sequence position
        self.current_sequence_index = 0
        self.current_power_boost = PowerBoostLevel.X1
        self.target_attacks_for_level = self.power_boost_sequence[0]["attacks"]
        self.attacks_in_current_level = 0
        
        # Stats
        self.attacks_completed = 0
        self.raids_completed = 0
        self.fixed_ui_taps_completed = 0
        
        # Timing configuration
        self.action_delay = self.config.get('action_delay', 0.5)
        self.detection_interval = self.config.get('detection_interval', 1.0)
        
        # Debug mode
        self.debug_mode = self.config.get('debug_mode', False)
        
        # Last detection results
        self.last_dynamic_detections = []
        self.last_fixed_ui_results = {}
        
        logger.info("Game controller initialized")
    
    def start(self):
        """Start the game controller."""
        if self.running:
            logger.warning("Game controller is already running")
            return
        
        logger.info("Starting game controller")
        self.running = True
        self.state = GameState.INITIALIZING
        
        # Start action thread
        self.action_thread = threading.Thread(target=self._game_loop, daemon=True)
        self.action_thread.start()
    
    def stop(self):
        """Stop the game controller."""
        logger.info("Stopping game controller")
        self.running = False
        self.state = GameState.STOPPED
        
        # Wait for action thread to stop
        if self.action_thread and self.action_thread.is_alive():
            self.action_thread.join(timeout=2.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current game stats.
        
        Returns:
            Dictionary with stats
        """
        return {
            "state": self.state.value,
            "current_power_boost": self.current_power_boost.value,
            "current_sequence_position": self.current_sequence_index + 1,
            "sequence_length": len(self.power_boost_sequence),
            "target_attacks_for_level": self.target_attacks_for_level,
            "attacks_in_current_level": self.attacks_in_current_level,
            "attacks_completed": self.attacks_completed,
            "raids_completed": self.raids_completed,
            "fixed_ui_taps_completed": self.fixed_ui_taps_completed
        }
    
    def _game_loop(self):
        """Main game loop."""
        try:
            # Initialize counter for tracking consecutive no-detection cycles
            self.no_detection_count = 0
            
            # Track the last time we had a successful detection
            self.last_successful_detection_time = time.time()
            
            while self.running:
                # IMPORTANT: Capture a fresh screen each iteration
                screen = self.adb.capture_screen()

                # Periodically check and update power boost level
                current_time = time.time()
                if current_time - self.last_power_boost_detection_time > 5:  # Check every 5 seconds
                    self.last_power_boost_detection_time = current_time
                    
                    # Detect current power boost level
                    level, confidence = self.power_boost_manager.detect_power_boost_level(screen)
                    
                    if level and confidence >= 0.7:  # Lower threshold for better detection
                        if level != self.current_power_boost_level:
                            logger.info(f"Power boost level changed from {self.current_power_boost_level} to {level}")
                            self.current_power_boost_level = level
                
                if screen is None:
                    logger.error("Failed to capture screen")
                    self.state = GameState.ERROR
                    time.sleep(2.0)  # Wait before retrying
                    continue
                
                # CRITICAL: Perform fresh detection EVERY time through the loop
                if self.using_hybrid_detection:
                    # Explicitly clear old detection results first
                    self.last_dynamic_detections = []
                    self.last_fixed_ui_results = {}
                    
                    # Check if screen capture is valid
                    if screen is None:
                        logger.error("Screen capture failed - attempting to re-capture")
                        screen = self.adb.capture_screen()
                        if screen is None:
                            logger.error("Second screen capture attempt failed")
                            self.state = GameState.ERROR
                            time.sleep(2.0)  # Wait before retrying
                            continue
                    
                    # Perform fresh detections with error handling
                    try:
                        self.last_dynamic_detections = self.yolo_detector.detect(screen)
                        logger.info(f"YOLO detection found {len(self.last_dynamic_detections)} objects")
                        
                        self.last_fixed_ui_results = self.fixed_ui_detector.detect(screen)
                        fixed_count = sum(1 for r in self.last_fixed_ui_results.values() if r["detected"])
                        logger.info(f"Fixed UI detection found {fixed_count} elements")
                    except Exception as e:
                        logger.error(f"Detection error: {str(e)}")
                        self.state = GameState.ERROR
                        time.sleep(2.0)
                        continue
                    
                    # Track which elements are handled in this iteration
                    handled_elements = set()
                else:
                    # Clear old results and perform fresh YOLO detection
                    self.last_dynamic_detections = []
                    
                    try:
                        self.last_dynamic_detections = self.yolo_detector.detect(screen)
                        logger.info(f"YOLO-only detection found {len(self.last_dynamic_detections)} objects")
                    except Exception as e:
                        logger.error(f"YOLO detection error: {str(e)}")
                        self.state = GameState.ERROR
                        time.sleep(2.0)
                        continue
                        
                    self.last_fixed_ui_results = {}
                    handled_elements = set()
                
                # Enhanced debug logging to track detection results
                if self.debug_mode:
                    dynamic_count = len(self.last_dynamic_detections) if self.last_dynamic_detections else 0
                    fixed_count = sum(1 for r in self.last_fixed_ui_results.values() if r["detected"]) if self.last_fixed_ui_results else 0
                    logger.debug(f"Detected: {dynamic_count} dynamic objects, {fixed_count} fixed UI elements")
                    
                    # Log specific detected objects for debugging
                    if dynamic_count > 0:
                        object_names = [det.class_name for det in self.last_dynamic_detections[:5]]  # List first 5 for brevity
                        logger.debug(f"YOLO detected objects: {', '.join(object_names)}{' ...' if dynamic_count > 5 else ''}")
                    
                    if fixed_count > 0:
                        ui_names = [name for name, res in self.last_fixed_ui_results.items() if res["detected"]]
                        logger.debug(f"Fixed UI detected elements: {', '.join(ui_names)}")
                
                # Check if any elements were detected in this cycle
                any_detections = (len(self.last_dynamic_detections) > 0 or 
                                sum(1 for r in self.last_fixed_ui_results.values() if r["detected"]) > 0)
                
                if any_detections:
                    # Update last successful detection time
                    self.last_successful_detection_time = time.time()
                    self.no_detection_count = 0
                else:
                    # Check how long since last successful detection
                    time_since_detection = time.time() - self.last_successful_detection_time
                    
                    # If it's been more than 10 seconds with no detections
                    if time_since_detection > 10:
                        logger.warning(f"No successful detections for {time_since_detection:.1f} seconds")
                        
                        # After 20 seconds, try complete system restart
                        if time_since_detection > 20:
                            logger.error("Detection appears to be completely stalled")
                            self.restart_detection_system()
                            self.last_successful_detection_time = time.time()  # Reset timer
                            
                        # If detection has completely stalled for 30+ seconds
                        if time_since_detection > 30:
                            logger.critical("Critical detection failure - attempting ADB reset")
                            try:
                                # Try to restart ADB connection
                                self.adb.initialize()
                                time.sleep(2.0)
                                
                                # Take a fresh screenshot to verify connectivity
                                test_screen = self.adb.capture_screen()
                                if test_screen is not None:
                                    logger.info("ADB connection successfully reset")
                                    # Restart detection system
                                    self.restart_detection_system()
                                else:
                                    logger.error("ADB reset failed - unable to capture screen")
                            except Exception as e:
                                logger.error(f"Error during ADB reset: {str(e)}")
                
                # STEP 1: Check for spin buttons that need long press in fixed UI
                spin_buttons_handled = False
                if self.using_hybrid_detection and self.last_fixed_ui_results:
                    # Look specifically for spin buttons that need long press
                    spin_buttons = []
                    for name, result in self.last_fixed_ui_results.items():
                        if result["detected"] and "spin" in name.lower() and result.get("long_press", False):
                            spin_buttons.append((name, result["center"][0], result["center"][1]))
                    
                    if spin_buttons:
                        spin_buttons_handled = True
                        # Found spin button(s) that need long press
                        logger.info(f"Found {len(spin_buttons)} spin button(s) for long press")
                        
                        # Save previous state to return to
                        previous_state = self.state
                        self.state = GameState.TAPPING_FIXED_UI
                        
                        # Long press each spin button using improved approach
                        for name, x, y in spin_buttons:
                            try:
                                logger.info(f"Long pressing spin button: {name} at ({int(x)}, {int(y)})")
                                
                                # Mark as handled
                                position_key = f"{int(x)}-{int(y)}"
                                handled_elements.add(position_key)
                                
                                # Approach 1: Standard tap first
                                logger.info("Step 1: Initial tap")
                                self.adb.tap(int(x), int(y))
                                time.sleep(1.5)  # Longer wait to ensure game responds
                                
                                # Approach 2: Direct shell command for reliability
                                logger.info("Step 2: Direct long press command (3 seconds)")
                                cmd = f"input swipe {int(x)} {int(y)} {int(x)} {int(y)} 3000"
                                self.adb.run_shell_command(cmd)
                                
                                # Longer wait after long press
                                logger.info("Waiting for auto-spin to activate...")
                                time.sleep(5.0)  # Much longer wait
                            except Exception as e:
                                logger.error(f"Error executing long press: {str(e)}")
                        
                        # After auto-spin activation, force multiple detection cycles
                        logger.info("Auto-spin activated - forcing complete detection reset")
                        
                        # Clear all cached data
                        self.last_dynamic_detections = []
                        self.last_fixed_ui_results = {}
                        
                        # Force multiple detection cycles with delays in between
                        for i in range(3):  # Try 3 times
                            logger.info(f"Forced detection cycle {i+1}/3")
                            
                            # Capture fresh screen
                            fresh_screen = self.adb.capture_screen()
                            if fresh_screen is None:
                                logger.error(f"Failed to capture screen in reset cycle {i+1}")
                                time.sleep(1.0)
                                continue
                                
                            # Run detection
                            try:
                                # Run YOLO detection
                                new_dynamic_detections = self.yolo_detector.detect(fresh_screen)
                                logger.info(f"Cycle {i+1}: Found {len(new_dynamic_detections)} YOLO objects")
                                
                                # Run fixed UI detection
                                if self.fixed_ui_detector:
                                    new_fixed_ui_results = self.fixed_ui_detector.detect(fresh_screen)
                                    fixed_count = sum(1 for r in new_fixed_ui_results.values() if r["detected"])
                                    logger.info(f"Cycle {i+1}: Found {fixed_count} fixed UI elements")
                                    
                                    # Update our stored results if we found something
                                    if len(new_dynamic_detections) > 0 or fixed_count > 0:
                                        self.last_dynamic_detections = new_dynamic_detections
                                        self.last_fixed_ui_results = new_fixed_ui_results
                                        logger.info("Detection successful - updating stored results")
                                        break
                            except Exception as e:
                                logger.error(f"Error in detection reset cycle {i+1}: {str(e)}")
                            
                            # Delay between cycles
                            time.sleep(1.0)
                        
                        # Return to normal state
                        self.state = previous_state
                        
                        # Skip to next iteration with a brief delay
                        time.sleep(self.action_delay)
                        continue
                
                # STEP 2: Handle regular tappable fixed UI elements
                fixed_ui_handled = False
                if self.using_hybrid_detection and self.last_fixed_ui_results:
                    # Get tap targets, excluding spin buttons (we already handled them with long press)
                    fixed_ui_tap_targets = []
                    for name, result in self.last_fixed_ui_results.items():
                        if (result["detected"] and 
                            result.get("tappable", False) and 
                            not (result.get("long_press", False) and "spin" in name.lower())):
                            fixed_ui_tap_targets.append((name, result["center"][0], result["center"][1]))
                    
                    if fixed_ui_tap_targets:
                        fixed_ui_handled = True
                        # Found tappable fixed UI elements
                        logger.info(f"Found {len(fixed_ui_tap_targets)} tappable fixed UI elements")
                        
                        # Save previous state
                        previous_state = self.state
                        self.state = GameState.TAPPING_FIXED_UI
                        
                        # Tap each fixed UI element
                        for name, x, y in fixed_ui_tap_targets:
                            logger.info(f"Tapping fixed UI element: {name} at ({int(x)}, {int(y)})")
                            
                            # Mark as handled
                            position_key = f"{int(x)}-{int(y)}"
                            handled_elements.add(position_key)
                            
                            self.adb.tap(int(x), int(y))
                            self.fixed_ui_taps_completed += 1
                            time.sleep(self.action_delay)
                        
                        # Return to previous state
                        self.state = previous_state
                
                # If we handled fixed UI elements, skip to the next iteration with a fresh screen capture
                if fixed_ui_handled:
                    logger.info("Fixed UI elements handled - skipping to next iteration for a fresh detection")
                    time.sleep(self.action_delay)
                    continue
                
                # STEP 3: Handle YOLO detected elements
                yolo_handled = False
                if self.last_dynamic_detections:
                    # Get tap targets from YOLO detector
                    dynamic_tap_targets = self.yolo_detector.get_tap_targets(self.last_dynamic_detections)
                    
                    # Count detected target types for logging/debugging
                    aiming_icons_count = sum(1 for t in dynamic_tap_targets if "attack_aiming_icon" in t[0].lower())
                    logger.info(f"YOLO detected {len(dynamic_tap_targets)} potential tap targets (including {aiming_icons_count} aiming icons)")
                    
                    # Set game state based on detected elements if needed
                    if aiming_icons_count > 0:
                        # Detect specific game mode based on context
                        if any("raid" in det.class_name.lower() for det in self.last_dynamic_detections):
                            logger.info("Raid targeting detected - setting state to RAIDING")
                            self.state = GameState.RAIDING
                        else:
                            logger.info("Attack targeting detected - setting state to ATTACKING")
                            self.state = GameState.ATTACKING
                    
                    # Filter out any elements that were already handled or spin buttons
                    filtered_dynamic_targets = []
                    for target in dynamic_tap_targets:
                        name, x, y = target
                        position_key = f"{int(x)}-{int(y)}"
                        
                        # Skip spin buttons
                        if "spin" in name.lower():
                            logger.info(f"Skipping YOLO-detected spin button {name} to prevent conflict with fixed UI handler")
                            continue
                        
                        # Skip elements already handled by fixed UI
                        if position_key in handled_elements:
                            logger.info(f"Skipping YOLO element {name} at ({int(x)}, {int(y)}) - already handled")
                            continue
                        
                        filtered_dynamic_targets.append(target)
                    
                    if filtered_dynamic_targets:
                        logger.info(f"Processing {len(filtered_dynamic_targets)} YOLO-detected elements in state: {self.state.value}")
                        yolo_handled = self._handle_dynamic_tap_targets(filtered_dynamic_targets)
                        
                        if yolo_handled:
                            logger.info("YOLO element tapped successfully")
                            time.sleep(self.action_delay * 2)  # Adding longer delay to let game react
                        else:
                            logger.warning("YOLO handler didn't tap any elements")
                
                # If we handled YOLO elements, skip to the next iteration with a fresh screen capture
                if yolo_handled:
                    logger.info("YOLO elements handled - continuing to next detection cycle")
                    time.sleep(self.action_delay)
                    self.no_detection_count = 0  # Reset no-detection counter after successful handling
                    continue
                
                # Add periodic forced refresh if no elements detected
                no_elements_detected = (len(self.last_dynamic_detections) == 0 and 
                                    sum(1 for r in self.last_fixed_ui_results.values() if r["detected"]) == 0)

                if no_elements_detected:
                    self.no_detection_count += 1
                    
                    # If several cycles with no detections, force a refresh
                    if self.no_detection_count >= 3:
                        logger.warning(f"No elements detected for {self.no_detection_count} cycles - forcing refresh")
                        refresh_success = self._force_refresh_detection()
                        self.no_detection_count = 0
                        
                        if refresh_success:
                            # Skip to next iteration with fresh detection results
                            logger.info("Forced refresh successful - continuing with new detection")
                            continue
                else:
                    # Reset counter when elements are detected
                    self.no_detection_count = 0
                
                # If no elements detected or handled, proceed with state-based actions
                if self.state == GameState.INITIALIZING:
                    self._initialize_game(screen)
                elif self.state == GameState.WAITING:
                    self._check_game_screen(screen)
                elif self.state == GameState.IDLE:
                    self._handle_idle_state(screen)
                elif self.state == GameState.ATTACKING:
                    # Check if we actually have attack elements currently visible
                    attack_elements = [det for det in self.last_dynamic_detections if "attack" in det.class_name.lower()]
                    if not attack_elements:
                        logger.info("No attack elements visible - returning to IDLE state")
                        self.state = GameState.IDLE
                    else:
                        self._handle_attack_state(screen)
                elif self.state == GameState.RAIDING:
                    # Check if we actually have raid elements currently visible
                    raid_elements = [det for det in self.last_dynamic_detections if "raid" in det.class_name.lower()]
                    if not raid_elements:
                        logger.info("No raid elements visible - returning to IDLE state")
                        self.state = GameState.IDLE
                    else:
                        self._handle_raid_state(screen)
                elif self.state == GameState.ERROR:
                    # Try to recover from error
                    self._try_recover(screen)
                
                # Short delay to prevent excessive CPU usage
                logger.debug("End of loop iteration - waiting for next detection cycle")
                time.sleep(self.detection_interval)
                
        except Exception as e:
            logger.error(f"Error in game loop: {str(e)}", exc_info=True)
            self.state = GameState.ERROR
        

    def _force_refresh_detection(self):
            """Force a complete refresh of screen capture and detection."""
            logger.info("Forcing complete detection refresh")
            
            # Clear all detection data
            self.last_dynamic_detections = []
            self.last_fixed_ui_results = {}
            
            # Capture a fresh screen
            screen = self.adb.capture_screen()
            if screen is None:
                logger.error("Failed to capture screen during forced refresh")
                return False
            
            # Perform fresh detections
            try:
                if self.using_hybrid_detection:
                    self.last_dynamic_detections = self.yolo_detector.detect(screen)
                    self.last_fixed_ui_results = self.fixed_ui_detector.detect(screen)
                    
                    # Log detection results
                    dynamic_count = len(self.last_dynamic_detections)
                    fixed_count = sum(1 for r in self.last_fixed_ui_results.values() if r["detected"])
                    logger.info(f"Forced refresh: detected {dynamic_count} YOLO objects and {fixed_count} fixed UI elements")
                    
                    # If in debug mode, log more details
                    if self.debug_mode and dynamic_count > 0:
                        object_names = [det.class_name for det in self.last_dynamic_detections[:5]]
                        logger.debug(f"YOLO objects after refresh: {', '.join(object_names)}{' ...' if dynamic_count > 5 else ''}")
                else:
                    self.last_dynamic_detections = self.yolo_detector.detect(screen)
                    logger.info(f"Forced refresh: detected {len(self.last_dynamic_detections)} YOLO objects")
                
                return True
            except Exception as e:
                logger.error(f"Detection error during forced refresh: {str(e)}")
                return False

    

    def _initialize_game(self, screen):
            """Initialize game state."""
            logger.info("Initializing game state")
            
            # Update state
            self.state = GameState.WAITING

    def _check_and_update_power_boost(self, screen):
        """
        Check the current power boost level and update if needed.
        
        Args:
            screen: Current screen image
        """
        # Only check periodically to avoid performance impact
        current_time = time.time()
        if current_time - self.last_power_boost_detection_time < self.power_boost_detection_interval:
            return
        
        self.last_power_boost_detection_time = current_time
        
        # Detect current power boost level
        level, confidence = self.power_boost_manager.detect_power_boost_level(screen)
        
        if level and confidence >= 0.8:  # Only accept high confidence detections
            logger.info(f"Detected power boost level: {level} (confidence: {confidence:.2f})")
            
            # Update current level if different
            if level != self.current_power_boost_level:
                logger.info(f"Power boost level changed from {self.current_power_boost_level} to {level}")
                self.current_power_boost_level = level
    
    def _check_game_screen(self, screen):
        """Check if we're in the main game screen."""
        logger.debug("Checking game screen")
        
        # Determine if we have any detections using the last results
        dynamic_objects_detected = len(self.last_dynamic_detections) > 0
        fixed_ui_detected = self.using_hybrid_detection and any(r["detected"] for r in self.last_fixed_ui_results.values())
        
        if dynamic_objects_detected or fixed_ui_detected:
            logger.info("Game screen detected")
            self.state = GameState.IDLE
        else:
            logger.debug("Waiting for game screen...")
    
    def _get_bbox_center(self, detection):
        """Get the center point of a detection's bounding box.
        Handles different detection object formats."""
        try:
            # First try to get bbox attribute (standard format)
            if hasattr(detection, 'bbox'):
                x1, y1, x2, y2 = detection.bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                return center_x, center_y
            
            # Try box attribute (alternative format)
            elif hasattr(detection, 'box'):
                x1, y1, x2, y2 = detection.box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                return center_x, center_y
            
            # Try xyxy attribute (YOLOv5 format)
            elif hasattr(detection, 'xyxy'):
                x1, y1, x2, y2 = detection.xyxy[0]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                return center_x, center_y
            
            # Try center_x and center_y attributes (another format)
            elif hasattr(detection, 'center_x') and hasattr(detection, 'center_y'):
                return detection.center_x, detection.center_y
            
            # Last resort - get coordinates from tap target
            elif hasattr(self.yolo_detector, 'get_tap_targets'):
                tap_targets = self.yolo_detector.get_tap_targets([detection])
                if tap_targets:
                    return tap_targets[0][1], tap_targets[0][2]
            
            # If none of the above worked, log a warning and return a default
            logger.warning(f"Could not determine center for detection: {detection}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting detection center: {str(e)}")
            return None
        

    def debug_power_boost_detection(self):
        """Debug power boost detection with explicit detector creation."""
        logger.info("DEBUG: Testing power boost detection")
        
        # Capture a fresh screenshot
        screen = self.adb.capture_screen()
        if screen is None:
            logger.error("DEBUG: Failed to capture screen for power boost detection")
            return None, 0
        
        # Save screenshot for analysis
        debug_dir = "debug"
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(f"{debug_dir}/power_boost_debug.png", screen)
        logger.info(f"DEBUG: Saved screenshot to {debug_dir}/power_boost_debug.png")
        
        # 1. Try detection with the existing power boost manager
        level1, confidence1 = self.power_boost_manager.detect_power_boost_level(screen)
        logger.info(f"DEBUG: Current detector result: {level1} (confidence: {confidence1:.2f})")
        
        # 2. Create a fresh detector with lower threshold
        from power_boost_detector import PowerBoostDetector
        fresh_detector = PowerBoostDetector("power_boost_templates.json")
        fresh_detector.set_threshold(0.7)  # Lower threshold for testing
        level2, confidence2 = fresh_detector.detect(screen)
        logger.info(f"DEBUG: Fresh detector result: {level2} (confidence: {confidence2:.2f})")
        
        # 3. Draw detected template on image for visual verification
        if level2 and level2 in fresh_detector.templates:
            roi = fresh_detector.templates[level2]["roi"]
            template = fresh_detector.templates[level2]["template"]
            logger.info(f"DEBUG: Template size: {template.shape}")
            logger.info(f"DEBUG: Template ROI: {roi}")
            
            # Draw ROI on image
            debug_img = screen.copy()
            x1, y1, x2, y2 = roi
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_img, f"{level2}: {confidence2:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save debug image
            cv2.imwrite(f"{debug_dir}/power_boost_detection.png", debug_img)
            logger.info(f"DEBUG: Saved detection visualization to {debug_dir}/power_boost_detection.png")
        
        # Find power boost button as well
        try:
            logger.info("DEBUG: Attempting to find power boost button")
            self.power_boost_manager.find_power_boost_button(screen, self.last_fixed_ui_results)
            logger.info(f"DEBUG: Power boost button at {self.power_boost_manager.power_boost_button}")
            
            # Draw button on debug image
            if level2:
                button_img = debug_img.copy() if 'debug_img' in locals() else screen.copy()
                button_x, button_y = self.power_boost_manager.power_boost_button
                cv2.circle(button_img, (button_x, button_y), 15, (0, 0, 255), -1)
                cv2.putText(button_img, "Power Boost Button", (button_x-60, button_y-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imwrite(f"{debug_dir}/power_boost_button.png", button_img)
                logger.info(f"DEBUG: Saved button visualization to {debug_dir}/power_boost_button.png")
        except Exception as e:
            logger.error(f"DEBUG: Error finding button: {str(e)}")
        
        # Test a tap at the button location
        try:
            if self.power_boost_manager.power_boost_button:
                logger.info("DEBUG: Testing tap at power boost button")
                x, y = self.power_boost_manager.power_boost_button
                tap_result = self.adb.tap(x, y)
                logger.info(f"DEBUG: Tap result: {tap_result}")
        except Exception as e:
            logger.error(f"DEBUG: Error tapping button: {str(e)}")
        
        return level2, confidence2 if level2 else (level1, confidence1)
    
    # Add this method to the GameLogicIntegrator class in game_logic_integrator.py
    # Add to game_logic_integrator.py
    def update_power_boost_sequence(self, new_sequence):
        """
        Update the power boost sequence with new configuration.
        
        Args:
            new_sequence: New sequence from configuration
            
        Returns:
            bool: Success status
        """
        try:
            if not new_sequence:
                logger.warning("Empty power boost sequence received")
                return False
            
            # Log the update
            sequence_str = ', '.join([f"{item['level']}:{item['attacks']}" for item in new_sequence])
            logger.info(f"Updating GameLogicIntegrator power boost sequence to: {sequence_str}")
            
            # 1. Update stored sequence directly
            self.power_boost_sequence = new_sequence
            
            # 2. Reset sequence position
            self.current_sequence_index = 0
            self.attacks_in_current_level = 0
            
            # 3. Update current level and target
            if new_sequence:
                next_item = new_sequence[0]
                self.current_power_boost_level = next_item["level"]
                self.target_attacks_for_level = next_item["attacks"]
            
            logger.info(f"GameLogicIntegrator sequence updated: {len(new_sequence)} levels")
            logger.info(f"Current level: {self.current_power_boost_level}, target attacks: {self.target_attacks_for_level}")
            
            return True
        except Exception as e:
            logger.error(f"Error updating GameLogicIntegrator sequence: {str(e)}")
            return False
    
    def _handle_idle_state(self, screen):
        """Handle the idle state - decide what to do next."""
        logger.debug("Handling idle state")
        
        # Get tap targets from YOLO detector
        # We use the tap_targets method if it exists, otherwise we extract them ourselves
        if hasattr(self.yolo_detector, 'get_tap_targets'):
            dynamic_tap_targets = self.yolo_detector.get_tap_targets(self.last_dynamic_detections)
        else:
            # Extract tap targets manually
            dynamic_tap_targets = []
            for det in self.last_dynamic_detections:
                class_name = det.class_name if hasattr(det, 'class_name') else "unknown"
                # Skip if not in tap_elements
                if not hasattr(self.yolo_detector, 'tap_elements') or class_name in getattr(self.yolo_detector, 'tap_elements', []):
                    center = self._get_bbox_center(det)
                    if center:
                        dynamic_tap_targets.append((class_name, center[0], center[1]))
        
        # Process YOLO detector results first
        if dynamic_tap_targets:
            # Check for special elements first
            spin_button = next((t for t in dynamic_tap_targets if t[0] == "spin_button"), None)
            attack_target = next((t for t in dynamic_tap_targets if t[0] == "attack_symbol"), None)
            raid_target = next((t for t in dynamic_tap_targets if t[0] == "raid_symbol"), None)
            
            if spin_button:
                logger.info("Spin button found, starting spin")
                self.state = GameState.SPINNING
                self._handle_spin(dynamic_tap_targets)
                return
            elif attack_target:
                logger.info("Attack target found")
                self.state = GameState.ATTACKING
                time.sleep(self.action_delay)
                self._tap_target(attack_target)
                return
            elif raid_target:
                logger.info("Raid target found")
                self.state = GameState.RAIDING
                time.sleep(self.action_delay)
                self._tap_target(raid_target)
                return
            else:
                # Tap any other dynamic targets
                target_name, x, y = dynamic_tap_targets[0]
                logger.info(f"Tapping dynamic element: {target_name} at ({int(x)}, {int(y)})")
                self.adb.tap(int(x), int(y))
                time.sleep(self.action_delay)
                return
        
        # If we reached here, no actions were taken
        logger.debug("No action taken in idle state")

    # Add this debug function to GameLogicIntegrator class
    # Update the debug_attack_counter method in CoinMasterBotGUI
    def debug_attack_counter(self):
        """Debug the attack counter across all game controller types"""
        if not self.game_controller:
            self.log("Game controller not available")
            return
        
        self.log("===== ATTACK COUNTER DEBUG =====")
        
        # Check controller type
        controller_type = self.game_controller.__class__.__name__
        self.log(f"Game controller type: {controller_type}")
        
        # Common attributes to check regardless of controller type
        common_attrs = [
            'attacks_completed', 
            'attacks_in_current_level',
            'target_attacks_for_level', 
            'current_power_boost_level',
            'current_sequence_index',
            'power_boost_sequence'
        ]
        
        # Check all common attributes
        for attr in common_attrs:
            if hasattr(self.game_controller, attr):
                value = getattr(self.game_controller, attr)
                self.log(f"{attr}: {value}")
                
                # For sequences, show more details
                if attr == 'power_boost_sequence':
                    if isinstance(value, list):
                        self.log(f"Sequence length: {len(value)}")
                        for idx, item in enumerate(value):
                            self.log(f"  [{idx}]: {item}")
                    elif hasattr(value, 'sequence'):
                        # This is likely a PowerBoostSequence object
                        sequence = value.sequence
                        self.log(f"PowerBoostSequence length: {len(sequence)}")
                        for idx, item in enumerate(sequence):
                            level = item['level']
                            level_name = level.name if hasattr(level, 'name') else str(level)
                            self.log(f"  [{idx}]: {level_name} - {item['attacks']} attacks")
            else:
                self.log(f"{attr}: Not found")
        
        # GameLogicIntegrator specific attributes
        if controller_type == "GameLogicIntegrator":
            if hasattr(self.game_controller, 'power_boost_manager'):
                pbm = self.game_controller.power_boost_manager
                self.log(f"Power boost button: {pbm.power_boost_button}")
                
                # Test if power boost button can be found
                if not pbm.power_boost_button:
                    self.log("Attempting to find power boost button...")
                    screen = self.adb.capture_screen()
                    if screen is not None:
                        pbm.find_power_boost_button(screen, getattr(self.game_controller, 'last_fixed_ui_results', None))
                        self.log(f"Power boost button after search: {pbm.power_boost_button}")
        
        # Check if power boost change would trigger
        attacks = getattr(self.game_controller, 'attacks_in_current_level', 0)
        target = getattr(self.game_controller, 'target_attacks_for_level', 1)
        
        try:
            attacks = int(attacks)
            target = int(target)
            self.log(f"Attack comparison: {attacks} >= {target} = {attacks >= target}")
            
            if attacks >= target:
                self.log("[SUCCESS] POWER BOOST CHANGE WOULD TRIGGER")
            else:
                self.log(f"[FAIL] Power boost change would NOT trigger (need {target-attacks} more attacks)")
        except (ValueError, TypeError) as e:
            self.log(f"Error comparing attack values: {e}")
        
        self.log("Attack counter debug completed")
        messagebox.showinfo("Attack Counter Debug", 
                        f"Current attacks: {attacks}/{target}\n" +
                        f"Power boost change would {'TRIGGER' if attacks >= target else 'NOT trigger'}")
        
    def check_power_boost_templates(self):
        """Check if power boost templates are loaded and working."""
        if not hasattr(self, 'power_boost_manager') or not self.power_boost_manager:
            logger.error("Power boost manager not initialized")
            return False
        
        # Check if templates are loaded
        if not hasattr(self.power_boost_manager.detector, 'templates'):
            logger.error("Power boost detector templates not initialized")
            return False
        
        # Check how many templates are loaded
        templates = self.power_boost_manager.detector.templates
        template_count = len(templates)
        
        if template_count == 0:
            logger.error("No power boost templates loaded!")
            return False
        
        # Log loaded templates
        logger.info(f"Loaded {template_count} power boost templates: {list(templates.keys())}")
        
        # Take a screenshot and test detection
        screen = self.adb.capture_screen()
        if screen is None:
            logger.error("Failed to capture screen for template testing")
            return False
        
        # Test detection
        level, confidence = self.power_boost_manager.detect_power_boost_level(screen)
        if level:
            logger.info(f"Detected power boost level: {level} (confidence: {confidence:.2f})")
            return True
        else:
            logger.warning("No power boost level detected in current screen")
            return False

    def _handle_attack_state(self, screen):
        # Find attack aiming icons - make sure this is exactly matching your model's output
        aiming_icons = [det for det in self.last_dynamic_detections if det.class_name == "attack_aiming_icon"]
        
        if aiming_icons:
            # Found an aiming icon - tap it
            icon = aiming_icons[0]
            x, y = int(icon.center_x), int(icon.center_y)
            
            logger.info(f"Tapping aiming icon at ({x}, {y})")
            self.adb.tap(x, y)
            
            # Register the attack - THIS IS A CRITICAL PART
            # Using the centralized register_attack method
            change_needed = self.register_attack()
            
            if change_needed:
                # Start thread for power boost change
                threading.Thread(target=self._execute_power_boost_change, daemon=True).start()

    
    def register_attack(self):
        """
        Properly register an attack with enhanced logging.
        
        Returns:
            bool: True if power boost change needed
        """
        # Log before state
        before_level = self.attacks_in_current_level
        before_total = self.attacks_completed
        
        # Increment BOTH counters
        self.attacks_completed += 1
        self.attacks_in_current_level += 1
        
        # Log after state with clear indicators
        logger.info(f"Attack registered: {before_level} → {self.attacks_in_current_level}/{self.target_attacks_for_level} (Total: {before_total} → {self.attacks_completed})")
        
        # Check if we need to change power boost level
        if self.attacks_in_current_level >= self.target_attacks_for_level:
            logger.info("Power boost threshold reached - change needed")
            return True
        
        return False


    def debug_counters(self):
        """Debug method to display all counter values."""
        logger.info("===== COUNTER DEBUG =====")
        logger.info(f"Current level: {self.current_power_boost_level}")
        logger.info(f"Current sequence index: {self.current_sequence_index}")
        logger.info(f"Attacks in current level: {self.attacks_in_current_level}")
        logger.info(f"Total attacks completed: {self.attacks_completed}")
        logger.info(f"Target attacks for level: {self.target_attacks_for_level}")
        logger.info(f"Power boost change threshold met: {self.attacks_in_current_level >= self.target_attacks_for_level}")

    # Add this method to GameLogicIntegrator
    def _periodic_power_boost_check(self):
        """Periodically check and correct power boost level if needed."""
        # This method runs in a separate thread
        while self.running:
            try:
                # Only run checks when in IDLE or SPINNING state to avoid interrupting attacks
                if self.state == GameState.IDLE or self.state == GameState.SPINNING:
                    screen = self.adb.capture_screen()
                    if screen is not None:
                        detected_level = self.power_boost_manager.auto_detect_and_verify_level(screen)
                        expected_level = self.current_power_boost_level
                        
                        # If detected level doesn't match expected level, correct it
                        if detected_level != expected_level:
                            logger.warning(f"Power boost sync issue detected: UI shows {detected_level}, should be {expected_level}")
                            # Fix the discrepancy
                            success = self.power_boost_manager.change_power_boost(detected_level, expected_level)
                            if success:
                                logger.info(f"Power boost auto-corrected to {expected_level}")
                            else:
                                logger.error("Failed to auto-correct power boost level")
                
                # Check every 30 seconds to avoid overhead
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in periodic power boost check: {str(e)}")
                time.sleep(30)  # Continue checks even after error


    # Add to GameLogicIntegrator class
    def _create_default_power_boost_templates(self):
        """Create default power boost templates as fallback."""
        try:
            logger.info("Creating default power boost templates")
            
            # Create templates directory
            os.makedirs("templates", exist_ok=True)
            
            # Define default template regions based on common screen sizes
            if self.adb and self.adb.screen_resolution:
                w, h = self.adb.screen_resolution
                # Calculate default region for power boost button (bottom center of screen)
                x1 = int(w * 0.35)
                y1 = int(h * 0.75)
                x2 = int(w * 0.65)
                y2 = int(h * 0.85)
                
                # Create default configuration
                default_config = {
                    "X1": {
                        "roi": [x1, y1, x2, y2],
                        "template_path": "templates/default_power_boost.png",
                        "size": [x2-x1, y2-y1]
                    }
                }
                
                # Take a screenshot to use as template
                screen = self.adb.capture_screen()
                if screen is not None:
                    # Save region as template
                    template = screen[y1:y2, x1:x2]
                    cv2.imwrite("templates/default_power_boost.png", template)
                    
                    # Save configuration
                    with open("power_boost_templates.json", 'w') as f:
                        json.dump(default_config, f, indent=4)
                    
                    logger.info("Default power boost template created successfully")
                    
                    # Reload templates in power boost manager
                    self.power_boost_manager.reload_templates()
                    return True
            
            logger.warning("Could not create default templates due to missing screen resolution")
            return False
            
        except Exception as e:
            logger.error(f"Error creating default templates: {str(e)}")
            return False

    def _execute_power_boost_change(self):
        """
        Execute power boost change using the proven manual method.
        This function becomes a simple wrapper around the manual change function.
        """
        try:
            # Get current sequence information
            current_index = self.current_sequence_index
            next_index = (current_index + 1) % len(self.power_boost_sequence)
            next_item = self.power_boost_sequence[next_index]
            target_level = next_item["level"]
            
            logger.info(f"Auto power boost change starting - using manual change method to level {target_level}")
            
            # Add a cooldown check to prevent too frequent changes
            current_time = time.time()
            if hasattr(self, 'last_power_boost_change_time') and (current_time - self.last_power_boost_change_time < 3):
                delay_needed = 3 - (current_time - self.last_power_boost_change_time)
                logger.info(f"Cooling down power boost change for {delay_needed:.1f} seconds")
                time.sleep(delay_needed)
            
            # Call the manual function directly - we know this works
            success = self.manual_change_power_boost(target_level)
            
            # Update last change time
            self.last_power_boost_change_time = time.time()
            
            if success:
                logger.info(f"Power boost successfully changed to {target_level}")
                
                # Only update counters after successful change
                self.attacks_in_current_level = 0
                self.current_sequence_index = next_index
                self.target_attacks_for_level = next_item["attacks"]
                
                # Debug the counter state after update
                self.debug_counters()
                
                # Small delay to let UI update
                time.sleep(1.0)
                
                return True
            else:
                logger.error(f"Power boost change to {target_level} failed")
                
                # Force the state to be checked on next iteration
                # This helps break out of loops where reality differs from internal state
                self.last_power_boost_detection_time = 0  
                
                return False
                
        except Exception as e:
            logger.error(f"Error in power boost change: {str(e)}")
            return False
        
    def restart_detection_system(self):
        """Restart the detection system if it becomes unresponsive."""
        logger.warning("Restarting detection system - clearing all cached data")
        
        try:
            # Clear all cached detection data
            self.last_dynamic_detections = []
            self.last_fixed_ui_results = {}
            
            # Reset power boost button location to force fresh detection
            if hasattr(self, 'power_boost_manager'):
                self.power_boost_manager.power_boost_button = None
            
            # Force a power boost level detection on next cycle
            self.last_power_boost_detection_time = 0
            
            # Force multiple detection cycles with delays in between
            for i in range(2):
                logger.info(f"Detection reset cycle {i+1}/2")
                
                # Capture fresh screen
                screen = self.adb.capture_screen()
                if screen is None:
                    logger.error(f"Failed to capture screen in reset cycle {i+1}")
                    time.sleep(1.0)
                    continue
                    
                # Run detection with error handling
                try:
                    if self.using_hybrid_detection:
                        self.last_dynamic_detections = self.yolo_detector.detect(screen)
                        self.last_fixed_ui_results = self.fixed_ui_detector.detect(screen)
                        
                        dynamic_count = len(self.last_dynamic_detections)
                        fixed_count = sum(1 for r in self.last_fixed_ui_results.values() if r["detected"])
                        logger.info(f"Reset cycle {i+1}: Found {dynamic_count} dynamic objects, {fixed_count} fixed UI elements")
                    else:
                        self.last_dynamic_detections = self.yolo_detector.detect(screen)
                        logger.info(f"Reset cycle {i+1}: Found {len(self.last_dynamic_detections)} dynamic objects")
                        
                except Exception as e:
                    logger.error(f"Error in detection reset cycle {i+1}: {str(e)}")
                
                # Delay between cycles
                time.sleep(1.0)
            
            # Also detect current power boost level
            try:
                screen = self.adb.capture_screen()
                if screen is not None:
                    level, confidence = self.power_boost_manager.detect_power_boost_level(screen)
                    if level and confidence >= 0.7:
                        if level != self.current_power_boost_level:
                            logger.warning(f"Power boost level mismatch detected during reset: UI shows {level}, code has {self.current_power_boost_level}")
                            # Consider updating internal state to match what's detected
                            # This will help break out of problematic loops
                            logger.info(f"Resetting power boost level to match UI: {level}")
                            self.current_power_boost_level = level
                    
            except Exception as e:
                logger.error(f"Error detecting power boost level during reset: {str(e)}")
            
            logger.info("Detection system restart completed")
            
        except Exception as e:
            logger.error(f"Error during detection system restart: {str(e)}")
            
    def _handle_spin(self, tap_targets):
        """Handle spin action."""
        spin_button = next((t for t in tap_targets if t[0] == "spin_button"), None)
        
        if spin_button:
            logger.info("Tapping spin button")
            self._tap_target(spin_button)
            time.sleep(self.action_delay * 3)  # Longer delay for spin
            self.state = GameState.IDLE
        else:
            logger.warning("Spin button not found in tap targets")
            self.state = GameState.IDLE

    def debug_attack_counter(self):
        """Debug the attack counter and power boost threshold."""
        logger.info("===== ATTACK COUNTER DEBUG =====")
        logger.info(f"Current level: {self.current_power_boost_level}")
        logger.info(f"Current sequence index: {self.current_sequence_index}")
        logger.info(f"Attacks in current level: {self.attacks_in_current_level}")
        logger.info(f"Target attacks for level: {self.target_attacks_for_level}")
        logger.info(f"Will trigger power boost change: {self.attacks_in_current_level >= self.target_attacks_for_level}")
        
        # Check power boost button coordinates
        if hasattr(self, 'power_boost_manager') and hasattr(self.power_boost_manager, 'power_boost_button'):
            logger.info(f"Power boost button at: {self.power_boost_manager.power_boost_button}")
        else:
            logger.info("No power boost button coordinates available")
    

    def _handle_dynamic_tap_targets(self, tap_targets):
        """
        Direct handler that taps attack_symbols and directly changes power boost levels
        after the required number of attacks is reached.
        """
        if not tap_targets:
            return False
        
        # Look for attack_symbol from YOLO model
        attack_found = False
        for target in tap_targets:
            # Get the target name (first element)
            target_name = target[0] if len(target) > 0 else ""
            
            # Check if this is an attack_symbol
            if target_name == "attack_symbol":
                attack_found = True
                logger.info(f"Attack symbol found: {target}")
                
                # Get coordinates
                x, y = int(target[1]), int(target[2])
                
                # Tap the attack symbol
                logger.info(f"Tapping attack symbol at ({x}, {y})")
                self.adb.tap(x, y)
                
                # Register the attack in our counter
                self.attacks_completed += 1
                self.attacks_in_current_level += 1
                
                # Check if we've reached the target number of attacks for this level
                logger.info(f"Attacks: {self.attacks_in_current_level}/{self.target_attacks_for_level}")
                
                if self.attacks_in_current_level >= self.target_attacks_for_level:
                    # Get the next level in the sequence
                    current_index = self.current_sequence_index
                    next_index = (current_index + 1) % len(self.power_boost_sequence)
                    next_item = self.power_boost_sequence[next_index]
                    target_level = next_item["level"]
                    
                    logger.info(f"Target attacks reached! Changing power boost to {target_level}")
                    
                    # Change power boost level directly
                    self._direct_power_boost_tap(target_level)
                    
                    # Update sequence position AFTER changing level
                    self.attacks_in_current_level = 0
                    self.current_sequence_index = next_index
                    self.target_attacks_for_level = next_item["attacks"]
                    self.current_power_boost_level = target_level
                    
                    logger.info(f"Sequence advanced to {target_level}, next target: {self.target_attacks_for_level} attacks")
                
                # No need to check other targets
                break
        
        # Return true if we found and handled an attack symbol
        return attack_found


    def _tap_target(self, target):
        """Tap a specific target."""
        name, x, y = target
        logger.info(f"Tapping {name} at ({int(x)}, {int(y)})")
        self.adb.tap(int(x), int(y))
        time.sleep(self.action_delay)
    
    def _advance_sequence(self):
        """Advance to the next level in the power boost sequence."""
        # Move to next position in sequence
        self.current_sequence_index = (self.current_sequence_index + 1) % len(self.power_boost_sequence)
        
        # Update current level and attacks target
        next_level = self.power_boost_sequence[self.current_sequence_index]
        self.current_power_boost = PowerBoostLevel.from_string(next_level["level"])
        self.target_attacks_for_level = next_level["attacks"]
        self.attacks_in_current_level = 0
        
        logger.info(f"Advanced to {self.current_power_boost.value} level, target: {self.target_attacks_for_level} attacks")
    
    def _try_recover(self, screen):
        """Try to recover from error state."""
        logger.info("Attempting to recover from error state")
        
        # Get tap targets from both detectors
        if hasattr(self.yolo_detector, 'get_tap_targets'):
            dynamic_tap_targets = self.yolo_detector.get_tap_targets(self.last_dynamic_detections)
        else:
            # Extract tap targets manually
            dynamic_tap_targets = []
            for det in self.last_dynamic_detections:
                class_name = det.class_name if hasattr(det, 'class_name') else "unknown"
                if not hasattr(self.yolo_detector, 'tap_elements') or class_name in getattr(self.yolo_detector, 'tap_elements', []):
                    center = self._get_bbox_center(det)
                    if center:
                        dynamic_tap_targets.append((class_name, center[0], center[1]))
                        
        fixed_ui_tap_targets = []
        
        if self.using_hybrid_detection:
            fixed_ui_tap_targets = self.fixed_ui_detector.get_tap_targets(self.last_fixed_ui_results)
        
        # Try tapping any detected elements to recover
        if fixed_ui_tap_targets:
            name, x, y = fixed_ui_tap_targets[0]
            logger.info(f"Recovery: Tapping fixed UI element {name} at ({int(x)}, {int(y)})")
            self.adb.tap(int(x), int(y))
            self.state = GameState.IDLE
            time.sleep(self.action_delay)
        elif dynamic_tap_targets:
            name, x, y = dynamic_tap_targets[0]
            logger.info(f"Recovery: Tapping dynamic element {name} at ({int(x)}, {int(y)})")
            self.adb.tap(int(x), int(y))
            self.state = GameState.IDLE
            time.sleep(self.action_delay)
        else:
            # Try pressing back button
            logger.info("No targets found, pressing back button")
            self.adb.press_key(self.adb.KEY_BACK)
            time.sleep(1.0)
            self.state = GameState.WAITING

    # Update the start method in GameLogicIntegrator
    def start(self):
        """Start the game controller with enhanced automatic features."""
        if self.running:
            logger.warning("Game controller is already running")
            return
        
        logger.info("Starting game controller with full power boost automation")
        self.running = True
        self.state = GameState.INITIALIZING
        
        # Check if power boost templates are loaded
        template_check = self.check_power_boost_templates()
        if not template_check:
            logger.warning("Power boost templates may not be properly loaded! Creating default templates...")
            # Initialize default templates if missing
            self._create_default_power_boost_templates()
        
        # Start main game loop thread
        self.action_thread = threading.Thread(target=self._game_loop, daemon=True)
        self.action_thread.start()
        
        # Start periodic power boost check thread for automatic correction
        self.power_boost_check_thread = threading.Thread(target=self._periodic_power_boost_check, daemon=True)
        self.power_boost_check_thread.start()
        
        logger.info("All automatic systems initialized and running")

    def manual_change_power_boost(self, target_level):
        """
        Manual power boost change with enhanced reliability.
        
        Args:
            target_level: Target power boost level (format: "X1", "X15", etc.)
            
        Returns:
            bool: Success status
        """
        logger.info(f"Manual power boost change to {target_level}")
        
        # Take a fresh screenshot
        screen = self.adb.capture_screen()
        if screen is None:
            logger.error("Failed to capture screen")
            return False
        
        # Detect current level
        current_level, confidence = self.power_boost_manager.detect_power_boost_level(screen)
        if not current_level:
            logger.warning("Current power boost level not detected, using X1 as default")
            current_level = "X1"
        else:
            logger.info(f"Current power boost level: {current_level} (confidence: {confidence:.2f})")
        
        # Make sure target level is in correct format (X followed by number)
        if not target_level.startswith("X"):
            target_level = "X" + target_level
        
        # Define the power boost level sequence
        boost_levels = ["X1", "X2", "X3", "X15", "X50", "X400", "X1500", "X6000", "X20000"]
        
        # Validate target level
        if target_level not in boost_levels:
            logger.error(f"Invalid target level: {target_level}. Valid levels: {boost_levels}")
            return False
        
        # For GameLogicIntegrator with power_boost_manager
        if hasattr(self, 'power_boost_manager'):
            # Get the boost button coordinates (reset first to force fresh detection)
            self.power_boost_manager.power_boost_button = None
            self.power_boost_manager.find_power_boost_button(screen, self.last_fixed_ui_results)
            
            if not self.power_boost_manager.power_boost_button:
                logger.warning("Power boost button not found, using default position (540, 1300)")
                # Use exact coordinates as provided
                button_x, button_y = 540, 1300
            else:
                button_x, button_y = self.power_boost_manager.power_boost_button
                logger.info(f"Power boost button found at {(button_x, button_y)}")
            
            # Calculate taps needed
            try:
                current_idx = boost_levels.index(current_level)
                target_idx = boost_levels.index(target_level)
                
                # Calculate taps using correct wrapping logic
                if target_idx > current_idx:
                    taps_needed = target_idx - current_idx
                else:
                    taps_needed = len(boost_levels) - current_idx + target_idx
                
                logger.info(f"Need {taps_needed} taps to change from {current_level} to {target_level}")
                
                # Execute taps with proper delays
                if taps_needed > 0:
                    for i in range(taps_needed):
                        logger.info(f"Power boost tap {i+1}/{taps_needed}")
                        self.adb.tap(button_x, button_y)
                        time.sleep(1.0)  # 1 second between taps for reliability
                    
                    # Update current level
                    self.current_power_boost_level = target_level
                    logger.info(f"Power boost successfully changed to {target_level}")
                    return True
                else:
                    logger.info(f"Already at the target level {target_level}")
                    return True
                    
            except ValueError as e:
                logger.error(f"Error calculating power boost taps: {str(e)}")
                return False
        else:
            logger.error("Power boost manager not available")
            return False
        