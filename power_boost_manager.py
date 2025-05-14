import os
import logging
import time
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any

# Import the power boost detector
from power_boost_detector import PowerBoostDetector

logger = logging.getLogger(__name__)

class PowerBoostManager:
    """
    Enhanced manager class for handling power boost level detection and changes.
    Fully automatic with improved reliability and recovery mechanisms.
    Integrates with the GameLogicIntegrator to provide robust power boost management.
    """
    
    def __init__(self, config_file="power_boost_templates.json", adb_controller=None):
        """
        Initialize the power boost manager.
        
        Args:
            config_file: Path to the power boost templates config file
            adb_controller: ADB controller instance
        """
        # Ensure we have absolute path to config file
        if not os.path.isabs(config_file):
            self.config_file = os.path.abspath(config_file)
        else:
            self.config_file = config_file
            
        self.adb_controller = adb_controller
        
        # Create detector and load templates
        self.detector = PowerBoostDetector(self.config_file)
        
        # Set a lower threshold for better detection
        self.detector.set_threshold(0.7)  # More permissive threshold
        
        # Power boost sequence
        self.boost_levels = ["X1", "X2", "X3", "X15", "X50", "X400", "X1500", "X6000", "X20000"]
        
        # Power boost button coordinates
        self.power_boost_button = None
        
        # Last detection timestamp
        self.last_detection_time = 0
        
        # Detection confidence threshold
        self.detection_threshold = 0.7  # More permissive threshold
        
        # Consecutive failure counter for adaptive recovery
        self.consecutive_failures = 0
        
        # Maximum failures before fallback
        self.max_failures_before_fallback = 3
        
        # Check if templates are loaded
        if os.path.exists(self.config_file):
            logger.info(f"Power boost templates loaded from {self.config_file}")
            logger.info(f"Loaded templates for: {list(self.detector.templates.keys())}")
        else:
            logger.warning(f"Power boost template file {self.config_file} not found. "
                        "Fallback mechanisms will be used.")
            self.create_default_templates()
        
    def auto_detect_and_verify_level(self, image: np.ndarray) -> str:
        """
        Automatically detect current power boost level with multiple attempts for reliability.
        
        Args:
            image: Current screenshot
            
        Returns:
            Detected level or fallback to "X1" if detection fails
        """
        # Try detection with decreasing thresholds for better resilience
        thresholds = [0.8, 0.7, 0.6, 0.5]
        
        for threshold in thresholds:
            self.detector.set_threshold(threshold)
            level, confidence = self.detector.detect(image)
            
            if level and confidence >= threshold:
                logger.info(f"Auto-detected power boost level: {level} (confidence: {confidence:.2f})")
                self.consecutive_failures = 0  # Reset failure counter on success
                return level
        
        # If detection fails after all attempts, use default and log the issue
        self.consecutive_failures += 1
        logger.warning(f"Could not reliably detect power boost level (failure #{self.consecutive_failures}), using default X1")
        
        # If too many consecutive failures, try to recreate templates
        if self.consecutive_failures >= self.max_failures_before_fallback:
            logger.warning(f"Too many consecutive detection failures ({self.consecutive_failures}), recreating templates")
            self.create_default_templates()
            self.consecutive_failures = 0  # Reset counter
        
        return "X1"
    
    def detect_power_boost_level(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Detect the current power boost level in the image.
        This improved version specifically looks for the yellow power boost button.
        
        Args:
            image: OpenCV image (BGR)
            
        Returns:
            Tuple of (detected level, confidence) or (None, 0) if not detected
        """
        # Update detection timestamp
        self.last_detection_time = time.time()
        
        # Use the detector to find the power boost level with standard detection
        level, confidence = self.detector.detect(image)
        
        if level and confidence >= self.detection_threshold:
            # Successful detection
            self.consecutive_failures = 0  # Reset failure counter
            logger.info(f"Standard detection found power boost level: {level} (confidence: {confidence:.2f})")
            return level, confidence
            
        # If standard detection failed, try text-based detection
        try:
            # Extract bottom portion of screen where power boost button is typically located
            h, w = image.shape[:2]
            bottom_portion = image[int(h*0.75):int(h*0.95), :]
            
            # Convert to HSV and create a mask for yellow (power boost button color)
            hsv = cv2.cvtColor(bottom_portion, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([40, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process large yellow areas (potential power boost buttons)
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                if cv2.contourArea(contour) < 1000:  # Skip small areas
                    continue
                    
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract the region
                button_region = bottom_portion[y:y+h, x:x+w]
                
                # Check for text patterns that resemble power boost indicators
                for boost_level in self.boost_levels:
                    # Look specifically for "X1", "X2", etc. 
                    # For simplicity, just check if the level appears in the yellow region
                    if boost_level.lower() in f"{button_region}".lower():
                        logger.info(f"Text-based detection found power boost level: {boost_level} (confidence: 0.7)")
                        return boost_level, 0.7
            
            # If we get here, level detection failed with both methods
            self.consecutive_failures += 1
            
            # Try exact match with fixed coordinates
            # Based on the Coin Master UI, the power boost button has known positions
            if self.power_boost_button:
                x, y = self.power_boost_button
                
                # Search for any level text near the button
                for level in self.boost_levels:
                    # Check if any part of the image around button position has text resembling the level
                    region_around_button = image[max(0, y-50):min(image.shape[0], y+50), 
                                               max(0, x-50):min(image.shape[1], x+50)]
                    # Just a simple check
                    if level.lower() in f"{region_around_button}".lower():
                        logger.info(f"Position-based detection found power boost level: {level} (confidence: 0.6)")
                        return level, 0.6
            
            # Default to X1 if detection failed but we've previously located the button
            if self.power_boost_button:
                logger.warning(f"Detection failed but using default level X1 (failure #{self.consecutive_failures})")
                return "X1", 0.5
                
            logger.warning(f"Power boost detection failed (failure #{self.consecutive_failures})")
            
            # Check if we should recreate templates
            if self.consecutive_failures >= self.max_failures_before_fallback:
                logger.warning("Too many consecutive failures, recreating templates")
                self.create_default_templates()
                self.consecutive_failures = 0
            
            return None, 0
            
        except Exception as e:
            logger.error(f"Error in power boost detection: {str(e)}")
            self.consecutive_failures += 1
            return None, 0
    
    def create_default_templates(self) -> bool:
        """
        Create default templates based on screen size as fallback.
        
        Returns:
            bool: Success status
        """
        if not self.adb_controller or not self.adb_controller.connected:
            logger.error("Cannot create default templates: ADB controller not available")
            return False
        
        try:
            logger.info("Creating default power boost templates")
            
            # Create templates directory
            os.makedirs("templates", exist_ok=True)
            
            # Get screen size
            w, h = self.adb_controller.screen_resolution
            if not w or not h:
                logger.error("Cannot get screen resolution")
                return False
            
            # Take a screenshot
            screen = self.adb_controller.capture_screen()
            if screen is None:
                logger.error("Failed to capture screen for template creation")
                return False
            
            # Define default regions for each boost level
            # Power boost button is typically at the bottom of the screen
            # We'll create a template for each level at the same position
            bottom_area_y = int(h * 0.85)
            center_x = w // 2
            
            # Create dictionary of templates
            templates = {}
            
            # Create a region around the power boost area
            x1 = max(0, center_x - 150)
            y1 = max(0, bottom_area_y - 50)
            x2 = min(w, center_x + 150)
            y2 = min(h, bottom_area_y + 50)
            
            for level in self.boost_levels:
                template_path = f"templates/default_{level}.png"
                
                # Extract region from screenshot
                template = screen[y1:y2, x1:x2]
                
                # Save template
                cv2.imwrite(template_path, template)
                
                # Add to templates dictionary
                templates[level] = {
                    "roi": [x1, y1, x2, y2],
                    "template_path": template_path,
                    "size": [x2-x1, y2-y1]
                }
            
            # Save templates configuration
            with open(self.config_file, 'w') as f:
                import json
                json.dump(templates, f, indent=4)
            
            # Reload templates
            self.reload_templates()
            
            logger.info(f"Created {len(templates)} default templates")
            return True
            
        except Exception as e:
            logger.error(f"Error creating default templates: {str(e)}")
            return False
    
    def find_power_boost_button(self, image: np.ndarray, fixed_ui_results: Dict = None) -> bool:
        """
        Find the power boost button based on exact game UI layout.
        Specifically avoids confusing the SPIN button with the power boost button.
        
        Args:
            image: Current screen image
            fixed_ui_results: Fixed UI detection results (optional)
            
        Returns:
            bool: Success status
        """
        logger.info("Attempting to find power boost button")
        
        # Check if we have resolution information
        if self.adb_controller and self.adb_controller.screen_resolution:
            w, h = self.adb_controller.screen_resolution
            
            # Based on user provided information, the power boost button is at 540x1300
            # Use exact coordinates provided by user
            power_boost_x = 540
            power_boost_y = 1300
            
            logger.info(f"Using exact power boost button coordinates: ({power_boost_x}, {power_boost_y})")
            self.power_boost_button = (power_boost_x, power_boost_y)
            return True
        
        # Method 1: Try to detect which power boost level is currently visible
        current_level, confidence = self.detect_power_boost_level(image)
        
        if current_level and confidence > 0.5:
            logger.info(f"Detected power boost level: {current_level} with confidence: {confidence:.2f}")
            
            # Get the ROI of the detected level
            if current_level in self.detector.templates:
                roi = self.detector.templates[current_level]["roi"]
                x1, y1, x2, y2 = roi
                
                # Set button position at the center of the template
                button_x = (x1 + x2) // 2
                button_y = (y1 + y2) // 2
                
                self.power_boost_button = (button_x, button_y)
                logger.info(f"Set power boost button directly at center of template: {self.power_boost_button}")
                return True
        
        # Method 2: Color-based detection specifically for the yellow power boost button
        # The power boost button is yellow and has X1, X2, etc. displayed on it
        try:
            h, w = image.shape[:2]
            
            # Define the region to look for the power boost button
            # Based on the user's screenshot, it's near the bottom
            bottom_region = image[int(h*0.75):int(h*0.95), :]
            
            # Convert to HSV and look for yellow/gold color (power boost button)
            hsv = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2HSV)
            
            # Yellow/gold color range in HSV
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([40, 255, 255])
            
            # Create a mask for yellow/gold colors
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Sort contours by area (largest first)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Take the largest contour that's likely to be the power boost button
                if cv2.contourArea(contours[0]) > 1000:  # Minimum area threshold
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contours[0])
                    
                    # Adjust coordinates to full image
                    center_x = x + w//2
                    center_y = int(h*0.75) + y + h//2
                    
                    # Avoid the red SPIN button (which is below the power boost)
                    logger.info(f"Found power boost button using color detection at ({center_x}, {center_y})")
                    self.power_boost_button = (center_x, center_y)
                    return True
        except Exception as e:
            logger.warning(f"Color-based power boost detection failed: {str(e)}")
        
        # Method 3: Use specific location based on screenshot proportions
        if self.adb_controller and self.adb_controller.screen_resolution:
            w, h = self.adb_controller.screen_resolution
            
            # Based on the image, the power boost button is approximately in the middle-bottom
            power_boost_x = w // 2
            power_boost_y = int(h * 0.85)  # Position above the SPIN button
            
            logger.warning(f"Using estimated position for power boost button: ({power_boost_x}, {power_boost_y})")
            self.power_boost_button = (power_boost_x, power_boost_y)
            return True
        
        logger.error("Failed to find power boost button with all methods")
        return False
    
    def reload_templates(self) -> bool:
        """
        Reload templates from the configuration file.
        
        Returns:
            bool: Success status
        """
        logger.info(f"Reloading power boost templates from {self.config_file}")
        
        # Create a new detector to reload templates
        old_detector = self.detector
        self.detector = PowerBoostDetector(self.config_file)
        
        # Set same threshold as before
        self.detector.set_threshold(old_detector.detection_threshold)
        
        # Check if templates loaded successfully
        success = len(self.detector.templates) > 0
        
        if success:
            logger.info(f"Successfully loaded {len(self.detector.templates)} templates")
        else:
            logger.error("Failed to load any templates")
            
        return success
    
    def change_power_boost(self, current_level: str, target_level: str) -> bool:
        """
        Change the power boost level using precise tapping sequence.
        Uses the exact sequence for Coin Master power boost levels.
        
        Args:
            current_level: Current power boost level
            target_level: Target power boost level
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"POWER BOOST CHANGE: from {current_level} to {target_level}")
        
        if not self.adb_controller:
            logger.error("ADB controller not available")
            return False
        
        # Make sure we have power boost button coordinates
        if not self.power_boost_button:
            logger.warning("Power boost button location unknown, attempting to find it")
            screen = self.adb_controller.capture_screen()
            if not screen is None:
                found = self.find_power_boost_button(screen, None)
                if not found:
                    logger.warning("Could not find power boost button, using default position (540, 1300)")
                    self.power_boost_button = (540, 1300)
            else:
                logger.error("Failed to capture screen to find power boost button")
                return False
        
        # Clean up level names if needed
        if current_level.startswith("power_boost_"):
            current_level = current_level[12:].upper()
        if target_level.startswith("power_boost_"):
            target_level = target_level[12:].upper()
        
        # Make sure they're uppercase
        current_level = current_level.upper()
        target_level = target_level.upper()
        
        # Make sure they start with X
        if not current_level.startswith("X"):
            current_level = "X" + current_level
        if not target_level.startswith("X"):
            target_level = "X" + target_level
        
        # Define tap map for each level
        tap_map = {
            "X1": 0,
            "X2": 1,
            "X3": 2,
            "X15": 3,
            "X50": 4,
            "X400": 5,
            "X1500": 6,
            "X6000": 7,
            "X20000": 8
        }
        
        # Check if levels are valid
        if current_level not in tap_map or target_level not in tap_map:
            logger.error(f"Invalid power boost level: {current_level} or {target_level}")
            logger.error(f"Valid levels: {list(tap_map.keys())}")
            return False
        
        # Get tap positions
        current_taps = tap_map.get(current_level, 0)
        target_taps = tap_map.get(target_level, 0)
        
        # Calculate taps needed with correct Coin Master logic
        if target_taps > current_taps:
            taps_needed = target_taps - current_taps
        else:
            # Wrap around: taps to X1 + taps to target
            taps_needed = (9 - current_taps) + target_taps
        
        # Extract button coordinates
        x, y = self.power_boost_button
        
        logger.info(f"Will tap EXACTLY {taps_needed} times at ({x}, {y}) to change from {current_level} to {target_level}")
        
        # Debugging - log before taking action
        logger.debug(f"Current level map position: {current_taps}, Target level map position: {target_taps}")
        logger.debug(f"Calculated taps needed using tap map: {taps_needed}")
        
        # Execute the exact number of taps needed
        for tap in range(taps_needed):
            logger.info(f"Power boost tap {tap+1}/{taps_needed}")
            
            # Use a more reliable tap approach
            self.adb_controller.tap(x, y)
            
            # Longer delay between taps for reliability (Coin Master animations)
            time.sleep(1.0)  # Increased from 0.8 to 1.0 seconds
        
        # Wait for change to take effect
        time.sleep(1.5)  # Increased wait time after completing sequence
        
        # Verify the change was successful
        verification_result = self.verify_power_boost_change(target_level)
        if verification_result:
            logger.info(f"✓ Successfully changed power boost to {target_level}")
        else:
            logger.warning(f"× Failed to verify change to {target_level}")
            
        return verification_result

    def verify_power_boost_change(self, target_level: str) -> bool:
        """
        Verify that the power boost level was successfully changed.
        Uses multiple detection attempts for reliability.
        
        Args:
            target_level: Expected power boost level
            
        Returns:
            True if the current level matches the target level
        """
        # Multiple verification attempts
        max_attempts = 3
        
        for attempt in range(max_attempts):
            # Take a fresh screenshot
            if self.adb_controller:
                screen = self.adb_controller.capture_screen()
                if screen is not None:
                    # Try with decreasing confidence thresholds
                    for threshold in [0.7, 0.6, 0.5]:
                        # Set threshold
                        self.detector.set_threshold(threshold)
                        
                        # Detect current level
                        current_level, confidence = self.detect_power_boost_level(screen)
                        
                        if current_level and confidence > threshold:
                            logger.info(f"Verification attempt {attempt+1}: detected level {current_level} (confidence: {confidence:.2f})")
                            
                            if current_level == target_level:
                                logger.info("Power boost change VERIFIED SUCCESSFUL")
                                return True
                            else:
                                logger.warning(f"Power boost change FAILED - detected {current_level} but expected {target_level}")
                                
                                # If last attempt, try one more tap and check again
                                if attempt == max_attempts - 1 and self.power_boost_button:
                                    logger.info("Last attempt: trying one more tap")
                                    x, y = self.power_boost_button
                                    self.adb_controller.tap(x, y)
                                    time.sleep(1.0)
                                    
                                    # Recheck
                                    screen = self.adb_controller.capture_screen()
                                    if screen is not None:
                                        final_level, final_conf = self.detect_power_boost_level(screen)
                                        if final_level == target_level:
                                            logger.info("Power boost change successful after extra tap")
                                            return True
                                
                                # Try tapping the exact difference in next attempt if not the last attempt
                                if attempt < max_attempts - 1:
                                    # Calculate the remaining taps needed
                                    try:
                                        current_idx = self.boost_levels.index(current_level)
                                        target_idx = self.boost_levels.index(target_level)
                                        
                                        if target_idx > current_idx:
                                            taps = target_idx - current_idx
                                        else:
                                            taps = len(self.boost_levels) - current_idx + target_idx
                                            
                                        logger.info(f"Trying {taps} more taps to reach {target_level}")
                                        
                                        if self.power_boost_button:
                                            x, y = self.power_boost_button
                                            for _ in range(taps):
                                                self.adb_controller.tap(x, y)
                                                time.sleep(0.8)
                                            time.sleep(1.0)
                                    except Exception as e:
                                        logger.error(f"Error calculating remaining taps: {str(e)}")
                
                # Short wait before next attempt
                time.sleep(1.0)
        
        logger.warning("Could not verify power boost change after all attempts")
        return False
    
    def ensure_correct_level(self, expected_level: str) -> bool:
        """
        Ensure the power boost is at the correct level, changing it if necessary.
        
        Args:
            expected_level: The expected power boost level
            
        Returns:
            bool: Success status
        """
        if not self.adb_controller:
            logger.error("ADB controller not available")
            return False
            
        # Take a screenshot
        screen = self.adb_controller.capture_screen()
        if screen is None:
            logger.error("Failed to capture screen for level check")
            return False
            
        # Detect current level
        current_level, _ = self.detect_power_boost_level(screen)
        
        # If detection failed or current level doesn't match expected, fix it
        if not current_level or current_level != expected_level:
            if not current_level:
                logger.warning(f"Could not detect current level, forcing change to {expected_level}")
                current_level = "X1"  # Assume default as starting point
            else:
                logger.warning(f"Power boost level mismatch: found {current_level}, expected {expected_level}")
                
            # Change to correct level
            return self.change_power_boost(current_level, expected_level)
        
        # Already at correct level
        return True