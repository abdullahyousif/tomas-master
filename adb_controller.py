import os
import subprocess
import time
import numpy as np
import cv2
import logging
from typing import Tuple, List, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

class ADBController:
    """
    Enhanced ADB controller with better error handling and more functionality.
    Handles all ADB operations including screen capture, tap simulation, and device management.
    """
    
    def __init__(self, device_id: str = None, adb_path: str = "adb"):
        """
        Initialize the ADB controller.
        
        Args:
            device_id: Specific device ID to connect to. If None, uses the first available device.
            adb_path: Path to ADB executable
        """
        self.device_id = device_id
        self.adb_path = adb_path
        self.connected = False
        self.screen_resolution = None
        self.device_info = {}
        self.initialize()
    
    def initialize(self) -> bool:
        """
        Initialize ADB connection and get device information.
        
        Returns:
            bool: True if initialization successful, False otherwise.
        """
        try:
            # Get list of connected devices
            result = subprocess.run(
                [self.adb_path, "devices"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Parse devices
            lines = result.stdout.strip().split('\n')[1:]  # Skip the first line (header)
            devices = [line.split('\t')[0] for line in lines if '\t' in line and 'device' in line]
            
            if not devices:
                logger.error("No devices connected")
                return False
            
            # Use specified device or first available
            if self.device_id is None:
                self.device_id = devices[0]
                logger.info(f"Using device: {self.device_id}")
            elif self.device_id not in devices:
                logger.error(f"Specified device {self.device_id} not found")
                return False
            
            # Get screen resolution
            self.screen_resolution = self._get_screen_resolution()
            if not self.screen_resolution:
                logger.error("Failed to get screen resolution")
                return False
            
            # Get device information
            self._get_device_info()
            
            logger.info(f"Device initialized with resolution: {self.screen_resolution}")
            logger.info(f"Device model: {self.device_info.get('model', 'Unknown')}")
            logger.info(f"Android version: {self.device_info.get('android_version', 'Unknown')}")
            
            self.connected = True
            return True
            
        except subprocess.SubprocessError as e:
            logger.error(f"ADB initialization error: {str(e)}")
            return False
    
    def _get_screen_resolution(self) -> Optional[Tuple[int, int]]:
        """
        Get screen resolution of the connected device.
        
        Returns:
            Tuple[int, int]: Width and height of the screen, or None if failed.
        """
        try:
            result = subprocess.run(
                [self.adb_path, "-s", self.device_id, "shell", "wm", "size"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Parse resolution
            output = result.stdout.strip()
            if "Physical size:" in output:
                size_str = output.split("Physical size:")[1].strip()
            else:
                size_str = output.split("Override size:")[1].strip()
            
            width, height = map(int, size_str.split('x'))
            return (width, height)
            
        except (subprocess.SubprocessError, ValueError, IndexError) as e:
            logger.error(f"Failed to get screen resolution: {str(e)}")
            return None
    
    def _get_device_info(self) -> None:
        """
        Get detailed information about the connected device.
        """
        try:
            # Get device model
            result = subprocess.run(
                [self.adb_path, "-s", self.device_id, "shell", "getprop", "ro.product.model"],
                capture_output=True,
                text=True,
                check=True
            )
            self.device_info["model"] = result.stdout.strip()
            
            # Get Android version
            result = subprocess.run(
                [self.adb_path, "-s", self.device_id, "shell", "getprop", "ro.build.version.release"],
                capture_output=True,
                text=True,
                check=True
            )
            self.device_info["android_version"] = result.stdout.strip()
            
            # Get device brand
            result = subprocess.run(
                [self.adb_path, "-s", self.device_id, "shell", "getprop", "ro.product.brand"],
                capture_output=True,
                text=True,
                check=True
            )
            self.device_info["brand"] = result.stdout.strip()
            
            # Get device serial
            result = subprocess.run(
                [self.adb_path, "-s", self.device_id, "shell", "getprop", "ro.serialno"],
                capture_output=True,
                text=True,
                check=True
            )
            self.device_info["serial"] = result.stdout.strip()
            
            # Get SDK version
            result = subprocess.run(
                [self.adb_path, "-s", self.device_id, "shell", "getprop", "ro.build.version.sdk"],
                capture_output=True,
                text=True,
                check=True
            )
            try:
                self.device_info["sdk_version"] = int(result.stdout.strip())
            except ValueError:
                self.device_info["sdk_version"] = 0
            
        except subprocess.SubprocessError as e:
            logger.warning(f"Failed to get complete device info: {str(e)}")
    
    def capture_screen(self, method: str = "auto") -> Optional[np.ndarray]:
        """
        Capture the current screen using ADB.
        
        Args:
            method: Capture method ('screencap', 'screenshot', 'auto')
            
        Returns:
            np.ndarray: OpenCV image of the screen, or None if capture failed.
        """
        if not self.connected:
            logger.error("Device not connected")
            return None
        
        # Try different methods based on what was requested or what works
        if method == "auto":
            # Try screencap first, fall back to screenshot
            img = self._capture_screen_via_screencap()
            if img is None:
                logger.info("Screencap failed, trying screenshot method")
                img = self._capture_screen_via_screenshot()
            return img
        elif method == "screencap":
            return self._capture_screen_via_screencap()
        elif method == "screenshot":
            return self._capture_screen_via_screenshot()
        else:
            logger.error(f"Unknown capture method: {method}")
            return None
    
    def _capture_screen_via_screencap(self) -> Optional[np.ndarray]:
        """
        Capture screen using the screencap command.
        
        Returns:
            np.ndarray: OpenCV image of the screen, or None if capture failed.
        """
        try:
            # Use pipe to avoid temporary files
            pipe = subprocess.Popen(
                [self.adb_path, "-s", self.device_id, "shell", "screencap", "-p"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Read raw screenshot data
            raw_screenshot, err = pipe.communicate()
            
            if pipe.returncode != 0:
                logger.error(f"Screencap error: {err.decode('utf-8')}")
                return None
            
            # Fix line endings for windows
            if os.name == "nt":
                raw_screenshot = raw_screenshot.replace(b'\r\n', b'\n')
            
            # Convert to numpy array
            np_arr = np.frombuffer(raw_screenshot, np.uint8)
            
            # Decode as image
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Failed to decode screenshot image")
                return None
                
            return img
            
        except Exception as e:
            logger.error(f"Screen capture via screencap error: {str(e)}")
            return None
    
    def _capture_screen_via_screenshot(self) -> Optional[np.ndarray]:
        """
        Capture screen using temporary file and pull.
        
        Returns:
            np.ndarray: OpenCV image of the screen, or None if capture failed.
        """
        try:
            # Use adb to capture screen to a temporary file
            subprocess.run(
                [self.adb_path, "-s", self.device_id, "shell", "screencap", "-p", "/sdcard/screen.png"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Pull the file from device
            subprocess.run(
                [self.adb_path, "-s", self.device_id, "pull", "/sdcard/screen.png", "screen.png"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Remove the file from device
            subprocess.run(
                [self.adb_path, "-s", self.device_id, "shell", "rm", "/sdcard/screen.png"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Read the image
            img = cv2.imread("screen.png")
            
            # Remove the local file
            if os.path.exists("screen.png"):
                os.remove("screen.png")
            
            if img is None:
                logger.error("Failed to read captured screen image")
                return None
                
            return img
            
        except Exception as e:
            logger.error(f"Screen capture via screenshot error: {str(e)}")
            return None
    
    def tap(self, x: int, y: int) -> bool:
        """
        Simulate a tap at the specified coordinates with enhanced reliability.
        
        Args:
            x: X coordinate for the tap
            y: Y coordinate for the tap
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            logger.error("Device not connected")
            return False
        
        # Ensure coordinates are integers
        x, y = int(x), int(y)
        
        # Validate coordinates against screen dimensions
        if self.screen_resolution:
            max_x, max_y = self.screen_resolution
            if x < 0 or x >= max_x or y < 0 or y >= max_y:
                logger.warning(f"Tap coordinates ({x}, {y}) out of bounds ({max_x}x{max_y})")
                # Clamp to valid range
                x = max(0, min(x, max_x - 1))
                y = max(0, min(y, max_y - 1))
                logger.info(f"Adjusted coordinates to ({x}, {y})")
        
        try:
            # Try multiple tap approaches for reliability
            
            # Method 1: Standard input tap
            logger.info(f"Tap method 1: Standard tap at ({x}, {y})")
            result1 = subprocess.run(
                [self.adb_path, "-s", self.device_id, "shell", "input", "tap", str(x), str(y)],
                capture_output=True,
                text=True
            )
            
            if result1.returncode != 0:
                logger.warning(f"Tap method 1 failed: {result1.stderr}")
                
                # Method 2: Direct shell command
                logger.info(f"Tap method 2: Direct shell at ({x}, {y})")
                cmd = f"input tap {x} {y}"
                self.run_shell_command(cmd)
                
                # Method 3: Short swipe (more reliable on some devices)
                logger.info(f"Tap method 3: Short swipe at ({x}, {y})")
                cmd = f"input swipe {x} {y} {x} {y} 100"
                self.run_shell_command(cmd)
            
            logger.info(f"Tap completed at ({x}, {y})")
            return True
            
        except Exception as e:
            logger.error(f"Tap error at ({x}, {y}): {str(e)}")
            return False
        
    def power_boost_tap_sequence(self, x: int, y: int, taps_needed: int) -> bool:
        """
        Execute a specialized tap sequence for changing power boost levels.
        
        Args:
            x: X coordinate for the power boost button
            y: Y coordinate for the power boost button
            taps_needed: Number of taps needed to reach the target level
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            logger.error("Device not connected")
            return False
        
        try:
            # Ensure coordinates are integers
            x, y = int(x), int(y)
            
            # Log operation details
            logger.info(f"Power boost tap sequence: {taps_needed} taps at ({x}, {y})")
            
            # Check if coordinates are within screen boundaries
            if self.screen_resolution:
                max_x, max_y = self.screen_resolution
                if x < 0 or x >= max_x or y < 0 or y >= max_y:
                    logger.warning(f"Power boost coordinates ({x}, {y}) outside screen bounds ({max_x}, {max_y})")
                    # Clamp to screen boundaries
                    x = max(0, min(x, max_x - 1))
                    y = max(0, min(y, max_y - 1))
                    logger.info(f"Adjusted coordinates to ({x}, {y})")
            
            # STEP 1: Initial activation tap with multiple methods
            logger.info("STEP 1: Initial activation tap")
            
            # Method 1: Standard tap
            self.tap(x, y)
            
            # Method 2: Direct shell command
            self.run_shell_command(f"input tap {x} {y}")
            
            # Method 3: Touchscreen input (sometimes more reliable)
            self.run_shell_command(f"input touchscreen tap {x} {y}")
            
            # Wait for UI to respond
            time.sleep(1.5)
            
            # STEP 2: Execute the required number of taps
            logger.info(f"STEP 2: Executing {taps_needed} taps")
            
            for i in range(taps_needed):
                logger.info(f"Power boost tap {i+1}/{taps_needed}")
                
                # Try multiple tap methods for each tap
                
                # Method 1: Standard tap
                self.tap(x, y)
                
                # Short delay between methods
                time.sleep(0.3)
                
                # Method 2: Tap and hold briefly for better recognition
                cmd = f"input swipe {x} {y} {x} {y} 100"
                self.run_shell_command(cmd)
                
                # Longer delay between taps
                time.sleep(1.0)
                
                # Log progress
                logger.info(f"Completed tap {i+1}")
            
            # STEP 3: Final verification tap with longer hold
            if taps_needed > 0:
                logger.info("STEP 3: Final verification tap")
                cmd = f"input swipe {x} {y} {x} {y} 300"
                self.run_shell_command(cmd)
                
                # Wait for UI to stabilize
                time.sleep(2.0)
            
            logger.info("Power boost tap sequence completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Power boost tap sequence error: {str(e)}")
            return False
        
# Add to adb_controller.py
    def power_boost_tap_sequence(self, x, y, taps_needed):
        """
        Specialized method for power boost taps with enhanced reliability.
        """
        logger.info(f"Executing power boost tap sequence: {taps_needed} taps at ({x}, {y})")
        
        try:
            # Make sure coordinates are integers
            x, y = int(x), int(y)
            
            # First tap to activate power boost UI
            self.tap(x, y)
            time.sleep(1.0)  # Wait for UI to appear
            
            # Execute precise number of taps with pauses between
            for i in range(taps_needed):
                logger.info(f"Power boost tap {i+1}/{taps_needed}")
                
                # Try multiple tap approaches for reliability
                self.tap(x, y)  # Standard tap
                time.sleep(0.3)  # Short pause
                
                # Also try direct shell command for reliability
                self.run_shell_command(f"input tap {x} {y}")
                
                # Wait between taps to let UI respond
                time.sleep(0.8)
                
            # Final verification tap and longer wait
            logger.info("Final verification tap")
            self.tap(x, y)
            time.sleep(1.5)
            
            return True
            
        except Exception as e:
            logger.error(f"Power boost tap sequence error: {str(e)}")
            return False
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> bool:
        """
        Perform a swipe gesture from (x1, y1) to (x2, y2).
        
        Args:
            x1: Starting X coordinate
            y1: Starting Y coordinate
            x2: Ending X coordinate
            y2: Ending Y coordinate
            duration_ms: Duration of the swipe in milliseconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            logger.error("Device not connected")
            return False
        
        try:
            # Use direct shell command for more reliable execution
            command = f"input swipe {x1} {y1} {x2} {y2} {duration_ms}"
            logger.debug(f"Executing swipe command: {command}")
            
            result = self.run_shell_command(command)
            if result is None:
                logger.error("Swipe command failed")
                return False
                
            logger.debug(f"Swipe from ({x1}, {y1}) to ({x2}, {y2}) in {duration_ms}ms")
            return True
            
        except Exception as e:
            logger.error(f"Swipe error: {str(e)}")
            return False

    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> bool:
        """
        Perform a long press at the specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            duration_ms: Duration of the press in milliseconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            logger.error("Device not connected")
            return False
        
        try:
            # Use input swipe command with same start and end coordinates instead of default long_press
            command = f"input swipe {x} {y} {x} {y} {duration_ms}"
            logger.debug(f"Executing long press command: {command}")
            
            result = self.run_shell_command(command)
            if result is None:
                logger.error("Long press command failed")
                return False
                
            logger.debug(f"Long press at ({x}, {y}) for {duration_ms}ms")
            return True
            
        except Exception as e:
            logger.error(f"Long press error: {str(e)}")
            return False
        
    def activate_auto_spin(self, x: int, y: int) -> bool:
        """
        Specialized method to activate auto-spin by trying multiple approaches.
        
        Args:
            x: X coordinate of the spin button
            y: Y coordinate of the spin button
            
        Returns:
            bool: True if commands executed successfully, False otherwise
        """
        if not self.connected:
            logger.error("Device not connected")
            return False
        
        try:
            # Try several approaches in sequence
            
            # First, try a regular tap
            logger.info(f"Auto-spin activation: Initial tap at ({x}, {y})")
            self.tap(x, y)
            time.sleep(0.5)
            
            # Then, try a direct swipe command with longer duration
            logger.info(f"Auto-spin activation: Long press via swipe at ({x}, {y})")
            command = f"input swipe {x} {y} {x} {y} 5000"
            self.run_shell_command(command)
            time.sleep(1.0)
            
            # Try another approach using monkey tool if available
            logger.info("Auto-spin activation: Trying alternative method")
            alt_command = f"input touchscreen swipe {x} {y} {x} {y} 5000"
            self.run_shell_command(alt_command)
            
            # Give time for auto-spin to activate
            time.sleep(3.0)
            
            return True
            
        except Exception as e:
            logger.error(f"Auto-spin activation error: {str(e)}")
            return False
        
    def force_auto_spin_activation(self, x: int, y: int) -> bool:
        """
        Force auto-spin activation using multiple aggressive techniques.
        This specialized method tries various approaches to ensure auto-spin activates.
        
        Args:
            x: X coordinate of the spin button
            y: Y coordinate of the spin button
            
        Returns:
            bool: True if commands executed successfully, False otherwise
        """
        logger.info(f"Force auto-spin activation at ({x}, {y}) using multiple methods")
        
        try:
            # Method 1: Direct tap to ensure button is responsive
            logger.info("Method 1: Initial tap")
            self.tap(x, y)
            time.sleep(1.5)
            
            # Method 2: Standard long press with extended duration
            logger.info("Method 2: Standard long press (10 seconds)")
            self.long_press(x, y, duration_ms=10000)
            time.sleep(2.0)
            
            # Method 3: Force tap and hold with even longer duration
            logger.info("Method 3: Force tap and hold (15 seconds)")
            self.force_tap_and_hold(x, y, duration_ms=15000)
            time.sleep(2.0)
            
            # Method 4: Using touchscreen input directly
            logger.info("Method 4: Direct touchscreen input method")
            alt_cmd = f"input touchscreen swipe {x} {y} {x} {y} 12000"
            self.run_shell_command(alt_cmd)
            time.sleep(2.0)
            
            # Method 5: Try using sendevent for low-level touch events (if available)
            try:
                logger.info("Method 5: Trying sendevent approach")
                # Get device touch input device path
                devices_cmd = "getevent -p | grep -e 'add device' -e 'ABS_MT_POSITION'"
                devices_result = self.run_shell_command(devices_cmd)
                
                if devices_result and "ABS_MT_POSITION" in devices_result:
                    # Extract device path
                    device_lines = devices_result.split('\n')
                    device_path = None
                    
                    for i, line in enumerate(device_lines):
                        if "add device" in line:
                            for j in range(i+1, len(device_lines)):
                                if "ABS_MT_POSITION" in device_lines[j]:
                                    device_path = line.split(":")[-1].strip()
                                    break
                    
                    if device_path:
                        logger.info(f"Using touch device: {device_path}")
                        # Send touch down event
                        self.run_shell_command(f"sendevent {device_path} 3 57 0")
                        self.run_shell_command(f"sendevent {device_path} 3 53 {x}")
                        self.run_shell_command(f"sendevent {device_path} 3 54 {y}")
                        self.run_shell_command(f"sendevent {device_path} 1 330 1")
                        self.run_shell_command(f"sendevent {device_path} 0 0 0")
                        
                        # Hold for 10 seconds
                        logger.info("Holding touch via sendevent for 10 seconds")
                        time.sleep(10.0)
                        
                        # Send touch up event
                        self.run_shell_command(f"sendevent {device_path} 3 57 4294967295")
                        self.run_shell_command(f"sendevent {device_path} 1 330 0")
                        self.run_shell_command(f"sendevent {device_path} 0 0 0")
                else:
                    logger.info("Could not find touch input device for sendevent approach")
            except Exception as e:
                logger.warning(f"Sendevent approach failed: {str(e)}")
            
            # Wait to see if any method worked
            logger.info("Waiting for auto-spin to activate...")
            time.sleep(5.0)
            
            return True
        except Exception as e:
            logger.error(f"Force auto-spin activation error: {str(e)}")
            return False


    def force_tap_and_hold(self, x: int, y: int, duration_ms: int = 8000) -> bool:
        """
        Force tap and hold at specified coordinates using multiple approaches for maximum reliability.
        
        Args:
            x: X coordinate to tap
            y: Y coordinate to tap
            duration_ms: Duration to hold in milliseconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            logger.error("Device not connected")
            return False
        
        try:
            # Log the attempt
            logger.info(f"Executing force tap and hold at ({x}, {y}) for {duration_ms}ms")
            
            # Approach 1: Standard swipe command
            swipe_cmd = f"input swipe {x} {y} {x} {y} {duration_ms}"
            self.run_shell_command(swipe_cmd)
            logger.info("Completed swipe command approach")
            
            # Allow some time between approaches
            time.sleep(0.5)
            
            # Approach 2: Touchscreen command (more reliable on some devices)
            touch_cmd = f"input touchscreen swipe {x} {y} {x} {y} {duration_ms}"
            self.run_shell_command(touch_cmd)
            logger.info("Completed touchscreen command approach")
            
            # Log completion
            logger.info(f"Force tap and hold completed at ({x}, {y})")
            return True
        
        except Exception as e:
            logger.error(f"Force tap and hold error: {str(e)}")
            return False
    
    def get_connected_devices(self) -> List[str]:
        """
        Get a list of all connected devices.
        
        Returns:
            List[str]: Device IDs of connected devices
        """
        try:
            result = subprocess.run(
                [self.adb_path, "devices"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            lines = result.stdout.strip().split('\n')[1:]
            devices = [line.split('\t')[0] for line in lines if '\t' in line and 'device' in line]
            return devices
            
        except subprocess.SubprocessError as e:
            logger.error(f"Error getting devices: {str(e)}")
            return []
    
    def set_device(self, device_id: str) -> bool:
        """
        Set a new device to control.
        
        Args:
            device_id: Device ID to switch to
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.device_id = device_id
        self.connected = False
        return self.initialize()
    
    def run_shell_command(self, command: str) -> Optional[str]:
        """
        Run a shell command on the device.
        
        Args:
            command: Shell command to run
            
        Returns:
            Command output if successful, None otherwise
        """
        if not self.connected:
            logger.error("Device not connected")
            return None
            
        try:
            result = subprocess.run(
                [self.adb_path, "-s", self.device_id, "shell", command],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
            
        except subprocess.SubprocessError as e:
            logger.error(f"Shell command error: {str(e)}")
            return None
    
    def push_file(self, local_path: str, device_path: str) -> bool:
        """
        Push a file to the device.
        
        Args:
            local_path: Path to local file
            device_path: Path on the device
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error("Device not connected")
            return False
            
        try:
            subprocess.run(
                [self.adb_path, "-s", self.device_id, "push", local_path, device_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info(f"Pushed {local_path} to {device_path}")
            return True
            
        except subprocess.SubprocessError as e:
            logger.error(f"Push file error: {str(e)}")
            return False
    
    def pull_file(self, device_path: str, local_path: str) -> bool:
        """
        Pull a file from the device.
        
        Args:
            device_path: Path on the device
            local_path: Path to local file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error("Device not connected")
            return False
            
        try:
            subprocess.run(
                [self.adb_path, "-s", self.device_id, "pull", device_path, local_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info(f"Pulled {device_path} to {local_path}")
            return True
            
        except subprocess.SubprocessError as e:
            logger.error(f"Pull file error: {str(e)}")
            return False
    
    def start_app(self, package_name: str, activity_name: Optional[str] = None) -> bool:
        """
        Start an app on the device.
        
        Args:
            package_name: Package name of the app
            activity_name: Activity name to start (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error("Device not connected")
            return False
            
        try:
            if activity_name:
                # Start specific activity
                command = f"am start -n {package_name}/{activity_name}"
            else:
                # Start main activity
                command = f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1"
                
            result = self.run_shell_command(command)
            
            if result is None or "Error" in result:
                logger.error(f"Failed to start app: {result}")
                return False
                
            logger.info(f"Started app {package_name}")
            return True
            
        except Exception as e:
            logger.error(f"Start app error: {str(e)}")
            return False
    
    def stop_app(self, package_name: str) -> bool:
        """
        Stop an app on the device.
        
        Args:
            package_name: Package name of the app
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error("Device not connected")
            return False
            
        try:
            command = f"am force-stop {package_name}"
            result = self.run_shell_command(command)
            
            logger.info(f"Stopped app {package_name}")
            return True
            
        except Exception as e:
            logger.error(f"Stop app error: {str(e)}")
            return False
    
    def press_key(self, key_code: int) -> bool:
        """
        Simulate pressing a key on the device.
        
        Args:
            key_code: Android key code
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error("Device not connected")
            return False
            
        try:
            subprocess.run(
                [self.adb_path, "-s", self.device_id, "shell", "input", "keyevent", str(key_code)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.debug(f"Pressed key {key_code}")
            return True
            
        except subprocess.SubprocessError as e:
            logger.error(f"Key press error: {str(e)}")
            return False
    
    def get_battery_info(self) -> Dict[str, Any]:
        """
        Get battery information from the device.
        
        Returns:
            Dictionary with battery information (level, temperature, status)
        """
        if not self.connected:
            logger.error("Device not connected")
            return {}
            
        try:
            result = self.run_shell_command("dumpsys battery")
            
            if result is None:
                return {}
                
            # Parse battery info
            battery_info = {}
            for line in result.split('\n'):
                line = line.strip()
                if "level:" in line:
                    try:
                        battery_info["level"] = int(line.split("level:")[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif "temperature:" in line:
                    try:
                        temp = int(line.split("temperature:")[1].strip())
                        battery_info["temperature"] = temp / 10.0  # Convert to degrees Celsius
                    except (ValueError, IndexError):
                        pass
                elif "status:" in line:
                    try:
                        status_code = int(line.split("status:")[1].strip())
                        statuses = {
                            1: "Unknown",
                            2: "Charging",
                            3: "Discharging",
                            4: "Not charging",
                            5: "Full"
                        }
                        battery_info["status"] = statuses.get(status_code, "Unknown")
                    except (ValueError, IndexError):
                        pass
            
            return battery_info
            
        except Exception as e:
            logger.error(f"Get battery info error: {str(e)}")
            return {}
    
    def get_device_temperature(self) -> Dict[str, float]:
        """
        Get device temperature information.
        
        Returns:
            Dictionary with temperature readings
        """
        if not self.connected:
            logger.error("Device not connected")
            return {}
            
        try:
            result = self.run_shell_command("dumpsys thermalservice")
            
            if result is None:
                return {}
                
            # Parse thermal info
            temperatures = {}
            
            # Look for the thermal status section
            if "Current temperatures:" in result:
                temp_section = result.split("Current temperatures:")[1]
                temp_section = temp_section.split("Current cooling devices:")[0]
                
                for line in temp_section.split('\n'):
                    line = line.strip()
                    if line and "=" in line:
                        parts = line.split('=')
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            try:
                                value = float(parts[1].strip().split(' ')[0])
                                temperatures[name] = value
                            except (ValueError, IndexError):
                                pass
            
            return temperatures
            
        except Exception as e:
            logger.error(f"Get device temperature error: {str(e)}")
            return {}
    
    # Common Android key codes for convenience
    KEY_BACK = 4
    KEY_HOME = 3
    KEY_MENU = 82
    KEY_POWER = 26
    KEY_VOLUME_UP = 24
    KEY_VOLUME_DOWN = 25