import logging
import time
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional, Any, Callable
import threading

from adb_controller import ADBController
from detector import YOLODetector, Detection

logger = logging.getLogger(__name__)

class GameState(Enum):
    """
    Enum representing the different states of the game
    """
    IDLE = auto()
    SPINNING = auto()
    ATTACKING = auto()
    RAIDING = auto()
    ADJUSTING_POWER_BOOST = auto()
    ERROR_RECOVERY = auto()      # New state for handling errors
    RESTARTING_GAME = auto()     # New state for restarting the game if needed


class PowerBoostLevel(Enum):
    """
    Enum representing the different power boost levels and their tap counts
    """
    X1 = 0       # Base level
    X2 = 1       # 1 tap from base
    X3 = 2       # 2 taps from base
    X15 = 3      # 3 taps from base
    X50 = 4      # 4 taps from base
    X400 = 5     # 5 taps from base
    X1500 = 6    # 6 taps from base
    X6000 = 7    # 7 taps from base
    X20000 = 8   # 8 taps from base


class PowerBoostStrategy(Enum):
    """
    Different strategies for power boost management
    """
    SEQUENTIAL = auto()      # Follow sequence in order
    ADAPTIVE = auto()        # Adapt based on available coins
    CONSERVATIVE = auto()    # Focus on lower boosts for efficiency
    AGGRESSIVE = auto()      # Focus on higher boosts for max damage


class PowerBoostSequence:
    """
    Class to manage the power boost sequence with different strategies
    """
    
    def __init__(self, config: Dict[str, Any] = None, strategy: PowerBoostStrategy = PowerBoostStrategy.SEQUENTIAL):
        """
        Initialize the power boost sequence manager.
        
        Args:
            config: Configuration dictionary with custom sequence settings
            strategy: Power boost strategy to use
        """
        # Default sequence as specified in the requirements
        self.default_sequence = [
            {"level": PowerBoostLevel.X1, "attacks": 8},
            {"level": PowerBoostLevel.X15, "attacks": 3},
            {"level": PowerBoostLevel.X50, "attacks": 4},
            {"level": PowerBoostLevel.X400, "attacks": 3},
            {"level": PowerBoostLevel.X1500, "attacks": 1},
            {"level": PowerBoostLevel.X6000, "attacks": 1},
            {"level": PowerBoostLevel.X20000, "attacks": 1}
        ]
        
        # Strategies
        self.strategies = {
            PowerBoostStrategy.SEQUENTIAL: self.default_sequence,
            PowerBoostStrategy.CONSERVATIVE: [
                {"level": PowerBoostLevel.X1, "attacks": 12},
                {"level": PowerBoostLevel.X15, "attacks": 6},
                {"level": PowerBoostLevel.X50, "attacks": 3},
                {"level": PowerBoostLevel.X400, "attacks": 2}
            ],
            PowerBoostStrategy.AGGRESSIVE: [
                {"level": PowerBoostLevel.X15, "attacks": 4},
                {"level": PowerBoostLevel.X50, "attacks": 4},
                {"level": PowerBoostLevel.X400, "attacks": 3},
                {"level": PowerBoostLevel.X1500, "attacks": 2},
                {"level": PowerBoostLevel.X6000, "attacks": 2},
                {"level": PowerBoostLevel.X20000, "attacks": 2}
            ]
        }
        
        # Use provided config or the selected strategy
        self.sequence = None
        if config and "power_boost_sequence" in config:
            # Convert string level names to PowerBoostLevel enum values
            raw_sequence = config.get("power_boost_sequence", [])
            self.sequence = []
            
            for item in raw_sequence:
                # Check if level is already a PowerBoostLevel enum
                if isinstance(item["level"], PowerBoostLevel):
                    level = item["level"]
                else:
                    # Convert string to enum
                    try:
                        level_str = item["level"]
                        # Handle string with or without 'X' prefix
                        if level_str.startswith('X'):
                            enum_name = level_str  # Already has X prefix
                        else:
                            enum_name = 'X' + level_str
                        
                        # Get enum value
                        level = PowerBoostLevel[enum_name]
                    except (KeyError, ValueError) as e:
                        logger.error(f"Invalid power boost level: {item['level']}, using X1")
                        level = PowerBoostLevel.X1
                
                # Add to sequence
                self.sequence.append({
                    "level": level,
                    "attacks": item["attacks"]
                })
        
        # If no sequence was provided or conversion failed, use strategy
        if not self.sequence:
            self.strategy = strategy
            self.sequence = self.strategies.get(strategy, self.default_sequence)
        
        # Initialize index and counters
        self.current_index = 0
        self.attacks_completed = 0
        
        # Debug log
        logger_msg = f"PowerBoostSequence initialized with {len(self.sequence)} levels: "
        logger_msg += ", ".join([f"{item['level'].name}:{item['attacks']}" for item in self.sequence])
        logger.info(logger_msg)
    
    def get_current_level(self) -> PowerBoostLevel:
        """
        Get the current power boost level in the sequence.
        
        Returns:
            Current PowerBoostLevel
        """
        if not self.sequence or self.current_index >= len(self.sequence):
            return PowerBoostLevel.X1  # Default if sequence is invalid
            
        return self.sequence[self.current_index]["level"]
    
    def get_target_attacks(self) -> int:
        """
        Get the target number of attacks for the current level.
        
        Returns:
            Number of attacks to complete before moving to next level
        """
        if not self.sequence or self.current_index >= len(self.sequence):
            return 1  # Default if sequence is invalid
        
        return self.sequence[self.current_index]["attacks"]
    
    def get_taps_to_reach_level(self, target_level: PowerBoostLevel, current_level: PowerBoostLevel) -> int:
        """
        Calculate the number of taps needed to reach the target level from current level.
        
        Args:
            target_level: Target PowerBoostLevel
            current_level: Current PowerBoostLevel
            
        Returns:
            Number of taps needed
        """
        return target_level.value - current_level.value
    
    def register_attack(self) -> bool:
        """
        Register a completed attack and check if it's time to move to the next boost level.
        
        Returns:
            True if it's time to move to the next level, False otherwise
        """
        self.attacks_completed += 1
        logger.info(f"Attack registered: {self.attacks_completed}/{self.get_target_attacks()}")
        
        # Check if target attacks reached
        if self.attacks_completed >= self.get_target_attacks():
            # Reset counter
            self.attacks_completed = 0
            
            # Move to next level
            self.current_index = (self.current_index + 1) % len(self.sequence)
            logger.info(f"Moving to next level: {self.get_current_level().name}")
            return True
        
        return False
    
    def reset(self) -> None:
        """
        Reset the sequence to the beginning.
        """
        self.current_index = 0
        self.attacks_completed = 0
        logger.info("Power boost sequence reset")
    
    def change_strategy(self, strategy: PowerBoostStrategy) -> None:
        """
        Change the power boost strategy.
        
        Args:
            strategy: New PowerBoostStrategy to use
        """
        self.strategy = strategy
        self.sequence = self.strategies.get(strategy, self.default_sequence)
        self.reset()
        logger.info(f"Power boost strategy changed to {strategy.name}")
    
    def update_sequence(self, new_sequence):
        """
        Update the power boost sequence.
        
        Args:
            new_sequence: New sequence to use
        """
        # Convert string level names to PowerBoostLevel enum values
        self.sequence = []
        
        for item in new_sequence:
            # Check if level is already a PowerBoostLevel enum
            if isinstance(item["level"], PowerBoostLevel):
                level = item["level"]
            else:
                # Convert string to enum
                try:
                    level_str = item["level"]
                    # Handle string with or without 'X' prefix
                    if level_str.startswith('X'):
                        enum_name = level_str  # Already has X prefix
                    else:
                        enum_name = 'X' + level_str
                    
                    # Get enum value
                    level = PowerBoostLevel[enum_name]
                except (KeyError, ValueError) as e:
                    logger.error(f"Invalid power boost level: {item['level']}, using X1")
                    level = PowerBoostLevel.X1
            
            # Add to sequence
            self.sequence.append({
                "level": level,
                "attacks": item["attacks"]
            })
        
        # Reset index and counter
        self.current_index = 0
        self.attacks_completed = 0
        
        # Debug log
        logger_msg = f"PowerBoostSequence updated with {len(self.sequence)} levels: "
        logger_msg += ", ".join([f"{item['level'].name}:{item['attacks']}" for item in self.sequence])
        logger.info(logger_msg)


class GameController:
    """
    Enhanced controller class for the Coin Master game automation
    """
    
    def __init__(self, 
                 adb_controller: ADBController, 
                 detector: YOLODetector,
                 config: Dict[str, Any] = None):
        """
        Initialize the game controller.
        
        Args:
            adb_controller: ADBController instance
            detector: YOLODetector instance
            config: Configuration dictionary
        """
        self.adb = adb_controller
        self.detector = detector
        self.config = config or {}
        
        # Initialize game state
        self.state = GameState.IDLE
        self.current_power_boost = PowerBoostLevel.X1
        
        # Initialize power boost strategy and sequence
        strategy_name = self.config.get("power_boost_strategy", "SEQUENTIAL")
        try:
            strategy = PowerBoostStrategy[strategy_name]
        except (KeyError, ValueError):
            strategy = PowerBoostStrategy.SEQUENTIAL
            
        self.power_boost_sequence = PowerBoostSequence(config, strategy)
        
        # Stats tracking
        self.stats = {
            "attacks_completed": 0,
            "raids_completed": 0,
            "spins_completed": 0,
            "power_boosts_changed": 0,
            "errors_recovered": 0,
            "game_restarts": 0,
            "session_start_time": time.time(),
            "last_detection_time": 0,
            "detection_success_rate": 1.0,
            "coins_earned_estimate": 0
        }
        
        # Get screen dimensions
        if self.adb.screen_resolution:
            self.screen_width, self.screen_height = self.adb.screen_resolution
        else:
            logger.warning("Screen resolution not available, using default 1080x1920")
            self.screen_width, self.screen_height = 1080, 1920
            
        # Helper variables
        self.running = False
        self.last_action_time = 0
        self.action_delay = config.get("action_delay", 0.5) if config else 0.5
        self.max_error_recovery_attempts = config.get("max_error_recovery_attempts", 3)
        self.current_error_recovery_attempts = 0
        self.state_transition_time = time.time()
        self.state_history = []
        
        # Coordinates for power boost button (will be determined dynamically)
        self.power_boost_coords = None
        
        # Detection reliability tracking
        self.detection_history = []
        self.max_detection_history = 100
        
        # Callbacks
        self.state_change_callback = None
        self.screenshot_callback = None
        self.error_callback = None
        
        # Auto-restart flags
        self.auto_restart_enabled = config.get("auto_restart_enabled", False)
        self.auto_restart_interval = config.get("auto_restart_interval", 3600)  # 1 hour default
        self.last_restart_time = time.time()
        
        # Detection failure handling
        self.consecutive_detection_failures = 0
        self.max_detection_failures = config.get("max_detection_failures", 5)
        
        # Thread for game loop
        self.game_thread = None
        
    def start(self) -> None:
        """
        Start the bot.
        """
        logger.info("Starting Coin Master Bot")
        self.running = True
        self.state = GameState.SPINNING
        self.state_transition_time = time.time()
        self.stats["session_start_time"] = time.time()
        
        # Start in a separate thread
        self.game_thread = threading.Thread(target=self.run_game_loop, daemon=True)
        self.game_thread.start()
    
    def stop(self) -> None:
        """
        Stop the bot.
        """
        logger.info("Stopping Coin Master Bot")
        self.running = False
        self.state = GameState.IDLE
        
        # Wait for thread to finish
        if self.game_thread and self.game_thread.is_alive():
            self.game_thread.join(timeout=2.0)
    
    def run_game_loop(self) -> None:
        """
        Main game loop. Continuously captures the screen, detects objects, and takes actions.
        """
        while self.running:
            try:
                # Check if we need to restart the game
                if self.auto_restart_enabled and time.time() - self.last_restart_time > self.auto_restart_interval:
                    logger.info("Auto-restart interval reached, restarting game")
                    self.state = GameState.RESTARTING_GAME
                    self.last_restart_time = time.time()
                
                # Capture screen
                screen = self.adb.capture_screen()
                if screen is None:
                    logger.error("Failed to capture screen, attempting recovery...")
                    self._handle_detection_failure()
                    continue
                
                # Detect objects
                detections = self.detector.detect(screen)
                
                # Update detection tracking
                self._update_detection_tracking(len(detections) > 0)
                
                # Count objects by class
                object_counts = self.detector.count_objects_by_class(detections)
                logger.debug(f"Detected objects: {object_counts}")
                
                # Save screenshot if callback is set
                if self.screenshot_callback:
                    self.screenshot_callback(screen, detections, self.state)
                
                # Check for elements that should be tapped immediately
                tap_targets = self.detector.get_tap_targets(detections)
                if tap_targets:
                    self.consecutive_detection_failures = 0  # Reset failure counter on successful detection
                    for class_name, x, y in tap_targets:
                        logger.info(f"Tapping {class_name} at ({int(x)}, {int(y)})")
                        self.adb.tap(int(x), int(y))
                        time.sleep(self.action_delay)
                
                # Process game state
                if self.state != GameState.ERROR_RECOVERY and self.state != GameState.RESTARTING_GAME:
                    self._process_game_state(detections, object_counts)
                else:
                    # Handle recovery states
                    if self.state == GameState.ERROR_RECOVERY:
                        self._handle_error_recovery()
                    elif self.state == GameState.RESTARTING_GAME:
                        self._restart_game()
                
                # Check for state timeout
                self._check_state_timeout()
                
                # Short delay to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in game loop: {str(e)}")
                self._enter_error_recovery(f"Exception: {str(e)}")
                time.sleep(1)
    
    def _update_detection_tracking(self, detection_success: bool) -> None:
        """
        Update detection success tracking.
        
        Args:
            detection_success: Whether object detection was successful
        """
        # Add current detection result to history
        self.detection_history.append(detection_success)
        
        # Trim history if it exceeds maximum length
        if len(self.detection_history) > self.max_detection_history:
            self.detection_history.pop(0)
        
        # Calculate success rate
        if self.detection_history:
            self.stats["detection_success_rate"] = sum(self.detection_history) / len(self.detection_history)
        
        # Update last detection time if successful
        if detection_success:
            self.stats["last_detection_time"] = time.time()
            self.consecutive_detection_failures = 0
        else:
            self.consecutive_detection_failures += 1
    
    def _handle_detection_failure(self) -> None:
        """
        Handle screen capture or detection failure.
        """
        logger.warning(f"Detection failure #{self.consecutive_detection_failures}")
        
        # If too many consecutive failures, enter error recovery
        if self.consecutive_detection_failures >= self.max_detection_failures:
            self._enter_error_recovery("Too many consecutive detection failures")
            return
        
        # Try to reset ADB connection
        if self.consecutive_detection_failures % 3 == 0:
            logger.info("Attempting to reset ADB connection")
            try:
                self.adb.initialize()
            except Exception as e:
                logger.error(f"Failed to reset ADB connection: {str(e)}")
        
        # Short delay before retrying
        time.sleep(1)
    
    def _enter_error_recovery(self, reason: str) -> None:
        """
        Enter error recovery state.
        
        Args:
            reason: Reason for entering error recovery
        """
        logger.warning(f"Entering error recovery mode: {reason}")
        self.state = GameState.ERROR_RECOVERY
        self.state_transition_time = time.time()
        self.current_error_recovery_attempts = 0
        self.stats["errors_recovered"] += 1
        
        # Call error callback if set
        if self.error_callback:
            self.error_callback(reason)
    
    def _handle_error_recovery(self) -> None:
        """
        Handle error recovery state.
        """
        self.current_error_recovery_attempts += 1
        logger.info(f"Error recovery attempt {self.current_error_recovery_attempts}/{self.max_error_recovery_attempts}")
        
        # Try different recovery actions based on attempt number
        if self.current_error_recovery_attempts == 1:
            # First attempt: Just wait a bit and go back to spinning
            logger.info("Recovery action: Waiting and returning to spinning state")
            time.sleep(2)
            self.state = GameState.SPINNING
            
        elif self.current_error_recovery_attempts == 2:
            # Second attempt: Try tapping in the center of the screen
            logger.info("Recovery action: Tapping screen center")
            self.adb.tap(self.screen_width // 2, self.screen_height // 2)
            time.sleep(1)
            
            # Try closing any possible dialogs by tapping the usual X button locations
            x_locations = [(self.screen_width - 50), (self.screen_width - 100)]
            y_locations = [50, 100, 150]
            
            for x in x_locations:
                for y in y_locations:
                    logger.info(f"Recovery action: Tapping possible X button at ({x}, {y})")
                    self.adb.tap(x, y)
                    time.sleep(0.5)
            
            self.state = GameState.SPINNING
            
        elif self.current_error_recovery_attempts == 3:
            # Third attempt: Try swiping and more taps
            logger.info("Recovery action: Swiping across screen")
            self.adb.swipe(
                self.screen_width // 4, 
                self.screen_height // 2,
                self.screen_width * 3 // 4, 
                self.screen_height // 2
            )
            time.sleep(1)
            
            # Tap in all four corners
            corners = [
                (50, 50),  # Top-left
                (self.screen_width - 50, 50),  # Top-right
                (50, self.screen_height - 50),  # Bottom-left
                (self.screen_width - 50, self.screen_height - 50)  # Bottom-right
            ]
            
            for x, y in corners:
                logger.info(f"Recovery action: Tapping corner at ({x}, {y})")
                self.adb.tap(x, y)
                time.sleep(0.5)
                
            self.state = GameState.SPINNING
            
        else:
            # Final attempt: Restart the game
            logger.info("Recovery action: Restarting the game")
            self.state = GameState.RESTARTING_GAME
            self.stats["game_restarts"] += 1
    
    def _restart_game(self) -> None:
        """
        Restart the Coin Master app.
        """
        logger.info("Restarting Coin Master")
        
        try:
            # Close the app
            logger.info("Closing Coin Master app")
            self.adb.run_shell_command("am force-stop com.moonactive.coinmaster")
            time.sleep(2)
            
            # Start the app
            logger.info("Starting Coin Master app")
            self.adb.run_shell_command("am start -n com.moonactive.coinmaster/com.moonactive.coinmaster.MainActivity")
            
            # Wait for app to load
            logger.info("Waiting for app to load")
            time.sleep(10)
            
            # Reset state
            self.state = GameState.SPINNING
            self.last_restart_time = time.time()
            self.state_transition_time = time.time()
            self.consecutive_detection_failures = 0
            
        except Exception as e:
            logger.error(f"Failed to restart game: {str(e)}")
            # If we can't restart, go back to error recovery
            self.state = GameState.ERROR_RECOVERY
    
    def _check_state_timeout(self) -> None:
        """
        Check if the current state has timed out and needs recovery.
        """
        # Define timeouts for different states
        timeouts = {
            GameState.SPINNING: 60,  # 1 minute
            GameState.ATTACKING: 30,  # 30 seconds
            GameState.RAIDING: 30,    # 30 seconds
            GameState.ADJUSTING_POWER_BOOST: 15,  # 15 seconds
            GameState.ERROR_RECOVERY: 60,  # 1 minute
            GameState.RESTARTING_GAME: 60  # 1 minute
        }
        
        current_timeout = timeouts.get(self.state, 30)  # Default timeout
        time_in_state = time.time() - self.state_transition_time
        
        # If state has timed out
        if time_in_state > current_timeout:
            logger.warning(f"State {self.state.name} timed out after {time_in_state:.1f} seconds")
            
            # Record state transition
            self._transition_state(GameState.ERROR_RECOVERY)
            
            # Reset state timeout
            self.state_transition_time = time.time()
    
    def _transition_state(self, new_state: GameState) -> None:
        """
        Transition to a new state with proper tracking.
        
        Args:
            new_state: New game state
        """
        if new_state != self.state:
            old_state = self.state
            self.state = new_state
            self.state_transition_time = time.time()
            
            # Record in history
            self.state_history.append((old_state, new_state, time.time()))
            if len(self.state_history) > 50:  # Keep only recent history
                self.state_history.pop(0)
            
            logger.info(f"State transition: {old_state.name} -> {new_state.name}")
            
            # Call state change callback if set
            if self.state_change_callback:
                self.state_change_callback(old_state, new_state)
    
    # In the _process_game_state method, modify the SPINNING state section

    def _process_game_state(self, detections: List[Detection], object_counts: Dict[str, int]) -> None:
        """
        Process the current game state and take appropriate actions.
        
        Args:
            detections: List of Detection objects
            object_counts: Dictionary mapping class names to counts
        """
        if self.state == GameState.SPINNING:
            # Check for attack condition (3 Thor hammers)
            if object_counts.get("attack_symbol", 0) >= 3:
                logger.info("Attack detected! (3 hammers)")
                self._transition_state(GameState.ATTACKING)
                time.sleep(self.action_delay * 2)  # Wait for animation
                return
                
            # Check for raid condition (3 pigs)
            if object_counts.get("raid_symbol", 0) >= 3:
                logger.info("Raid detected! (3 pigs)")
                self._transition_state(GameState.RAIDING)
                time.sleep(self.action_delay * 2)  # Wait for animation
                return
            
            # Check if auto-spin is active by looking for STOP button or auto-spin indicator
            stop_buttons = [det for det in detections if det.class_name == "stop_button" or 
                                                    det.class_name == "auto_spin_active" or 
                                                    det.class_name == "autospin_active"]
            if stop_buttons:
                # Auto-spin is already active, do nothing
                logger.info("Auto-spin already active (STOP or auto-spin indicator detected)")
                return
            
            # Look for autospin_button first (new button we've trained the model to recognize)
            autospin_buttons = [det for det in detections if det.class_name == "autospin_button"]
            if autospin_buttons:
                # Found the dedicated auto-spin button, tap it directly
                autospin_button = autospin_buttons[0]
                x, y = int(autospin_button.center_x), int(autospin_button.center_y)
                logger.info(f"Tapping dedicated auto-spin button at ({x}, {y})")
                self.adb.tap(x, y)  # Just a tap, not a long press
                self.last_action_time = time.time()
                self.stats["spins_completed"] += 1
                time.sleep(self.action_delay)
                return
                
            # If autospin button not found, check for regular spin button and force auto-spin mode
            spin_buttons = [det for det in detections if det.class_name == "spin_button"]
            if spin_buttons:
                # Found regular spin button, need to activate auto-spin
                spin_button = spin_buttons[0]
                x, y = int(spin_button.center_x), int(spin_button.center_y)
                
                logger.info(f"Found spin button at ({x}, {y}) - FORCING AUTO-SPIN")
                
                # Wait for any previous actions to complete
                time.sleep(1.0)
                
                try:
                    # Try multiple approaches in sequence with pauses between them
                    
                    # First, try a regular tap to make sure the button is responsive
                    logger.info("Step 1: Initial tap")
                    self.adb.tap(x, y)
                    time.sleep(1.5)  # Longer wait to ensure game responds
                    
                    # Then, try a direct swipe command with longer duration
                    logger.info("Step 2: Direct long press command (10 seconds)")
                    cmd = f"input swipe {x} {y} {x} {y} 10000"
                    self.adb.run_shell_command(cmd)
                    
                    # Long wait after the long press to let the game respond
                    logger.info("Waiting for auto-spin to activate...")
                    time.sleep(5.0)  # Much longer wait
                    
                    # Record the action
                    self.last_action_time = time.time()
                    self.stats["spins_completed"] += 1
                
                except Exception as e:
                    logger.error(f"Auto-spin activation error: {str(e)}")
                    
        elif self.state == GameState.ATTACKING:
            # Look for attack aiming icons and tap one
            aiming_icons = [det for det in detections if det.class_name == "attack_aiming_icon"]
            ok_buttons = [det for det in detections if det.class_name == "ok_button"]

            # Flag to track if we've just tapped an aiming icon
            just_tapped_aiming = False

            if aiming_icons:
                # Add this debug line
                logger.info(f"Found {len(aiming_icons)} aiming icons in attack mode")
        
                # Pick the first aiming icon
                icon = aiming_icons[0]
                x, y = int(icon.center_x), int(icon.center_y)
        
                # Tap the icon directly here instead of waiting for tap_targets
                logger.info(f"Tapping aiming icon at ({x}, {y})")
                self.adb.tap(x, y)
                just_tapped_aiming = True
                self.last_action_time = time.time()
                
                # Count this as a successful attack when tapping aiming icon
                logger.info("Attack performed")
                self.stats["attacks_completed"] += 1
                
                # Safely call the coin estimation function if it exists
                if hasattr(self, '_estimate_attack_coins') and callable(getattr(self, '_estimate_attack_coins')):
                    self.stats["coins_earned_estimate"] += self._estimate_attack_coins()
                
                # Register the attack and check if we need to change power boost level
                logger.info("Registering attack completion")
                if self.power_boost_sequence.register_attack():
                    logger.info("Time to change power boost level")
                    self._transition_state(GameState.ADJUSTING_POWER_BOOST)
                else:
                    # Since we've tapped an aiming icon, just wait for the next detection cycle
                    # Don't switch states yet
                    pass
                
                time.sleep(self.action_delay)  # Give time for the tap to register

            # If OK button is found, it means the attack animation is complete
            elif ok_buttons:
                # Add this debug line
                logger.info(f"OK button found, attack animation complete")
        
                # Tap the OK button directly
                ok_button = ok_buttons[0]
                x, y = int(ok_button.center_x), int(ok_button.center_y)
                logger.info(f"Tapping OK button at ({x}, {y})")
                self.adb.tap(x, y)
                time.sleep(self.action_delay)  # Give time for the tap to register
                
                # Go back to spinning after tapping OK
                logger.info("Returning to spinning state")
                self._transition_state(GameState.SPINNING)

            # If no attack elements are found and we didn't just tap an aiming icon,
            # we might have missed the attack screen
            elif not aiming_icons and not just_tapped_aiming and time.time() - self.last_action_time > 5:
                logger.warning("No attack elements found, returning to spinning")
                self._transition_state(GameState.SPINNING)
            
        elif self.state == GameState.RAIDING:
            # Look for raid hole icons and tap one
            raid_holes = [det for det in detections if det.class_name == "raid_hole_icon"]
            raid_x_icons = [det for det in detections if det.class_name == "raid_x_icon"]
            
            # Flag to track if we've just tapped
            just_tapped = False

            if raid_holes:
                # Log detection
                logger.info(f"Found {len(raid_holes)} raid holes in raid mode")
                
                # Pick the first raid hole
                raid_hole = raid_holes[0]
                x, y = int(raid_hole.center_x), int(raid_hole.center_y)
                
                # Tap the raid hole directly
                logger.info(f"Tapping raid hole at ({x}, {y})")
                self.adb.tap(x, y)
                just_tapped = True
                self.last_action_time = time.time()
                time.sleep(self.action_delay * 2)  # Give time for the tap to register
            
            # If X icon is found, it means we should exit the raid
            elif raid_x_icons:
                logger.info(f"Found raid X icon, tapping to exit raid")
                
                # Tap the X button directly
                raid_x = raid_x_icons[0]
                x, y = int(raid_x.center_x), int(raid_x.center_y)
                logger.info(f"Tapping raid X icon at ({x}, {y})")
                self.adb.tap(x, y)
                time.sleep(self.action_delay)  # Give time for the tap to register
                
                # Count this as a completed raid
                self.stats["raids_completed"] += 1
                
                # Go back to spinning after tapping X
                logger.info("Returning to spinning state after raid")
                self._transition_state(GameState.SPINNING)
            
            # If no raid elements are found and we didn't just tap something,
            # we might have missed the raid screen
            elif not just_tapped and time.time() - self.last_action_time > 5:
                logger.warning("No raid elements found, returning to spinning")
                self._transition_state(GameState.SPINNING)
        
        elif self.state == GameState.ADJUSTING_POWER_BOOST:
            # Get target power boost level
            target_level = self.power_boost_sequence.get_current_level()
            
            # Find the power boost button if not already known
            if not self.power_boost_coords:
                power_boost_buttons = [det for det in detections if det.class_name == "power_boost_button"]
                if power_boost_buttons:
                    button = power_boost_buttons[0]
                    self.power_boost_coords = (int(button.center_x), int(button.center_y))
                    logger.info(f"Found power boost button at {self.power_boost_coords}")
            
            # If we have the button coordinates, tap to adjust power boost
            if self.power_boost_coords:
                # Calculate taps needed
                taps_needed = self.power_boost_sequence.get_taps_to_reach_level(
                    target_level, self.current_power_boost
                )
                
                logger.info(f"Adjusting power boost from {self.current_power_boost.name} to {target_level.name} ({taps_needed} taps)")
                
                # Tap the power boost button the required number of times
                for _ in range(taps_needed):
                    self.adb.tap(*self.power_boost_coords)
                    time.sleep(self.action_delay)
                    self.stats["power_boosts_changed"] += 1
                
                # Update current power boost level
                self.current_power_boost = target_level
                
                # Go back to spinning
                self._transition_state(GameState.SPINNING)
            else:
                # If can't find power boost button, go back to spinning
                logger.warning("Cannot find power boost button, returning to spinning")
                self._transition_state(GameState.SPINNING)
                # Add to game_logic.py in GameController class
    def update_power_boost_sequence(self, new_sequence):
        """
        Update the power boost sequence from configuration.
        
        Args:
            new_sequence: New sequence configuration
            
        Returns:
            bool: Success status
        """
        try:
            if not new_sequence:
                logger.warning("Empty power boost sequence received")
                return False
                
            # Convert string level names to PowerBoostLevel enum if needed
            processed_sequence = []
            
            for item in new_sequence:
                # Check level format and convert if needed
                if isinstance(item["level"], str):
                    try:
                        level_str = item["level"]
                        if level_str.startswith('X'):
                            enum_name = level_str
                        else:
                            enum_name = 'X' + level_str
                        
                        level = PowerBoostLevel[enum_name]
                    except (KeyError, ValueError):
                        logger.warning(f"Invalid power boost level string: {item['level']}, using X1")
                        level = PowerBoostLevel.X1
                elif isinstance(item["level"], PowerBoostLevel):
                    level = item["level"]
                else:
                    logger.warning(f"Unknown level type: {type(item['level'])}, using X1")
                    level = PowerBoostLevel.X1
                    
                processed_sequence.append({
                    "level": level,
                    "attacks": item["attacks"]
                })
            
            # Log the update
            sequence_str = ', '.join([f"{item['level']}:{item['attacks']}" for item in processed_sequence])
            logger.info(f"Updating GameController power boost sequence to: {sequence_str}")
            
            # Update the sequence in PowerBoostSequence
            if hasattr(self, 'power_boost_sequence'):
                self.power_boost_sequence.sequence = processed_sequence
                self.power_boost_sequence.current_index = 0
                self.power_boost_sequence.attacks_completed = 0
                logger.info("Successfully updated power_boost_sequence object")
            
            return True
        except Exception as e:
            logger.error(f"Error updating GameController sequence: {str(e)}")
            return False