import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "device_id": None,  # Use first available device
    "model_path": "models/my_model.pt",
    "detection_confidence": 0.5,
    "action_delay": 0.5,  # Delay between actions in seconds
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
    "debug_mode": False  # When true, shows detection visualization
}


class ConfigHandler:
    """
    Handles loading, saving, and accessing configuration
    """
    
    def __init__(self, config_path: str = "config/settings.json"):
        """
        Initialize the config handler.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or create default if not exists.
        
        Returns:
            Configuration dictionary
        """
        try:
            # Check if config file exists
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                
                # Merge with default config to ensure all required fields are present
                merged_config = DEFAULT_CONFIG.copy()
                merged_config.update(config)
                return merged_config
            else:
                # Create config directory if it doesn't exist
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                
                # Save default config
                self.save_config(DEFAULT_CONFIG)
                logger.info(f"Created default configuration at {self.config_path}")
                return DEFAULT_CONFIG.copy()
        
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            logger.info("Using default configuration")
            return DEFAULT_CONFIG.copy()
    
    def save_config(self, config: Dict[str, Any] = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save (uses current config if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create config directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Save config
            with open(self.config_path, 'w') as f:
                json.dump(config or self.config, f, indent=4)
            
            logger.info(f"Saved configuration to {self.config_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary with updates
        """
        self.config.update(updates)
    
    def save(self) -> bool:
        """
        Save current configuration to file.
        
        Returns:
            True if successful, False otherwise
        """
        return self.save_config(self.config)
    
    def reset_to_default(self) -> None:
        """
        Reset configuration to default.
        """
        self.config = DEFAULT_CONFIG.copy()
        self.save()
        logger.info("Reset configuration to default")
        
    def update_power_boost_sequence(self, new_sequence) -> bool:
        """
        Update power boost sequence in configuration and save to disk.
        
        Args:
            new_sequence: New power boost sequence list of dicts with 'level' and 'attacks'
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not new_sequence:
                logger.warning("Empty power boost sequence received, ignoring")
                return False
                
            # Log the update
            sequence_str = ', '.join([f"{item['level']}:{item['attacks']}" for item in new_sequence])
            logger.info(f"Updating power boost sequence in config: {sequence_str}")
            
            # Validate sequence
            for item in new_sequence:
                if 'level' not in item or 'attacks' not in item:
                    logger.error(f"Invalid sequence item: {item}")
                    return False
                    
                # Ensure attacks is an integer
                try:
                    item['attacks'] = int(item['attacks'])
                except (ValueError, TypeError):
                    logger.error(f"Invalid attacks value in {item}")
                    return False
            
            # 1. Update in-memory config
            self.config['power_boost_sequence'] = new_sequence
            
            # 2. Save to disk immediately
            result = self.save_config(self.config)
            
            logger.info(f"Power boost sequence updated in config: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to update power boost sequence in config: {str(e)}")
            return False