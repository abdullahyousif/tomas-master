import json
import os
import sys
import time
import logging
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from adb_controller import ADBController
from detector import YOLODetector
from fixed_ui_detector import FixedUIDetector, ROISelector
from game_logic import GameController, GameState, PowerBoostLevel
from power_boost_configurator import PowerBoostConfigurator
from power_boost_detector import PowerBoostTemplateEditor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("coin_master_bot.log")
    ]
)
logger = logging.getLogger(__name__)

class LogHandler(logging.Handler):
    """Custom logging handler that forwards logs to the GUI"""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        
    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
            self.text_widget.configure(state='disabled')
        # Call append from the main thread
        self.text_widget.after(0, append)

class CoinMasterBotGUI:
    """Simple GUI for the Coin Master Bot"""
    
    def __init__(self, root):
        """Initialize the GUI"""
        self.root = root
        self.root.title("Coin Master Bot")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Variables
        self.adb = None
        self.detector = None
        self.fixed_ui_detector = None
        self.game_controller = None
        self.roi_selector_window = None
        
        self.running = False
        self.bot_running = False
        self.model_path = tk.StringVar(value="models/my_model.pt")
        self.device_id = tk.StringVar()
        self.status_var = tk.StringVar(value="Not Connected")
        self.power_boost_var = tk.StringVar(value="X1")
        self.attacks_var = tk.StringVar(value="0/0")
        self.raids_var = tk.StringVar(value="0")
        self.screen_capture_var = tk.BooleanVar(value=True)
        self.use_fixed_ui_var = tk.BooleanVar(value=True)
        
        # Debug variable
        self.last_debug_time = 0
        
        # Load configuration if available
        self.config = self.load_config_file()
        
        # Create GUI elements
        self.create_menu()
        self.create_layout()
        
        # Setup log handler
        log_handler = LogHandler(self.log_text)
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(log_handler)
        
        # Initialize directory structure
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("templates", exist_ok=True)
        os.makedirs("config", exist_ok=True)
        
        # Add welcome message
        self.log("Welcome to Coin Master Bot")
        self.log("Connect to a device and load a model to begin")

    def open_power_boost_template_editor(self):
        """Open the Power Boost Template Editor in a new window"""
        self.log("Opening Power Boost Template Editor")
        
        try:
            # Create a new window
            boost_editor_window = tk.Toplevel(self.root)
            boost_editor_window.title("Power Boost Template Editor")
            boost_editor_window.geometry("1200x800")
            
            # Create Template Editor and pass ADB controller reference
            editor = PowerBoostTemplateEditor(boost_editor_window, adb_controller=self.adb)
            
            # If we have a device connected, the screenshot functionality will be enabled
            if self.adb and self.adb.connected:
                self.log("Device connected - screenshot button enabled in Power Boost Template Editor")
            else:
                # No device connected, show message
                messagebox.showinfo("No Device", 
                                "No device connected. You can still load images manually in the editor.\n"
                                "Connect a device in the main interface to enable the Take Screenshot button.")
        
        except Exception as e:
            self.log(f"Error opening Power Boost Template Editor: {str(e)}")
            messagebox.showerror("Error", f"Failed to open Power Boost Template Editor: {str(e)}")
    
    def load_config_file(self):
        """Load configuration from file if available."""
        config_path = "config/settings.json"
        
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                return config
            except Exception as e:
                print(f"Error loading configuration: {str(e)}")
        
        # Default configuration
        return {
            "device_id": None,
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
            "debug_mode": False
        }
    
    def create_menu(self):
        """Create the application menu"""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Model", command=self.load_model_dialog)
        file_menu.add_command(label="Save Config", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Device menu
        device_menu = tk.Menu(menubar, tearoff=0)
        device_menu.add_command(label="Refresh Devices", command=self.refresh_devices)
        menubar.add_cascade(label="Device", menu=device_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Fixed UI Editor", command=self.open_fixed_ui_editor)
        tools_menu.add_command(label="Power Boost Config", command=self.open_power_boost_config)
        tools_menu.add_command(label="Power Boost Template Editor", command=self.open_power_boost_template_editor)  # Add this line
        tools_menu.add_command(label="Debug Power Boost", command=self.debug_power_boost_sequence)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_layout(self):
        """Create the main layout"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side (controls and status)
        left_frame = ttk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Model section
        model_frame = ttk.LabelFrame(left_frame, text="Model", padding="10")
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="Model Path:").pack(anchor=tk.W)
        model_path_frame = ttk.Frame(model_frame)
        model_path_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(model_path_frame, textvariable=self.model_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_path_frame, text="Browse", command=self.load_model_dialog).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(model_frame, text="Load Model", command=self.load_model).pack(fill=tk.X)
        
        # Device section
        device_frame = ttk.LabelFrame(left_frame, text="Device", padding="10")
        device_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(device_frame, text="Device:").pack(anchor=tk.W)
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_id)
        self.device_combo.pack(fill=tk.X, pady=5)
        
        device_buttons = ttk.Frame(device_frame)
        device_buttons.pack(fill=tk.X)
        ttk.Button(device_buttons, text="Refresh", command=self.refresh_devices).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(device_buttons, text="Connect", command=self.connect_device).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Control section
        control_frame = ttk.LabelFrame(left_frame, text="Control", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        # Add Fixed UI Editor button
        ttk.Button(
            control_frame, 
            text="Fixed UI Editor", 
            command=self.open_fixed_ui_editor
        ).pack(fill=tk.X, pady=2)
        
        # Add Power Boost Config button
        ttk.Button(
            control_frame,
            text="Power Boost Config",
            command=self.open_power_boost_config
        ).pack(fill=tk.X, pady=2)
        
        # Add Power Boost Template Editor button
        ttk.Button(
            control_frame,
            text="Power Boost Template Editor",
            command=self.open_power_boost_template_editor
        ).pack(fill=tk.X, pady=2)
        

        ttk.Button(
            control_frame,
            text="Check & Sync Sequence Everywhere",
            command=self.check_and_sync_sequence
        ).pack(fill=tk.X, pady=2)

        ## Replace the current button with this in ui_handler.py
        ttk.Button(
            control_frame,
            text="Debug Attack Counter",
            command=self.debug_attack_counter  # Call our own method instead of directly calling game controller
        ).pack(fill=tk.X, pady=2)

        # Add this button to control_frame in ui_handler.py
        ttk.Button(
            control_frame,
            text="Sync Counter with Sequence",
            command=self.sync_attack_counter_with_sequence
        ).pack(fill=tk.X, pady=2)
        
        # Add Test Power Boost Change button
        ttk.Button(
            control_frame,
            text="Test Power Boost Change",
            command=self.test_power_boost_change
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            control_frame,
            text="Reload Power Boost Templates",
            command=self.reload_power_boost_templates
        ).pack(fill=tk.X, pady=2)
        
        # Add Fixed UI detection checkbox
        ttk.Checkbutton(
            control_frame, 
            text="Use Fixed UI Detection", 
            variable=self.use_fixed_ui_var
        ).pack(anchor=tk.W, pady=2)

        self.start_button = ttk.Button(control_frame, text="Start Bot", command=self.start_bot, state=tk.DISABLED)
        self.start_button.pack(fill=tk.X, pady=2)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Bot", command=self.stop_bot, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)
        
        ttk.Checkbutton(control_frame, text="Enable Screen Capture", variable=self.screen_capture_var).pack(anchor=tk.W)
        
        # Status section
        status_frame = ttk.LabelFrame(left_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(status_frame, text="Bot Status:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(status_frame, text="Power Boost:").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.power_boost_var).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(status_frame, text="Attacks:").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.attacks_var).grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(status_frame, text="Raids:").grid(row=3, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.raids_var).grid(row=3, column=1, sticky=tk.W)
        
        for child in status_frame.winfo_children():
            child.grid_configure(padx=5, pady=2)
        
        # Right side (screen capture and logs)
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Screen capture
        self.screen_frame = ttk.LabelFrame(right_frame, text="Screen Capture")
        self.screen_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.screen_label = ttk.Label(self.screen_frame)
        self.screen_label.pack(fill=tk.BOTH, expand=True)
        
        # Logs
        log_frame = ttk.LabelFrame(right_frame, text="Logs")
        log_frame.pack(fill=tk.BOTH, pady=(0, 5))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.statusbar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)


    # And add the corresponding method:
    def reload_power_boost_templates(self):
        """Reload power boost templates from file."""
        if not self.game_controller:
            self.log("Game controller not available")
            return
            
        try:
            if hasattr(self.game_controller, 'power_boost_manager'):
                success = self.game_controller.power_boost_manager.reload_templates()
                if success:
                    self.log("Power boost templates reloaded successfully")
                else:
                    self.log("Failed to reload power boost templates - file not found")
            else:
                self.log("Power boost manager not initialized")
        except Exception as e:
            self.log(f"Error reloading power boost templates: {str(e)}")

    def reload_power_boost_templates(self):
        """Reload power boost templates from file."""
        if not self.game_controller:
            self.log("Game controller not available")
            return
                
        try:
            if hasattr(self.game_controller, 'power_boost_manager'):
                success = self.game_controller.power_boost_manager.reload_templates()
                if success:
                    self.log("Power boost templates reloaded successfully")
                else:
                    self.log("Failed to reload power boost templates - file not found")
            else:
                self.log("Power boost manager not initialized")
        except Exception as e:
            self.log(f"Error reloading power boost templates: {str(e)}")

    def force_full_sequence_update(self):
        """Force update of sequence in all locations"""
        try:
            # Get current sequence
            sequence = self.config.get('power_boost_sequence', [])
            if not sequence:
                self.log("No sequence in current config")
                return
            
            sequence_str = ', '.join([f"{item['level']}:{item['attacks']}" for item in sequence])
            self.log(f"Forcing update of sequence: {sequence_str}")
            
            # 1. Update root config.json
            try:
                with open("config.json", 'r') as f:
                    root_config = json.load(f)
                
                root_config['power_boost_sequence'] = sequence
                
                with open("config.json", 'w') as f:
                    json.dump(root_config, f, indent=2)
                
                self.log("✓ Updated root config.json")
            except Exception as e:
                self.log(f"Error updating root config.json: {str(e)}")
            
            # 2. Update config/settings.json
            try:
                os.makedirs("config", exist_ok=True)
                
                if os.path.exists("config/settings.json"):
                    with open("config/settings.json", 'r') as f:
                        settings_config = json.load(f)
                else:
                    settings_config = {}
                
                settings_config['power_boost_sequence'] = sequence
                
                with open("config/settings.json", 'w') as f:
                    json.dump(settings_config, f, indent=2)
                
                self.log("✓ Updated config/settings.json")
            except Exception as e:
                self.log(f"Error updating config/settings.json: {str(e)}")
            
            # 3. Update game controller
            if self.game_controller:
                if hasattr(self.game_controller, 'update_power_boost_sequence'):
                    result = self.game_controller.update_power_boost_sequence(sequence)
                    self.log(f"✓ Updated game controller sequence: {result}")
                else:
                    # Direct update
                    self.log("Using direct game controller update")
                    self.game_controller.power_boost_sequence = sequence
                    self.game_controller.current_sequence_index = 0
                    self.game_controller.attacks_in_current_level = 0
                    self.log("✓ Direct game controller update completed")
            
            self.log("Full sequence update completed")
            messagebox.showinfo("Success", "Power boost sequence updated in all locations")
        except Exception as e:
            self.log(f"Error in full sequence update: {str(e)}")
            messagebox.showerror("Error", f"Sequence update failed: {str(e)}")
    
    def refresh_devices(self):
        """Refresh the list of connected devices"""
        self.log("Refreshing devices...")
        
        try:
            if self.adb is None:
                self.adb = ADBController()
            
            devices = self.adb.get_connected_devices()
            
            if not devices:
                self.log("No devices found")
                messagebox.showinfo("No Devices", "No devices connected.\nMake sure ADB is installed and your device is connected with USB debugging enabled.")
                self.device_combo['values'] = []
                return
            
            self.log(f"Found {len(devices)} device(s)")
            self.device_combo['values'] = devices
            
            # Select first device if none selected
            if not self.device_id.get() and devices:
                self.device_id.set(devices[0])
                
        except Exception as e:
            self.log(f"Error refreshing devices: {str(e)}")
            messagebox.showerror("Error", f"Failed to refresh devices: {str(e)}")
    
    def connect_device(self):
        """Connect to the selected device"""
        device_id = self.device_id.get()
        
        if not device_id:
            messagebox.showinfo("No Device", "Please select a device first.")
            return
        
        self.log(f"Connecting to device: {device_id}")
        
        try:
            if self.adb is None:
                self.adb = ADBController(device_id)
            else:
                self.adb.set_device(device_id)
            
            if not self.adb.connected:
                self.log("Failed to connect to device")
                messagebox.showerror("Connection Failed", "Failed to connect to the selected device.")
                return
            
            self.log(f"Connected to {device_id}")
            self.log(f"Screen resolution: {self.adb.screen_resolution}")
            
            # Take a test screenshot
            self.log("Taking test screenshot...")
            self.take_screenshot()
            
            # Initialize fixed UI detector if enabled
            if self.use_fixed_ui_var.get() and self.fixed_ui_detector is None:
                self.log("Initializing fixed UI detector...")
                self.fixed_ui_detector = FixedUIDetector()
            
            # Enable model loading if not already loaded
            if self.detector is None:
                self.status_var.set("Connected to device")
            else:
                # If model already loaded, enable start button
                self.start_button.config(state=tk.NORMAL)
                self.status_var.set("Ready")
            
        except Exception as e:
            self.log(f"Error connecting to device: {str(e)}")
            messagebox.showerror("Error", f"Failed to connect to device: {str(e)}")
    
    def load_model_dialog(self):
        """Open file dialog to select model file"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pt"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.model_path.set(file_path)
    
    def load_model(self):
        """Load the YOLOv5/YOLOv11 model"""
        model_path = self.model_path.get()
        
        if not model_path:
            messagebox.showinfo("No Model", "Please specify a model file path.")
            return
        
        if not os.path.exists(model_path):
            messagebox.showerror("File Not Found", f"Model file not found at {model_path}")
            return
        
        self.log(f"Loading model from {model_path}...")
        
        try:
            self.detector = YOLODetector(
                model_path=model_path,
                conf_threshold=0.5
            )
            
            self.log("Model loaded successfully")
            
            # Enable start button if device is connected
            if self.adb and self.adb.connected:
                self.start_button.config(state=tk.NORMAL)
                self.status_var.set("Ready")
            else:
                self.status_var.set("Device not connected")
            
        except Exception as e:
            self.log(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def open_fixed_ui_editor(self):
        """Open the Fixed UI Editor in a new window"""
        if self.roi_selector_window is not None and self.roi_selector_window.winfo_exists():
            # If window already exists, just raise it
            self.roi_selector_window.lift()
            self.roi_selector_window.focus_force()
            return
        
        self.log("Opening Fixed UI Editor")
        
        try:
            # Create a new window
            self.roi_selector_window = tk.Toplevel(self.root)
            self.roi_selector_window.title("Fixed UI Editor")
            self.roi_selector_window.geometry("1200x800")
            
            # Create ROI Selector and pass ADB controller reference
            self.roi_selector = ROISelector(self.roi_selector_window, adb_controller=self.adb)
            
            # If we have a device connected, we don't need to immediately take a screenshot
            # as the user can now do this directly from the Fixed UI Editor interface
            if self.adb and self.adb.connected:
                self.log("Device connected - screenshot button enabled in Fixed UI Editor")
            else:
                # No device connected, show message
                messagebox.showinfo("No Device", "No device connected. You can still load images manually in the editor.\nConnect a device in the main interface to enable the Take Screenshot button.")
        
        except Exception as e:
            self.log(f"Error opening Fixed UI Editor: {str(e)}")
            messagebox.showerror("Error", f"Failed to open Fixed UI Editor: {str(e)}")
            # Reset the window reference
            self.roi_selector_window = None
    
    def open_power_boost_config(self):
        """Open the Power Boost Configuration dialog"""
        self.log("Opening Power Boost Configuration")
        
        try:
            # Create Power Boost Configurator
            PowerBoostConfigurator(self.root, self.config, self.on_config_save)
            
        except Exception as e:
            self.log(f"Error opening Power Boost Configurator: {str(e)}")
            messagebox.showerror("Error", f"Failed to open Power Boost Configurator: {str(e)}")
    
    def on_config_save(self, updated_config):
        """Handle saving of updated configuration from Power Boost Configurator"""
        # Update the config
        self.config = updated_config
        
        # Log the sequence being saved
        sequence = self.config.get('power_boost_sequence', [])
        sequence_str = ', '.join([f"{item['level']}:{item['attacks']}" for item in sequence])
        self.log(f"Power Boost Config saved: {sequence_str}")
        
        # Save to file
        self.save_config()
        
        # Update game controller if running
        if self.game_controller and self.bot_running:
            self.log("Updating game controller with new sequence")
            
            # Reset attack counters
            if hasattr(self.game_controller, 'attacks_in_current_level'):
                self.game_controller.attacks_in_current_level = 0
                self.log("Reset attacks_in_current_level to 0")
                
            if hasattr(self.game_controller, 'current_sequence_index'):
                self.game_controller.current_sequence_index = 0
                self.log("Reset current_sequence_index to 0")
                
            # Update sequence
            self.update_power_boost_sequence()
            
            # Reset to first level in sequence
            if sequence and hasattr(self.game_controller, 'current_power_boost_level'):
                self.game_controller.current_power_boost_level = sequence[0]['level']
                self.log(f"Reset current_power_boost_level to {sequence[0]['level']}")
                
            if sequence and hasattr(self.game_controller, 'target_attacks_for_level'):
                self.game_controller.target_attacks_for_level = sequence[0]['attacks']
                self.log(f"Set target_attacks_for_level to {sequence[0]['attacks']}")
            
            # Show debug info
            self.debug_attack_counter()
        else:
            self.log("Game controller not running, changes will apply on next start")

    def test_power_boost_change(self):
        """Test power boost change function."""
        if not self.game_controller:
            self.log("Game controller not available")
            return
            
        try:
            # Ask user for target level
            from tkinter import simpledialog
            target_level = simpledialog.askstring(
                "Power Boost Test", 
                "Enter target power boost level (e.g. X15):",
                initialvalue="X15"
            )
            
            if not target_level:
                return
                
            # Call the manual function
            if hasattr(self.game_controller, 'manual_change_power_boost'):
                success = self.game_controller.manual_change_power_boost(target_level)
                if success:
                    self.log(f"Power boost changed to {target_level}")
                else:
                    self.log(f"Failed to change power boost to {target_level}")
            else:
                self.log("Manual power boost change function not available")
                
                # Fallback to direct power boost manager call if available
                if hasattr(self.game_controller, 'power_boost_manager'):
                    pbm = self.game_controller.power_boost_manager
                    
                    # Take a screenshot to find button if needed
                    screen = self.adb.capture_screen()
                    if not pbm.power_boost_button:
                        self.log("Looking for power boost button")
                        pbm.find_power_boost_button(screen, getattr(self.game_controller, 'last_fixed_ui_results', None))
                    
                    if pbm.power_boost_button:
                        self.log(f"Power boost button found at {pbm.power_boost_button}")
                        current_level = getattr(self.game_controller, 'current_power_boost_level', "X1")
                        self.log(f"Attempting to change from {current_level} to {target_level}")
                        success = pbm.change_power_boost(current_level, target_level)
                        
                        if success:
                            self.log(f"Power boost changed to {target_level}")
                        else:
                            self.log("Failed to change power boost")
                    else:
                        self.log("Power boost button not found")
                else:
                    self.log("Power boost manager not available")
        except Exception as e:
            self.log(f"Error in power boost test: {str(e)}")
    
    def update_power_boost_sequence(self):
        """Update power boost sequence in the game controller if it's running"""
        if not self.game_controller or not self.bot_running:
            self.log("Game controller not running, power boost sequence will be applied on next start")
            return
        
        try:
            # Get the current sequence from config
            sequence = self.config.get('power_boost_sequence', [])
            
            # Log the sequence we're updating to
            sequence_str = ', '.join([f"{item['level']}:{item['attacks']}" for item in sequence])
            self.log(f"Updating power boost sequence to: {sequence_str}")
            
            # Update the game controller's power boost sequence
            if hasattr(self.game_controller, 'power_boost_sequence'):
                # Check if it has the update_sequence method we added
                if hasattr(self.game_controller.power_boost_sequence, 'update_sequence'):
                    self.game_controller.power_boost_sequence.update_sequence(sequence)
                    self.log("Power boost sequence updated successfully")
                else:
                    # Fallback if update_sequence method is not available
                    self.game_controller.power_boost_sequence.sequence = sequence
                    self.game_controller.power_boost_sequence.current_index = 0
                    self.game_controller.power_boost_sequence.attacks_completed = 0
                    self.log("Power boost sequence updated using fallback method")
        except Exception as e:
            self.log(f"Error updating power boost sequence: {str(e)}")

    def check_config_files(self):
        """Check config files for sequence consistency"""
        self.log("===== CHECKING CONFIGURATION FILES =====")
        
        # Check root config.json
        try:
            with open("config.json", 'r') as f:
                root_config = json.load(f)
                root_sequence = root_config.get('power_boost_sequence', [])
                self.log(f"Root config.json sequence: {len(root_sequence)} levels")
                for item in root_sequence:
                    self.log(f"  {item['level']}: {item['attacks']} attacks")
        except Exception as e:
            self.log(f"Error reading root config.json: {str(e)}")
        
        # Check config/settings.json
        try:
            if os.path.exists("config/settings.json"):
                with open("config/settings.json", 'r') as f:
                    config_settings = json.load(f)
                    settings_sequence = config_settings.get('power_boost_sequence', [])
                    self.log(f"config/settings.json sequence: {len(settings_sequence)} levels")
                    for item in settings_sequence:
                        self.log(f"  {item['level']}: {item['attacks']} attacks")
            else:
                self.log("config/settings.json does not exist")
        except Exception as e:
            self.log(f"Error reading config/settings.json: {str(e)}")
        
        # Check in-memory config
        self.log("\nIn-memory config sequence:")
        mem_sequence = self.config.get('power_boost_sequence', [])
        for item in mem_sequence:
            self.log(f"  {item['level']}: {item['attacks']} attacks")
        
        self.log("Configuration file check completed")

    def check_and_sync_sequence(self):
        """Check and synchronize power boost sequence across all locations"""
        self.log("===== CHECKING AND SYNCING SEQUENCE =====")
        
        # 1. First check if configurations match
        self.check_config_files()
        self.check_game_controller_sequence()
        
        # Ask user if they want to sync all locations
        if messagebox.askyesno("Sequence Sync", 
                            "Do you want to synchronize the sequence across all locations?\n" +
                            "This will copy the current in-memory sequence to all files and the game controller."):
            
            # Get current sequence
            sequence = self.config.get('power_boost_sequence', [])
            if not sequence:
                self.log("No sequence in current config")
                return
            
            sequence_str = ', '.join([f"{item['level']}:{item['attacks']}" for item in sequence])
            self.log(f"Syncing sequence: {sequence_str}")
            
            # 1. Update root config.json
            try:
                config_path = "config.json"
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        root_config = json.load(f)
                else:
                    root_config = {}
                
                root_config['power_boost_sequence'] = sequence
                
                with open(config_path, 'w') as f:
                    json.dump(root_config, f, indent=2)
                
                self.log(f"✓ Updated {config_path}")
            except Exception as e:
                self.log(f"Error updating root config.json: {str(e)}")
            
            # 2. Update config/settings.json
            try:
                config_dir = "config"
                os.makedirs(config_dir, exist_ok=True)
                
                settings_path = f"{config_dir}/settings.json"
                if os.path.exists(settings_path):
                    with open(settings_path, 'r') as f:
                        settings_config = json.load(f)
                else:
                    settings_config = {}
                
                settings_config['power_boost_sequence'] = sequence
                
                with open(settings_path, 'w') as f:
                    json.dump(settings_config, f, indent=2)
                
                self.log(f"✓ Updated {settings_path}")
            except Exception as e:
                self.log(f"Error updating config/settings.json: {str(e)}")
            
            # 3. Update game controller
            if self.game_controller:
                try:
                    if hasattr(self.game_controller, 'update_power_boost_sequence'):
                        result = self.game_controller.update_power_boost_sequence(sequence)
                        self.log(f"✓ Updated game controller sequence using method: {result}")
                    else:
                        # Direct update
                        self.log("Using direct game controller attributes update")
                        self.game_controller.power_boost_sequence = sequence
                        self.game_controller.current_sequence_index = 0
                        self.game_controller.attacks_in_current_level = 0
                        if hasattr(self.game_controller, 'target_attacks_for_level') and len(sequence) > 0:
                            self.game_controller.target_attacks_for_level = sequence[0]['attacks']
                        self.log("✓ Direct game controller update completed")
                except Exception as e:
                    self.log(f"Error updating game controller: {str(e)}")
            
            self.log("Sequence sync completed")
            messagebox.showinfo("Success", "Power boost sequence synchronized in all locations")
            
            # Recheck to confirm update
            self.check_config_files()
            self.check_game_controller_sequence()
    
    def debug_power_boost_sequence(self):
        """Debug the power boost sequence"""
        try:
            sequence = self.config.get('power_boost_sequence', [])
            self.log(f"Config sequence: {sequence}")
            
            if self.game_controller and hasattr(self.game_controller, 'power_boost_sequence'):
                pbs = self.game_controller.power_boost_sequence
                
                if hasattr(pbs, 'sequence'):
                    gc_sequence = pbs.sequence
                    self.log(f"Game controller sequence: {gc_sequence}")
                    
                    # Display each level
                    for i, item in enumerate(gc_sequence):
                        level = item['level']
                        attacks = item['attacks']
                        level_name = level.name if hasattr(level, 'name') else str(level)
                        self.log(f"  {i}: {level_name} - {attacks} attacks")
                    
                    self.log(f"Current index: {pbs.current_index}")
                    self.log(f"Attacks completed: {pbs.attacks_completed}")
                    self.log(f"Target attacks: {pbs.get_target_attacks()}")
        except Exception as e:
            self.log(f"Error debugging power boost sequence: {str(e)}")

    # Replace update_power_boost_sequence in ui_handler.py
    def update_power_boost_sequence(self):
        """Update power boost sequence in the game controller if it's running"""
        if not self.game_controller or not self.bot_running:
            self.log("Game controller not running, power boost sequence will be applied on next start")
            return
        
        try:
            # Get the current sequence from config
            sequence = self.config.get('power_boost_sequence', [])
            
            # Log the sequence we're updating to
            sequence_str = ', '.join([f"{item['level']}:{item['attacks']}" for item in sequence])
            self.log(f"Updating power boost sequence to: {sequence_str}")
            
            # Update sequence in game controller
            if hasattr(self.game_controller, 'power_boost_sequence'):
                if isinstance(self.game_controller.power_boost_sequence, list):
                    # Direct sequence list (GameLogicIntegrator)
                    self.game_controller.power_boost_sequence = sequence
                    self.log("Direct sequence update successful")
                    
                    # Reset counters to start from first level
                    if hasattr(self.game_controller, 'current_sequence_index'):
                        self.game_controller.current_sequence_index = 0
                    if hasattr(self.game_controller, 'attacks_in_current_level'):
                        self.game_controller.attacks_in_current_level = 0
                    
                    # Set initial targets
                    if sequence:
                        if hasattr(self.game_controller, 'target_attacks_for_level'):
                            self.game_controller.target_attacks_for_level = sequence[0]['attacks']
                        if hasattr(self.game_controller, 'current_power_boost_level'):
                            self.game_controller.current_power_boost_level = sequence[0]['level']
                            
                elif hasattr(self.game_controller.power_boost_sequence, 'update_sequence'):
                    # PowerBoostSequence object (GameController)
                    self.game_controller.power_boost_sequence.update_sequence(sequence)
                    self.log("PowerBoostSequence update successful")
                    
                else:
                    self.log("Unknown power_boost_sequence type, using fallback")
                    self.game_controller.power_boost_sequence = sequence
                
                self.log("Power boost sequence updated successfully")
            else:
                self.log("Game controller does not have power_boost_sequence attribute")
                
        except Exception as e:
            self.log(f"Error updating power boost sequence: {str(e)}")

        
    def sync_attack_counter_with_sequence(self):
        """Synchronize attack counter with the current power boost sequence"""
        if not self.game_controller:
            self.log("Game controller not available")
            return
        
        self.log("===== SYNCHRONIZING ATTACK COUNTER WITH SEQUENCE =====")
        
        try:
            # Get sequence
            sequence = None
            if hasattr(self.game_controller, 'power_boost_sequence'):
                if isinstance(self.game_controller.power_boost_sequence, list):
                    sequence = self.game_controller.power_boost_sequence
                elif hasattr(self.game_controller.power_boost_sequence, 'sequence'):
                    sequence = self.game_controller.power_boost_sequence.sequence
            
            if not sequence or len(sequence) == 0:
                self.log("No power boost sequence found")
                return
            
            # Reset counters
            if hasattr(self.game_controller, 'current_sequence_index'):
                old_index = self.game_controller.current_sequence_index
                self.game_controller.current_sequence_index = 0
                self.log(f"Reset current_sequence_index from {old_index} to 0")
            
            if hasattr(self.game_controller, 'attacks_in_current_level'):
                old_attacks = self.game_controller.attacks_in_current_level
                self.game_controller.attacks_in_current_level = 0
                self.log(f"Reset attacks_in_current_level from {old_attacks} to 0")
            
            # Set target values from first sequence item
            first_item = sequence[0]
            if isinstance(first_item, dict) and 'level' in first_item and 'attacks' in first_item:
                level = first_item['level']
                attacks = first_item['attacks']
                
                if hasattr(self.game_controller, 'current_power_boost_level'):
                    old_level = self.game_controller.current_power_boost_level
                    self.game_controller.current_power_boost_level = level
                    self.log(f"Set current_power_boost_level from {old_level} to {level}")
                
                if hasattr(self.game_controller, 'target_attacks_for_level'):
                    old_target = self.game_controller.target_attacks_for_level
                    self.game_controller.target_attacks_for_level = attacks
                    self.log(f"Set target_attacks_for_level from {old_target} to {attacks}")
            
            # Debug the result
            self.debug_attack_counter()
            
            self.log("Attack counter synchronization complete")
            messagebox.showinfo("Synchronization Complete", 
                            "Attack counter has been synchronized with power boost sequence")
        except Exception as e:
            self.log(f"Error synchronizing attack counter: {str(e)}")
            messagebox.showerror("Synchronization Error", str(e))


    def debug_power_boost_detection(self):
        """Debug power boost detection."""
        if not self.game_controller:
            self.log("Game controller not available")
            return
                
        try:
            # Check if direct debug method exists on controller
            if hasattr(self.game_controller, 'debug_power_boost_detection'):
                self.log("Running power boost detection debug...")
                level, confidence = self.game_controller.debug_power_boost_detection()
                
                result_message = ""
                if level:
                    self.log(f"Power boost debug detection: {level} (confidence: {confidence:.2f})")
                    result_message = f"Detected level: {level}\nConfidence: {confidence:.2f}"
                else:
                    self.log("Power boost debug detection: No level detected")
                    result_message = "No power boost level detected."
                
                # Check if button was found
                button_info = ""
                if hasattr(self.game_controller, 'power_boost_manager') and \
                hasattr(self.game_controller.power_boost_manager, 'power_boost_button') and \
                self.game_controller.power_boost_manager.power_boost_button:
                    button_x, button_y = self.game_controller.power_boost_manager.power_boost_button
                    button_info = f"\n\nPower boost button found at: ({button_x}, {button_y})"
                    self.log(f"Power boost button found at: ({button_x}, {button_y})")
                else:
                    button_info = "\n\nPower boost button not found!"
                    self.log("Power boost button not found")
                        
                # Show results in message box
                messagebox.showinfo("Power Boost Detection Debug", 
                                result_message + button_info + 
                                "\n\nDebug images saved to 'debug' folder")
                    
                # Try to open the debug folder
                try:
                    import subprocess
                    import os
                    debug_dir = os.path.abspath("debug")
                    if os.path.exists(debug_dir):
                        self.log(f"Opening debug folder: {debug_dir}")
                        if os.name == 'nt':  # Windows
                            os.startfile(debug_dir)
                        elif os.name == 'posix':  # macOS, Linux
                            subprocess.call(['xdg-open', debug_dir])
                except Exception as folder_e:
                    self.log(f"Could not open debug folder: {str(folder_e)}")
            else:
                # FALLBACK: Implement direct detection if method not available on controller
                self.log("Controller debug method not available, using UI fallback")
                self.fallback_debug_power_boost_detection()
        except Exception as e:
            self.log(f"Error in power boost detection debug: {str(e)}")
            messagebox.showerror("Debug Error", f"Error in power boost detection debug: {str(e)}")

    def fallback_debug_power_boost_detection(self):
        """Fallback debug method if controller doesn't have the method"""
        self.log("FALLBACK: Direct power boost debug from UI")
        
        try:
            # Create debug directory
            debug_dir = "debug"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Capture fresh screenshot
            screen = None
            if self.adb:
                screen = self.adb.capture_screen()
                if screen is not None:
                    # Save screenshot
                    cv2.imwrite(f"{debug_dir}/fallback_debug.png", screen)
                    self.log(f"Screenshot saved to {debug_dir}/fallback_debug.png")
                    
                    # Try to detect power boost level directly
                    # Import here to avoid circular imports
                    from power_boost_detector import PowerBoostDetector
                    
                    detector = PowerBoostDetector("power_boost_templates.json")
                    detector.set_threshold(0.7)  # Lower threshold for testing
                    level, confidence = detector.detect(screen)
                    
                    if level:
                        self.log(f"Detected level: {level} (confidence: {confidence:.2f})")
                        
                        # Draw detection on image
                        debug_img = detector.draw_detection(screen, level)
                        cv2.imwrite(f"{debug_dir}/fallback_detection.png", debug_img)
                        self.log(f"Detection visualization saved to {debug_dir}/fallback_detection.png")
                        
                        messagebox.showinfo("Fallback Detection", 
                                        f"Detected level: {level}\nConfidence: {confidence:.2f}\n\n" +
                                        f"Debug images saved to {debug_dir}")
                    else:
                        self.log("No power boost level detected")
                        messagebox.showinfo("Fallback Detection", "No power boost level detected")
                else:
                    self.log("Failed to capture screen")
                    messagebox.showerror("Error", "Failed to capture screen")
            else:
                self.log("ADB controller not available")
                messagebox.showerror("Error", "ADB controller not available")
        except Exception as e:
            self.log(f"Fallback debug error: {str(e)}")
            messagebox.showerror("Fallback Debug Error", str(e))
    
    def debug_power_boost_sequence(self):
        """Debug all power boost sequence instances in the system"""
        self.log("===== POWER BOOST SEQUENCE DEBUG =====")
        
        # 1. Check config file
        try:
            with open("config/settings.json", 'r') as f:
                config_data = json.load(f)
                config_sequence = config_data.get('power_boost_sequence', [])
                self.log("1. Config File Sequence:")
                for item in config_sequence:
                    self.log(f"   {item['level']}: {item['attacks']} attacks")
        except Exception as e:
            self.log(f"Error reading config file: {str(e)}")
        
        # 2. Check in-memory config
        self.log("\n2. In-Memory Config Sequence:")
        mem_sequence = self.config.get('power_boost_sequence', [])
        for item in mem_sequence:
            self.log(f"   {item['level']}: {item['attacks']} attacks")
        
        # 3. Check GameLogicIntegrator
        if self.game_controller:
            self.log("\n3. Game Controller Sequence:")
            
            # A. Direct sequence in GameLogicIntegrator
            if hasattr(self.game_controller, 'power_boost_sequence'):
                gc_sequence = self.game_controller.power_boost_sequence
                
                if isinstance(gc_sequence, list):
                    self.log("   Game Controller has direct sequence list:")
                    for item in gc_sequence:
                        self.log(f"   {item['level']}: {item['attacks']} attacks")
                    
                    # Additional state info
                    if hasattr(self.game_controller, 'current_sequence_index'):
                        self.log(f"   Current index: {self.game_controller.current_sequence_index}")
                    if hasattr(self.game_controller, 'attacks_in_current_level'):
                        self.log(f"   Attacks in level: {self.game_controller.attacks_in_current_level}")
                    if hasattr(self.game_controller, 'target_attacks_for_level'):
                        self.log(f"   Target attacks: {self.game_controller.target_attacks_for_level}")
                
                # B. PowerBoostSequence object
                elif hasattr(gc_sequence, 'sequence'):
                    self.log("   Game Controller has PowerBoostSequence object:")
                    try:
                        pbs_sequence = gc_sequence.sequence
                        for item in pbs_sequence:
                            level = item['level']
                            level_name = level.name if hasattr(level, 'name') else str(level)
                            self.log(f"   {level_name}: {item['attacks']} attacks") 
                        
                        self.log(f"   Current index: {gc_sequence.current_index}")
                        self.log(f"   Attacks completed: {gc_sequence.attacks_completed}")
                    except Exception as e:
                        self.log(f"   Error examining PowerBoostSequence: {str(e)}")
            else:
                self.log("   Game Controller does not have power_boost_sequence attribute")
        else:
            self.log("\n3. Game Controller not initialized")
        
        self.log("\nSequence debug completed")

    # Add this function to ui_handler.py
    def force_update_config_file(self):
        """Force update of config.json file with current sequence"""
        try:
            # Get current sequence
            sequence = self.config.get('power_boost_sequence', [])
            
            # Display what we're about to do
            sequence_str = ', '.join([f"{item['level']}:{item['attacks']}" for item in sequence])
            self.log(f"Forcing update of config.json with sequence: {sequence_str}")
            
            # Read existing config.json to preserve other settings
            config_path = "config.json"  # Direct path to your config
            
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                self.log("config.json not found or invalid, creating new one")
                config_data = {}
            
            # Update sequence in the config data
            config_data['power_boost_sequence'] = sequence
            
            # Write back to config.json
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            self.log(f"✓ Successfully updated {config_path}")
            messagebox.showinfo("Success", f"Power boost sequence saved to {config_path}")
            
            return True
        except Exception as e:
            self.log(f"Error updating config file: {str(e)}")
            messagebox.showerror("Error", f"Failed to update config.json: {str(e)}")
            return False

    def test_register_attack(self):
        """Test registering an attack"""
        if not self.game_controller or not hasattr(self.game_controller, 'power_boost_sequence'):
            self.log("Game controller not available")
            return
        
        try:
            pbs = self.game_controller.power_boost_sequence
            before_index = pbs.current_index
            before_attacks = pbs.attacks_completed
            
            # Register an attack
            should_advance = pbs.register_attack()
            
            self.log(f"Before: index={before_index}, attacks={before_attacks}")
            self.log(f"After: index={pbs.current_index}, attacks={pbs.attacks_completed}")
            self.log(f"Should advance to next level: {should_advance}")
        except Exception as e:
            self.log(f"Error testing attack registration: {str(e)}")
    
    def start_bot(self):
        """Start the bot"""
        if not self.adb or not self.adb.connected:
            messagebox.showinfo("Not Connected", "Please connect to a device first.")
            return
            
        if not self.detector:
            messagebox.showinfo("No Model", "Please load a model first.")
            return
        
        self.log("Starting bot...")
        
        try:
            # Initialize fixed UI detector if enabled and not already initialized
            if self.use_fixed_ui_var.get():
                if self.fixed_ui_detector is None:
                    self.log("Initializing fixed UI detector...")
                    self.fixed_ui_detector = FixedUIDetector(config_file="fixed_ui_elements.json")
                    
                    # Verify that we have fixed UI elements configured
                    if not self.fixed_ui_detector.roi_elements:
                        self.log("Warning: No fixed UI elements configured")
                        messagebox.showwarning(
                            "No Fixed UI Elements", 
                            "Fixed UI detection is enabled but no elements are configured.\n\n"
                            "Click 'Fixed UI Editor' to add UI elements before starting."
                        )
                        return
                else:
                    # Reload configuration in case it was updated in the editor
                    self.log("Reloading fixed UI elements configuration...")
                    self.fixed_ui_detector.load_config()
            
            # Make sure we have the current config
            config = self.config.copy()
            
            # Update with current UI settings
            config.update({
                "device_id": self.adb.device_id,
                "model_path": self.model_path.get(),
                "detection_confidence": 0.5,
                "action_delay": 0.5,
                "use_fixed_ui": self.use_fixed_ui_var.get(),
                "debug_mode": True  # Enable debug mode for visualization
            })
            
            # Log the power boost sequence that's being used
            try:
                sequence = config.get('power_boost_sequence', [])
                sequence_str = ', '.join([f"{item['level']}:{item['attacks']}" for item in sequence])
                self.log(f"Using power boost sequence: {sequence_str}")
            except Exception as e:
                self.log(f"Error logging power boost sequence: {str(e)}")
            
            # Initialize game controller if not already done
            if not self.game_controller:
                # Initialize with appropriate controllers based on settings
                if self.use_fixed_ui_var.get() and self.fixed_ui_detector:
                    self.log("Using hybrid detection (YOLO + Fixed UI)")
                    
                    # Import necessary components
                    from game_logic_integrator import GameLogicIntegrator
                    
                    # Use game logic integrator
                    self.game_controller = GameLogicIntegrator(
                        adb=self.adb,
                        detector=self.detector,
                        fixed_ui_detector=self.fixed_ui_detector,
                        config=config
                    )
                else:
                    self.log("Using YOLO detection only")
                    self.game_controller = GameController(self.adb, self.detector, config)
            else:
                # Reset the game controller with new configuration
                self.game_controller.stop()
                
                # Initialize with new configuration
                if self.use_fixed_ui_var.get() and self.fixed_ui_detector:
                    self.log("Using hybrid detection (YOLO + Fixed UI)")
                    
                    # Import necessary components
                    from game_logic_integrator import GameLogicIntegrator
                    
                    # Use game logic integrator
                    self.game_controller = GameLogicIntegrator(
                        adb=self.adb,
                        detector=self.detector,
                        fixed_ui_detector=self.fixed_ui_detector,
                        config=config
                    )
                else:
                    self.log("Using YOLO detection only")
                    self.game_controller = GameController(self.adb, self.detector, config)
            
            # Start the bot
            self.game_controller.start()
            self.bot_running = True
            
            # Update UI
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("Running")
            
            # Start the update loop if not already running
            if not self.running:
                self.running = True
                threading.Thread(target=self.update_loop, daemon=True).start()
            
        except Exception as e:
            self.log(f"Error starting bot: {str(e)}")
            messagebox.showerror("Error", f"Failed to start bot: {str(e)}")
    
    def stop_bot(self):
        """Stop the bot"""
        if not self.game_controller:
            return
        
        self.log("Stopping bot...")
        
        try:
            # Stop the bot
            self.game_controller.stop()
            self.bot_running = False
            
            # Update UI
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_var.set("Stopped")
            
        except Exception as e:
            self.log(f"Error stopping bot: {str(e)}")
            messagebox.showerror("Error", f"Failed to stop bot: {str(e)}")
    
    def take_screenshot(self):
        """Take a screenshot and update the UI"""
        if not self.adb or not self.adb.connected:
            return None
        
        try:
            # Capture screen
            screen = self.adb.capture_screen()
            
            if screen is not None:
                # Convert to PhotoImage and update label
                screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
                
                # Resize to fit the frame while maintaining aspect ratio
                frame_width = self.screen_frame.winfo_width()
                frame_height = self.screen_frame.winfo_height()
                
                if frame_width > 100 and frame_height > 100:  # Ensure the frame has been rendered
                    h, w = screen_rgb.shape[:2]
                    
                    # Calculate scaling factor
                    scale = min(frame_width / w, frame_height / h)
                    
                    # Calculate new dimensions
                    new_width = int(w * scale)
                    new_height = int(h * scale)
                    
                    # Resize image
                    resized = cv2.resize(screen_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # Convert to PhotoImage
                    img = Image.fromarray(resized)
                    photo = ImageTk.PhotoImage(image=img)
                    
                    # Update label
                    self.screen_label.config(image=photo)
                    self.screen_label.image = photo  # Keep a reference to prevent garbage collection
                
                return screen
            
        except Exception as e:
            self.log(f"Error taking screenshot: {str(e)}")
            return None
    
    def update_stats(self):
        """Update the status display with current game stats"""
        if not self.game_controller:
            return
        
        try:
            # Get current stats
            stats = self.game_controller.get_stats()
            
            # Update status variables
            self.status_var.set(stats['state'])
            self.power_boost_var.set(stats['current_power_boost'])
            self.attacks_var.set(f"{stats['attacks_in_current_level']}/{stats['target_attacks_for_level']} (Total: {stats['attacks_completed']})")
            self.raids_var.set(str(stats['raids_completed']))
            
            # Update status bar with additional info
            self.statusbar.config(text=f"Status: {stats['state']} | Power Boost: {stats['current_power_boost']} | " +
                                f"Sequence: {stats['current_sequence_position']}/{stats['sequence_length']} | " +
                                f"Attacks: {stats['attacks_in_current_level']}/{stats['target_attacks_for_level']}")
                
        except Exception as e:
            self.log(f"Error updating stats: {str(e)}")
    
    def update_loop(self):
        """Main update loop for the GUI"""
        while self.running:
            try:
                # Take screenshot if enabled
                if self.screen_capture_var.get():
                    screen = self.take_screenshot()
                    
                    # If bot is running and screen was captured, draw detections
                    if self.bot_running and screen is not None:
                        try:
                            # Determine what type of game controller we're using
                            if hasattr(self.game_controller, 'hybrid_detector') and self.game_controller.hybrid_detector:
                                # Using hybrid detection from GameLogicIntegrator
                                # Create a hybrid results dictionary that matches what draw_results expects
                                hybrid_results = {
                                    "dynamic_objects": self.game_controller.last_dynamic_detections,
                                    "fixed_ui": self.game_controller.last_fixed_ui_results
                                }
                                
                                # Use the hybrid detector's draw_results method
                                display_img = self.game_controller.hybrid_detector.draw_results(screen, hybrid_results)
                            
                            elif self.detector:
                                # Using only YOLO detection
                                detections = self.detector.detect(screen)
                                display_img = self.detector.draw_detections(screen, detections)
                            
                            else:
                                display_img = screen
                        
                        except Exception as e:
                            # If visualization fails, log the error and just show the raw screenshot
                            self.log(f"Error drawing detections: {str(e)}")
                            display_img = screen
                        
                        # Convert to PhotoImage and update label
                        display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                        
                        # Resize to fit the frame while maintaining aspect ratio
                        frame_width = self.screen_frame.winfo_width()
                        frame_height = self.screen_frame.winfo_height()
                        
                        if frame_width > 100 and frame_height > 100:  # Ensure the frame has been rendered
                            h, w = display_rgb.shape[:2]
                            
                            # Calculate scaling factor
                            scale = min(frame_width / w, frame_height / h)
                            
                            # Calculate new dimensions
                            new_width = int(w * scale)
                            new_height = int(h * scale)
                            
                            # Resize image
                            resized = cv2.resize(display_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
                            
                            # Convert to PhotoImage
                            img = Image.fromarray(resized)
                            photo = ImageTk.PhotoImage(image=img)
                            
                            # Update label
                            self.screen_label.config(image=photo)
                            self.screen_label.image = photo  # Keep a reference
                
                # Update stats if bot is running
                if self.bot_running:
                    self.update_stats()
                
                # Sleep to prevent excessive CPU usage
                time.sleep(1)
                
            except Exception as e:
                self.log(f"Error in update loop: {str(e)}")
                time.sleep(5)  # Longer delay on error

                # Add this method to CoinMasterBotGUI in ui_handler.py
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
                self.log("✓ POWER BOOST CHANGE WOULD TRIGGER")
            else:
                self.log(f"✗ Power boost change would NOT trigger (need {target-attacks} more attacks)")
        except (ValueError, TypeError) as e:
            self.log(f"Error comparing attack values: {e}")
        
        self.log("Attack counter debug completed")
        messagebox.showinfo("Attack Counter Debug", 
                        f"Current attacks: {attacks}/{target}\n" +
                        f"Power boost change would {'TRIGGER' if attacks >= target else 'NOT trigger'}")
        
    # Add this method to CoinMasterBotGUI in ui_handler.py
    def force_register_attack(self):
        """Manually register an attack for testing"""
        if not self.game_controller:
            self.log("Game controller not available")
            return
        
        self.log("Manually registering an attack...")
        
        try:
            # For GameLogicIntegrator
            if hasattr(self.game_controller, 'attacks_in_current_level'):
                self.game_controller.attacks_in_current_level += 1
                self.log(f"Incremented attacks_in_current_level to {self.game_controller.attacks_in_current_level}")
            
            if hasattr(self.game_controller, 'attacks_completed'):
                self.game_controller.attacks_completed += 1
                self.log(f"Incremented attacks_completed to {self.game_controller.attacks_completed}")
            
            # For GameController with PowerBoostSequence
            if hasattr(self.game_controller, 'power_boost_sequence') and hasattr(self.game_controller.power_boost_sequence, 'register_attack'):
                result = self.game_controller.power_boost_sequence.register_attack()
                self.log(f"PowerBoostSequence.register_attack() result: {result}")
            
            # Check if we should trigger power boost change
            self.debug_attack_counter()
            
            # If ready to change, offer to force power boost change
            attacks = getattr(self.game_controller, 'attacks_in_current_level', 0)
            target = getattr(self.game_controller, 'target_attacks_for_level', 1)
            
            if attacks >= target:
                if messagebox.askyesno("Power Boost Change", 
                                    "Attack threshold reached! Do you want to force a power boost change?"):
                    self.force_power_boost_change()
        except Exception as e:
            self.log(f"Error registering attack: {str(e)}")

    
    # Update the force_power_boost_change method in CoinMasterBotGUI
    def force_power_boost_change(self):
        """Force a power boost change to the next level"""
        if not self.game_controller:
            self.log("Game controller not available")
            return
        
        self.log("===== FORCING POWER BOOST CHANGE =====")
        
        try:
            # Get current sequence info
            controller = self.game_controller
            
            # For GameLogicIntegrator
            if hasattr(controller, 'power_boost_sequence') and isinstance(controller.power_boost_sequence, list):
                current_index = getattr(controller, 'current_sequence_index', 0)
                current_level = getattr(controller, 'current_power_boost_level', "X1")
                
                # Move to next level
                next_index = (current_index + 1) % len(controller.power_boost_sequence)
                next_item = controller.power_boost_sequence[next_index]
                target_level = next_item["level"]
                
                self.log(f"Changing from level {current_level} to {target_level}")
                
                # Reset attack counter
                controller.attacks_in_current_level = 0
                controller.current_sequence_index = next_index
                controller.target_attacks_for_level = next_item["attacks"]
                
                # For GameLogicIntegrator with power_boost_manager
                if hasattr(controller, 'power_boost_manager'):
                    pbm = controller.power_boost_manager
                    
                    # Find button if needed
                    if not pbm.power_boost_button:
                        screen = self.adb.capture_screen()
                        if screen is not None:
                            pbm.find_power_boost_button(screen, getattr(controller, 'last_fixed_ui_results', None))
                    
                    # Change level
                    if pbm.power_boost_button:
                        x, y = pbm.power_boost_button
                        self.log(f"Power boost button found at ({x}, {y})")
                        
                        # Use power_boost_manager method
                        success = pbm.change_power_boost(current_level, target_level)
                        
                        if success:
                            controller.current_power_boost_level = target_level
                            self.log(f"[SUCCESS] Changed to {target_level} successfully")
                        else:
                            self.log("[FAIL] Power boost change failed")
                    else:
                        self.log("[FAIL] Power boost button not found")
                else:
                    self.log("No power_boost_manager available")
            
            # For GameController
            elif hasattr(controller, 'power_boost_sequence') and hasattr(controller.power_boost_sequence, 'sequence'):
                pbs = controller.power_boost_sequence
                current_level = pbs.get_current_level()
                
                # Register enough attacks to trigger level change
                while pbs.attacks_completed < pbs.get_target_attacks():
                    pbs.register_attack()
                    
                self.log(f"Registered enough attacks to change from {current_level.name} to next level")
                
                # The next level should be set by register_attack
                new_level = pbs.get_current_level()
                self.log(f"New level is {new_level.name}")
            else:
                self.log("Unknown game controller configuration")
                
            # Update display
            self.debug_attack_counter()
            
        except Exception as e:
            self.log(f"Error forcing power boost change: {str(e)}")
        
    def log(self, message):
            """Add a message to the log"""
            logger.info(message)
    
    def save_config(self):
        """Save current configuration"""
        # Update config with current UI values
        self.config.update({
            "device_id": self.device_id.get(),
            "model_path": self.model_path.get(),
            "detection_confidence": 0.5,
            "action_delay": 0.5,
            "use_fixed_ui": self.use_fixed_ui_var.get(),
            "fixed_ui_config": "fixed_ui_elements.json",
            "log_level": "INFO",
            "debug_mode": False
        })
        
        # Make sure power_boost_sequence exists in config
        if "power_boost_sequence" not in self.config:
            self.config["power_boost_sequence"] = [
                {"level": "X1", "attacks": 8},
                {"level": "X15", "attacks": 3},
                {"level": "X50", "attacks": 4},
                {"level": "X400", "attacks": 3},
                {"level": "X1500", "attacks": 1},
                {"level": "X6000", "attacks": 1},
                {"level": "X20000", "attacks": 1}
            ]
        
        try:
            os.makedirs("config", exist_ok=True)
            
            with open("config/settings.json", 'w') as f:
                import json
                json.dump(self.config, f, indent=2)
            
            self.log("Configuration saved to config/settings.json")
            messagebox.showinfo("Success", "Configuration saved successfully.")
            
        except Exception as e:
            self.log(f"Error saving configuration: {str(e)}")
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

    def test_attack_registration(self):
        """Test the attack registration logic directly."""
        if not self.game_controller:
            self.log("Game controller not available")
            return
        
        self.log("===== TESTING ATTACK REGISTRATION =====")
        
        try:
            # Record before state
            current_level = getattr(self.game_controller, 'current_power_boost_level', "Unknown")
            attacks_before = getattr(self.game_controller, 'attacks_in_current_level', 0)
            total_before = getattr(self.game_controller, 'attacks_completed', 0)
            target = getattr(self.game_controller, 'target_attacks_for_level', 0)
            
            self.log(f"Before state: Level {current_level}, Attacks {attacks_before}/{target} (Total: {total_before})")
            
            # Directly call register_attack
            if hasattr(self.game_controller, 'register_attack'):
                result = self.game_controller.register_attack()
                self.log(f"register_attack() called, result: {result}")
            else:
                self.log("register_attack method not found, using direct increments")
                if hasattr(self.game_controller, 'attacks_in_current_level'):
                    self.game_controller.attacks_in_current_level += 1
                if hasattr(self.game_controller, 'attacks_completed'):
                    self.game_controller.attacks_completed += 1
            
            # Get after state
            attacks_after = getattr(self.game_controller, 'attacks_in_current_level', 0)
            total_after = getattr(self.game_controller, 'attacks_completed', 0)
            
            self.log(f"After state: Attacks {attacks_after}/{target} (Total: {total_after})")
            
            # Show results
            if attacks_after > attacks_before and total_after > total_before:
                self.log("SUCCESS: Both counters incremented correctly")
            else:
                self.log("FAILED: Counters did not increment properly")
                
            # Check threshold
            if attacks_after >= target:
                self.log("Power boost change threshold REACHED")
                if messagebox.askyesno("Power Boost Change", 
                                "Attack threshold reached! Force power boost change?"):
                    self.force_power_boost_change()
            
        except Exception as e:
            self.log(f"Error in attack registration test: {str(e)}")
    
    def test_fixed_ui_autospin(self):
        """Test the auto-spin activation using fixed UI detection"""
        if not self.adb or not self.adb.connected:
            self.log("No device connected")
            return
        
        try:
            # Take screenshot
            screen = self.adb.capture_screen()
            
            # Detect fixed UI elements
            if self.fixed_ui_detector:
                fixed_ui_results = self.fixed_ui_detector.detect(screen)
                
                # Look for spin button in fixed UI results
                spin_button = None
                for name, result in fixed_ui_results.items():
                    if name.lower().find("spin") >= 0 and result["detected"]:
                        spin_button = (name, result["center"][0], result["center"][1])
                        break
                
                if spin_button:
                    name, x, y = spin_button
                    self.log(f"Found fixed UI spin button '{name}' at ({int(x)}, {int(y)}) - testing AUTO-SPIN")
                    
                    # Perform long press
                    self.log("Performing long press for 8 seconds...")
                    
                    # Try multiple approaches
                    try:
                        # Use force_tap_and_hold
                        if hasattr(self.adb, 'force_tap_and_hold'):
                            self.adb.force_tap_and_hold(int(x), int(y), duration_ms=8000)
                        else:
                            # Alternative: Use long_press
                            self.adb.long_press(int(x), int(y), duration_ms=8000)
                        
                        self.log("Auto-spin test commands sent successfully")
                    except Exception as e:
                        self.log(f"Error in long press: {str(e)}")
                else:
                    self.log("No spin button found in fixed UI elements")
                    self.log("Available fixed UI elements:")
                    for name, result in fixed_ui_results.items():
                        if result["detected"]:
                            self.log(f"  - {name} at ({int(result['center'][0])}, {int(result['center'][1])})")
            else:
                self.log("Fixed UI detector not initialized")
                
        except Exception as e:
            self.log(f"Error in fixed UI auto-spin test: {str(e)}")

    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About Coin Master Bot",
            "Coin Master Automation Bot\n\n"
            "A Python bot that automates Coin Master gameplay using ADB and object detection.\n\n"
            "Controls the game through ADB and detects game elements using both YOLO and fixed UI detection."
        )
    
    def debug_power_boost_detection(self):
        """Debug power boost detection."""
        if not self.game_controller:
            self.log("Game controller not available")
            return
            
        try:
            if hasattr(self.game_controller, 'debug_power_boost_detection'):
                self.log("Running power boost detection debug...")
                level, confidence = self.game_controller.debug_power_boost_detection()
                
                result_message = ""
                if level:
                    self.log(f"Power boost debug detection: {level} (confidence: {confidence:.2f})")
                    result_message = f"Detected level: {level}\nConfidence: {confidence:.2f}"
                else:
                    self.log("Power boost debug detection: No level detected")
                    result_message = "No power boost level detected."
                
                # Check if button was found
                button_info = ""
                if hasattr(self.game_controller, 'power_boost_manager') and \
                hasattr(self.game_controller.power_boost_manager, 'power_boost_button') and \
                self.game_controller.power_boost_manager.power_boost_button:
                    button_x, button_y = self.game_controller.power_boost_manager.power_boost_button
                    button_info = f"\n\nPower boost button found at: ({button_x}, {button_y})"
                    self.log(f"Power boost button found at: ({button_x}, {button_y})")
                else:
                    button_info = "\n\nPower boost button not found!"
                    self.log("Power boost button not found")
                    
                # Show results in message box
                messagebox.showinfo("Power Boost Detection Debug", 
                                result_message + button_info + 
                                "\n\nDebug images saved to 'debug' folder")
                
                # Try to open the debug folder
                try:
                    import subprocess
                    import os
                    debug_dir = os.path.abspath("debug")
                    if os.path.exists(debug_dir):
                        self.log(f"Opening debug folder: {debug_dir}")
                        if os.name == 'nt':  # Windows
                            os.startfile(debug_dir)
                        elif os.name == 'posix':  # macOS, Linux
                            subprocess.call(['xdg-open', debug_dir])
                except Exception as folder_e:
                    self.log(f"Could not open debug folder: {str(folder_e)}")
            else:
                self.log("Debug power boost detection not available")
                messagebox.showinfo("Debug Not Available", 
                                "Debug power boost detection is not available.\n\n" +
                                "Please update the game_logic_integrator.py file with the debug method.")
        except Exception as e:
            self.log(f"Error in power boost detection debug: {str(e)}")
            messagebox.showerror("Debug Error", 
                                f"Error in power boost detection debug:\n{str(e)}")
            
    # Add this method to your CoinMasterBotGUI class in ui_handler.py
    def check_game_controller_sequence(self):
        """Check game controller sequence"""
        if not self.game_controller:
            self.log("Game controller not available")
            return
            
        self.log("===== CHECKING GAME CONTROLLER SEQUENCE =====")
        
        # Check direct sequence
        if hasattr(self.game_controller, 'power_boost_sequence'):
            sequence = self.game_controller.power_boost_sequence
            if isinstance(sequence, list):
                self.log(f"Game controller direct sequence: {len(sequence)} levels")
                for item in sequence:
                    self.log(f"  {item['level']}: {item['attacks']} attacks")
            elif hasattr(sequence, 'sequence'):
                # PowerBoostSequence object
                self.log("Game controller has PowerBoostSequence object")
                pbs_sequence = sequence.sequence
                self.log(f"PowerBoostSequence: {len(pbs_sequence)} levels")
                for item in pbs_sequence:
                    level = item['level']
                    level_name = level.name if hasattr(level, 'name') else str(level)
                    self.log(f"  {level_name}: {item['attacks']} attacks")
        else:
            self.log("Game controller does not have power_boost_sequence attribute")
        
        # Additional controller attributes
        if hasattr(self.game_controller, 'current_sequence_index'):
            self.log(f"Current sequence index: {self.game_controller.current_sequence_index}")
        
        if hasattr(self.game_controller, 'attacks_in_current_level'):
            self.log(f"Attacks in current level: {self.game_controller.attacks_in_current_level}")
        
        if hasattr(self.game_controller, 'target_attacks_for_level'):
            self.log(f"Target attacks for level: {self.game_controller.target_attacks_for_level}")
        
        if hasattr(self.game_controller, 'current_power_boost_level'):
            self.log(f"Current power boost level: {self.game_controller.current_power_boost_level}")
        
        self.log("Game controller check completed")

    def on_close(self):
        """Handle window close"""
        if self.bot_running:
            if not messagebox.askyesno("Quit", "The bot is still running. Are you sure you want to quit?"):
                return
            
            # Stop the bot
            self.stop_bot()
        
        # Stop the update loop
        self.running = False
        
        # Close any open dialogs
        if self.roi_selector_window and self.roi_selector_window.winfo_exists():
            self.roi_selector_window.destroy()
        
        # Close the window
        self.root.destroy()
    

def main():
    """Main entry point"""
    try:
        root = tk.Tk()
        app = CoinMasterBotGUI(root)
        root.mainloop()
        return 0
    except Exception as e:
        print(f"Critical error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())