import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import logging

logger = logging.getLogger(__name__)

class PowerBoostConfigurator:
    """
    Dialog window for configuring the power boost sequence.
    Allows users to set the number of attacks for each power boost level.
    """
    
    def __init__(self, parent, config=None, on_save=None):
        """
        Initialize the configurator.
        
        Args:
            parent: Parent window
            config: Current configuration (optional)
            on_save: Callback function to call when saving the configuration
        """
        self.parent = parent
        self.config = config or {}
        self.on_save = on_save
        
        # Default power boost sequence
        self.default_sequence = [
            {"level": "X1", "attacks": 8},
            {"level": "X2", "attacks": 3},
            {"level": "X3", "attacks": 3},
            {"level": "X15", "attacks": 3},
            {"level": "X50", "attacks": 4},
            {"level": "X400", "attacks": 3},
            {"level": "X1500", "attacks": 1},
            {"level": "X6000", "attacks": 1},
            {"level": "X20000", "attacks": 1}
        ]
        
        # Load current sequence from config
        self.power_boost_sequence = self.config.get('power_boost_sequence', self.default_sequence)
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Power Boost Sequence Configuration")
        self.dialog.geometry("500x500")
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)  # Make dialog modal
        self.dialog.grab_set()  # Make dialog modal
        
        # Setup UI
        self.create_ui()
        
        # Center dialog on parent
        self.center_dialog()
        
    
    def center_dialog(self):
        """Center the dialog on the parent window."""
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        # Wait for dialog to be drawn before getting its size
        self.dialog.update_idletasks()
        
        # Get actual dialog size
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        # Calculate position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        # Set position
        self.dialog.geometry(f"+{x}+{y}")
    
    def create_ui(self):
        """Create the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Heading
        ttk.Label(main_frame, text="Configure Power Boost Sequence", font=("Helvetica", 12, "bold")).pack(pady=10)
        ttk.Label(main_frame, text="Set the number of attacks to perform at each power boost level").pack(pady=5)
        
        # Create entry widgets for each power boost level
        entry_frame = ttk.Frame(main_frame)
        entry_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create a frame for each entry
        self.entry_widgets = {}
        
        # Add column headers
        ttk.Label(entry_frame, text="Power Boost Level", font=("Helvetica", 10, "bold")).grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        ttk.Label(entry_frame, text="Attacks", font=("Helvetica", 10, "bold")).grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
        ttk.Label(entry_frame, text="Status", font=("Helvetica", 10, "bold")).grid(row=0, column=2, padx=10, pady=5, sticky=tk.W)
        
        # Get all boost levels
        boost_levels = ["X1", "X2", "X3", "X15", "X50", "X400", "X1500", "X6000", "X20000"]
        
        # Create entries for each level
        for i, level in enumerate(boost_levels):
            # Initial values
            attacks = 0
            enabled = False
            
            # Look for level in current sequence
            for item in self.power_boost_sequence:
                if item["level"] == level:
                    attacks = item["attacks"]
                    enabled = True
                    break
            
            # Level label
            ttk.Label(entry_frame, text=level).grid(row=i+1, column=0, padx=10, pady=5, sticky=tk.W)
            
            # Attacks entry
            attacks_var = tk.StringVar(value=str(attacks))
            attacks_entry = ttk.Spinbox(entry_frame, from_=1, to=100, textvariable=attacks_var, width=5)
            attacks_entry.grid(row=i+1, column=1, padx=10, pady=5, sticky=tk.W)
            
            # Status indicator (enabled/disabled)
            enabled_var = tk.BooleanVar(value=enabled)
            enabled_check = ttk.Checkbutton(entry_frame, text="Enabled", variable=enabled_var)
            enabled_check.grid(row=i+1, column=2, padx=10, pady=5, sticky=tk.W)
            
            # Store the widgets
            self.entry_widgets[level] = {
                "attacks_var": attacks_var,
                "enabled_var": enabled_var
            }
        
        # Add buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Save", command=self.save_config).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Reset to Default", command=self.reset_to_default).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    # In power_boost_configurator.py
    def save_config(self):
        """Save the configuration."""
        try:
            # Build the sequence from the entry widgets
            sequence = []
            
            for level, widgets in self.entry_widgets.items():
                # Only include enabled levels
                if widgets["enabled_var"].get():
                    try:
                        attacks = int(widgets["attacks_var"].get())
                        if attacks <= 0:
                            raise ValueError(f"Attacks for {level} must be positive")
                            
                        sequence.append({
                            "level": level,
                            "attacks": attacks
                        })
                    except ValueError as e:
                        messagebox.showerror("Invalid Input", f"Invalid number of attacks for {level}: {str(e)}")
                        return
            
            if not sequence:
                messagebox.showerror("Invalid Sequence", "You must have at least one enabled power boost level")
                return
            
            # Update the configuration
            self.config['power_boost_sequence'] = sequence
            
            # Save directly to config.json
            try:
                config_file = "config.json"  # Direct path to root config
                
                # Read existing config to preserve other settings
                try:
                    with open(config_file, 'r') as f:
                        existing_config = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    existing_config = {}
                
                # Update just the sequence
                existing_config['power_boost_sequence'] = sequence
                
                # Write back to file
                with open(config_file, 'w') as f:
                    json.dump(existing_config, f, indent=2)
                    
                logger.info(f"Saved power boost sequence to {config_file}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save to config.json: {str(e)}")
                logger.error(f"Error saving to config.json: {str(e)}")
            
            # Save to file if needed
            if self.on_save:
                try:
                    self.on_save(self.config)
                    logger.info("On-save callback executed successfully")
                except Exception as e:
                    logger.error(f"Error in on-save callback: {str(e)}")
            
            # Show success message
            messagebox.showinfo("Success", "Power boost sequence saved")
            
            # Close the dialog
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
            logger.error(f"Failed to save power boost configuration: {str(e)}", exc_info=True)

    def on_config_save(self, updated_config):
        """Handle saving of updated configuration from Power Boost Configurator"""
        # Update the config
        self.config = updated_config
        
        # Log the sequence being saved
        sequence = self.config.get('power_boost_sequence', [])
        sequence_str = ', '.join([f"{item['level']}:{item['attacks']}" for item in sequence])
        self.log(f"Power Boost Config saved: {sequence_str}")
        
        # 1. Save to standard config path
        self.save_config()
        
        # 2. Also save directly to config.json in root
        try:
            config_file = "config.json"  # Direct path
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.log(f"✓ Also saved to {config_file}")
        except Exception as e:
            self.log(f"⚠ Error saving to {config_file}: {str(e)}")
        
        # 3. Update game controller if running
        self.update_power_boost_sequence()
    
    def reset_to_default(self):
        """Reset the configuration to default."""
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset to the default sequence?"):
            # Reset all levels to disabled
            for level, widgets in self.entry_widgets.items():
                widgets["attacks_var"].set("0")
                widgets["enabled_var"].set(False)
            
            # Set values for levels in the default sequence
            for item in self.default_sequence:
                level = item["level"]
                attacks = item["attacks"]
                
                if level in self.entry_widgets:
                    widgets = self.entry_widgets[level]
                    widgets["attacks_var"].set(str(attacks))
                    widgets["enabled_var"].set(True)