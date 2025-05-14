import cv2
import numpy as np
import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import logging
import threading
import time
import subprocess
from typing import Dict, List, Tuple, Optional, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ROISelector:
    """
    GUI tool for selecting and managing regions of interest (ROIs) in game screenshots.
    These ROIs represent fixed UI elements that can be detected without deep learning.
    """
    
    def __init__(self, root, adb_controller=None):
        """Initialize the ROI Selector tool."""
        self.root = root
        self.root.title("Fixed UI Element Detector")
        self.root.geometry("1200x800")
        
        # Store ADB controller reference if provided
        self.adb_controller = adb_controller
        
        # Variables
        self.current_image_path = None
        self.current_image = None
        self.current_image_display = None
        self.original_image = None
        self.roi_elements = {}  # Dict to store ROIs: {name: {"roi": (x1, y1, x2, y2), "template": img_array, "tappable": bool}}
        self.selected_roi = None
        self.drawing = False
        self.roi_start_x = 0
        self.roi_start_y = 0
        self.scale_factor = 1.0
        self.detection_threshold = 0.8
        self.screenshot_in_progress = False
        
        # Create main frames
        self.setup_ui()
        
        # Initialize OpenCV variables
        self.cv_image = None
        
        # Load saved ROIs if available
        self.config_file = "fixed_ui_elements.json"
        self.load_config()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main layout frames
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Image frame with scrollbars
        self.image_frame = ttk.Frame(self.root)
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Canvas for the image with scrollbars
        self.canvas_frame = ttk.Frame(self.image_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        self.scrollbar_y = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.scrollbar_x = ttk.Scrollbar(self.image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set, yscrollcommand=self.scrollbar_y.set)
        
        # ROI list frame
        roi_frame = ttk.LabelFrame(self.root, text="ROI Elements", padding=10)
        roi_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Control buttons - first row
        ttk.Button(control_frame, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=5)
        
        # Add Take Screenshot button
        self.screenshot_button = ttk.Button(control_frame, text="Take Screenshot", command=self.take_screenshot)
        self.screenshot_button.grid(row=0, column=1, padx=5)
        # Disable the button if no ADB controller is available
        if self.adb_controller is None:
            self.screenshot_button.config(state=tk.DISABLED)
            tooltip_msg = "ADB Controller not available. Connect a device in the main interface."
            
            # Create tooltip (basic implementation)
            def show_tooltip(event):
                x, y = event.x_root, event.y_root
                tw = tk.Toplevel(self.root)
                tw.wm_overrideredirect(True)
                tw.wm_geometry(f"+{x+15}+{y+10}")
                label = ttk.Label(tw, text=tooltip_msg, background="#ffffe0", relief="solid", borderwidth=1)
                label.pack()
                self.tooltip = tw
                
            def hide_tooltip(event):
                if hasattr(self, 'tooltip'):
                    self.tooltip.destroy()
                    del self.tooltip
                    
            self.screenshot_button.bind("<Enter>", show_tooltip)
            self.screenshot_button.bind("<Leave>", hide_tooltip)
            
        ttk.Button(control_frame, text="Save Config", command=self.save_config).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Test Detection", command=self.test_detection).grid(row=0, column=3, padx=5)
        
        # ROI controls - second row
        ttk.Label(control_frame, text="Element Name:").grid(row=1, column=0, padx=5, pady=(10, 0))
        self.roi_name_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.roi_name_var, width=20).grid(row=1, column=1, padx=5, pady=(10, 0))
        
        # Add tappable checkbox
        self.roi_tappable_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Tappable", variable=self.roi_tappable_var).grid(row=1, column=2, padx=5, pady=(10, 0))
        
        # Add long press checkbox
        self.roi_long_press_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Long Press", variable=self.roi_long_press_var).grid(row=1, column=3, padx=5, pady=(10, 0))
        
        ttk.Button(control_frame, text="Add ROI", command=self.add_roi).grid(row=1, column=4, padx=5, pady=(10, 0))
        ttk.Button(control_frame, text="Delete Selected", command=self.delete_selected_roi).grid(row=1, column=5, padx=5, pady=(10, 0))
        
        # Threshold slider - third row
        ttk.Label(control_frame, text="Detection Threshold:").grid(row=2, column=0, padx=5, pady=(10, 0))
        self.threshold_var = tk.DoubleVar(value=0.8)
        threshold_slider = ttk.Scale(control_frame, from_=0.5, to=1.0, 
                                    variable=self.threshold_var, orient=tk.HORIZONTAL,
                                    length=200, command=self.update_threshold)
        threshold_slider.grid(row=2, column=1, columnspan=2, padx=5, pady=(10, 0), sticky=tk.EW)
        self.threshold_label = ttk.Label(control_frame, text="0.80")
        self.threshold_label.grid(row=2, column=3, padx=5, pady=(10, 0))
        
        # ROI Listbox
        self.roi_listbox_frame = ttk.Frame(roi_frame)
        self.roi_listbox_frame.pack(fill=tk.BOTH)
        
        self.roi_listbox = tk.Listbox(self.roi_listbox_frame, height=6)
        self.roi_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        listbox_scrollbar = ttk.Scrollbar(self.roi_listbox_frame, orient=tk.VERTICAL, 
                                        command=self.roi_listbox.yview)
        listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.roi_listbox.configure(yscrollcommand=listbox_scrollbar.set)
        
        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.roi_listbox.bind("<<ListboxSelect>>", self.on_roi_select)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                                relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set("Ready")

    def add_roi(self):
        """Add the current selection as a new ROI."""
        if self.selected_roi is None or self.cv_image is None:
            messagebox.showwarning("Warning", "Please make a selection first")
            return
        
        name = self.roi_name_var.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter a name for the ROI")
            return
        
        if name in self.roi_elements:
            if not messagebox.askyesno("Confirm", f"ROI '{name}' already exists. Replace it?"):
                return
        
        # Convert scaled coordinates back to original image size
        x1, y1, x2, y2 = map(int, self.selected_roi)
        
        # Extract the template from the original image
        template = self.cv_image[y1:y2, x1:x2].copy()
        
        # Store the ROI with the tappable and long_press flags
        tappable = self.roi_tappable_var.get()
        long_press = self.roi_long_press_var.get()
        
        self.roi_elements[name] = {
            "roi": (x1, y1, x2, y2),
            "template": template,
            "size": (template.shape[1], template.shape[0]),
            "tappable": tappable,
            "long_press": long_press
        }
        
        # Update the listbox
        self.refresh_roi_listbox()
        
        # Draw the ROI on the canvas
        self.refresh_rois()
        
        # Clear selection
        self.selected_roi = None
        self.canvas.delete("selection")
        self.roi_name_var.set("")
        
        # Prepare status message with properties
        properties = []
        if tappable:
            properties.append("tappable")
        if long_press:
            properties.append("long press")
        properties_str = ", ".join(properties) if properties else "non-tappable"
        
        self.status_var.set(f"Added ROI: {name} ({template.shape[1]}x{template.shape[0]}) [{properties_str}]")

    def refresh_roi_listbox(self):
        """Update the ROI listbox with current ROIs."""
        self.roi_listbox.delete(0, tk.END)
        for name in sorted(self.roi_elements.keys()):
            roi = self.roi_elements[name]
            indicators = []
            if roi.get("tappable", False):
                indicators.append("⊕")
            if roi.get("long_press", False):
                indicators.append("⏱")
            indicator_str = "".join(indicators)
            
            self.roi_listbox.insert(tk.END, f"{name} - {roi['size'][0]}x{roi['size'][1]} {indicator_str}")

    def save_config(self):
        """Save the ROI configuration to a JSON file."""
        if not self.roi_elements:
            messagebox.showinfo("Info", "No ROIs to save")
            return
        
        try:
            # Convert templates to file paths
            config_data = {}
            os.makedirs("templates", exist_ok=True)
            
            for name, roi_data in self.roi_elements.items():
                template_path = f"templates/{name}.png"
                cv2.imwrite(template_path, roi_data["template"])
                
                config_data[name] = {
                    "roi": roi_data["roi"],
                    "template_path": template_path,
                    "size": roi_data["size"],
                    "tappable": roi_data.get("tappable", False),
                    "long_press": roi_data.get("long_press", False)
                }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            messagebox.showinfo("Success", f"Configuration saved to {self.config_file}")
            self.status_var.set(f"Configuration saved: {len(self.roi_elements)} ROIs")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
            logger.error(f"Failed to save configuration: {str(e)}")

    def on_roi_select(self, event):
        """Handle ROI selection from the listbox."""
        if not self.roi_listbox.curselection():
            return
        
        index = self.roi_listbox.curselection()[0]
        name = list(sorted(self.roi_elements.keys()))[index]
        
        # Highlight the selected ROI
        self.canvas.delete("roi_highlight")
        roi = self.roi_elements[name]["roi"]
        x1, y1, x2, y2 = roi
        
        scaled_x1 = x1 * self.scale_factor
        scaled_y1 = y1 * self.scale_factor
        scaled_x2 = x2 * self.scale_factor
        scaled_y2 = y2 * self.scale_factor
        
        self.canvas.create_rectangle(
            scaled_x1, scaled_y1, scaled_x2, scaled_y2,
            outline="blue", width=3, tags="roi_highlight"
        )
        
        # Update ROI name entry field and checkboxes
        self.roi_name_var.set(name)
        self.roi_tappable_var.set(self.roi_elements[name].get("tappable", False))
        self.roi_long_press_var.set(self.roi_elements[name].get("long_press", False))
        self.selected_roi = roi


    def on_mouse_down(self, event):
        """Handle mouse button press event."""
        if self.cv_image is None:
            return
        
        # Start drawing a new ROI
        self.drawing = True
        self.roi_start_x = self.canvas.canvasx(event.x)
        self.roi_start_y = self.canvas.canvasy(event.y)
        
        # Clear previous selection rectangle
        self.canvas.delete("selection")

    def on_mouse_drag(self, event):
        """Handle mouse drag event to draw selection rectangle."""
        if not self.drawing:
            return
        
        # Get current mouse position (adjusted for canvas scrolling)
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Update selection rectangle
        self.canvas.delete("selection")
        self.canvas.create_rectangle(self.roi_start_x, self.roi_start_y, x, y,
                                    outline="red", width=2, tags="selection")

    def on_mouse_up(self, event):
        """Handle mouse button release event to finalize selection."""
        if not self.drawing:
            return
        
        self.drawing = False
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Ensure we have a valid selection (not just a click)
        if abs(x - self.roi_start_x) < 5 or abs(y - self.roi_start_y) < 5:
            self.canvas.delete("selection")
            return
        
        # Store the coordinates for adding ROI later
        self.selected_roi = (
            min(self.roi_start_x, x) / self.scale_factor,
            min(self.roi_start_y, y) / self.scale_factor,
            max(self.roi_start_x, x) / self.scale_factor,
            max(self.roi_start_y, y) / self.scale_factor
        )
        
        self.status_var.set(f"Selection: ({int(self.selected_roi[0])}, {int(self.selected_roi[1])}) - "
                        f"({int(self.selected_roi[2])}, {int(self.selected_roi[3])})")


    def refresh_rois(self):
        """Refresh ROI rectangles on the canvas."""
        # Clear previous ROIs
        self.canvas.delete("roi")
        
        if self.cv_image is None:
            return
        
        # Draw all ROIs
        for name, roi_data in self.roi_elements.items():
            x1, y1, x2, y2 = roi_data["roi"]
            scaled_x1 = x1 * self.scale_factor
            scaled_y1 = y1 * self.scale_factor
            scaled_x2 = x2 * self.scale_factor
            scaled_y2 = y2 * self.scale_factor
            
            # Use different colors for tappable/non-tappable ROIs
            outline_color = "green" if roi_data.get("tappable", False) else "orange"
            
            # Add visual indication for long press
            if roi_data.get("long_press", False):
                outline_color = "magenta"  # Use magenta for long-press elements
            
            self.canvas.create_rectangle(
                scaled_x1, scaled_y1, scaled_x2, scaled_y2,
                outline=outline_color, width=2, tags=("roi", f"roi_{name}")
            )
            self.canvas.create_text(
                scaled_x1 + 5, scaled_y1 + 5,
                text=name, anchor=tk.NW, fill="yellow",
                tags=("roi", f"roi_text_{name}")
            )
        
    def delete_selected_roi(self):
            """Delete the currently selected ROI."""
            if not self.roi_listbox.curselection():
                messagebox.showinfo("Info", "Please select an ROI to delete")
                return
            
            index = self.roi_listbox.curselection()[0]
            name = list(sorted(self.roi_elements.keys()))[index]
            
            if messagebox.askyesno("Confirm", f"Delete ROI '{name}'?"):
                del self.roi_elements[name]
                self.refresh_roi_listbox()
                self.refresh_rois()
                self.canvas.delete("roi_highlight")
                self.roi_name_var.set("")
                self.selected_roi = None
    
    def save_config(self):
        """Save the ROI configuration to a JSON file."""
        if not self.roi_elements:
            messagebox.showinfo("Info", "No ROIs to save")
            return
        
        try:
            # Convert templates to file paths
            config_data = {}
            os.makedirs("templates", exist_ok=True)
            
            for name, roi_data in self.roi_elements.items():
                template_path = f"templates/{name}.png"
                cv2.imwrite(template_path, roi_data["template"])
                
                config_data[name] = {
                    "roi": roi_data["roi"],
                    "template_path": template_path,
                    "size": roi_data["size"],
                    "tappable": roi_data.get("tappable", False),  # Save tappable flag
                    "long_press": roi_data.get("long_press", False)  # Save long_press flag
                }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            messagebox.showinfo("Success", f"Configuration saved to {self.config_file}")
            self.status_var.set(f"Configuration saved: {len(self.roi_elements)} ROIs")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
            logger.error(f"Failed to save configuration: {str(e)}")
    
    def load_config(self):
        """Load ROI configuration from a JSON file."""
        if not os.path.exists(self.config_file):
            logger.info(f"Configuration file {self.config_file} not found")
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            self.roi_elements = {}
            for name, roi_data in config_data.items():
                template_path = roi_data["template_path"]
                
                if os.path.exists(template_path):
                    template = cv2.imread(template_path)
                    
                    self.roi_elements[name] = {
                        "roi": tuple(roi_data["roi"]),
                        "template": template,
                        "size": tuple(roi_data["size"]),
                        "tappable": roi_data.get("tappable", False)  # Load tappable flag with default to False
                    }
                else:
                    logger.warning(f"Template file {template_path} not found for ROI {name}")
            
            # Update UI
            self.refresh_roi_listbox()
            self.status_var.set(f"Loaded {len(self.roi_elements)} ROIs from configuration")
            logger.info(f"Loaded {len(self.roi_elements)} ROIs from configuration")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
            logger.error(f"Failed to load configuration: {str(e)}")
    
    def load_image(self):
        """Load an image from file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        
        if not file_path:
            return
        
        self.load_image_from_path(file_path)

    def take_screenshot(self):
        """Take a screenshot from the connected device."""
        if self.adb_controller is None:
            messagebox.showinfo("No Device", "No device connected. Please connect a device in the main interface.")
            return
        
        if self.screenshot_in_progress:
            return
                
        self.screenshot_in_progress = True
        self.status_var.set("Taking screenshot...")
        self.screenshot_button.config(state=tk.DISABLED)
        
        # Run in a separate thread to avoid freezing UI
        threading.Thread(target=self._take_screenshot_thread, daemon=True).start()

    def _take_screenshot_thread(self):
        """Thread function to take screenshot without freezing UI."""
        try:
            # Capture screen
            screen = self.adb_controller.capture_screen()
            
            if screen is not None:
                # Save to temporary file
                temp_file = "temp_screenshot.png"
                cv2.imwrite(temp_file, screen)
                
                # Update UI from main thread
                self.root.after(0, lambda: self._load_screenshot(temp_file))
            else:
                # Update UI from main thread
                self.root.after(0, lambda: self._screenshot_failed("Failed to capture screenshot from device."))
                
        except Exception as e:
            logger.error(f"Screenshot error: {str(e)}")
            # Update UI from main thread
            self.root.after(0, lambda: self._screenshot_failed(f"Screenshot error: {str(e)}"))

    def _load_screenshot(self, file_path):
        """Load the captured screenshot."""
        self.load_image_from_path(file_path)
        self.status_var.set(f"Screenshot captured from device")
        self.screenshot_button.config(state=tk.NORMAL)
        self.screenshot_in_progress = False

    def _screenshot_failed(self, error_message):
        """Handle screenshot failure."""
        messagebox.showerror("Error", error_message)
        self.status_var.set("Screenshot failed")
        self.screenshot_button.config(state=tk.NORMAL)
        self.screenshot_in_progress = False

    def load_image_from_path(self, file_path):
        """Load an image from a specified path."""
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", f"Image file not found: {file_path}")
            return
            
        try:
            # Load the image with OpenCV
            self.cv_image = cv2.imread(file_path)
            if self.cv_image is None:
                raise ValueError("Could not load image")
            
            # Convert to RGB for display
            rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            self.original_image = rgb_image.copy()
            
            # Calculate scale factor for display (maintain aspect ratio)
            img_height, img_width = self.cv_image.shape[:2]
            max_display_width = 1000
            max_display_height = 700
            
            width_scale = max_display_width / img_width if img_width > max_display_width else 1
            height_scale = max_display_height / img_height if img_height > max_display_height else 1
            self.scale_factor = min(width_scale, height_scale)
            
            if self.scale_factor < 1:
                display_width = int(img_width * self.scale_factor)
                display_height = int(img_height * self.scale_factor)
                display_image = cv2.resize(rgb_image, (display_width, display_height))
            else:
                self.scale_factor = 1.0
                display_image = rgb_image
            
            # Create a PhotoImage object for Tkinter
            self.current_image = Image.fromarray(display_image)
            self.current_image_display = ImageTk.PhotoImage(self.current_image)
            
            # Update canvas
            self.canvas.config(scrollregion=(0, 0, display_image.shape[1], display_image.shape[0]))
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image_display, tags="image")
            
            # Display image info
            self.current_image_path = file_path
            self.status_var.set(f"Loaded: {os.path.basename(file_path)} - Original size: {img_width}x{img_height}")
            
            # Reset drawing state
            self.drawing = False
            self.selected_roi = None
            self.refresh_rois()
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            logger.error(f"Failed to load image: {str(e)}")

    def update_threshold(self, *args):
        """Update the detection threshold value."""
        value = self.threshold_var.get()
        self.detection_threshold = value
        self.threshold_label.config(text=f"{value:.2f}")
    
    def test_detection(self):
        """Test detection on the current image."""
        if self.cv_image is None:
            messagebox.showinfo("Info", "Please load an image first")
            return
        
        if not self.roi_elements:
            messagebox.showinfo("Info", "No ROI elements defined")
            return
        
        try:
            # Make a copy of the image for visualization
            detection_img = self.cv_image.copy()
            results = {}
            
            # Perform template matching for each ROI
            for name, roi_data in self.roi_elements.items():
                template = roi_data["template"]
                
                # Perform template matching
                result = cv2.matchTemplate(detection_img, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                # If match is above threshold, mark it as detected
                if max_val >= self.detection_threshold:
                    x, y = max_loc
                    w, h = template.shape[1], template.shape[0]
                    
                    # Use different colors for tappable/non-tappable ROIs
                    color = (0, 255, 0) if roi_data.get("tappable", False) else (0, 165, 255)  # Green vs Orange
                    
                    # Draw rectangle on the image
                    cv2.rectangle(detection_img, (x, y), (x+w, y+h), color, 2)
                    
                    # Add tappable indicator to label
                    tappable_indicator = "⊕" if roi_data.get("tappable", False) else "⊙"
                    cv2.putText(detection_img, f"{name} {tappable_indicator}: {max_val:.2f}", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    results[name] = {
                        "detected": True,
                        "confidence": max_val,
                        "location": (x, y),
                        "center": (x + w//2, y + h//2),
                        "size": (w, h),
                        "tappable": roi_data.get("tappable", False)
                    }
                else:
                    results[name] = {
                        "detected": False,
                        "confidence": max_val,
                        "tappable": roi_data.get("tappable", False)
                    }
            
            # Display results
            rgb_result = cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB)
            
            if self.scale_factor < 1:
                display_width = int(rgb_result.shape[1] * self.scale_factor)
                display_height = int(rgb_result.shape[0] * self.scale_factor)
                display_image = cv2.resize(rgb_result, (display_width, display_height))
            else:
                display_image = rgb_result
            
            result_image = Image.fromarray(display_image)
            self.current_image_display = ImageTk.PhotoImage(result_image)
            
            self.canvas.delete("image")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image_display, tags="image")
            
            # Print detection results
            detected_count = sum(1 for r in results.values() if r["detected"])
            tappable_count = sum(1 for r in results.values() if r["detected"] and r["tappable"])
            self.status_var.set(f"Detection results: {detected_count}/{len(results)} elements detected ({tappable_count} tappable)")
            
            # Display detailed results
            result_text = "Detection Results:\n"
            for name, result in results.items():
                tappable_marker = "⊕" if result["tappable"] else "⊙"
                if result["detected"]:
                    result_text += f"✓ {name} {tappable_marker}: Confidence {result['confidence']:.2f}, Center: {result['center']}\n"
                else:
                    result_text += f"✗ {name} {tappable_marker}: Not detected (max confidence: {result['confidence']:.2f})\n"
            
            messagebox.showinfo("Detection Results", result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            logger.error(f"Detection failed: {str(e)}", exc_info=True)


class FixedUIDetector:
    """
    Class for detecting fixed UI elements in game screenshots.
    Can be used independently of the GUI.
    """
    
    def __init__(self, config_file="fixed_ui_elements.json"):
        """Initialize the detector."""
        self.config_file = config_file
        self.roi_elements = {}
        self.detection_threshold = 0.8
        self.load_config()
    
    def load_config(self):
        """Load ROI configuration from a JSON file."""
        if not os.path.exists(self.config_file):
            logger.warning(f"Configuration file {self.config_file} not found")
            return False
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            self.roi_elements = {}
            for name, roi_data in config_data.items():
                template_path = roi_data["template_path"]
                
                if os.path.exists(template_path):
                    template = cv2.imread(template_path)
                    
                    self.roi_elements[name] = {
                        "roi": tuple(roi_data["roi"]),
                        "template": template,
                        "size": tuple(roi_data["size"]),
                        "tappable": roi_data.get("tappable", False),  # Load tappable flag
                        "long_press": roi_data.get("long_press", False)  # Load long_press flag
                    }
                else:
                    logger.warning(f"Template file {template_path} not found for ROI {name}")
            
            logger.info(f"Loaded {len(self.roi_elements)} ROIs from configuration")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return False
    
    def set_threshold(self, threshold):
        """Set the detection threshold."""
        self.detection_threshold = threshold
    
    def detect(self, image, elements=None):
        """
        Detect fixed UI elements in an image.
        
        Args:
            image: OpenCV image (BGR)
            elements: List of element names to detect (None for all)
            
        Returns:
            Dict of detection results by element name
        """
        results = {}
        
        # Filter elements if specified
        roi_elements = self.roi_elements
        if elements:
            roi_elements = {name: data for name, data in self.roi_elements.items() if name in elements}
        
        # Process each element
        for name, roi_data in roi_elements.items():
            template = roi_data["template"]
            
            # Perform template matching
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # If match is above threshold, mark it as detected
            if max_val >= self.detection_threshold:
                x, y = max_loc
                w, h = template.shape[1], template.shape[0]
                
                results[name] = {
                    "detected": True,
                    "confidence": float(max_val),
                    "location": (int(x), int(y)),
                    "center": (int(x + w//2), int(y + h//2)),
                    "size": (int(w), int(h)),
                    "tappable": roi_data.get("tappable", False),
                    "long_press": roi_data.get("long_press", False)
                }
            else:
                results[name] = {
                    "detected": False,
                    "confidence": float(max_val),
                    "tappable": roi_data.get("tappable", False),
                    "long_press": roi_data.get("long_press", False)
                }
        
        return results
    
    def get_tap_targets(self, results=None, detect_image=None):
        """
        Get a list of elements that should be tapped from the detection results.
        
        Args:
            results: Detection results from detect() method. If None and detect_image provided, perform detection
            detect_image: Image to perform detection on if results not provided
            
        Returns:
            List of tuples (element_name, center_x, center_y) for detected tappable elements
        """
        tap_targets = []
        
        # Get detection results if not provided
        if results is None:
            if detect_image is None:
                logger.error("Either results or detect_image must be provided")
                return []
            results = self.detect(detect_image)
        
        # Find all detected tappable elements (excluding long-press elements)
        for name, result in results.items():
            if result["detected"] and result.get("tappable", False) and not result.get("long_press", False):
                tap_targets.append((name, result["center"][0], result["center"][1]))
        
        return tap_targets
    
    def get_long_press_targets(self, results=None, detect_image=None):
        """
        Get a list of elements that should be long-pressed from the detection results.
        
        Args:
            results: Detection results from detect() method. If None and detect_image provided, perform detection
            detect_image: Image to perform detection on if results not provided
            
        Returns:
            List of tuples (element_name, center_x, center_y) for detected long press elements
        """
        long_press_targets = []
        
        # Get detection results if not provided
        if results is None:
            if detect_image is None:
                logger.error("Either results or detect_image must be provided")
                return []
            results = self.detect(detect_image)
        
        # Find all detected elements with 'long_press' property set to True
        # NOTE: Changed to ignore tappable flag and only check long_press flag
        for name, result in results.items():
            if result["detected"] and result.get("long_press", False):
                long_press_targets.append((name, result["center"][0], result["center"][1]))
        
        return long_press_targets
    
    def draw_detections(self, image, results=None):
        """
        Draw detection results on an image.
        
        Args:
            image: OpenCV image (BGR)
            results: Detection results (if None, perform detection)
            
        Returns:
            Image with drawn detections
        """
        if results is None:
            results = self.detect(image)
        
        img_copy = image.copy()
        
        for name, result in results.items():
            if result["detected"]:
                x, y = result["location"]
                w, h = result["size"]
                
                # Different colors for different types of elements
                if result.get("long_press", False):
                    color = (255, 0, 255)  # Magenta for long press elements
                elif result.get("tappable", False):
                    color = (0, 255, 0)  # Green for tappable elements
                else:
                    color = (0, 165, 255)  # Orange for non-tappable elements
                
                # Draw rectangle
                cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, 2)
                
                # Add indicators to label
                indicators = []
                if result.get("tappable", False):
                    indicators.append("⊕")
                if result.get("long_press", False):
                    indicators.append("⏱")
                indicator_str = "".join(indicators)
                
                # Add label with confidence
                cv2.putText(
                    img_copy, 
                    f"{name} {indicator_str}: {result['confidence']:.2f}", 
                    (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    2
                )
                
                # Mark center point
                center_x, center_y = result["center"]
                cv2.circle(img_copy, (center_x, center_y), 5, (0, 0, 255), -1)
        
        return img_copy