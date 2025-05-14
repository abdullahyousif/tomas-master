import os
import sys
import cv2
import numpy as np
import logging
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PowerBoostDetector:
    """
    Class for detecting power boost levels in Coin Master.
    Uses template matching for reliable detection.
    """
    
    def __init__(self, config_file="power_boost_templates.json"):
        """Initialize the detector with configuration file."""
        self.config_file = config_file
        self.templates = {}
        self.detection_threshold = 0.8
        self.load_config()
    
    def load_config(self):
        """Load template configuration from file."""
        if not os.path.exists(self.config_file):
            logger.warning(f"Config file {self.config_file} not found")
            return False
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            self.templates = {}
            for level, data in config_data.items():
                template_path = data["template_path"]
                
                if os.path.exists(template_path):
                    template = cv2.imread(template_path)
                    self.templates[level] = {
                        "template": template,
                        "roi": tuple(data["roi"]),
                        "size": tuple(data["size"])
                    }
                else:
                    logger.warning(f"Template file {template_path} not found for level {level}")
            
            logger.info(f"Loaded {len(self.templates)} power boost templates")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return False
    
    def set_threshold(self, threshold):
        """Set the detection threshold."""
        self.detection_threshold = threshold
    
    def detect(self, image):
        """
        Detect power boost level in the image.
        
        Args:
            image: OpenCV image (BGR)
            
        Returns:
            Detected power boost level and confidence, or None if not detected
        """
        results = {}
        
        # Process each level template
        for level, data in self.templates.items():
            template = data["template"]
            
            # Extract ROI if specified
            x1, y1, x2, y2 = data["roi"]
            if x1 < x2 and y1 < y2 and x2 <= image.shape[1] and y2 <= image.shape[0]:
                roi = image[y1:y2, x1:x2]
            else:
                # Use full image if ROI is invalid
                roi = image
            
            # Perform template matching
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Store result for this level
            results[level] = {
                "confidence": float(max_val),
                "detected": max_val >= self.detection_threshold,
                "location": (x1 + max_loc[0], y1 + max_loc[1]) if x1 < x2 and y1 < y2 else max_loc
            }
        
        # Find best match
        best_level = None
        best_confidence = 0
        
        for level, result in results.items():
            if result["detected"] and result["confidence"] > best_confidence:
                best_level = level
                best_confidence = result["confidence"]
        
        if best_level:
            return best_level, best_confidence
        else:
            return None, 0
    
    def draw_detection(self, image, level=None, results=None):
        """
        Draw detection results on the image.
        
        Args:
            image: OpenCV image (BGR)
            level: Detected level (if None, will be determined)
            results: Detection results (if None, detection will be performed)
            
        Returns:
            Image with detection visualization
        """
        img_copy = image.copy()
        
        # If no results provided, detect
        if results is None:
            level, confidence = self.detect(image)
            if not level:
                return img_copy
        
        # Draw ROI and label for the detected level
        if level and level in self.templates:
            x1, y1, x2, y2 = self.templates[level]["roi"]
            template = self.templates[level]["template"]
            w, h = template.shape[1], template.shape[0]
            
            # Draw the ROI
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(
                img_copy, 
                f"Power Boost: {level}", 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
        
        return img_copy


class PowerBoostTemplateEditor:
    """GUI for creating and editing power boost level templates."""
    
    def __init__(self, root, adb_controller=None):
        """Initialize the template editor."""
        self.root = root
        self.root.title("Power Boost Template Editor")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Store ADB controller reference if provided
        self.adb_controller = adb_controller
        
        # Variables
        self.current_image_path = None
        self.current_image = None
        self.cv_image = None
        self.templates = {}  # Level -> template data mapping
        self.selected_roi = None
        self.drawing = False
        self.roi_start_x = 0
        self.roi_start_y = 0
        self.scale_factor = 1.0
        self.detection_threshold = 0.7
        self.screenshot_in_progress = False
        
        # Power boost levels
        self.power_boost_levels = [
            "X1", "X2", "X3", "X15", "X50", "X400", "X1500", "X6000", "X20000"
        ]
        self.selected_level = tk.StringVar(value=self.power_boost_levels[0])
        
        # UI setup
        self.setup_ui()
        
        # Load saved templates if available
        self.config_file = "power_boost_templates.json"
        self.load_config()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main layout - split into left panel (controls) and right panel (image display)
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        left_frame = ttk.Frame(main_paned, padding=10, width=300)
        main_paned.add(left_frame, weight=1)
        
        # Right panel - Image display
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        
        # === Left panel controls ===
        # Control buttons
        control_frame = ttk.LabelFrame(left_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=2)
        
        # Add Take Screenshot button
        self.screenshot_button = ttk.Button(control_frame, text="Take Screenshot", command=self.take_screenshot)
        self.screenshot_button.pack(fill=tk.X, pady=2)
        
        # Disable if no ADB controller
        if self.adb_controller is None:
            self.screenshot_button.config(state=tk.DISABLED)
        
        ttk.Button(control_frame, text="Save Templates", command=self.save_config).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Test Detection", command=self.test_detection).pack(fill=tk.X, pady=2)
        
        # Power boost level selection
        level_frame = ttk.LabelFrame(left_frame, text="Power Boost Level", padding=10)
        level_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(level_frame, text="Select Level:").pack(anchor=tk.W)
        level_combo = ttk.Combobox(level_frame, textvariable=self.selected_level, values=self.power_boost_levels)
        level_combo.pack(fill=tk.X, pady=5)
        
        ttk.Button(level_frame, text="Add Template", command=self.add_template).pack(fill=tk.X)
        ttk.Button(level_frame, text="Delete Template", command=self.delete_template).pack(fill=tk.X, pady=2)
        
        # Template list
        templates_frame = ttk.LabelFrame(left_frame, text="Templates", padding=10)
        templates_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.template_listbox = tk.Listbox(templates_frame, height=10)
        self.template_listbox.pack(fill=tk.BOTH, expand=True)
        self.template_listbox.bind("<<ListboxSelect>>", self.on_template_select)
        
        # Threshold slider
        threshold_frame = ttk.LabelFrame(left_frame, text="Detection Threshold", padding=10)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        self.threshold_var = tk.DoubleVar(value=0.8)
        threshold_slider = ttk.Scale(
            threshold_frame, 
            from_=0.5, 
            to=1.0, 
            variable=self.threshold_var,
            orient=tk.HORIZONTAL,
            command=self.update_threshold
        )
        threshold_slider.pack(fill=tk.X)
        self.threshold_label = ttk.Label(threshold_frame, text="0.80")
        self.threshold_label.pack(anchor=tk.E)
        
        # Right panel - image display
        self.canvas_frame = ttk.Frame(right_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        self.scrollbar_y = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.scrollbar_x = ttk.Scrollbar(right_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set, yscrollcommand=self.scrollbar_y.set)
        
        # Logs
        log_frame = ttk.LabelFrame(right_frame, text="Logs", padding=5)
        log_frame.pack(fill=tk.X, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, wrap=tk.WORD, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Add log handler
        log_handler = logging.StreamHandler()
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(log_handler)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
    
    def log(self, message):
        """Add a message to the log."""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')
        logger.info(message)
    
    def load_image(self):
        """Load an image from file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        
        if not file_path:
            return
        
        self.load_image_from_path(file_path)
    
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
            
            # Calculate scale factor for display
            img_height, img_width = self.cv_image.shape[:2]
            max_display_width = 800
            max_display_height = 600
            
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
            
            # Store image path
            self.current_image_path = file_path
            self.status_var.set(f"Loaded: {os.path.basename(file_path)} - Original size: {img_width}x{img_height}")
            self.log(f"Loaded image: {os.path.basename(file_path)}")
            
            # Draw existing templates
            self.refresh_templates()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            logger.error(f"Failed to load image: {str(e)}")
    
    def take_screenshot(self):
        """Take a screenshot using ADB controller."""
        if self.adb_controller is None:
            messagebox.showinfo("Info", "ADB controller not available")
            return
        
        if self.screenshot_in_progress:
            return
        
        self.screenshot_in_progress = True
        self.status_var.set("Taking screenshot...")
        self.screenshot_button.config(state=tk.DISABLED)
        
        # Run in a separate thread
        threading.Thread(target=self._take_screenshot_thread, daemon=True).start()
    
    def _take_screenshot_thread(self):
        """Thread function for taking screenshots."""
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
                self.root.after(0, lambda: self._screenshot_failed("Failed to capture screenshot"))
        except Exception as e:
            logger.error(f"Screenshot error: {str(e)}")
            self.root.after(0, lambda: self._screenshot_failed(f"Error: {str(e)}"))
    
    def _load_screenshot(self, file_path):
        """Load the captured screenshot."""
        self.load_image_from_path(file_path)
        self.status_var.set("Screenshot captured")
        self.screenshot_button.config(state=tk.NORMAL)
        self.screenshot_in_progress = False
    
    def _screenshot_failed(self, error_message):
        """Handle screenshot failure."""
        messagebox.showerror("Error", error_message)
        self.status_var.set("Screenshot failed")
        self.screenshot_button.config(state=tk.NORMAL)
        self.screenshot_in_progress = False
    
    def update_threshold(self, *args):
        """Update the threshold value."""
        value = self.threshold_var.get()
        self.detection_threshold = value
        self.threshold_label.config(text=f"{value:.2f}")
    
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
        self.canvas.create_rectangle(
            self.roi_start_x, self.roi_start_y, x, y,
            outline="red", width=2, tags="selection"
        )
    
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
        
        # Calculate corrected coordinates
        x1 = int(min(self.roi_start_x, x) / self.scale_factor)
        y1 = int(min(self.roi_start_y, y) / self.scale_factor)
        x2 = int(max(self.roi_start_x, x) / self.scale_factor)
        y2 = int(max(self.roi_start_y, y) / self.scale_factor)
        
        # Store the coordinates
        self.selected_roi = (x1, y1, x2, y2)
        
        # Debug log
        self.log(f"ROI coordinates: ({x1}, {y1}, {x2}, {y2})")
        
        width = x2 - x1
        height = y2 - y1
        
        self.status_var.set(f"Selection: ({x1}, {y1}) - ({x2}, {y2}) - Size: {width}x{height}")
    
    def add_template(self):
        """Add the current selection as a template for the selected power boost level."""
        if self.cv_image is None:
            messagebox.showinfo("Info", "Please load an image first")
            return
        
        if self.selected_roi is None:
            messagebox.showinfo("Info", "Please make a selection first")
            return
        
        level = self.selected_level.get()
        if not level:
            messagebox.showinfo("Info", "Please select a power boost level")
            return
        
        if level in self.templates:
            if not messagebox.askyesno("Confirm", f"Template for level {level} already exists. Replace it?"):
                return
        
        # Extract the template from the image
        x1, y1, x2, y2 = self.selected_roi
        
        # Verify coordinates are within image bounds
        height, width = self.cv_image.shape[:2]
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            self.log(f"WARNING: ROI coordinates ({x1}, {y1}, {x2}, {y2}) exceed image bounds ({width}, {height})")
            # Clamp coordinates to image bounds
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))
            self.selected_roi = (x1, y1, x2, y2)
            self.log(f"Clamped ROI to: ({x1}, {y1}, {x2}, {y2})")
        
        # Log the final coordinates used for template extraction
        self.log(f"Extracting template for {level} from ROI: ({x1}, {y1}, {x2}, {y2})")
        
        template = self.cv_image[y1:y2, x1:x2].copy()
        
        # Create templates directory if it doesn't exist
        os.makedirs("templates", exist_ok=True)
        
        # Save template image
        template_path = f"templates/power_boost_{level}.png"
        cv2.imwrite(template_path, template)
        
        # Store template data
        self.templates[level] = {
            "roi": self.selected_roi,
            "template_path": template_path,
            "size": (template.shape[1], template.shape[0])
        }
        
        # Update template list
        self.refresh_template_list()
        
        # Refresh canvas
        self.refresh_templates()
        
        # Clear selection
        self.selected_roi = None
        self.canvas.delete("selection")
        
        self.log(f"Added template for level {level}")
        self.status_var.set(f"Added template for level {level} - Size: {template.shape[1]}x{template.shape[0]}")
    
    def delete_template(self):
        """Delete the template for the selected power boost level."""
        level = self.selected_level.get()
        
        if level not in self.templates:
            messagebox.showinfo("Info", f"No template found for level {level}")
            return
        
        if not messagebox.askyesno("Confirm", f"Delete template for level {level}?"):
            return
        
        # Delete template image if it exists
        template_path = self.templates[level]["template_path"]
        if os.path.exists(template_path):
            try:
                os.remove(template_path)
            except Exception as e:
                logger.warning(f"Failed to delete template file: {str(e)}")
        
        # Remove template data
        del self.templates[level]
        
        # Update template list
        self.refresh_template_list()
        
        # Refresh canvas
        self.refresh_templates()
        
        self.log(f"Deleted template for level {level}")
        self.status_var.set(f"Deleted template for level {level}")
    
    def refresh_template_list(self):
        """Update the template listbox."""
        self.template_listbox.delete(0, tk.END)
        
        for level in sorted(self.templates.keys(), key=lambda x: self.power_boost_levels.index(x) if x in self.power_boost_levels else 999):
            roi = self.templates[level]["roi"]
            width = roi[2] - roi[0]
            height = roi[3] - roi[1]
            self.template_listbox.insert(tk.END, f"{level} - Size: {width}x{height}")
    
    def on_template_select(self, event):
        """Handle template selection from the listbox."""
        if not self.template_listbox.curselection():
            return
        
        index = self.template_listbox.curselection()[0]
        level = sorted(self.templates.keys(), key=lambda x: self.power_boost_levels.index(x) if x in self.power_boost_levels else 999)[index]
        
        # Update selected level
        self.selected_level.set(level)
        
        # Highlight template on canvas
        self.canvas.delete("roi_highlight")
        
        roi = self.templates[level]["roi"]
        x1, y1, x2, y2 = roi
        
        # Scale coordinates to display size
        scaled_x1 = x1 * self.scale_factor
        scaled_y1 = y1 * self.scale_factor
        scaled_x2 = x2 * self.scale_factor
        scaled_y2 = y2 * self.scale_factor
        
        self.canvas.create_rectangle(
            scaled_x1, scaled_y1, scaled_x2, scaled_y2,
            outline="blue", width=2, tags="roi_highlight"
        )
        
        self.status_var.set(f"Selected template: {level}")
    
    def refresh_templates(self):
        """Refresh template visualization on the canvas."""
        # Clear previous ROIs
        self.canvas.delete("template_roi")
        
        if self.cv_image is None:
            return
        
        # Draw all templates
        for level, data in self.templates.items():
            x1, y1, x2, y2 = data["roi"]
            
            # Scale coordinates to display size
            scaled_x1 = x1 * self.scale_factor
            scaled_y1 = y1 * self.scale_factor
            scaled_x2 = x2 * self.scale_factor
            scaled_y2 = y2 * self.scale_factor
            
            # Draw rectangle
            self.canvas.create_rectangle(
                scaled_x1, scaled_y1, scaled_x2, scaled_y2,
                outline="green", width=2, tags=("template_roi", f"template_{level}")
            )
            
            # Draw label
            self.canvas.create_text(
                scaled_x1 + 5, scaled_y1 + 5,
                text=level, anchor=tk.NW, fill="yellow",
                tags=("template_roi", f"template_text_{level}")
            )
    
    def save_config(self):
        """Save templates to configuration file."""
        if not self.templates:
            messagebox.showinfo("Info", "No templates to save")
            return
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.templates, f, indent=4)
            
            self.log(f"Saved {len(self.templates)} templates to {self.config_file}")
            messagebox.showinfo("Success", f"Templates saved to {self.config_file}")
            self.status_var.set(f"Templates saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save templates: {str(e)}")
            messagebox.showerror("Error", f"Failed to save templates: {str(e)}")

    # Add to game_logic_integrator.py
    
    def load_config(self):
        """Load templates from configuration file."""
        if not os.path.exists(self.config_file):
            self.log(f"Configuration file {self.config_file} not found")
            return
        
        try:
            with open(self.config_file, 'r') as f:
                self.templates = json.load(f)
            
            # Verify template files exist
            valid_templates = {}
            for level, data in self.templates.items():
                template_path = data["template_path"]
                
                if os.path.exists(template_path):
                    valid_templates[level] = data
                else:
                    logger.warning(f"Template file {template_path} not found for level {level}")
            
            self.templates = valid_templates
            self.refresh_template_list()
            
            self.log(f"Loaded {len(self.templates)} templates from {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
    
    def test_detection(self):
        """Test detection of power boost levels in the current image."""
        if self.cv_image is None:
            messagebox.showinfo("Info", "Please load an image first")
            return
        
        if not self.templates:
            messagebox.showinfo("Info", "No templates defined")
            return
        
        try:
            # Create PowerBoostDetector
            detector = PowerBoostDetector(self.config_file)
            detector.set_threshold(self.detection_threshold)
            
            # Detect level
            level, confidence = detector.detect(self.cv_image)
            
            if level:
                # Draw detection result
                result_image = detector.draw_detection(self.cv_image, level)
                
                # Convert to RGB for display
                rgb_result = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                
                # Resize for display
                if self.scale_factor < 1:
                    display_width = int(rgb_result.shape[1] * self.scale_factor)
                    display_height = int(rgb_result.shape[0] * self.scale_factor)
                    display_image = cv2.resize(rgb_result, (display_width, display_height))
                else:
                    display_image = rgb_result
                
                # Update canvas
                result_pil = Image.fromarray(display_image)
                result_photo = ImageTk.PhotoImage(result_pil)
                
                self.canvas.delete("image")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=result_photo, tags="image")
                self.canvas.image = result_photo  # Keep a reference
                
                self.log(f"Detected power boost level: {level} with confidence {confidence:.2f}")
                self.status_var.set(f"Detected level: {level} (confidence: {confidence:.2f})")
                
                # Show message box
                messagebox.showinfo("Detection Result", f"Detected Power Boost Level: {level}\nConfidence: {confidence:.2f}")
            else:
                self.log("No power boost level detected")
                self.status_var.set("No power boost level detected")
                messagebox.showinfo("Detection Result", "No power boost level detected")
        
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            messagebox.showerror("Error", f"Detection error: {str(e)}")


def main():
    # Create root window
    root = tk.Tk()
    
    # Create the editor
    editor = PowerBoostTemplateEditor(root)
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()
