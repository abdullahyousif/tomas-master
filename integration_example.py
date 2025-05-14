import cv2
import numpy as np
import time
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import os
from typing import List, Dict, Tuple, Optional, Any

# Import both detection systems
from detector import YOLODetector, Detection
from fixed_ui_detector import FixedUIDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridDetector:
    """
    Hybrid detector that combines YOLO for dynamic elements and 
    template matching for fixed UI elements.
    """
    
    def __init__(self, 
                 yolo_model_path: str = "models/my_model.pt",
                 fixed_ui_config: str = "fixed_ui_elements.json",
                 yolo_conf_threshold: float = 0.5,
                 template_threshold: float = 0.8):
        """
        Initialize the hybrid detector.
        
        Args:
            yolo_model_path: Path to YOLO model
            fixed_ui_config: Path to fixed UI configuration file
            yolo_conf_threshold: Confidence threshold for YOLO
            template_threshold: Threshold for template matching
        """
        # Initialize YOLO detector
        logger.info("Initializing YOLO detector...")
        self.yolo_detector = YOLODetector(
            model_path=yolo_model_path,
            conf_threshold=yolo_conf_threshold
        )
        
        # Initialize fixed UI detector
        logger.info("Initializing fixed UI detector...")
        self.fixed_detector = FixedUIDetector(config_file=fixed_ui_config)
        self.fixed_detector.set_threshold(template_threshold)
        
        # Store thresholds
        self.yolo_conf_threshold = yolo_conf_threshold
        self.template_threshold = template_threshold
        
        # Class mapping for YOLO detector
        self.class_names = self.yolo_detector.class_names
        
        logger.info("Hybrid detector initialized")
    
    def detect(self, image: np.ndarray, detect_dynamic: bool = True, detect_fixed: bool = True,
               fixed_elements: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform detection using both detectors.
        
        Args:
            image: Input image (BGR format)
            detect_dynamic: Whether to detect dynamic elements with YOLO
            detect_fixed: Whether to detect fixed UI elements
            fixed_elements: List of specific fixed elements to detect (None for all)
            
        Returns:
            Dictionary containing detection results from both detectors
        """
        start_time = time.time()
        results = {
            "dynamic_objects": [],
            "fixed_ui": {},
            "detection_time": 0
        }
        
        # Run YOLO detector for dynamic objects
        if detect_dynamic:
            dynamic_results = self.yolo_detector.detect(image)
            results["dynamic_objects"] = dynamic_results
        
        # Run fixed UI detector
        if detect_fixed:
            fixed_results = self.fixed_detector.detect(image, elements=fixed_elements)
            results["fixed_ui"] = fixed_results
        
        # Calculate total detection time
        results["detection_time"] = time.time() - start_time
        
        return results
    
    def get_all_tap_targets(self, results: Dict[str, Any]) -> List[Tuple[str, float, float]]:
        """
        Get a unified list of all elements that should be tapped from both detectors.
        
        Args:
            results: Detection results from detect() method
            
        Returns:
            List of tuples (element_name, center_x, center_y) for all tappable elements
        """
        tap_targets = []
        
        # Get tap targets from YOLO detector
        if "dynamic_objects" in results and results["dynamic_objects"]:
            dynamic_targets = self.yolo_detector.get_tap_targets(results["dynamic_objects"])
            tap_targets.extend(dynamic_targets)
        
        # Get tap targets from fixed UI detector (all tappable elements)
        if "fixed_ui" in results:
            for name, result in results["fixed_ui"].items():
                if result["detected"] and result.get("tappable", True):  # Default to True to tap all detected elements
                    tap_targets.append((name, result["center"][0], result["center"][1]))
        
        return tap_targets
    
    def draw_results(self, image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Draw detection results on the image.
        
        Args:
            image: Input image (BGR format)
            results: Detection results from detect()
            
        Returns:
            Image with visualization of detection results
        """
        img_copy = image.copy()
        
        # Draw YOLO detections
        if "dynamic_objects" in results and results["dynamic_objects"]:
            for detection in results["dynamic_objects"]:
                x1, y1, x2, y2 = detection.bbox
                label = f"{detection.class_name}: {detection.confidence:.2f}"
                
                # Determine if this is a tappable element
                is_tappable = detection.class_name in self.yolo_detector.tap_elements
                color = (0, 0, 255) if not is_tappable else (0, 255, 0)  # Red for non-tappable, Green for tappable
                
                # Draw bounding box
                cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw label background
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(
                    img_copy, 
                    (int(x1), int(y1) - text_size[1] - 10), 
                    (int(x1) + text_size[0], int(y1)), 
                    color, 
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    img_copy, 
                    label, 
                    (int(x1), int(y1) - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2
                )
                
                # Mark center point for tappable elements
                if is_tappable:
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    cv2.circle(img_copy, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Draw fixed UI detections
        if "fixed_ui" in results:
            for name, result in results["fixed_ui"].items():
                if result["detected"]:
                    x, y = result["location"]
                    w, h = result["size"]
                    
                    # Use different colors for tappable vs non-tappable elements
                    is_tappable = result.get("tappable", True)  # Default to True for backward compatibility
                    color = (0, 255, 0) if is_tappable else (0, 165, 255)  # Green for tappable, Orange for non-tappable
                    
                    # Draw bounding box
                    cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, 2)
                    
                    # Add tappable indicator to label
                    tappable_indicator = "⊕" if is_tappable else "⊙"
                    label = f"{name} {tappable_indicator}: {result['confidence']:.2f}"
                    
                    # Draw label background
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(
                        img_copy, 
                        (x, y - text_size[1] - 10), 
                        (x + text_size[0], y), 
                        color, 
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        img_copy, 
                        label, 
                        (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (255, 255, 255), 
                        2
                    )
                    
                    # Mark center point for tappable elements
                    if is_tappable:
                        center_x, center_y = result["center"]
                        cv2.circle(img_copy, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Add detection time
        if "detection_time" in results:
            time_text = f"Detection time: {results['detection_time']:.3f}s"
            cv2.putText(
                img_copy, 
                time_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
        
        return img_copy


class HybridDetectorGUI:
    """
    GUI for the Hybrid Detector, allowing users to:
    1. Load and view images
    2. Configure detection parameters
    3. Run detection and view results
    4. Open the Fixed UI Selector for configuring fixed UI elements
    """
    
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Game Object Hybrid Detector")
        self.root.geometry("1280x800")
        
        # Variables
        self.current_image_path = None
        self.current_image = None
        self.current_results = None
        self.is_processing = False
        self.scale_factor = 1.0
        
        # Configure detector
        self.yolo_model_path = tk.StringVar(value="models/my_model.pt")
        self.fixed_ui_config = tk.StringVar(value="fixed_ui_elements.json")
        self.yolo_conf_threshold = tk.DoubleVar(value=0.5)
        self.template_threshold = tk.DoubleVar(value=0.8)
        self.detect_dynamic = tk.BooleanVar(value=True)
        self.detect_fixed = tk.BooleanVar(value=True)
        self.show_tap_targets = tk.BooleanVar(value=True)
        
        # Create detector instance (will be initialized when running detection)
        self.detector = None
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main layout - split into left panel (controls) and right panel (image display)
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        left_frame = ttk.Frame(main_paned, padding=10)
        main_paned.add(left_frame, weight=1)
        
        # Right panel - Image display
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=4)
        
        # === Left panel controls ===
        # Model configuration
        model_frame = ttk.LabelFrame(left_frame, text="Detection Configuration", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        # YOLO model path
        ttk.Label(model_frame, text="YOLO Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(model_frame, textvariable=self.yolo_model_path).grid(row=0, column=1, sticky=tk.EW, pady=2)
        ttk.Button(model_frame, text="Browse", command=self.browse_yolo_model).grid(row=0, column=2, pady=2)
        
        # Fixed UI config
        ttk.Label(model_frame, text="Fixed UI Config:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(model_frame, textvariable=self.fixed_ui_config).grid(row=1, column=1, sticky=tk.EW, pady=2)
        ttk.Button(model_frame, text="Browse", command=self.browse_fixed_config).grid(row=1, column=2, pady=2)
        ttk.Button(model_frame, text="Edit UI Elements", command=self.open_ui_selector).grid(row=1, column=3, pady=2)
        
        # Thresholds
        ttk.Label(model_frame, text="YOLO Confidence:").grid(row=2, column=0, sticky=tk.W, pady=2)
        yolo_threshold_slider = ttk.Scale(model_frame, from_=0.1, to=1.0, variable=self.yolo_conf_threshold, 
                                          orient=tk.HORIZONTAL, length=150)
        yolo_threshold_slider.grid(row=2, column=1, sticky=tk.EW, pady=2)
        self.yolo_threshold_label = ttk.Label(model_frame, text="0.50")
        self.yolo_threshold_label.grid(row=2, column=2, pady=2)
        yolo_threshold_slider.configure(command=self.update_yolo_threshold)
        
        ttk.Label(model_frame, text="Template Confidence:").grid(row=3, column=0, sticky=tk.W, pady=2)
        template_threshold_slider = ttk.Scale(model_frame, from_=0.5, to=1.0, variable=self.template_threshold, 
                                              orient=tk.HORIZONTAL, length=150)
        template_threshold_slider.grid(row=3, column=1, sticky=tk.EW, pady=2)
        self.template_threshold_label = ttk.Label(model_frame, text="0.80")
        self.template_threshold_label.grid(row=3, column=2, pady=2)
        template_threshold_slider.configure(command=self.update_template_threshold)
        
        # Detection options
        ttk.Checkbutton(model_frame, text="Detect Dynamic Objects", variable=self.detect_dynamic).grid(
            row=4, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(model_frame, text="Detect Fixed UI", variable=self.detect_fixed).grid(
            row=5, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(model_frame, text="Show Tap Targets", variable=self.show_tap_targets).grid(
            row=6, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Detection control buttons
        ttk.Button(left_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=5)
        ttk.Button(left_frame, text="Run Detection", command=self.run_detection).pack(fill=tk.X, pady=5)
        ttk.Button(left_frame, text="Save Results", command=self.save_results).pack(fill=tk.X, pady=5)
        ttk.Button(left_frame, text="Test Tap Sequence", command=self.test_tap_sequence).pack(fill=tk.X, pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(left_frame, text="Detection Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a text widget with scrollbar for results
        result_scroll = ttk.Scrollbar(results_frame)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text = tk.Text(results_frame, wrap=tk.WORD, height=20, width=30)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        result_scroll.config(command=self.result_text.yview)
        self.result_text.config(yscrollcommand=result_scroll.set)
        self.result_text.config(state=tk.DISABLED)
        
        # === Right panel - Image display ===
        # Canvas for image display with scrollbars
        self.canvas_frame = ttk.Frame(right_frame)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        self.scrollbar_y = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.scrollbar_x = ttk.Scrollbar(right_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set, yscrollcommand=self.scrollbar_y.set)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set("Ready")
    
    def update_yolo_threshold(self, *args):
        """Update the YOLO threshold display."""
        value = self.yolo_conf_threshold.get()
        self.yolo_threshold_label.config(text=f"{value:.2f}")
    
    def update_template_threshold(self, *args):
        """Update the template threshold display."""
        value = self.template_threshold.get()
        self.template_threshold_label.config(text=f"{value:.2f}")
    
    def browse_yolo_model(self):
        """Browse for YOLO model file."""
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("PyTorch Models", "*.pt"), ("All Files", "*.*")]
        )
        if file_path:
            self.yolo_model_path.set(file_path)
    
    def browse_fixed_config(self):
        """Browse for fixed UI configuration file."""
        file_path = filedialog.askopenfilename(
            title="Select Fixed UI Config",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if file_path:
            self.fixed_ui_config.set(file_path)
    
    def open_ui_selector(self):
        """Open the Fixed UI Element Selector."""
        try:
            # Create a new top-level window
            selector_window = tk.Toplevel(self.root)
            
            # Import is done here to avoid circular imports
            from fixed_ui_detector import ROISelector
            
            # Create ROI selector in the new window
            roi_selector = ROISelector(selector_window)
            
            # If we have a current image, load it in the selector
            if self.current_image_path:
                selector_window.after(500, lambda: roi_selector.load_image_from_path(self.current_image_path))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open UI Selector: {str(e)}")
            logger.error(f"Failed to open UI Selector: {str(e)}", exc_info=True)
    
    def load_image(self):
        """Load an image from file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        
        if not file_path:
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
            max_display_width = 900
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
            
            # Store image path
            self.current_image_path = file_path
            self.status_var.set(f"Loaded: {os.path.basename(file_path)} - Original size: {img_width}x{img_height}")
            
            # Clear previous results
            self.current_results = None
            self.clear_result_text()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            logger.error(f"Failed to load image: {str(e)}", exc_info=True)
    
    def initialize_detector(self):
        """Initialize or reinitialize the detector with current settings."""
        try:
            # Create a new detector instance with current settings
            self.detector = HybridDetector(
                yolo_model_path=self.yolo_model_path.get(),
                fixed_ui_config=self.fixed_ui_config.get(),
                yolo_conf_threshold=self.yolo_conf_threshold.get(),
                template_threshold=self.template_threshold.get()
            )
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize detector: {str(e)}")
            logger.error(f"Failed to initialize detector: {str(e)}", exc_info=True)
            return False
    
    def run_detection(self):
        """Run detection on the current image."""
        if self.cv_image is None:
            messagebox.showinfo("Info", "Please load an image first")
            return
        
        if self.is_processing:
            messagebox.showinfo("Info", "Detection is already running")
            return
        
        # Set processing flag
        self.is_processing = True
        self.status_var.set("Running detection...")
        
        # Run detection in a separate thread to keep UI responsive
        threading.Thread(target=self._run_detection_thread, daemon=True).start()
    
    def _run_detection_thread(self):
        """Thread function to run detection."""
        try:
            # Initialize detector if needed
            if self.detector is None:
                if not self.initialize_detector():
                    self.is_processing = False
                    return
            
            # Run detection
            results = self.detector.detect(
                image=self.cv_image,
                detect_dynamic=self.detect_dynamic.get(),
                detect_fixed=self.detect_fixed.get()
            )
            
            # Process and display results
            self.current_results = results
            self.display_results(results)
            
            # Create detection visualization
            result_image = self.detector.draw_results(self.cv_image, results)
            
            # Convert to RGB for display
            rgb_result = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            # Resize for display if needed
            if self.scale_factor < 1:
                display_width = int(rgb_result.shape[1] * self.scale_factor)
                display_height = int(rgb_result.shape[0] * self.scale_factor)
                display_image = cv2.resize(rgb_result, (display_width, display_height))
            else:
                display_image = rgb_result
            
            # Update canvas display
            self.root.after(0, lambda: self._update_display(display_image))
            
        except Exception as e:
            # Handle errors
            self.root.after(0, lambda: messagebox.showerror("Error", f"Detection failed: {str(e)}"))
            logger.error(f"Detection failed: {str(e)}", exc_info=True)
        
        finally:
            # Reset processing flag
            self.is_processing = False
            self.root.after(0, lambda: self.status_var.set("Detection completed"))
    
    def _update_display(self, display_image):
        """Update the image display on the canvas."""
        result_image = Image.fromarray(display_image)
        self.current_image_display = ImageTk.PhotoImage(result_image)
        
        self.canvas.delete("image")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image_display, tags="image")
    
    def display_results(self, results):
        """Display detection results in the text widget."""
        self.clear_result_text()
        
        self.result_text.config(state=tk.NORMAL)
        
        # Add detection time
        self.result_text.insert(tk.END, f"Detection Time: {results['detection_time']:.3f}s\n\n")
        
        # Add dynamic object results
        if "dynamic_objects" in results and self.detect_dynamic.get():
            dynamic_count = len(results["dynamic_objects"])
            self.result_text.insert(tk.END, f"Dynamic Objects: {dynamic_count}\n")
            
            if dynamic_count > 0:
                # Group by class
                by_class = {}
                for det in results["dynamic_objects"]:
                    class_name = det.class_name
                    if class_name not in by_class:
                        by_class[class_name]
                for det in results["dynamic_objects"]:
                    class_name = det.class_name
                    if class_name not in by_class:
                        by_class[class_name] = []
                    by_class[class_name].append(det)
                
                # List objects by class
                for class_name, detections in by_class.items():
                    is_tappable = class_name in self.detector.yolo_detector.tap_elements
                    tappable_mark = "⊕" if is_tappable else "⊙"
                    self.result_text.insert(tk.END, f"\n- {class_name} {tappable_mark}: {len(detections)}\n")
                    for i, det in enumerate(detections):
                        self.result_text.insert(
                            tk.END, 
                            f"  #{i+1}: Conf={det.confidence:.2f}, "
                            f"Pos=({int(det.bbox[0])},{int(det.bbox[1])})\n"
                        )
            self.result_text.insert(tk.END, "\n")
        
        # Add fixed UI results
        if "fixed_ui" in results and self.detect_fixed.get():
            fixed_results = results["fixed_ui"]
            detected_count = sum(1 for r in fixed_results.values() if r["detected"])
            tappable_count = sum(1 for r in fixed_results.values() if r["detected"] and r.get("tappable", True))
            total_count = len(fixed_results)
            
            self.result_text.insert(tk.END, f"Fixed UI Elements: {detected_count}/{total_count} ({tappable_count} tappable)\n\n")
            
            # List detected elements
            if detected_count > 0:
                self.result_text.insert(tk.END, "Detected Elements:\n")
                for name, result in fixed_results.items():
                    if result["detected"]:
                        tappable_mark = "⊕" if result.get("tappable", True) else "⊙"
                        center_x, center_y = result["center"]
                        self.result_text.insert(
                            tk.END, 
                            f"- {name} {tappable_mark}: Conf={result['confidence']:.2f}, "
                            f"Center=({center_x},{center_y})\n"
                        )
            
            # List missed elements
            missed_count = total_count - detected_count
            if missed_count > 0:
                self.result_text.insert(tk.END, "\nMissed Elements:\n")
                for name, result in fixed_results.items():
                    if not result["detected"]:
                        tappable_mark = "⊕" if result.get("tappable", True) else "⊙"
                        self.result_text.insert(
                            tk.END, 
                            f"- {name} {tappable_mark}: Max Conf={result['confidence']:.2f}\n"
                        )
        
        # Add tap target information
        if self.show_tap_targets.get():
            tap_targets = self.detector.get_all_tap_targets(results)
            
            if tap_targets:
                self.result_text.insert(tk.END, f"\nTap Targets ({len(tap_targets)}):\n")
                for i, (name, x, y) in enumerate(tap_targets):
                    self.result_text.insert(
                        tk.END,
                        f"{i+1}. {name} at ({int(x)}, {int(y)})\n"
                    )
            else:
                self.result_text.insert(tk.END, "\nNo tap targets detected\n")
        
        self.result_text.see(tk.END)
        self.result_text.config(state=tk.DISABLED)
    
    def clear_result_text(self):
        """Clear the result text widget."""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
    
    def test_tap_sequence(self):
        """Test the tap sequence by highlighting each tap target in order."""
        if not self.current_results:
            messagebox.showinfo("Info", "Run detection first to identify tap targets")
            return
        
        # Get all tap targets from detection results
        tap_targets = self.detector.get_all_tap_targets(self.current_results)
        
        if not tap_targets:
            messagebox.showinfo("Info", "No tap targets detected in the current image")
            return
        
        # Start tap sequence animation
        self.animate_tap_sequence(tap_targets, 0)
    
    def animate_tap_sequence(self, tap_targets, current_index):
        """Animate a tap sequence on the image."""
        if current_index >= len(tap_targets):
            # Animation completed, restore original display
            result_image = self.detector.draw_results(self.cv_image, self.current_results)
            
            # Highlight all tap targets
            if self.show_tap_targets.get():
                for i, (name, x, y) in enumerate(tap_targets):
                    # Draw a larger circle around the tap point with sequence number
                    cv2.circle(result_image, (int(x), int(y)), 20, (0, 255, 255), 2)
                    cv2.putText(
                        result_image,
                        f"{i+1}",
                        (int(x) - 5, int(y) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )
            
            # Display final result
            rgb_result = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            if self.scale_factor < 1:
                display_width = int(rgb_result.shape[1] * self.scale_factor)
                display_height = int(rgb_result.shape[0] * self.scale_factor)
                display_image = cv2.resize(rgb_result, (display_width, display_height))
            else:
                display_image = rgb_result
            
            self._update_display(display_image)
            
            # Update status
            self.status_var.set(f"Tap sequence completed: {len(tap_targets)} targets")
            return
        
        # Draw the current frame
        result_image = self.detector.draw_results(self.cv_image, self.current_results)
        
        # Highlight all tap targets
        for i, (name, x, y) in enumerate(tap_targets):
            # Draw circles for all targets
            if i < current_index:
                # Already tapped - green circle
                cv2.circle(result_image, (int(x), int(y)), 20, (0, 255, 0), 2)
                cv2.putText(
                    result_image,
                    f"{i+1}",
                    (int(x) - 5, int(y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            elif i == current_index:
                # Current tap target - pulsing red circle
                cv2.circle(result_image, (int(x), int(y)), 25, (0, 0, 255), 4)
                cv2.putText(
                    result_image,
                    f"{i+1}",
                    (int(x) - 5, int(y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
                
                # Display tap info
                name, x, y = tap_targets[current_index]
                self.status_var.set(f"Tapping {current_index+1}/{len(tap_targets)}: {name} at ({int(x)}, {int(y)})")
            else:
                # Future targets - yellow circles
                cv2.circle(result_image, (int(x), int(y)), 20, (0, 255, 255), 1)
                cv2.putText(
                    result_image,
                    f"{i+1}",
                    (int(x) - 5, int(y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    1
                )
        
        # Convert and display
        rgb_result = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        if self.scale_factor < 1:
            display_width = int(rgb_result.shape[1] * self.scale_factor)
            display_height = int(rgb_result.shape[0] * self.scale_factor)
            display_image = cv2.resize(rgb_result, (display_width, display_height))
        else:
            display_image = rgb_result
        
        self._update_display(display_image)
        
        # Schedule the next frame
        self.root.after(800, lambda: self.animate_tap_sequence(tap_targets, current_index + 1))
    
    def save_results(self):
        """Save detection results image to file."""
        if self.current_results is None:
            messagebox.showinfo("Info", "No detection results to save")
            return
        
        try:
            # Get save path
            save_path = filedialog.asksaveasfilename(
                title="Save Detection Results",
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")]
            )
            
            if not save_path:
                return
            
            # Generate result image at full resolution
            result_image = self.detector.draw_results(self.cv_image, self.current_results)
            
            # Highlight tap targets if enabled
            if self.show_tap_targets.get():
                tap_targets = self.detector.get_all_tap_targets(self.current_results)
                
                # Draw tap sequence
                for i, (name, x, y) in enumerate(tap_targets):
                    # Draw a larger circle around the tap point with sequence number
                    cv2.circle(result_image, (int(x), int(y)), 20, (0, 255, 255), 2)
                    cv2.putText(
                        result_image,
                        f"{i+1}",
                        (int(x) - 5, int(y) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )
            
            # Save the image
            cv2.imwrite(save_path, result_image)
            
            self.status_var.set(f"Results saved to {os.path.basename(save_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
            logger.error(f"Failed to save results: {str(e)}", exc_info=True)


def main():
    """Main function to start the application."""
    try:
        # Try to load the necessary modules
        try:
            from detector import YOLODetector, Detection
        except ImportError:
            # Create a simple mock for testing if YOLO detector is not available
            class MockDetection:
                def __init__(self, bbox, class_name, confidence):
                    self.bbox = bbox
                    self.class_name = class_name
                    self.confidence = confidence
            
            class MockYOLODetector:
                def __init__(self, model_path=None, conf_threshold=0.5):
                    self.model_path = model_path
                    self.conf_threshold = conf_threshold
                    self.class_names = ["player", "enemy", "item", "weapon"]
                    self.tap_elements = ["player", "item"]  # Define tappable elements
                    logger.warning("Using mock YOLO detector for testing")
                
                def detect(self, image):
                    # Return some mock detections for testing
                    return [
                        MockDetection([100, 100, 200, 200], "player", 0.95),
                        MockDetection([300, 300, 400, 400], "item", 0.85)
                    ]
                    
                def get_tap_targets(self, detections):
                    # Return mock tap targets
                    tap_targets = []
                    for det in detections:
                        if det.class_name in self.tap_elements:
                            center_x = (det.bbox[0] + det.bbox[2]) / 2
                            center_y = (det.bbox[1] + det.bbox[3]) / 2
                            tap_targets.append((det.class_name, center_x, center_y))
                    return tap_targets
            
            # Mock the imports
            import sys
            import types
            mock_module = types.ModuleType("detector")
            mock_module.YOLODetector = MockYOLODetector
            mock_module.Detection = MockDetection
            sys.modules["detector"] = mock_module
            
            from detector import YOLODetector, Detection
            logger.info("Loaded mock detector module for testing")
        
        # Start the application
        root = tk.Tk()
        app = HybridDetectorGUI(root)
        root.mainloop()
        
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}", exc_info=True)
        
        # Show error in a message box if possible
        try:
            import tkinter.messagebox as mb
            mb.showerror("Critical Error", f"Application failed to start: {str(e)}")
        except:
            print(f"Critical error: {str(e)}")


if __name__ == "__main__":
    main()
