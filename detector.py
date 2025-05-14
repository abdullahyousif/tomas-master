import os
import cv2
import torch
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class Detection:
    """
    Class to represent a single detected object
    """
    def __init__(self, class_id: int, class_name: str, confidence: float, box: Tuple[float, float, float, float]):
        """
        Initialize a detection object.
        
        Args:
            class_id: Class ID of the detected object
            class_name: Class name of the detected object
            confidence: Confidence score (0-1)
            box: Bounding box in format (x1, y1, x2, y2)
        """
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.box = box
        self.bbox = box  # Add bbox as an alias to box
        self.x1, self.y1, self.x2, self.y2 = box
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.center_x = (self.x1 + self.x2) / 2
        self.center_y = (self.y1 + self.y2) / 2
    
    def __str__(self) -> str:
        """String representation of the detection"""
        return f"{self.class_name} ({self.confidence:.2f}) at ({int(self.center_x)}, {int(self.center_y)})"


class YOLODetector:
    """
    YOLOv11 detector for Coin Master game elements
    """
    
    def __init__(self, 
                 model_path: str = "models/my_model.pt", 
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 device: str = None):
        """
        Initialize the YOLOv11 detector.
        
        Args:
            model_path: Path to the YOLOv11 model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
            device: Device to run inference on ('cpu', 'cuda:0', etc.). If None, auto-selects.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Auto-select device if not specified
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load the model
        self.load_model()
        
        # Elements that should be tapped when detected
        self.tap_elements = [
            "attack_aiming_icon",
            "close_pannels",
            "let_me_rest",
            "ok_button",
            "raid_x_icon"
        ]
        
    def load_model(self) -> None:
        """
        Load the YOLOv11 model.
        """
        try:
            logger.info(f"Loading YOLOv11 model from {self.model_path}")
            
            # First check if the file exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found at {self.model_path}")
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            # Try different loading methods
            try:
                # Try loading with ultralytics YOLO first (for newer versions)
                logger.info("Attempting to load with ultralytics YOLO...")
                self.model = YOLO(self.model_path)
                self.model_type = "ultralytics"
                logger.info("Model loaded successfully with ultralytics YOLO")
            except Exception as e1:
                logger.warning(f"Ultralytics YOLO loading failed: {str(e1)}")
                try:
                    # Try torch hub as fallback (for older YOLOv5 models)
                    logger.info("Attempting to load with torch.hub...")
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
                    self.model_type = "torch_hub"
                    # Move model to appropriate device
                    self.model.to(self.device)
                    # Set confidence and IoU thresholds
                    self.model.conf = self.conf_threshold
                    self.model.iou = self.iou_threshold
                    logger.info("Model loaded successfully with torch.hub")
                except Exception as e2:
                    logger.error(f"Torch hub loading also failed: {str(e2)}")
                    raise RuntimeError(f"Failed to load model using both methods. Original error: {str(e1)}, Second error: {str(e2)}")
            
            # Get class names based on model type
            if self.model_type == "ultralytics":
                self.class_names = self.model.names
            else:  # torch_hub
                self.class_names = self.model.names
                
            logger.info(f"Model loaded successfully with {len(self.class_names)} classes")
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv11 model: {str(e)}")
            raise
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run object detection on an image.
        
        Args:
            image: OpenCV image (BGR)
            
        Returns:
            List of Detection objects
        """
        try:
            # Convert image to RGB (YOLO expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            detections = []
            
            if self.model_type == "ultralytics":
                # Process with ultralytics YOLO
                results = self.model(rgb_image, conf=self.conf_threshold, iou=self.iou_threshold)
                
                # Extract detection data
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        class_name = self.class_names[cls]
                        
                        detection = Detection(
                            class_id=cls,
                            class_name=class_name,
                            confidence=conf,
                            box=(float(x1), float(y1), float(x2), float(y2))
                        )
                        
                        detections.append(detection)
            else:
                # Process with torch hub YOLOv5
                results = self.model(rgb_image, size=640)
                
                # Extract detection data from results
                result_data = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]
                
                for x1, y1, x2, y2, conf, cls in result_data:
                    class_id = int(cls)
                    class_name = self.class_names[class_id]
                    
                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(conf),
                        box=(float(x1), float(y1), float(x2), float(y2))
                    )
                    
                    detections.append(detection)
            
            return detections
        
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return []
    
    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw detection boxes on an image.
        
        Args:
            image: OpenCV image
            detections: List of Detection objects
            
        Returns:
            Image with drawn detections
        """
        img_copy = image.copy()
        
        for det in detections:
            try:
                x1, y1, x2, y2 = map(int, det.box)
                
                # Determine color based on class (green for tap elements, blue for others)
                if det.class_name in self.tap_elements:
                    color = (0, 255, 0)  # BGR: Green
                else:
                    color = (255, 0, 0)  # BGR: Blue
                
                # Draw bounding box
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                
                # Add label with confidence
                label = f"{det.class_name}: {det.confidence:.2f}"
                cv2.putText(
                    img_copy, 
                    label, 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    2
                )
                
                # Mark center point
                center_x, center_y = int(det.center_x), int(det.center_y)
                cv2.circle(img_copy, (center_x, center_y), 5, (0, 0, 255), -1)
            except Exception as e:
                logger.error(f"Error drawing detection for {det.class_name}: {str(e)}")
        
        return img_copy
    
    def count_objects_by_class(self, detections: List[Detection]) -> Dict[str, int]:
        """
        Count the number of objects detected for each class.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            Dictionary mapping class names to counts
        """
        counts = {}
        for det in detections:
            if det.class_name in counts:
                counts[det.class_name] += 1
            else:
                counts[det.class_name] = 1
        
        return counts
    
    def get_tap_targets(self, detections: List[Detection]) -> List[Tuple[str, int, int]]:
        """
        Get objects that should be immediately tapped based on detection results.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of tuples (class_name, x, y) for tap targets
        """
        # Log all detected objects for debugging
        for det in detections:
            logger.debug(f"Checking if {det.class_name} should be tapped (in tap_elements: {det.class_name in self.tap_elements})")
        
        # Define classes that should be immediately tapped when detected
        tap_classes = [
            "ok_button", 
            "close_button", 
            "claim_button", 
            "collect_button",
            "spin_reward",
            "attack_aiming_icon",  # Make sure this is exact match
            "raid_target",
        ]
        
        # Define classes that should be ignored by YOLO detector (handled by template system)
        power_boost_classes = [
            "power_boost_x1",
            "power_boost_x2",
            "power_boost_x3",
            "power_boost_x15",
            "power_boost_x50",
            "power_boost_x400",
            "power_boost_x1500",
            "power_boost_x6000",
            "power_boost_x20000",
            "power_boost_button"  # Add this if you have a generic class for the power boost button
        ]
        
        # Find all detections matching tap classes or in tap_elements
        tap_targets = []
        for det in detections:
            # Skip any spin buttons - let fixed UI detector handle these
            if det.class_name == "spin_button" or det.class_name == "autospin_button" or "spin" in det.class_name.lower():
                logger.info(f"Ignoring YOLO-detected {det.class_name} to prevent conflict with fixed UI detector")
                continue
            
            # Skip any power boost related elements - let template detector handle these
            if any(boost_class in det.class_name.lower() for boost_class in power_boost_classes):
                logger.info(f"Ignoring YOLO-detected {det.class_name} to prevent conflict with power boost template system")
                continue
                
            if det.class_name in tap_classes or det.class_name in self.tap_elements:
                tap_targets.append((det.class_name, det.center_x, det.center_y))
                logger.info(f"Added tap target: {det.class_name} at ({int(det.center_x)}, {int(det.center_y)})")
        
        return tap_targets