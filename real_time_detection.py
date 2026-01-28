import cv2
import time
import numpy as np
from ultralytics import YOLO

class RealTimeDetection:
    def __init__(self, model_name="yolov8n.pt"):
        print(f"Loading {model_name} for real-time detection...")
        self.model = YOLO(model_name)
        self.model_name = model_name
        
        # Performance tracking
        self.fps_list = []
        self.frame_count = 0
        self.start_time = time.time()
        
    def start_detection(self, camera_source=0):
        """Start real-time object detection from camera"""
        
        # Open camera
        self.cap = cv2.VideoCapture(camera_source)
        
        if not self.cap.isOpened():
            print("❌ Error: Cannot open camera")
            print("Trying alternative camera sources...")
            # Try different camera indices
            for i in range(1, 5):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"✓ Camera found at index {i}")
                    break
            else:
                print("❌ No camera found. Using test video...")
                # Use a test video file instead
                self.cap = cv2.VideoCapture('test_video.mp4')
                if not self.cap.isOpened():
                    # Create a synthetic video if no camera/video available
                    print("Creating synthetic video for demonstration...")
                    return self.create_synthetic_video()
        
        # Get camera properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✓ Camera opened successfully")
        print(f"✓ Resolution: {self.frame_width}x{self.frame_height}")
        print(f"✓ Model: {self.model_name}")
        print("Press 'q' to quit, 's' to save screenshot, '1-4' to switch models")
        
        self.run_detection()
    
    def create_synthetic_video(self):
        """Create a synthetic video when no camera is available"""
        print("Running with synthetic video feed...")
        self.run_synthetic_detection()
    
    def run_detection(self):
        """Main detection loop"""
        while True:
            # Read frame from camera
            ret, frame = self.cap.read()
            
            if not ret:
                print("❌ Cannot read frame from camera")
                break
            
            # Perform object detection
            results = self.model(frame, verbose=False)
            
            # Draw detections on frame
            annotated_frame = results[0].plot()
            
            # Calculate performance metrics
            current_time = time.time()
            current_fps = 1.0 / (current_time - self.start_time) if self.start_time > 0 else 0
            self.fps_list.append(current_fps)
            self.frame_count += 1
            self.start_time = current_time
            
            # Add performance overlay
            self.add_performance_overlay(annotated_frame, current_fps, results[0])
            
            # Display the frame
            cv2.imshow('Real-Time Object Detection - PBL Project', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"detection_screenshot_{timestamp}.jpg", annotated_frame)
                print(f"✓ Screenshot saved: detection_screenshot_{timestamp}.jpg")
            elif key == ord('1'):
                self.switch_model("yolov8n.pt")
            elif key == ord('2'):
                self.switch_model("yolov8s.pt")
            elif key == ord('3'):
                self.switch_model("yolov8m.pt")
            elif key == ord('4'):
                self.switch_model("yolov8l.pt")
        
        # Cleanup
        self.cleanup()
    
    def run_synthetic_detection(self):
        """Run detection on synthetic video when no camera is available"""
        print("Creating synthetic video stream...")
        
        while True:
            # Create a synthetic frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add some objects to detect
            cv2.rectangle(frame, (100, 100), (200, 300), (255, 0, 0), -1)  # Blue "object"
            cv2.circle(frame, (400, 200), 50, (0, 255, 0), -1)  # Green "object"
            cv2.rectangle(frame, (300, 50), (500, 150), (0, 0, 255), -1)  # Red "object"
            cv2.putText(frame, "SYNTHETIC VIDEO - No Camera Detected", (50, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Perform object detection
            results = self.model(frame, verbose=False)
            
            # Draw detections
            annotated_frame = results[0].plot()
            
            # Calculate FPS
            current_time = time.time()
            current_fps = 1.0 / (current_time - self.start_time) if self.start_time > 0 else 0
            self.fps_list.append(current_fps)
            self.frame_count += 1
            self.start_time = current_time
            
            # Add performance overlay
            self.add_performance_overlay(annotated_frame, current_fps, results[0])
            
            # Display
            cv2.imshow('Real-Time Object Detection - PBL Project (Synthetic)', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.switch_model("yolov8n.pt")
            elif key == ord('2'):
                self.switch_model("yolov8s.pt")
        
        self.cleanup()
    
    def switch_model(self, new_model):
        """Switch to a different YOLO model"""
        print(f"Switching to {new_model}...")
        self.model = YOLO(new_model)
        self.model_name = new_model
        print(f"✓ Model switched to {new_model}")
    
    def add_performance_overlay(self, frame, current_fps, results):
        """Add performance information to the frame"""
        # Performance info
        avg_fps = np.mean(self.fps_list[-30:]) if len(self.fps_list) > 0 else 0  # Last 30 frames
        detections = len(results.boxes) if results.boxes is not None else 0
        
        # Create performance text
        info_text = [
            f"Model: {self.model_name}",
            f"FPS: {current_fps:.1f} (Avg: {avg_fps:.1f})",
            f"Detections: {detections}",
            f"Resolution: {self.frame_width}x{self.frame_height}",
            "Controls: Q=Quit, S=Save, 1-4=Switch Models"
        ]
        
        # Draw background for text
        y_offset = 30
        for i, text in enumerate(info_text):
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (10, y_offset * i + 5), 
                         (15 + text_size[0], y_offset * (i + 1)), 
                         (0, 0, 0), -1)
            
            # Draw text
            color = (0, 255, 0) if i == 1 else (255, 255, 255)  # FPS in green
            cv2.putText(frame, text, (10, y_offset * (i + 1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print performance summary
        if self.fps_list:
            avg_fps = np.mean(self.fps_list)
            print(f"\n📊 Performance Summary:")
            print(f"Total frames processed: {self.frame_count}")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Final model: {self.model_name}")

def main():
    """Main function to run real-time detection"""
    print("=" * 60)
    print("REAL-TIME OBJECT DETECTION - PBL PROJECT")
    print("=" * 60)
    print("This will open your camera and show live object detection")
    print("Available models: yolov8n.pt (fastest), yolov8s.pt, yolov8m.pt, yolov8l.pt (most accurate)")
    print()
    
    # Choose model
    model_choice = input("Choose model (1=yolov8n, 2=yolov8s, 3=yolov8m, 4=yolov8l) [1]: ") or "1"
    
    models = {
        "1": "yolov8n.pt",
        "2": "yolov8s.pt", 
        "3": "yolov8m.pt",
        "4": "yolov8l.pt"
    }
    
    selected_model = models.get(model_choice, "yolov8n.pt")
    
    # Camera source choice
    print("\nCamera options:")
    print("0 = Default webcam")
    print("1 = External USB camera") 
    print("2 = IP camera (enter URL)")
    
    cam_choice = input("Choose camera source [0]: ") or "0"
    
    if cam_choice == "2":
        camera_url = input("Enter IP camera URL: ")
        detector = RealTimeDetection(selected_model)
        detector.start_detection(camera_url)
    else:
        camera_index = int(cam_choice)
        detector = RealTimeDetection(selected_model)
        detector.start_detection(camera_index)

if __name__ == "__main__":
    main()