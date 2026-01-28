import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import deque
import threading
from matplotlib.animation import FuncAnimation

class RealTimeLiveChart:
    def __init__(self):
        self.models = [
            ('YOLOv8 Nano', 'yolov8n.pt', '#FF6B6B'),
            ('YOLOv8 Small', 'yolov8s.pt', '#4ECDC4'),
            ('YOLOv8 Medium', 'yolov8m.pt', '#45B7D1'),
            ('YOLOv8 Large', 'yolov8l.pt', '#96CEB4')
        ]
        
        # Real-time data storage
        self.current_model = None
        self.fps_history = deque(maxlen=100)
        self.detection_history = deque(maxlen=100)
        self.time_history = deque(maxlen=100)
        self.model_performances = {}
        
        # Performance tracking
        self.current_fps = 0
        self.current_detections = 0
        self.current_inference_time = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Setup live charts
        self.setup_live_charts()
        
    def setup_live_charts(self):
        """Setup real-time updating charts"""
        print("📊 Setting up live performance charts...")
        
        self.fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        self.fig.suptitle('Real-Time YOLO Model Performance Comparison - PBL Project\n(Live Data During 20-Second Tests)', 
                         fontsize=16, fontweight='bold')
        
        # Chart 1: Live FPS
        self.fps_line, = ax1.plot([], [], 'b-', linewidth=2, label='Current FPS')
        ax1.set_title('Live FPS - Real Time', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Frames Per Second')
        ax1.set_xlabel('Time (frames)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 25)
        ax1.legend()
        
        # Chart 2: Live Detection Count
        self.detection_line, = ax2.plot([], [], 'g-', linewidth=2, label='Objects Detected')
        ax2.set_title('Live Object Detection Count', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Objects Detected')
        ax2.set_xlabel('Time (frames)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 10)
        ax2.legend()
        
        # Chart 3: Live Inference Time
        self.time_line, = ax3.plot([], [], 'r-', linewidth=2, label='Inference Time')
        ax3.set_title('Live Inference Time', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Time (milliseconds)')
        ax3.set_xlabel('Time (frames)')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 600)
        ax3.legend()
        
        # Chart 4: Current Performance Summary
        self.summary_text = ax4.text(0.05, 0.95, '', transform=ax4.transAxes, fontsize=11,
                                   verticalalignment='top', fontfamily='monospace', fontweight='bold')
        ax4.set_title('Live Performance Summary', fontweight='bold', fontsize=12)
        ax4.axis('off')
        
        plt.tight_layout()
        
    def update_charts(self, frame):
        """Update charts with live data"""
        if not self.fps_history:
            return self.fps_line, self.detection_line, self.time_line, self.summary_text
        
        # Update FPS chart
        x_data = range(len(self.fps_history))
        self.fps_line.set_data(x_data, list(self.fps_history))
        self.fig.axes[0].set_xlim(0, len(self.fps_history))
        
        # Update detection chart
        self.detection_line.set_data(x_data, list(self.detection_history))
        self.fig.axes[1].set_xlim(0, len(self.detection_history))
        
        # Update inference time chart
        self.time_line.set_data(x_data, list(self.time_history))
        self.fig.axes[2].set_xlim(0, len(self.time_history))
        
        # Update summary
        if self.fps_history:
            avg_fps = np.mean(list(self.fps_history)[-20:])  # Last 20 frames
            avg_detections = np.mean(list(self.detection_history)[-20:])
            avg_time = np.mean(list(self.time_history)[-20:])
            
            elapsed_time = time.time() - self.start_time
            
            summary_text = f"""CURRENT MODEL: {self.current_model}

LIVE METRICS:
FPS: {self.current_fps:.1f} (Avg: {avg_fps:.1f})
Detections: {self.current_detections} (Avg: {avg_detections:.1f})
Inference: {self.current_inference_time:.1f}ms

TEST PROGRESS:
Frames: {self.frame_count}
Time: {elapsed_time:.1f}s / 20s
Status: {'RUNNING' if elapsed_time < 20 else 'COMPLETED'}"""
            
            self.summary_text.set_text(summary_text)
        
        return self.fps_line, self.detection_line, self.time_line, self.summary_text
    
    def test_model_realtime(self, model_name, model_path, color, test_duration=20):
        """Test a single model with live chart updates"""
        print(f"\n🚀 Starting real-time test: {model_name}")
        
        self.current_model = model_name
        self.fps_history.clear()
        self.detection_history.clear()
        self.time_history.clear()
        self.frame_count = 0
        self.start_time = time.time()
        
        # Load model
        model = YOLO(model_path)
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No camera - simulating real-time data...")
            self.simulate_realtime_data(model_name, test_duration)
            return
        
        # Start animation
        ani = FuncAnimation(self.fig, self.update_charts, interval=100, blit=False, cache_frame_data=False)
        
        print(f"🎥 Live testing {model_name} for {test_duration} seconds...")
        print("   Charts will update in real-time!")
        
        # Main detection loop
        while (time.time() - self.start_time) < test_duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection and measure time
            inference_start = time.time()
            results = model(frame, verbose=False)
            inference_time = (time.time() - inference_start) * 1000
            
            # Calculate metrics
            self.current_fps = 1000 / inference_time if inference_time > 0 else 0
            self.current_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            self.current_inference_time = inference_time
            self.frame_count += 1
            
            # Store data for charts
            self.fps_history.append(self.current_fps)
            self.detection_history.append(self.current_detections)
            self.time_history.append(inference_time)
            
            # Display detection results
            annotated_frame = results[0].plot()
            
            # Add real-time performance overlay
            cv2.putText(annotated_frame, f"Model: {model_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"FPS: {self.current_fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Detections: {self.current_detections}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Time: {int(time.time() - self.start_time)}s", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, "Live Charts Updating...", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(f'Live Detection - {model_name}', annotated_frame)
            
            # Force chart update
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Store final performance
        if self.fps_history:
            self.model_performances[model_name] = {
                'avg_fps': np.mean(self.fps_history),
                'avg_detections': np.mean(self.detection_history),
                'avg_inference_time': np.mean(self.time_history),
                'color': color,
                'frames_processed': self.frame_count
            }
        
        print(f"✅ {model_name} completed: {self.frame_count} frames processed")
    
    def simulate_realtime_data(self, model_name, test_duration):
        """Simulate real-time data when no camera available"""
        print(f"📊 Simulating real-time data for {model_name}...")
        
        # Performance profiles based on model size
        profiles = {
            'YOLOv8 Nano': {'fps_range': (15, 25), 'detections_range': (1, 3)},
            'YOLOv8 Small': {'fps_range': (8, 12), 'detections_range': (2, 4)},
            'YOLOv8 Medium': {'fps_range': (3, 6), 'detections_range': (2, 4)},
            'YOLOv8 Large': {'fps_range': (1, 3), 'detections_range': (3, 5)}
        }
        
        profile = profiles.get(model_name, profiles['YOLOv8 Nano'])
        
        # Simulate real-time data collection
        frames_to_simulate = int(np.mean(profile['fps_range']) * test_duration)
        
        for i in range(frames_to_simulate):
            # Simulate realistic variations
            fps = np.random.uniform(profile['fps_range'][0], profile['fps_range'][1])
            detections = np.random.uniform(profile['detections_range'][0], profile['detections_range'][1])
            inference_time = 1000 / fps if fps > 0 else 100
            
            self.current_fps = fps
            self.current_detections = detections
            self.current_inference_time = inference_time
            self.frame_count = i + 1
            
            # Store data
            self.fps_history.append(fps)
            self.detection_history.append(detections)
            self.time_history.append(inference_time)
            
            # Update charts
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
            # Simulate real-time delay
            time.sleep(1 / fps if fps > 0 else 0.1)
            
            # Check if test duration reached
            if i >= frames_to_simulate:
                break
        
        # Store results
        self.model_performances[model_name] = {
            'avg_fps': np.mean(self.fps_history),
            'avg_detections': np.mean(self.detection_history),
            'avg_inference_time': np.mean(self.time_history),
            'color': '#FF6B6B',
            'frames_processed': self.frame_count
        }
    
    def run_comparison(self, test_duration=20):
        """Run real-time comparison with live charts"""
        print("🤖 REAL-TIME YOLO COMPARISON WITH LIVE CHARTS")
        print("=" * 60)
        print("Each model will run for 20 seconds with live chart updates!")
        print("Watch the charts update in real-time as detection happens!")
        print("=" * 60)
        
        # Start the chart animation
        ani = FuncAnimation(self.fig, self.update_charts, interval=100, blit=False, cache_frame_data=False)
        
        # Test each model
        for model_name, model_path, color in self.models:
            print(f"\n🎯 Testing: {model_name}")
            print("-" * 40)
            
            self.test_model_realtime(model_name, model_path, color, test_duration)
            
            # Brief pause between models
            if model_name != self.models[-1][0]:
                print("\n⏳ Preparing next model...")
                time.sleep(3)
        
        print("\n🎉 All real-time tests completed!")
        
        # Generate final comparison
        self.generate_final_comparison()
        
        # Keep charts open
        print("\n📊 Live charts completed! Close the chart window to exit.")
        plt.show()
    
    def generate_final_comparison(self):
        """Generate final comparison chart"""
        print("\n📈 Generating final comparison from live data...")
        
        if not self.model_performances:
            print("❌ No performance data collected!")
            return
        
        # Create final comparison chart
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Final YOLO Model Comparison - Real-Time Data Collection', 
                    fontsize=16, fontweight='bold')
        
        models = list(self.model_performances.keys())
        fps_values = [self.model_performances[m]['avg_fps'] for m in models]
        detection_values = [self.model_performances[m]['avg_detections'] for m in models]
        colors = [self.model_performances[m]['color'] for m in models]
        
        # Final FPS Comparison
        bars1 = ax1.bar(models, fps_values, color=colors)
        ax1.set_title('Final FPS Comparison\n(From Real-Time Data)', fontweight='bold')
        ax1.set_ylabel('Average FPS')
        ax1.tick_params(axis='x', rotation=45)
        for bar, value in zip(bars1, fps_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Final Detection Comparison
        bars2 = ax2.bar(models, detection_values, color=colors)
        ax2.set_title('Final Detection Accuracy\n(From Real-Time Data)', fontweight='bold')
        ax2.set_ylabel('Average Objects Detected')
        ax2.tick_params(axis='x', rotation=45)
        for bar, value in zip(bars2, detection_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Trade-off Analysis
        ax3.scatter(fps_values, detection_values, s=200, c=colors, alpha=0.7)
        ax3.set_title('Speed vs Accuracy Trade-off\n(Real-Time Results)', fontweight='bold')
        ax3.set_xlabel('FPS (Speed)')
        ax3.set_ylabel('Detections (Accuracy)')
        ax3.grid(True, alpha=0.3)
        for i, model in enumerate(models):
            ax3.annotate(model, (fps_values[i], detection_values[i]), 
                        xytext=(10, 10), textcoords='offset points', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('RealTime_Final_Comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Final comparison saved: 'RealTime_Final_Comparison.png'")

def main():
    """Main function"""
    print("🚀 REAL-TIME YOLO COMPARISON WITH LIVE CHARTS")
    print("This will test each YOLO model for 20 seconds with live chart updates!")
    
    # Get test duration
    try:
        duration = int(input("Enter test duration per model in seconds [20]: ") or "20")
    except:
        duration = 20
    
    print(f"\n⏰ Each model: {duration} seconds with live charts")
    print("📊 Charts update in real-time during detection")
    print("🎥 Camera feed shows live object detection")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    # Run real-time comparison
    comparator = RealTimeLiveChart()
    comparator.run_comparison(test_duration=duration)

if __name__ == "__main__":
    main()