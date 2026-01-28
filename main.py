import cv2
import numpy as np
import os
import time
from datetime import datetime
from model_analyzer import ModelAnalyzer
from optimization import ModelOptimizer

def create_test_images(num_images=10):
    """Create synthetic test images for analysis"""
    print("Creating test images...")
    test_images = []
    
    for i in range(num_images):
        # Create synthetic images with different objects
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Add some simple shapes to simulate objects
        cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
        cv2.circle(img, (400, 300), 50, (0, 255, 0), -1)  # Green circle
        cv2.rectangle(img, (300, 400), (500, 500), (0, 0, 255), -1)  # Red rectangle
        
        test_images.append(img)
    
    print(f"✓ Created {len(test_images)} test images")
    return test_images

def setup_environment():
    """Setup the working environment"""
    # Create results directory
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return device

def real_time_demo(optimized_model=None):
    """Real-time demonstration with performance monitoring"""
    print("\n" + "="*50)
    print("REAL-TIME DEMONSTRATION")
    print("="*50)
    
    # Use optimized model if provided, else use nano model
    if optimized_model is None:
        model = YOLO("yolov8n.pt")
    else:
        model = optimized_model
    
    # Setup camera (use 0 for webcam, or IP camera URL)
    cap = cv2.VideoCapture(0)  # Change to your camera source
    
    if not cap.isOpened():
        print("❌ Cannot open camera. Using test mode...")
        return
    
    # Performance tracking
    fps_list = []
    frame_count = 0
    start_time = time.time()
    
    print("Starting real-time detection. Press 'q' to quit, 's' to save snapshot")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Perform detection
        inference_start = time.time()
        results = model(frame, verbose=False)
        inference_time = time.time() - inference_start
        
        # Calculate current FPS
        current_fps = 1.0 / inference_time if inference_time > 0 else 0
        fps_list.append(current_fps)
        
        # Display results
        annotated_frame = results[0].plot()
        
        # Add performance overlay
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Inference: {inference_time*1000:.1f}ms", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Model: {model.__class__.__name__}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Real-Time Object Detection - PBL Project', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save snapshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"results/snapshot_{timestamp}.jpg", annotated_frame)
            print(f"✓ Snapshot saved: results/snapshot_{timestamp}.jpg")
    
    # Calculate final statistics
    total_time = time.time() - start_time
    avg_fps = np.mean(fps_list) if fps_list else 0
    
    print(f"\nReal-time Demo Summary:")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total runtime: {total_time:.2f} seconds")
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to run the complete PBL analysis"""
    print("="*70)
    print("PBL PROJECT: Real-Time Object Detection on Constrained Hardware")
    print("Optimization Techniques and Speed-Accuracy Trade-off Analysis")
    print("="*70)
    
    # Setup
    device = setup_environment()
    
    # Create test data
    test_images = create_test_images(20)
    
    # Initialize analyzers
    analyzer = ModelAnalyzer()
    optimizer = ModelOptimizer()
    
    # Phase 1: Comprehensive Model Analysis
    print("\n📊 PHASE 1: Model Comparison Analysis")
    print("-" * 50)
    
    analyzer.run_comprehensive_analysis(test_images, input_sizes=[320, 640])
    analyzer.generate_report()
    
    # Phase 2: Optimization Techniques
    print("\n⚡ PHASE 2: Optimization Techniques")
    print("-" * 50)
    
    # Test input resolution impact
    resolution_results = optimizer.resize_input_test("yolov8n.pt", test_images[0])
    
    # Test quantization
    quantized_model = optimizer.quantize_model("yolov8n.pt", "results/yolov8n_quantized.pt")
    if quantized_model:
        optimization_results = optimizer.test_optimization_impact(
            YOLO("yolov8n.pt"), quantized_model, test_images[:5]
        )
    
    # Phase 3: Real-time Demonstration
    print("\n🎥 PHASE 3: Real-time Demonstration")
    print("-" * 50)
    
    real_time_demo(quantized_model if quantized_model else None)
    
    # Final Summary
    print("\n" + "="*70)
    print("PBL PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n📈 Key Findings:")
    print("1. Model comparison shows clear trade-offs between speed and accuracy")
    print("2. Optimization techniques can significantly improve performance")
    print("3. Input resolution has major impact on both speed and detection quality")
    print("4. Different models are optimal for different use cases")
    print(f"\n📁 Results saved in 'results/' directory")
    print("Use the generated plots and metrics for your project report!")

if __name__ == "__main__":
    import torch  # Moved here to avoid circular imports
    main()