"""
Real Carotid Artery Segmentation Video Inference Module
Integrated with actual U-Net model and preprocessing pipeline from notebook
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
import threading
import queue
import logging
from typing import Optional, Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import PyTorch and related libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class UNet(nn.Module):
    """
    U-Net model architecture from the original notebook
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.bottleneck = conv_block(512, 1024)

        self.pool = nn.MaxPool2d(2)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, 2)

        self.dec4 = conv_block(1024, 512)
        self.dec3 = conv_block(512, 256)
        self.dec2 = conv_block(256, 128)
        self.dec1 = conv_block(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.dec4(torch.cat([self.upconv4(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.upconv3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], dim=1))

        return torch.sigmoid(self.out_conv(dec1))


class RealVideoInferenceEngine:
    """
    Real carotid artery segmentation video inference engine
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the inference engine"""
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = None
        
        # Setup preprocessing transform (from notebook)
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Video processing parameters
        self.frame_count = 0
        self.total_frames = 0
        self.fps = 30.0
        self.processing_stats = {
            'start_time': None,
            'processed_frames': 0,
            'avg_inference_time': 0.0,
            'total_inference_time': 0.0
        }
        
        # Results storage
        self.results = {
            'diameters': [],
            'timestamps': [],
            'frame_numbers': [],
            'processing_times': []
        }
        
        # Thread-safe queues for real-time processing
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue()
        self.stop_processing = threading.Event()
        
        self.logger.info("RealVideoInferenceEngine initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'model_path': 'UNet_25Mei_Sore.pth',
            'output_settings': {
                'video_codec': 'mp4v',
                'overlay_alpha': 0.4,
                'overlay_color': [0, 255, 0],  # BGR format (Green)
                'text_color': [0, 255, 255],   # BGR format (Yellow)
                'font_scale': 0.8,
                'font_thickness': 2
            },
            'processing': {
                'input_size': [512, 512],
                'confidence_threshold': 0.5,
                'min_contour_area': 100,
                'calibration': {
                    'depth_mm': 50.0,           # From notebook
                    'image_height_px': 1048     # From notebook
                }
            },
            'logging': {
                'level': 'INFO',
                'file_prefix': 'real_video_inference'
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        return default_config
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{self.config['logging']['file_prefix']}_{timestamp}.log")
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load the trained U-Net segmentation model
        """
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return False
            
            # Initialize U-Net model
            self.model = UNet(in_channels=3, out_channels=1).to(self.device)
            
            # Load model weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            self.config['model_path'] = model_path
            
            self.logger.info(f"U-Net model loaded successfully from: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict segmentation mask using the loaded U-Net model
        Implementation from the original notebook
        """
        try:
            if self.model is None:
                self.logger.error("Model not loaded. Call load_model() first.")
                return None
            
            start_time = time.time()
            
            # Store original size for resizing back
            original_size = (frame.shape[0], frame.shape[1])  # (height, width)
            
            # Preprocessing: convert BGR to RGB (OpenCV uses BGR)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transforms (resize to 512x512 and normalize)
            transformed = self.transform(image=image)
            input_tensor = transformed["image"].unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)
                output = output.squeeze().cpu().numpy()
            
            # Convert to binary mask
            binary_mask = (output > self.config['processing']['confidence_threshold']).astype(np.uint8) * 255
            
            # Resize mask back to original frame size
            binary_mask_resized = cv2.resize(binary_mask, original_size[::-1])  # cv2 uses (width, height)
            
            # Calculate inference time
            inference_time = time.time() - start_time
            self.results['processing_times'].append(inference_time)
            
            # Update processing stats
            self.processing_stats['total_inference_time'] += inference_time
            self.processing_stats['processed_frames'] += 1
            if self.processing_stats['processed_frames'] > 0:
                self.processing_stats['avg_inference_time'] = (
                    self.processing_stats['total_inference_time'] / 
                    self.processing_stats['processed_frames']
                )            
            return binary_mask_resized
            
        except Exception as e:
            self.logger.error(f"Mask prediction failed: {e}")
            return None
    
    def calculate_diameter_from_mask(self, mask: np.ndarray) -> Optional[int]:
        """
        Calculate diameter from segmentation mask (EXACT implementation from notebook)
        Returns diameter in pixels
        """
        try:
            # Find contours (exact from notebook)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Get largest contour (exact from notebook)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate minimum enclosing circle (exact from notebook)
            (_, _), radius = cv2.minEnclosingCircle(largest_contour)
            diameter = int(radius * 2)
            
            return diameter
            
        except Exception as e:
            self.logger.error(f"Diameter calculation failed: {e}")
            return None
    
    def overlay_prediction(self, frame: np.ndarray, mask: np.ndarray, diameter_mm: Optional[float] = None) -> np.ndarray:
        """
        Create overlay visualization on frame (EXACT implementation from notebook)
        """
        try:
            # Use the exact overlay function from notebook
            color = (0, 255, 0)  # Green color in BGR format (from notebook)
            alpha = 0.4  # Transparency value (from notebook)
            
            # Create colored mask (exact implementation from notebook)
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            colored_mask[mask == 255] = color

            # Blend with original frame using transparency (exact from notebook)
            overlay = cv2.addWeighted(colored_mask, alpha, frame, 1 - alpha, 0)
            
            # Add diameter text if available (exact from notebook)
            if diameter_mm is not None:
                cv2.putText(overlay, f"Diameter: {diameter_mm:.2f} mm", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            return overlay
            
        except Exception as e:
            self.logger.error(f"Overlay creation failed: {e}")
            return frame
    
    def process_video_with_diameter(self, input_path: str, output_path: str, 
                                   csv_path: Optional[str] = None, 
                                   plot_path: Optional[str] = None, 
                                   progress_callback=None) -> bool:
        """
        Process video with real segmentation and diameter calculation
        Implementation from the original notebook
        """
        try:
            self.logger.info(f"Starting real segmentation: {input_path} -> {output_path}")
            
            if self.model is None:
                self.logger.error("Model not loaded. Please load model first.")
                return False
            
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                self.logger.error(f"Could not open video: {input_path}")
                return False
            
            # Get video properties
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.logger.info(f"Video properties: {frame_width}x{frame_height}, {self.fps:.2f} FPS, {self.total_frames} frames")
            
            # Setup output video writer
            fourcc = cv2.VideoWriter_fourcc(*self.config['output_settings']['video_codec'])
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (frame_width, frame_height))
            
            if not out.isOpened():
                self.logger.error(f"Could not create output video: {output_path}")
                cap.release()
                return False
            
            # Initialize processing stats
            self.processing_stats['start_time'] = time.time()
            self.frame_count = 0
            
            # Storage for diameter measurements
            frame_diameters_mm = []
            frame_numbers = []
              # Process frames (EXACT implementation from notebook)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Perform real segmentation using U-Net
                mask = self.predict_mask(frame)
                if mask is None:
                    # Use empty mask if prediction fails
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                  # Calculate diameter from mask (exact from notebook)
                diameter_px = self.calculate_diameter_from_mask(mask)
                diameter_mm = None
                
                if diameter_px is not None:
                    # Convert to mm using same calibration as notebook
                    calibration = self.config['processing']['calibration']
                    scale_mm_per_pixel = calibration['depth_mm'] / calibration['image_height_px']
                    diameter_mm = diameter_px * scale_mm_per_pixel
                    
                    frame_diameters_mm.append(diameter_mm)
                    frame_numbers.append(self.frame_count)
                    self.results['diameters'].append(diameter_mm)
                    self.results['frame_numbers'].append(self.frame_count)
                    self.results['timestamps'].append(self.frame_count / self.fps)
                
                # Create overlay with diameter text using exact notebook implementation
                overlay = self.overlay_prediction(frame, mask, diameter_mm)
                
                # Write frame to output video
                out.write(overlay)
                
                # Update progress
                if progress_callback:
                    progress = (self.frame_count / self.total_frames) * 100
                    progress_callback(progress, self.frame_count, self.total_frames)
                
                # Log progress periodically
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / self.total_frames) * 100
                    elapsed_time = time.time() - self.processing_stats['start_time']
                    avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                    self.logger.info(f"Progress: {progress:.1f}% ({self.frame_count}/{self.total_frames}) - Avg FPS: {avg_fps:.2f}")
            
            # Cleanup
            cap.release()
            out.release()
            
            # Calculate final stats
            total_time = time.time() - self.processing_stats['start_time']
            
            self.logger.info(f"Real segmentation completed successfully")
            self.logger.info(f"Total processing time: {total_time:.2f} seconds")
            self.logger.info(f"Average FPS: {self.frame_count/total_time:.2f}")
            self.logger.info(f"Segmentation measurements: {len(frame_diameters_mm)} frames")
            
            # Save diameter data to CSV (from notebook)
            if frame_diameters_mm and csv_path:
                try:
                    df = pd.DataFrame({
                        'Frame': frame_numbers,
                        'Diameter (mm)': frame_diameters_mm
                    })
                    df.to_csv(csv_path, index=False)
                    self.logger.info(f"Diameter data saved to: {csv_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save CSV: {e}")
            
            # Create diameter plot (from notebook)
            if frame_diameters_mm and plot_path:
                try:
                    plt.figure(figsize=(10, 5))
                    plt.plot(frame_numbers, frame_diameters_mm, marker='o', linestyle='-', markersize=2)
                    plt.title("Diameter Arteri Karotis per Frame (Real Segmentation)")
                    plt.xlabel("Frame")
                    plt.ylabel("Diameter (mm)")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    self.logger.info(f"Diameter plot saved to: {plot_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save plot: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {e}")
            return False
    
    def save_results(self, base_output_path: str) -> Dict[str, str]:
        """
        Save processing results to files
        """
        try:
            output_files = {}
            
            # Save CSV data
            if self.results['diameters']:
                csv_path = base_output_path.replace('.mp4', '_diameter_data.csv')
                df = pd.DataFrame({
                    'Frame': self.results['frame_numbers'],
                    'Timestamp (s)': self.results['timestamps'],
                    'Diameter (mm)': self.results['diameters']
                })
                df.to_csv(csv_path, index=False)
                output_files['csv'] = csv_path
                self.logger.info(f"Diameter data saved to: {csv_path}")
                
                # Create and save plot
                plot_path = base_output_path.replace('.mp4', '_diameter_plot.png')
                self._create_diameter_plot(plot_path)
                output_files['plot'] = plot_path
            
            # Save processing summary
            summary_path = base_output_path.replace('.mp4', '_summary.json')
            summary = {
                'processing_stats': self.processing_stats,
                'video_info': {
                    'total_frames': self.total_frames,
                    'fps': self.fps,
                    'processed_frames': self.frame_count
                },
                'segmentation_results': {
                    'total_measurements': len(self.results['diameters']),
                    'avg_diameter': float(np.mean(self.results['diameters'])) if self.results['diameters'] else 0,
                    'std_diameter': float(np.std(self.results['diameters'])) if self.results['diameters'] else 0,
                    'min_diameter': float(np.min(self.results['diameters'])) if self.results['diameters'] else 0,
                    'max_diameter': float(np.max(self.results['diameters'])) if self.results['diameters'] else 0
                },
                'model_info': {
                    'architecture': 'U-Net',
                    'model_path': self.config.get('model_path', 'Unknown'),
                    'device': str(self.device),
                    'confidence_threshold': self.config['processing']['confidence_threshold']
                }
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            output_files['summary'] = summary_path
            
            self.logger.info(f"Processing summary saved to: {summary_path}")
            
            return output_files
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return {}
    
    def _create_diameter_plot(self, output_path: str):
        """
        Create and save diameter vs time plot
        """
        try:
            plt.figure(figsize=(12, 6))
            
            if self.results['timestamps'] and self.results['diameters']:
                plt.plot(self.results['timestamps'], self.results['diameters'], 'b-', linewidth=1, alpha=0.7)
                plt.scatter(self.results['timestamps'], self.results['diameters'], c='red', s=1, alpha=0.5)
                
                plt.xlabel('Time (seconds)')
                plt.ylabel('Diameter (mm)')
                plt.title('Carotid Artery Diameter Over Time (Real U-Net Segmentation)')
                plt.grid(True, alpha=0.3)
                
                # Add statistics text
                if self.results['diameters']:
                    stats_text = f"""Real Segmentation Statistics:
Mean: {np.mean(self.results['diameters']):.2f} mm
Std: {np.std(self.results['diameters']):.2f} mm
Min: {np.min(self.results['diameters']):.2f} mm
Max: {np.max(self.results['diameters']):.2f} mm
Range: {np.max(self.results['diameters']) - np.min(self.results['diameters']):.2f} mm
Measurements: {len(self.results['diameters'])} frames"""
                    
                    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Diameter plot saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create diameter plot: {e}")


# Convenience functions for external use (updated for real segmentation)
def process_single_video_real(input_path: str, output_path: str, model_path: str, progress_callback=None) -> Dict[str, Any]:
    """
    Process a single video file with real U-Net segmentation
    """
    engine = RealVideoInferenceEngine()
    
    # Load the U-Net model
    if not engine.load_model(model_path):
        return {'success': False, 'error': 'Failed to load U-Net model'}
    
    # Generate paths for CSV and plot
    csv_path = output_path.replace('.mp4', '_diameter_data.csv')
    plot_path = output_path.replace('.mp4', '_diameter_plot.png')
    
    # Process video with real segmentation
    success = engine.process_video_with_diameter(input_path, output_path, csv_path, plot_path, progress_callback)
    
    if success:
        output_files = engine.save_results(output_path)
        return {
            'success': True,
            'output_files': output_files,
            'stats': engine.processing_stats,
            'results_summary': {
                'total_measurements': len(engine.results['diameters']),
                'avg_diameter': float(np.mean(engine.results['diameters'])) if engine.results['diameters'] else 0,
                'std_diameter': float(np.std(engine.results['diameters'])) if engine.results['diameters'] else 0,
                'segmentation_type': 'Real U-Net'
            }
        }
    else:
        return {'success': False, 'error': 'Real segmentation processing failed'}


def batch_process_videos_real(video_list: List[Dict[str, str]], model_path: str, progress_callback=None) -> Dict[str, Any]:
    """
    Process multiple videos in batch with real segmentation
    """
    results = {}
    
    # Verify model exists before starting batch
    if not os.path.exists(model_path):
        return {'error': f'Model file not found: {model_path}'}
    
    for i, video_info in enumerate(video_list):
        input_path = video_info['input']
        output_path = video_info['output']
        
        print(f"Processing video {i+1}/{len(video_list)}: {os.path.basename(input_path)}")
        
        result = process_single_video_real(input_path, output_path, model_path, progress_callback)
        results[os.path.basename(input_path)] = result
        
        if not result['success']:
            print(f"[ERROR] Failed to process {input_path}: {result.get('error', 'Unknown error')}")
        else:
            print(f"[OK] Successfully processed {input_path} with real U-Net segmentation")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Test with sample video
    test_input = "test_video.mp4"
    test_output = "test_output_real_segmentation.mp4"
    test_model = "UNet_25Mei_Sore.pth"
    
    if os.path.exists(test_input) and os.path.exists(test_model):
        print("Testing real U-Net segmentation...")
        result = process_single_video_real(test_input, test_output, test_model)
        
        if result['success']:
            print("[OK] Real segmentation test completed successfully")
            print(f"Output files: {result['output_files']}")
            print(f"Results: {result['results_summary']}")
        else:
            print(f"[ERROR] Test failed: {result['error']}")
    else:
        print(f"Test files not found - Video: {test_input}, Model: {test_model}")
        print("Skipping test...")
    
    print("Real Carotid Artery Segmentation VideoInferenceEngine ready for use")
