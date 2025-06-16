#!/usr/bin/env python3
"""
Real Carotid Artery Segmentation Video Inference Module
Integrated with actual U-Net model from notebook
"""

import os
import sys
import argparse
import json
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# Import the real segmentation engine
from video_inference_real import RealVideoInferenceEngine, process_single_video_real, batch_process_videos_real


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


def get_available_subjects():
    """Get list of available subjects from data_uji directory"""
    subjects = []
    data_dir = "data_uji"
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Directory {data_dir} not found")
        return subjects
    
    for item in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, item)) and item.startswith("Subjek"):
            subjects.append(item)
    
    subjects.sort()
    return subjects

def check_subject_data(subject_name):
    """Check what data is available for a subject"""
    subject_dir = os.path.join("data_uji", subject_name)
    data_status = {
        'video': False,
        'csv': False,
        'timestamps': False,
        'directory_exists': os.path.exists(subject_dir)
    }
    
    if not data_status['directory_exists']:
        return data_status
    
    files = os.listdir(subject_dir)
    
    # Check for video file
    video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov'))]
    data_status['video'] = len(video_files) > 0
    
    # Check for CSV files
    csv_files = [f for f in files if f.endswith('.csv')]
    data_status['csv'] = len(csv_files) > 0
    
    # Check for timestamps
    data_status['timestamps'] = 'timestamps.csv' in files
    
    return data_status

def process_selected_subject(subject_name, model_file=None, enhanced=False):
    """
    Process video inference for selected subject with high-quality overlay
    
    Args:
        subject_name (str): Name of the subject (e.g., 'Subjek1')
        model_file (str): Model file to use for inference
        enhanced (bool): Enable enhanced processing with pressure integration
    
    Returns:
        dict: Processing result with status and output paths
    """
    print(f"Starting enhanced video inference for {subject_name}")
    
    # Check if subject data exists
    data_status = check_subject_data(subject_name)
    if not data_status['directory_exists']:
        return {
            'status': 'error',
            'message': f'Subject directory not found: {subject_name}',
            'output_paths': {}
        }
    
    if not data_status['video']:
        return {
            'status': 'error', 
            'message': f'No video file found for {subject_name}',
            'output_paths': {}
        }
    
    # Setup paths
    subject_dir = os.path.join("data_uji", subject_name)
    output_dir = os.path.join("inference_results", subject_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find video file
    files = os.listdir(subject_dir)
    video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov'))]
    video_file = os.path.join(subject_dir, video_files[0])
    
    print(f"Processing video: {video_file}")
    print(f"Output directory: {output_dir}")
    
    # Model setup
    if model_file is None:
        # Find available models
        model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        if not model_files:
            return {
                'status': 'error',
                'message': 'No model files (.pth) found',
                'output_paths': {}
            }
        model_file = model_files[0]  # Use first available model
    
    print(f"Using model: {model_file}")
    
    if enhanced:
        print("Enhanced processing enabled - High-quality overlay with pressure integration")
    
    try:
        # Load and process video with high-quality settings
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            return {
                'status': 'error',
                'message': f'Cannot open video file: {video_file}',
                'output_paths': {}
            }
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {total_frames} frames, {fps} FPS, {width}x{height}")
        
        # Create output paths
        output_paths = {
            'directory': output_dir,
            'video_output': os.path.join(output_dir, f"{subject_name}_segmented.mp4"),
            'video_overlay': os.path.join(output_dir, f"{subject_name}_segmented_video.mp4"),
            'diameter_data': os.path.join(output_dir, f"{subject_name}_diameter.csv"),
            'processing_log': os.path.join(output_dir, f"{subject_name}_log.txt")
        }
        
        # Create processing log
        with open(output_paths['processing_log'], 'w') as log_file:
            log_file.write(f"Enhanced Video Processing Started: {datetime.now()}\n")
            log_file.write(f"Subject: {subject_name}\n") 
            log_file.write(f"Model: {model_file}\n")
            log_file.write(f"Enhanced: {enhanced}\n")
            log_file.write(f"Video: {video_file}\n")
            log_file.write(f"Resolution: {width}x{height}\n")
            log_file.write(f"Total frames: {total_frames}\n")
            log_file.write(f"FPS: {fps}\n")
            log_file.write("-" * 50 + "\n")
        
        # Setup high-quality video writer for overlay output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        overlay_writer = cv2.VideoWriter(
            output_paths['video_overlay'], 
            fourcc, 
            fps, 
            (width, height),
            True  # Color output
        )
        
        # Setup regular video writer for segmented output
        seg_writer = cv2.VideoWriter(
            output_paths['video_output'], 
            fourcc, 
            fps, 
            (width, height),
            True
        )
        
        # Load pressure data if available and enhanced mode is enabled
        pressure_data = None
        if enhanced and data_status['csv']:
            try:
                csv_files = [f for f in files if f.endswith('.csv')]
                if csv_files:
                    pressure_file = os.path.join(subject_dir, csv_files[0])
                    pressure_data = pd.read_csv(pressure_file)
                    print(f"Loaded pressure data: {len(pressure_data)} points")
            except Exception as e:
                print(f"Warning: Could not load pressure data: {e}")
          # Process video using real U-Net segmentation
        print("Starting real U-Net segmentation processing...")
        
        # Use real segmentation engine
        engine = RealVideoInferenceEngine()
        
        # Load the U-Net model
        if not engine.load_model(model_file):
            return {
                'status': 'error',
                'message': f'Failed to load U-Net model: {model_file}',
                'output_paths': {}
            }
        
        # Process video with real segmentation
        success = engine.process_video_with_diameter(
            video_file, 
            output_paths['video_overlay'],
            output_paths['diameter_data'],
            None,  # Plot will be created by save_results
            None   # No progress callback for now
        )
        
        if not success:
            return {
                'status': 'error',
                'message': 'Real segmentation processing failed',
                'output_paths': {}
            }        
        # Save additional results
        output_files = engine.save_results(output_paths['video_overlay'])
        
        # Update output paths with additional files
        for key, path in output_files.items():
            output_paths[f'result_{key}'] = path
        
        # Get processing stats from engine
        processed_frames = engine.frame_count
        measurements_count = len(engine.results['diameters'])
        
        # Final log update
        with open(output_paths['processing_log'], 'a') as log_file:
            log_file.write("-" * 50 + "\n")
            log_file.write(f"Real U-Net Processing completed: {datetime.now()}\n")
            log_file.write(f"Frames processed: {processed_frames}\n")
            log_file.write(f"Measurements: {measurements_count}\n")
            log_file.write(f"Model: U-Net (PyTorch)\n")
            log_file.write(f"Device: {engine.device}\n")
            log_file.write(f"Output files created: {len(output_paths)}\n")
            log_file.write(f"Segmented video: {output_paths['video_overlay']}\n")
        
        print(f"Real U-Net segmentation completed successfully")
        print(f"Generated {measurements_count} real segmentation measurements")
        print(f"Output: {output_paths['video_overlay']}")
        
        return {
            'status': 'success',
            'message': f'Successfully processed {subject_name} with real U-Net segmentation ({measurements_count} measurements)',
            'output_paths': output_paths
        }
        
    except Exception as e:
        error_msg = f"Error during enhanced processing: {str(e)}"
        print(f"ERROR: {error_msg}")
        
        # Log error if possible
        try:
            with open(output_paths['processing_log'], 'a') as log_file:
                log_file.write(f"ERROR: {error_msg}\n")
        except:
            pass
        
        return {
            'status': 'error',
            'message': error_msg,
            'output_paths': {}
        }

def process_frame_with_model(frame, model_file, frame_idx, enhanced=False):
    """
    Process single frame with model inference (enhanced quality)
    
    Args:
        frame: Input frame
        model_file: Model to use
        frame_idx: Frame index
        enhanced: Enhanced processing flag
    
    Returns:
        tuple: (segmented_frame, diameter_mm)
    """
    # Simulate high-quality model inference
    # In real implementation, load PyTorch model and run inference
    
    height, width = frame.shape[:2]
    
    # Create enhanced segmentation mask
    # Simulate carotid artery detection with better quality
    center_x, center_y = width // 2, height // 2
    
    # Simulate varying diameter based on cardiac cycle
    base_diameter = 50 + 10 * np.sin(frame_idx * 0.1)  # Simulated pulsation
    diameter_mm = 5.0 + 2.0 * np.sin(frame_idx * 0.1)  # Convert to mm
    
    # Create high-quality segmentation mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, (center_x, center_y), 
                (int(base_diameter), int(base_diameter * 0.8)), 
                0, 0, 360, 255, -1)
    
    # Apply Gaussian blur for smoother edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Create segmented frame with better visualization
    segmented_frame = frame.copy()
    
    # Apply colored overlay for segmentation
    overlay = np.zeros_like(frame)
    overlay[mask > 0] = [0, 255, 0]  # Green for artery
    
    # Blend with original frame
    alpha = 0.3
    segmented_frame = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
    
    # Add contour for better visibility
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(segmented_frame, contours, -1, (0, 255, 255), 2)
    
    return segmented_frame, diameter_mm

def create_enhanced_overlay(original_frame, segmented_frame, diameter_mm, 
                           frame_idx, pressure_data, fps, enhanced=False):
    """
    Create high-quality overlay with annotations and measurements
    
    Args:
        original_frame: Original video frame
        segmented_frame: Segmented frame
        diameter_mm: Diameter measurement
        frame_idx: Current frame index
        pressure_data: Pressure data (if available)
        fps: Video frame rate
        enhanced: Enhanced mode flag
    
    Returns:
        Enhanced overlay frame
    """
    # Start with segmented frame
    overlay_frame = segmented_frame.copy()
    
    height, width = overlay_frame.shape[:2]
    
    # Enhanced text overlay settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)  # Black background
    
    # Add semi-transparent information panel
    panel_height = 120
    panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
    panel[:] = [0, 0, 0]  # Black background
    
    # Add transparency
    alpha = 0.7
    roi = overlay_frame[height-panel_height:height, 0:width]
    overlay_frame[height-panel_height:height, 0:width] = cv2.addWeighted(
        roi, 1-alpha, panel, alpha, 0
    )
    
    # Current measurements
    timestamp = frame_idx / fps
    y_offset = height - panel_height + 25
    
    # Frame info
    frame_text = f"Frame: {frame_idx+1} | Time: {timestamp:.2f}s"
    cv2.putText(overlay_frame, frame_text, (10, y_offset), 
                font, font_scale-0.2, text_color, thickness-1)
    
    # Diameter measurement with enhanced formatting
    diameter_text = f"Diameter: {diameter_mm:.3f} mm"
    cv2.putText(overlay_frame, diameter_text, (10, y_offset + 30), 
                font, font_scale, (0, 255, 0), thickness)
    
    # Pressure info if available
    if enhanced and pressure_data is not None:
        try:
            # Interpolate pressure value for current frame
            pressure_times = pressure_data.iloc[:, 0].values
            pressure_values = pressure_data.iloc[:, 1].values
            current_pressure = np.interp(timestamp, pressure_times, pressure_values)
            
            pressure_text = f"Pressure: {current_pressure:.2f}"
            cv2.putText(overlay_frame, pressure_text, (10, y_offset + 60), 
                        font, font_scale, (255, 0, 0), thickness)
        except:
            pass
    
    # Enhanced mode indicator
    if enhanced:
        mode_text = "Enhanced Mode: ON"
        cv2.putText(overlay_frame, mode_text, (width-200, y_offset), 
                    font, font_scale-0.3, (0, 255, 255), thickness-1)
    
    # Add measurement crosshairs at artery center
    center_x, center_y = width // 2, height // 2
    cross_size = 20
    cv2.line(overlay_frame, 
             (center_x - cross_size, center_y), 
             (center_x + cross_size, center_y), 
             (255, 255, 0), 2)
    cv2.line(overlay_frame, 
             (center_x, center_y - cross_size), 
             (center_x, center_y + cross_size), 
             (255, 255, 0), 2)
    
    # Add diameter measurement line
    diameter_pixels = int(diameter_mm * 10)  # Scale for visualization
    cv2.line(overlay_frame,
             (center_x - diameter_pixels//2, center_y + 40),
             (center_x + diameter_pixels//2, center_y + 40),
             (0, 255, 0), 3)
    
    # Add measurement endpoints
    cv2.circle(overlay_frame, (center_x - diameter_pixels//2, center_y + 40), 3, (0, 255, 0), -1)
    cv2.circle(overlay_frame, (center_x + diameter_pixels//2, center_y + 40), 3, (0, 255, 0), -1)
    
    return overlay_frame

def batch_process_subjects(subjects_list, model_file=None, enhanced=True):
    """
    Process multiple subjects with real U-Net segmentation
    
    Args:
        subjects_list: List of subject names to process
        model_file: Model file path (default: auto-detect)
        enhanced: Use enhanced processing
    
    Returns:
        dict: Results for each subject
    """
    print(f"Starting batch processing for {len(subjects_list)} subjects...")
    print(f"Enhanced mode: {enhanced}")
    
    # Auto-detect model if not specified
    if model_file is None:
        model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        if not model_files:
            return {'error': 'No model files (.pth) found'}
        model_file = model_files[0]
    
    print(f"Using model: {model_file}")
    
    # Verify model exists
    if not os.path.exists(model_file):
        return {'error': f'Model file not found: {model_file}'}
    
    results = {}
    successful_count = 0
    failed_count = 0
    
    # Process each subject
    for i, subject in enumerate(subjects_list):
        print(f"\n[{i+1}/{len(subjects_list)}] Processing {subject}...")
        
        try:
            # Check if subject data exists
            data_status = check_subject_data(subject)
            if not data_status['directory_exists']:
                results[subject] = {
                    'status': 'error',
                    'message': f'Subject directory not found: {subject}',
                    'output_paths': {}
                }
                failed_count += 1
                continue
            
            if not data_status['video']:
                results[subject] = {
                    'status': 'error',
                    'message': f'No video file found for {subject}',
                    'output_paths': {}
                }
                failed_count += 1
                continue            # Process subject with real segmentation
            result = process_selected_subject(subject, model_file, enhanced)
            results[subject] = result
            
            if result['status'] == 'success':
                successful_count += 1
                print(f"[OK] {subject} processed successfully")
            else:
                failed_count += 1
                print(f"[ERROR] {subject} failed: {result['message']}")
                
        except Exception as e:
            results[subject] = {
                'status': 'error',
                'message': f'Exception during processing: {str(e)}',
                'output_paths': {}
            }
            failed_count += 1
            print(f"[ERROR] {subject} exception: {str(e)}")
    
    # Summary
    print(f"\n" + "="*50)
    print(f"BATCH PROCESSING SUMMARY")
    print(f"="*50)
    print(f"Total subjects: {len(subjects_list)}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    print(f"Model used: {model_file}")
    print(f"Enhanced mode: {enhanced}")
    
    # Create batch summary report
    batch_summary = {
        'total_subjects': len(subjects_list),
        'successful': successful_count,
        'failed': failed_count,
        'model_used': model_file,
        'enhanced_mode': enhanced,
        'processing_time': datetime.now().isoformat(),
        'results': results
    }
    
    # Save batch summary
    summary_path = f"batch_processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    print(f"Batch summary saved to: {summary_path}")
    
    return batch_summary


def process_all_available_subjects(model_file=None, enhanced=True):
    """
    Process all available subjects in data_uji directory
    
    Args:
        model_file: Model file path (default: auto-detect)
        enhanced: Use enhanced processing
    
    Returns:
        dict: Batch processing results
    """
    subjects = get_available_subjects()
    
    if not subjects:
        return {'error': 'No subjects found in data_uji directory'}
    
    print(f"Found {len(subjects)} subjects: {subjects}")
    
    return batch_process_subjects(subjects, model_file, enhanced)


# Convenience functions for external use
def process_single_video(input_path: str, output_path: str, model_path: str, progress_callback=None):
    """
    Process a single video file with real U-Net segmentation
    """
    return process_single_video_real(input_path, output_path, model_path, progress_callback)


def batch_process_videos(video_list, model_path: str, progress_callback=None):
    """
    Process multiple videos in batch with real segmentation
    """
    return batch_process_videos_real(video_list, model_path, progress_callback)


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Real Video Inference for Carotid Artery Segmentation')
    parser.add_argument('--model', type=str, help='U-Net model file (.pth)')
    parser.add_argument('--subject', type=str, help='Subject name (e.g., Subjek1)')
    parser.add_argument('--batch', action='store_true', help='Process all available subjects')
    parser.add_argument('--enhanced', action='store_true', help='Enable enhanced processing')
    
    args = parser.parse_args()
    
    print(f"Real U-Net Video Inference Started")
    print(f"Enhanced: {args.enhanced}")
    print("-" * 50)
    
    if args.batch:
        print("Processing all available subjects...")
        result = process_all_available_subjects(args.model, args.enhanced)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            return 1
        
        print(f"Batch processing completed:")
        print(f"  Total: {result['total_subjects']}")
        print(f"  Successful: {result['successful']}")
        print(f"  Failed: {result['failed']}")
        
        return 0 if result['failed'] == 0 else 1
    
    elif args.subject:
        print(f"Processing subject: {args.subject}")
        result = process_selected_subject(args.subject, args.model, args.enhanced)
        
        print("-" * 50)
        print(f"Result: {result['status']}")
        print(f"Message: {result['message']}")
        
        if result['status'] == 'success':
            print("Output files:")
            for key, path in result['output_paths'].items():
                if key != 'directory':
                    print(f"  {key}: {path}")
        
        return 0 if result['status'] == 'success' else 1
    
    else:
        print("ERROR: Please specify --subject or --batch")
        return 1


if __name__ == "__main__":
    sys.exit(main())
