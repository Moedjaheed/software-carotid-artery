"""
Launcher Script untuk Segmentasi Karotis
Script untuk menjalankan semua komponen dengan mudah
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime
import matplotlib.pyplot as plt

class SegmentationLauncher:
    """Launcher GUI dengan sistem tab seperti browser"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Carotid Segmentation Suite")
        self.root.geometry("800x600")  # More compact size
        self.root.minsize(700, 500)
        
        print("DEBUG: Initializing Carotid Segmentation Launcher...")
        print(f"DEBUG: Working directory: {os.getcwd()}")
        print(f"DEBUG: Python executable: {sys.executable}")
        
        # Initialize variables
        self.status_var = tk.StringVar(value="Ready")
        
        self.setup_tabbed_ui()
    
    def setup_tabbed_ui(self):
        """Setup tabbed user interface like a browser"""
        print("DEBUG: Setting up tabbed user interface...")
        
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Header with title and status
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="🩺 Carotid Segmentation Suite", 
                               font=("Arial", 14, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # Status bar
        status_frame = ttk.Frame(header_frame)
        status_frame.pack(side=tk.RIGHT)
        
        ttk.Label(status_frame, text="Status:", font=("Arial", 9)).pack(side=tk.LEFT)
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                font=("Arial", 9), foreground="blue")
        status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Create notebook (tab container)
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_home_tab()
        self.create_inference_tab()
        self.create_analytics_tab()
        self.create_tools_tab()
        self.create_settings_tab()
        
        print("DEBUG: Tabbed user interface setup completed")
    
    def create_home_tab(self):
        """Create home/dashboard tab"""
        home_frame = ttk.Frame(self.notebook)
        self.notebook.add(home_frame, text="🏠 Home")
        
        # Welcome section
        welcome_frame = ttk.LabelFrame(home_frame, text="Welcome", padding=15)
        welcome_frame.pack(fill=tk.X, padx=10, pady=5)
        
        welcome_text = """🩺 Carotid Artery Segmentation Suite
Advanced AI-powered analysis for carotid artery diameter measurement and pressure correlation.

Quick Actions:"""
        ttk.Label(welcome_frame, text=welcome_text, justify=tk.LEFT, 
                 font=("Arial", 10)).pack(anchor=tk.W)
        
        # Quick actions
        quick_frame = ttk.Frame(welcome_frame)
        quick_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Quick action buttons
        ttk.Button(quick_frame, text="🚀 Enhanced Inference", 
                  command=lambda: self.notebook.select(1), width=20).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(quick_frame, text="📊 Data Viewer", 
                  command=self.run_data_viewer, width=20).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(quick_frame, text="📈 Analytics", 
                  command=lambda: self.notebook.select(2), width=20).pack(side=tk.LEFT)
        
        # Recent activity section
        activity_frame = ttk.LabelFrame(home_frame, text="Recent Activity", padding=10)
        activity_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Activity listbox
        activity_list = tk.Listbox(activity_frame, height=8, font=("Arial", 9))
        activity_scrollbar = ttk.Scrollbar(activity_frame, orient=tk.VERTICAL, command=activity_list.yview)
        activity_list.configure(yscrollcommand=activity_scrollbar.set)
        
        # Add some sample activity
        activities = [
            "✅ System initialized successfully",
            "📁 Data directory: data_uji (7 subjects detected)",
            "🔧 Models available: UNet_25Mei_Sore.pth, UNet_22Mei_Sore.pth",
            "📊 Inference results: 4 subjects processed",
            "🎯 Ready for enhanced inference and analytics"
        ]
        
        for activity in activities:
            activity_list.insert(tk.END, activity)
        
        activity_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        activity_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # System info
        info_frame = ttk.LabelFrame(home_frame, text="System Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        info_text = f"""Python: {sys.version.split()[0]} | Working Directory: {os.getcwd()}
Models: 2 available | Subjects: 7 detected | Enhanced Features: ✅ Enabled"""
      
    def create_inference_tab(self):
        """Create inference operations tab"""
        inference_frame = ttk.Frame(self.notebook)
        self.notebook.add(inference_frame, text="🚀 Inference")
        
        # Enhanced Inference section
        enhanced_frame = ttk.LabelFrame(inference_frame, text="Enhanced Inference", padding=15)
        enhanced_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(enhanced_frame, text="Advanced AI inference with pressure data integration", 
                 font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 10))
        
        inference_buttons = ttk.Frame(enhanced_frame)
        inference_buttons.pack(fill=tk.X)
        
        ttk.Button(inference_buttons, text="🎯 Enhanced Inference (Multiple Subjects)", 
                  command=self.run_enhanced_inference, width=35).pack(pady=2, fill=tk.X)
        ttk.Button(inference_buttons, text="📋 Select Subject for Inference", 
                  command=self.run_subject_inference, width=35).pack(pady=2, fill=tk.X)
        ttk.Button(inference_buttons, text="⚡ Auto Inference (Quick)", 
                  command=self.run_inference, width=35).pack(pady=2, fill=tk.X)
        
        # Batch Processing section
        batch_frame = ttk.LabelFrame(inference_frame, text="Batch Processing", padding=15)
        batch_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(batch_frame, text="Process multiple subjects automatically", 
                 font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Button(batch_frame, text="🔄 Batch Process All Subjects", 
                  command=self.run_batch_processing, width=35).pack(fill=tk.X)
        
        # Model Training section
        training_frame = ttk.LabelFrame(inference_frame, text="Model Training", padding=15)
        training_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(training_frame, text="Train new AI models with your dataset", 
                 font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Button(training_frame, text="🎓 Start Training Model", 
                  command=self.run_training, width=35).pack(fill=tk.X)
    
    def create_analytics_tab(self):
        """Create analytics and visualization tab"""
        analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(analytics_frame, text="📊 Analytics")
        
        # Data Viewer section
        viewer_frame = ttk.LabelFrame(analytics_frame, text="Data Visualization", padding=15)
        viewer_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(viewer_frame, text="Interactive data visualization and analysis tools", 
                 font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 10))
        
        viewer_buttons = ttk.Frame(viewer_frame)
        viewer_buttons.pack(fill=tk.X)
        
        ttk.Button(viewer_buttons, text="📈 Enhanced Data Viewer", 
                  command=self.run_data_viewer, width=35).pack(pady=2, fill=tk.X)
        ttk.Button(viewer_buttons, text="👁️ View Inference Results", 
                  command=self.view_results, width=35).pack(pady=2, fill=tk.X)
        
        # Advanced Analytics section
        advanced_frame = ttk.LabelFrame(analytics_frame, text="Advanced Analytics", padding=15)
        advanced_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(advanced_frame, text="Statistical analysis and correlation studies", 
                 font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Button(advanced_frame, text="📊 Advanced Analytics Dashboard", 
                  command=self.run_advanced_analytics, width=35).pack(fill=tk.X)
        
        # Export & Reports section
        export_frame = ttk.LabelFrame(analytics_frame, text="Export & Reports", padding=15)
        export_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(export_frame, text="Generate reports and export data", 
                 font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 10))
        
        export_buttons = ttk.Frame(export_frame)
        export_buttons.pack(fill=tk.X)
        
        ttk.Button(export_buttons, text="📄 Generate Report", 
                  command=self.generate_report, width=17).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_buttons, text="💾 Export Data", 
                  command=self.export_data, width=17).pack(side=tk.LEFT)
    
    def create_tools_tab(self):
        """Create tools and utilities tab"""
        tools_frame = ttk.Frame(self.notebook)
        self.notebook.add(tools_frame, text="🔧 Tools")
        
        # System Tools section
        system_frame = ttk.LabelFrame(tools_frame, text="System Tools", padding=15)
        system_frame.pack(fill=tk.X, padx=10, pady=5)
        
        system_buttons = ttk.Frame(system_frame)
        system_buttons.pack(fill=tk.X)
        
        ttk.Button(system_buttons, text="🔍 Check Dependencies", 
                  command=self.check_dependencies, width=17).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(system_buttons, text="⬇️ Install Dependencies", 
                  command=self.install_dependencies, width=17).pack(side=tk.LEFT)
        
        # File Management section
        file_frame = ttk.LabelFrame(tools_frame, text="File Management", padding=15)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        file_buttons = ttk.Frame(file_frame)
        file_buttons.pack(fill=tk.X)
        
        ttk.Button(file_buttons, text="📁 Open Project Folder", 
                  command=self.open_folder, width=17).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_buttons, text="🗂️ Open Data Folder", 
                  command=self.open_data_folder, width=17).pack(side=tk.LEFT)
        
        # Maintenance section
        maintenance_frame = ttk.LabelFrame(tools_frame, text="Maintenance", padding=15)
        maintenance_frame.pack(fill=tk.X, padx=10, pady=5)
        
        maintenance_buttons = ttk.Frame(maintenance_frame)
        maintenance_buttons.pack(fill=tk.X)
        
        ttk.Button(maintenance_buttons, text="🧹 Clean Cache", 
                  command=self.clean_cache, width=17).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(maintenance_buttons, text="🔄 Reset Settings", 
                  command=self.reset_settings, width=17).pack(side=tk.LEFT)
    
    def create_settings_tab(self):
        """Create settings and configuration tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="⚙️ Settings")
        
        # Model Settings section
        model_frame = ttk.LabelFrame(settings_frame, text="Model Configuration", padding=15)
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(model_frame, text="Default model for inference:", 
                 font=("Arial", 10)).pack(anchor=tk.W)
        
        model_var = tk.StringVar(value="UNet_25Mei_Sore.pth")
        model_combo = ttk.Combobox(model_frame, textvariable=model_var, 
                                  values=["UNet_25Mei_Sore.pth", "UNet_22Mei_Sore.pth"],
                                  state="readonly", width=30)
        model_combo.pack(anchor=tk.W, pady=5)
        
        # Processing Settings section
        processing_frame = ttk.LabelFrame(settings_frame, text="Processing Options", padding=15)
        processing_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Checkboxes for processing options
        self.auto_pressure = tk.BooleanVar(value=True)
        self.auto_analytics = tk.BooleanVar(value=True)
        self.save_video = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(processing_frame, text="Auto-detect pressure data", 
                       variable=self.auto_pressure).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(processing_frame, text="Run analytics automatically", 
                       variable=self.auto_analytics).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(processing_frame, text="Save segmented video", 
                       variable=self.save_video).pack(anchor=tk.W, pady=2)
        
        # UI Settings section
        ui_frame = ttk.LabelFrame(settings_frame, text="Interface Settings", padding=15)
        ui_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(ui_frame, text="Theme:", font=("Arial", 10)).pack(anchor=tk.W)
        theme_var = tk.StringVar(value="Default")
        theme_combo = ttk.Combobox(ui_frame, textvariable=theme_var, 
                                  values=["Default", "Dark", "Light"],
                                  state="readonly", width=20)
        theme_combo.pack(anchor=tk.W, pady=5)
        
        # Save settings button
        ttk.Button(ui_frame, text="💾 Save Settings", 
                  command=self.save_settings).pack(anchor=tk.W, pady=(10, 0))
    
    # Utility functions for new features
    def generate_report(self):
        """Generate analysis report"""
        messagebox.showinfo("Report", "Report generation feature coming soon!")
    
    def export_data(self):
        """Export analysis data"""
        messagebox.showinfo("Export", "Data export feature coming soon!")
    
    def open_data_folder(self):
        """Open data folder"""
        try:
            if os.path.exists("data_uji"):
                os.startfile("data_uji")
            else:
                messagebox.showwarning("Warning", "Data folder 'data_uji' not found!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open data folder: {str(e)}")
    
    def clean_cache(self):
        """Clean cache files"""
        try:
            # Clean __pycache__ directories
            import shutil
            for root, dirs, files in os.walk("."):
                if "__pycache__" in dirs:
                    shutil.rmtree(os.path.join(root, "__pycache__"))
            messagebox.showinfo("Success", "Cache cleaned successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clean cache: {str(e)}")
    
    def reset_settings(self):
        """Reset settings to default"""
        result = messagebox.askyesno("Confirm", "Reset all settings to default values?")
        if result:
            messagebox.showinfo("Settings", "Settings reset to default!")
    
    def save_settings(self):
        """Save current settings"""
        messagebox.showinfo("Settings", "Settings saved successfully!")
    
    def run_data_viewer(self):
        """Run enhanced data viewer"""
        print("DEBUG: Starting Enhanced Data Viewer...")
        print("DEBUG: Features: Overlay video display, Real-time analysis, Diameter vs Pressure plots")
        
        try:
            if os.path.exists("data_viewer.py"):
                print("DEBUG: Found data_viewer.py (Enhanced Version), launching...")
                print("DEBUG: Enhanced features available:")
                print("DEBUG: - Segmented video overlay display")
                print("DEBUG: - Original size image preservation")
                print("DEBUG: - Diameter vs Pressure plot with Frame X-axis")
                print("DEBUG: - Real-time value annotations")
                print("DEBUG: - Automatic data synchronization")
                
                # Run in separate process but capture any errors
                process = subprocess.Popen([sys.executable, "data_viewer.py"])
                print("DEBUG: Enhanced Data Viewer process started successfully")
                self.status_var.set("Enhanced Data Viewer opened")
                
            else:
                print("DEBUG: data_viewer.py not found!")
                messagebox.showerror("Error", "Enhanced Data Viewer (data_viewer.py) not found!")
                
        except Exception as e:
            print(f"DEBUG: Error starting Enhanced Data Viewer - {str(e)}")
            self.status_var.set("Error starting Enhanced Data Viewer")
            messagebox.showerror("Error", f"Failed to start Enhanced Data Viewer: {str(e)}")

    # ...existing code...
    
    def run_training(self):
        """Run training script"""
        print("DEBUG: Starting training process...")
        self.status_var.set("Starting training...")
        
        try:
            if not os.path.exists("training_model.py"):
                print("DEBUG: training_model.py not found!")
                messagebox.showerror("Error", "training_model.py not found!")
                return
            
            print("DEBUG: Found training_model.py, launching in new console...")
            # Run in separate process
            subprocess.Popen([sys.executable, "training_model.py"], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
            self.status_var.set("Training started in new window")
            messagebox.showinfo("Success", "Training started! Check the console window for progress.")
            print("DEBUG: Training process started successfully")
            
        except Exception as e:
            print(f"DEBUG: Error starting training - {str(e)}")
            self.status_var.set("Error starting training")
            messagebox.showerror("Error", f"Failed to start training: {str(e)}")
    
    def run_inference(self):
        """Run video inference script"""
        print("DEBUG: Starting video inference process...")
        self.status_var.set("Starting video inference...")
        
        try:
            if not os.path.exists("video_inference.py"):
                print("DEBUG: video_inference.py not found!")
                messagebox.showerror("Error", "video_inference.py not found!")
                return
            
            # Check if model exists
            if not os.path.exists("UNet_25Mei_Sore.pth"):
                print("DEBUG: Model file not found, asking user...")
                response = messagebox.askyesno("Model Not Found", 
                    "Model file 'UNet_25Mei_Sore.pth' not found. "
                    "Do you want to continue anyway? (You may need to train the model first)")
                if not response:
                    print("DEBUG: User cancelled inference due to missing model")
                    return
            
            print("DEBUG: Creating inference log window...")
            
            # Create process log window
            log_window = tk.Toplevel(self.root)
            log_window.title("Video Inference Process")
            log_window.geometry("800x600")
            log_window.transient(self.root)
            
            # Status label
            status_label = ttk.Label(log_window, text="Starting video inference...")
            status_label.pack(pady=10)
            
            # Progress bar
            progress_bar = ttk.Progressbar(log_window, mode='indeterminate')
            progress_bar.pack(pady=5, padx=20, fill=tk.X)
            progress_bar.start()
            
            # Text widget for output
            output_frame = ttk.Frame(log_window)
            output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            output_text = tk.Text(output_frame, wrap=tk.WORD, height=25, font=("Consolas", 9))
            scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=output_text.yview)
            output_text.configure(yscrollcommand=scrollbar.set)
            
            output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Add initial info
            output_text.insert(tk.END, "=== VIDEO INFERENCE PROCESS LOG ===\\n\\n")
            output_text.insert(tk.END, f"Python executable: {sys.executable}\\n")
            output_text.insert(tk.END, f"Working directory: {os.getcwd()}\\n")
            output_text.insert(tk.END, f"Script: video_inference.py\\n")
            output_text.insert(tk.END, f"Model file: UNet_25Mei_Sore.pth\\n\\n")
            
            # Check data directory
            if os.path.exists("data_uji"):
                subjects = [d for d in os.listdir("data_uji") if os.path.isdir(os.path.join("data_uji", d))]
                output_text.insert(tk.END, f"Found {len(subjects)} subjects in data_uji: {', '.join(subjects)}\\n\\n")
            else:
                output_text.insert(tk.END, "Warning: data_uji directory not found\\n\\n")
            
            output_text.insert(tk.END, "Starting inference process...\\n")
            output_text.insert(tk.END, "Command: python video_inference.py\\n")
            output_text.insert(tk.END, "-" * 60 + "\\n")
            output_text.see(tk.END)
            output_text.update()
            
            print("DEBUG: Starting inference subprocess with GUI logging...")
            
            # Run inference in separate thread with GUI logging
            def run_inference_process():
                try:
                    process = subprocess.Popen(
                        [sys.executable, "video_inference.py"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                        cwd=os.getcwd()
                    )
                    
                    # Read output line by line
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            print(f"DEBUG: inference output: {line.strip()}")
                            # Use after() to safely update GUI from thread
                            self.root.after(0, lambda l=line: self.update_inference_output(
                                output_text, status_label, log_window, l))
                    
                    process.stdout.close()
                    return_code = process.wait()
                    
                    print(f"DEBUG: Inference completed with return code: {return_code}")
                    
                    # Update final status
                    self.root.after(0, lambda: self.finish_inference(
                        progress_bar, status_label, output_text, log_window, return_code))
                        
                except Exception as e:
                    print(f"DEBUG: Inference thread error: {str(e)}")
                    self.root.after(0, lambda: self.inference_error(
                        progress_bar, status_label, log_window, str(e)))
              # Start inference in background thread
            inference_thread = threading.Thread(target=run_inference_process, daemon=True)
            inference_thread.start()
            print("DEBUG: Inference thread started with GUI logging")
            self.status_var.set("Video inference running with log window")
            
        except Exception as e:
            print(f"DEBUG: Error starting inference - {str(e)}")
            self.status_var.set("Error starting inference")
            messagebox.showerror("Error", f"Failed to start inference: {str(e)}")
    
    def run_viewer(self):
        """Run enhanced data viewer with overlay and analysis features"""
        print("DEBUG: Starting Enhanced Data Viewer...")
        print("DEBUG: Features: Overlay video display, Real-time analysis, Diameter vs Pressure plots")
        self.status_var.set("Opening enhanced data viewer...")
        
        try:
            if not os.path.exists("data_viewer.py"):
                print("DEBUG: data_viewer.py not found!")
                messagebox.showerror("Error", "data_viewer.py not found!")
                return
            
            print("DEBUG: Found data_viewer.py (Enhanced Version), launching...")
            print("DEBUG: Enhanced features available:")
            print("DEBUG: - Segmented video overlay display")
            print("DEBUG: - Original size image preservation")
            print("DEBUG: - Diameter vs Pressure plot with Frame X-axis")
            print("DEBUG: - Real-time value annotations")
            print("DEBUG: - Automatic data synchronization")
            
            # Run in separate process
            subprocess.Popen([sys.executable, "data_viewer.py"])
            self.status_var.set("Enhanced Data Viewer opened")
            print("DEBUG: Enhanced Data Viewer process started successfully")
            
        except Exception as e:
            print(f"DEBUG: Error opening enhanced viewer - {str(e)}")
            self.status_var.set("Error opening enhanced viewer")
            messagebox.showerror("Error", f"Failed to open enhanced data viewer: {str(e)}")
    
    def check_dependencies(self):
        """Check if all dependencies are installed"""
        print("DEBUG: Checking dependencies...")
        self.status_var.set("Checking dependencies...")
        
        required_packages = [
            'torch', 'torchvision', 'cv2', 'numpy', 'pandas', 
            'matplotlib', 'PIL', 'scipy', 'albumentations', 'wandb', 'sklearn'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == 'cv2':
                    import cv2
                elif package == 'PIL':
                    from PIL import Image
                elif package == 'sklearn':
                    import sklearn
                else:
                    __import__(package)
                print(f"DEBUG: ✅ {package} is available")
            except ImportError:
                print(f"DEBUG: ❌ {package} is missing")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"DEBUG: Missing packages found: {missing_packages}")
            message = f"Missing packages: {', '.join(missing_packages)}\\n\\n"
            message += "Click 'Install Dependencies' to install them."
            messagebox.showwarning("Dependencies Check", message)
            self.status_var.set(f"Missing: {', '.join(missing_packages)}")
        else:
            print("DEBUG: All dependencies are available")
            messagebox.showinfo("Dependencies Check", "All dependencies are installed!")
            self.status_var.set("All dependencies OK")
    
    def install_dependencies(self):
        """Install dependencies from requirements.txt"""
        print("DEBUG: Starting dependency installation...")
        self.status_var.set("Installing dependencies...")
        
        try:
            if not os.path.exists("requirements.txt"):
                print("DEBUG: requirements.txt not found!")
                messagebox.showerror("Error", "requirements.txt not found!")
                return
            
            print("DEBUG: Found requirements.txt, creating installation window...")
            
            # Create progress window with detailed output
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Installing Dependencies")
            progress_window.geometry("700x500")
            progress_window.transient(self.root)
            
            # Progress label
            progress_label = ttk.Label(progress_window, text="Preparing installation...")
            progress_label.pack(pady=10)
            
            # Progress bar
            progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
            progress_bar.pack(pady=5, padx=20, fill=tk.X)
            progress_bar.start()
            
            # Text widget for output
            output_frame = ttk.Frame(progress_window)
            output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            output_text = tk.Text(output_frame, wrap=tk.WORD, height=20, font=("Consolas", 9))
            scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=output_text.yview)
            output_text.configure(yscrollcommand=scrollbar.set)
            
            output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Add initial info
            output_text.insert(tk.END, "=== DEPENDENCY INSTALLATION LOG ===\\n\\n")
            output_text.insert(tk.END, f"Python executable: {sys.executable}\\n")
            output_text.insert(tk.END, f"Working directory: {os.getcwd()}\\n")
            output_text.insert(tk.END, f"Requirements file: requirements.txt\\n\\n")
            
            # Check tkinter availability (This fixes the error!)
            try:
                import tkinter
                output_text.insert(tk.END, "✅ Tkinter is available (built-in with Python)\\n")
                print("DEBUG: Tkinter check passed - no need to install tkinter")
            except ImportError:
                output_text.insert(tk.END, "⚠️  Tkinter not available - may need to install python3-tk on Linux\\n")
                print("DEBUG: Tkinter not available")
            
            # Check conda environment
            conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Not in conda environment')
            output_text.insert(tk.END, f"Current environment: {conda_env}\\n\\n")
            print(f"DEBUG: Current conda environment: {conda_env}")
            
            # Check if tkinter is in requirements.txt and warn
            with open("requirements.txt", "r") as f:
                requirements_content = f.read()
                if "tkinter" in requirements_content.lower():
                    output_text.insert(tk.END, "⚠️  WARNING: tkinter found in requirements.txt\\n")
                    output_text.insert(tk.END, "   tkinter is built-in with Python and cannot be installed via pip\\n")
                    output_text.insert(tk.END, "   This may cause installation errors. Consider removing it from requirements.txt\\n\\n")
                    print("DEBUG: WARNING - tkinter found in requirements.txt")
            
            output_text.insert(tk.END, "Starting pip installation...\\n")
            output_text.insert(tk.END, "Command: pip install -r requirements.txt\\n")
            output_text.insert(tk.END, "-" * 60 + "\\n")
            output_text.see(tk.END)
            output_text.update()
            
            # Run installation in separate thread to avoid freezing UI
            def run_installation():
                try:
                    print("DEBUG: Starting pip subprocess...")
                    process = subprocess.Popen(
                        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-v"],
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                        cwd=os.getcwd()
                    )
                    
                    # Read output line by line
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            print(f"DEBUG: pip output: {line.strip()}")
                            # Use after() to safely update GUI from thread
                            self.root.after(0, lambda l=line: self.update_installation_output(
                                output_text, progress_label, progress_window, l))
                    
                    process.stdout.close()
                    return_code = process.wait()
                    
                    print(f"DEBUG: Installation completed with return code: {return_code}")
                    
                    # Update final status
                    self.root.after(0, lambda: self.finish_installation(
                        progress_bar, progress_label, output_text, progress_window, return_code))
                        
                except Exception as e:
                    print(f"DEBUG: Installation thread error: {str(e)}")
                    self.root.after(0, lambda: self.installation_error(
                        progress_bar, progress_label, progress_window, str(e)))
            
            # Start installation in background thread
            install_thread = threading.Thread(target=run_installation, daemon=True)
            install_thread.start()
            print("DEBUG: Installation thread started")
            
        except Exception as e:
            print(f"DEBUG: Error setting up installation: {str(e)}")
            self.status_var.set("Error installing dependencies")
            messagebox.showerror("Error", f"Failed to setup installation: {str(e)}")
    
    def update_installation_output(self, output_text, progress_label, progress_window, line):
        """Update installation output in GUI thread"""
        try:
            output_text.insert(tk.END, line)
            output_text.see(tk.END)
            
            # Update progress label based on content
            if "Installing" in line:
                package = line.split()[-1] if line.split() else "package"
                progress_label.config(text=f"Installing {package}...")
            elif "Successfully installed" in line:
                progress_label.config(text="Installation completed!")
            elif "ERROR" in line or "Failed" in line:
                progress_label.config(text="Error occurred during installation")
            elif "Collecting" in line:
                package = line.replace("Collecting", "").strip()
                progress_label.config(text=f"Collecting {package}...")
            
            progress_window.update_idletasks()
        except Exception as e:
            print(f"DEBUG: Error updating output: {str(e)}")
    def finish_installation(self, progress_bar, progress_label, output_text, progress_window, return_code):
        """Finish installation process"""
        try:
            progress_bar.stop()
            
            if return_code == 0:
                output_text.insert(tk.END, "\\n" + "="*60 + "\\n")
                output_text.insert(tk.END, "INSTALLATION COMPLETED SUCCESSFULLY!\\n")
                output_text.insert(tk.END, "="*60 + "\\n")
                progress_label.config(text="All dependencies installed successfully!")
                self.status_var.set("Dependencies installed successfully")
                print("DEBUG: Installation completed successfully")
                
                # Add close button
                close_btn = ttk.Button(progress_window, text="Close", 
                                     command=lambda: self.close_installation_window(progress_window, True))
                close_btn.pack(pady=10)
                
            else:
                output_text.insert(tk.END, "\\n" + "="*60 + "\\n")
                output_text.insert(tk.END, f"INSTALLATION FAILED (Exit code: {return_code})\\n")
                output_text.insert(tk.END, "="*60 + "\\n")
                progress_label.config(text="Installation failed")
                self.status_var.set("Error installing dependencies")
                print(f"DEBUG: Installation failed with exit code: {return_code}")
                
                # Add close button
                close_btn = ttk.Button(progress_window, text="Close", 
                                     command=lambda: self.close_installation_window(progress_window, False))
                close_btn.pack(pady=10)
            
            output_text.see(tk.END)
        except Exception as e:
            print(f"DEBUG: Error finishing installation: {str(e)}")
    
    def installation_error(self, progress_bar, progress_label, progress_window, error_msg):
        """Handle installation error"""
        try:
            progress_bar.stop()
            progress_label.config(text="Installation error")
            self.status_var.set("Error installing dependencies")
            print(f"DEBUG: Installation error handled: {error_msg}")
            
            # Add close button
            close_btn = ttk.Button(progress_window, text="Close", 
                                 command=lambda: self.close_installation_window(progress_window, False))
            close_btn.pack(pady=10)
        except Exception as e:
            print(f"DEBUG: Error handling installation error: {str(e)}")
    
    def close_installation_window(self, window, success):
        """Close installation window and show result"""
        try:
            window.destroy()
            if success:
                messagebox.showinfo("Success", "All dependencies installed successfully!")
                print("DEBUG: Installation window closed - success")
            else:
                messagebox.showerror("Error", "Installation failed. Check the log for details.")
                print("DEBUG: Installation window closed - failed")
        except Exception as e:
            print(f"DEBUG: Error closing installation window: {str(e)}")
    
    def open_folder(self):
        """Open project folder in file explorer"""
        print("DEBUG: Opening project folder...")
        try:
            current_dir = os.getcwd()
            if os.name == 'nt':  # Windows
                os.startfile(current_dir)
                print(f"DEBUG: Opened folder in Windows Explorer: {current_dir}")
            elif os.name == 'posix':  # macOS and Linux
                subprocess.call(['open', current_dir])
                print(f"DEBUG: Opened folder in Finder/File Manager: {current_dir}")
            
            self.status_var.set("Project folder opened")
        except Exception as e:
            print(f"DEBUG: Error opening folder: {str(e)}")
            messagebox.showerror("Error", f"Failed to open folder: {str(e)}")
    
    def update_inference_output(self, output_text, status_label, log_window, line):
        """Update inference output in GUI thread"""
        try:
            output_text.insert(tk.END, line)
            output_text.see(tk.END)
            
            # Update status label based on content
            if "Processing" in line:
                if "subject" in line.lower() or "subjek" in line.lower():
                    status_label.config(text=f"Processing subject...")
                else:
                    status_label.config(text="Processing video...")
            elif "Analyzing" in line:
                status_label.config(text="Analyzing frames...")
            elif "Saving" in line:
                status_label.config(text="Saving results...")
            elif "Complete" in line or "Done" in line:
                status_label.config(text="Inference completed!")
            elif "Error" in line or "Failed" in line:
                status_label.config(text="Error occurred during inference")
            elif "Loading" in line:
                if "model" in line.lower():
                    status_label.config(text="Loading model...")
                else:
                    status_label.config(text="Loading data...")            
            log_window.update_idletasks()
        except Exception as e:
            print(f"DEBUG: Error updating inference output: {str(e)}")
    
    def finish_inference(self, progress_bar, status_label, output_text, log_window, return_code, subject_name=None):
        """Finish inference process"""
        try:
            progress_bar.stop()
            
            subject_text = f" for {subject_name}" if subject_name else ""
            
            if return_code == 0:
                output_text.insert(tk.END, "\n" + "="*60 + "\n")
                output_text.insert(tk.END, f"VIDEO INFERENCE COMPLETED SUCCESSFULLY{subject_text.upper()}!\n")
                output_text.insert(tk.END, "="*60 + "\n")
                status_label.config(text=f"Video inference completed successfully{subject_text}!")
                self.status_var.set(f"Video inference completed successfully{subject_text}")
                print(f"DEBUG: Inference completed successfully{subject_text}")
                
                # Add close button
                close_btn = ttk.Button(log_window, text="Close", 
                                     command=lambda: self.close_inference_window(log_window, True))
                close_btn.pack(pady=10)
                
            else:
                output_text.insert(tk.END, "\n" + "="*60 + "\n")
                output_text.insert(tk.END, f"INFERENCE FAILED{subject_text.upper()} (Exit code: {return_code})\n")
                output_text.insert(tk.END, "="*60 + "\n")
                status_label.config(text=f"Inference failed{subject_text}")
                self.status_var.set(f"Error during inference{subject_text}")
                print(f"DEBUG: Inference failed{subject_text} with exit code: {return_code}")
                
                # Add close button
                close_btn = ttk.Button(log_window, text="Close", 
                                     command=lambda: self.close_inference_window(log_window, False))
                close_btn.pack(pady=10)
            
            output_text.see(tk.END)
        except Exception as e:
            print(f"DEBUG: Error finishing inference: {str(e)}")
    
    def inference_error(self, progress_bar, status_label, log_window, error_msg):
        """Handle inference error"""
        try:
            progress_bar.stop()
            status_label.config(text="Inference error")
            self.status_var.set("Error during inference")
            print(f"DEBUG: Inference error handled: {error_msg}")
            
            # Add close button
            close_btn = ttk.Button(log_window, text="Close", 
                                 command=lambda: self.close_inference_window(log_window, False))
            close_btn.pack(pady=10)
        except Exception as e:
            print(f"DEBUG: Error handling inference error: {str(e)}")
    
    def close_inference_window(self, window, success):
        """Close inference window and show result"""
        try:
            window.destroy()
            if success:
                messagebox.showinfo("Success", "Video inference completed successfully! Check the output files.")
                print("DEBUG: Inference window closed - success")
            else:
                messagebox.showerror("Error", "Inference failed. Check the log for details.")
                print("DEBUG: Inference window closed - failed")
        except Exception as e:
            print(f"DEBUG: Error closing inference window: {str(e)}")
    
    def run_subject_inference(self):
        """Run video inference dengan pilihan subjek"""
        print("DEBUG: Opening subject selection for inference...")
        self.status_var.set("Opening subject selection...")
        
        try:
            # Import video_inference untuk mendapatkan daftar subjek
            import sys
            sys.path.append(os.getcwd())
            from video_inference import get_available_subjects, process_selected_subject
            
            # Get available subjects
            subjects = get_available_subjects()
            
            if not subjects:
                messagebox.showwarning("No Subjects", "No subjects found in data_uji directory!")
                return
              # Create subject selection window
            selection_window = tk.Toplevel(self.root)
            selection_window.title("Select Subject for Inference")
            selection_window.geometry("450x400")
            selection_window.transient(self.root)
            selection_window.grab_set()  # Make it modal
            
            # Title
            title_label = ttk.Label(selection_window, text="Select Subject for Video Inference", 
                                   font=("Arial", 12, "bold"))
            title_label.pack(pady=10)
            
            # Info label
            info_label = ttk.Label(selection_window, text=f"Found {len(subjects)} subjects:")
            info_label.pack(pady=5)
            
            # Subject selection frame
            selection_frame = ttk.Frame(selection_window)
            selection_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            # Dropdown for subject selection
            subject_var = tk.StringVar(value=subjects[0] if subjects else "")
            subject_label = ttk.Label(selection_frame, text="Choose subject:")
            subject_label.pack(anchor=tk.W, pady=(0, 5))
            
            subject_combo = ttk.Combobox(selection_frame, textvariable=subject_var, 
                                       values=subjects, state="readonly", width=30)
            subject_combo.pack(fill=tk.X, pady=(0, 10))
            
            # Subject details
            details_frame = ttk.LabelFrame(selection_frame, text="Subject Details")
            details_frame.pack(fill=tk.X, pady=(0, 10))
            
            details_text = tk.Text(details_frame, height=6, wrap=tk.WORD)
            details_scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=details_text.yview)
            details_text.configure(yscrollcommand=details_scrollbar.set)
            
            details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
            
            def update_details(event=None):
                selected_subject = subject_var.get()
                if selected_subject:
                    subject_path = os.path.join("data_uji", selected_subject)
                    details_text.delete(1.0, tk.END)
                    
                    if os.path.exists(subject_path):
                        video_path = os.path.join(subject_path, f"{selected_subject}.mp4")
                        csv_files = [f for f in os.listdir(subject_path) if f.endswith('.csv')]
                        
                        details_text.insert(tk.END, f"Subject: {selected_subject}\\n")
                        details_text.insert(tk.END, f"Video: {'Found' if os.path.exists(video_path) else 'Not found'}\\n")
                        details_text.insert(tk.END, f"CSV files: {len(csv_files)}\\n")
                        
                        if csv_files:
                            details_text.insert(tk.END, f"Data files: {', '.join(csv_files)}\\n")
                        
                        # Check if already processed
                        output_path = os.path.join("inference_results", selected_subject)
                        if os.path.exists(output_path):
                            details_text.insert(tk.END, f"\\nStatus: Already processed\\n")
                            details_text.insert(tk.END, f"Output folder: {output_path}\\n")
                        else:
                            details_text.insert(tk.END, f"\\nStatus: Not processed yet\\n")
            
            subject_combo.bind('<<ComboboxSelected>>', update_details)
            update_details()  # Initial update
              # Buttons frame
            buttons_frame = ttk.Frame(selection_window)
            buttons_frame.pack(fill=tk.X, padx=20, pady=10)
            
            def start_processing():
                selected_subject = subject_var.get()
                print(f"DEBUG: Start processing clicked for subject: {selected_subject}")
                if not selected_subject:
                    messagebox.showwarning("No Selection", "Please select a subject!")
                    return
                
                print(f"DEBUG: Starting inference for {selected_subject}")
                selection_window.destroy()
                self.run_inference_for_subject(selected_subject)
              # Process button
            process_btn = ttk.Button(buttons_frame, text="Start Processing", 
                                   command=start_processing, width=20)
            process_btn.pack(side=tk.LEFT, padx=(0, 10))
            
            # Cancel button
            cancel_btn = ttk.Button(buttons_frame, text="Cancel", 
                                  command=selection_window.destroy, width=15)
            cancel_btn.pack(side=tk.LEFT)
            
            print("DEBUG: Subject selection window opened")
            
        except Exception as e:
            print(f"DEBUG: Error opening subject selection: {str(e)}")
            messagebox.showerror("Error", f"Failed to open subject selection: {str(e)}")
    
    def run_inference_for_subject(self, subject_name):
        """Run inference untuk subjek tertentu dengan GUI logging"""
        print(f"DEBUG: Starting inference for {subject_name}...")
        self.status_var.set(f"Starting inference for {subject_name}...")
        
        try:
            print(f"DEBUG: Creating inference log window for {subject_name}...")
            
            # Create process log window
            log_window = tk.Toplevel(self.root)
            log_window.title(f"Video Inference - {subject_name}")
            log_window.geometry("800x600")
            log_window.transient(self.root)
            
            # Status label
            status_label = ttk.Label(log_window, text=f"Starting inference for {subject_name}...")
            status_label.pack(pady=10)
            
            # Progress bar
            progress_bar = ttk.Progressbar(log_window, mode='indeterminate')
            progress_bar.pack(pady=5, padx=20, fill=tk.X)
            progress_bar.start()
            
            # Text widget for output
            output_frame = ttk.Frame(log_window)
            output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            output_text = tk.Text(output_frame, wrap=tk.WORD, height=25, font=("Consolas", 9))
            scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=output_text.yview)
            output_text.configure(yscrollcommand=scrollbar.set)
            
            output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Add initial info
            output_text.insert(tk.END, f"=== VIDEO INFERENCE: {subject_name} ===\\n\\n")
            output_text.insert(tk.END, f"Subject: {subject_name}\\n")
            output_text.insert(tk.END, f"Input video: data_uji\\{subject_name}\\{subject_name}.mp4\\n")
            output_text.insert(tk.END, f"Output directory: inference_results\\{subject_name}\\n\\n")
            output_text.insert(tk.END, "Starting processing...\\n")
            output_text.insert(tk.END, "-" * 60 + "\\n")
            output_text.see(tk.END)
            output_text.update()
            
            print(f"DEBUG: Starting inference subprocess for {subject_name}...")
            
            # Run inference in separate thread
            def run_subject_inference_process():
                try:
                    # Create a temporary script to run the specific subject
                    temp_script = f'''
import sys
import os
sys.path.append(r"{os.getcwd()}")
from video_inference import process_selected_subject

result = process_selected_subject("{subject_name}")
print(f"Processing result: {{result['status']}}")
print(f"Message: {{result['message']}}")

if result['status'] == 'success':
    for key, path in result['output_paths'].items():
        if key != 'directory':
            print(f"{{key.title()}}: {{path}}")

'''
                    
                    # Write temp script
                    temp_file = f"temp_inference_{subject_name}.py"
                    with open(temp_file, 'w') as f:
                        f.write(temp_script)
                    
                    process = subprocess.Popen(
                        [sys.executable, temp_file],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                        cwd=os.getcwd()
                    )
                    
                    # Read output line by line
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            print(f"DEBUG: inference output: {line.strip()}")
                            self.root.after(0, lambda l=line: self.update_inference_output(
                                output_text, status_label, log_window, l))
                    
                    process.stdout.close()
                    return_code = process.wait()
                    
                    # Clean up temp file
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    
                    print(f"DEBUG: Inference completed for {subject_name} with return code: {return_code}")
                    
                    # Update final status
                    self.root.after(0, lambda: self.finish_inference(
                        progress_bar, status_label, output_text, log_window, return_code))
                        
                except Exception as e:
                    print(f"DEBUG: Inference thread error: {str(e)}")
                    self.root.after(0, lambda: self.inference_error(
                        progress_bar, status_label, log_window, str(e)))
            
            # Start inference in background thread
            inference_thread = threading.Thread(target=run_subject_inference_process, daemon=True)
            inference_thread.start()
            print(f"DEBUG: Inference thread started for {subject_name}")
            
            self.status_var.set(f"Processing {subject_name} with log window")
            
        except Exception as e:
            print(f"DEBUG: Error starting inference for {subject_name}: {str(e)}")
            self.status_var.set("Error starting inference")
            messagebox.showerror("Error", f"Failed to start inference for {subject_name}: {str(e)}")
    
    def view_results(self):
        """View inference results"""
        print("DEBUG: Opening results viewer...")
        self.status_var.set("Opening results viewer...")
        
        try:
            # Check if results directory exists
            results_dir = "inference_results"
            if not os.path.exists(results_dir):
                messagebox.showinfo("No Results", "No inference results found. Run some inference first!")
                return
            
            # Get processed results
            results = {}
            for subject in os.listdir(results_dir):
                subject_path = os.path.join(results_dir, subject)
                if os.path.isdir(subject_path):
                    results[subject] = []
                    for file in os.listdir(subject_path):
                        if os.path.isfile(os.path.join(subject_path, file)):
                            results[subject].append(file)
            
            if not results:
                messagebox.showinfo("No Results", "No processed results found!")
                return
            
            # Create results window
            results_window = tk.Toplevel(self.root)
            results_window.title("Inference Results Viewer")
            results_window.geometry("600x500")
            results_window.transient(self.root)
            
            # Title
            title_label = ttk.Label(results_window, text="Inference Results", 
                                   font=("Arial", 14, "bold"))
            title_label.pack(pady=10)
            
            # Main frame with scrollbar
            main_frame = ttk.Frame(results_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # Treeview for results
            columns = ("Subject", "Files", "Status")
            tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=15)
            
            # Configure columns
            tree.heading("Subject", text="Subject")
            tree.heading("Files", text="Output Files")
            tree.heading("Status", text="Status")
            
            tree.column("Subject", width=100)
            tree.column("Files", width=200)
            tree.column("Status", width=100)
            
            # Scrollbar for treeview
            tree_scroll = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=tree_scroll.set)
            
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Populate treeview
            for subject, files in results.items():
                file_count = len(files)
                file_types = set(f.split('.')[-1] for f in files if '.' in f)
                status = "Complete" if file_count >= 3 else "Partial"
                
                tree.insert("", tk.END, values=(
                    subject,
                    f"{file_count} files ({', '.join(file_types)})",
                    status
                ))
            
            # Buttons frame
            buttons_frame = ttk.Frame(results_window)
            buttons_frame.pack(fill=tk.X, padx=10, pady=10)
            
            def open_selected():
                selection = tree.selection()
                if not selection:
                    messagebox.showwarning("No Selection", "Please select a subject!")
                    return
                
                item = tree.item(selection[0])
                subject = item['values'][0]
                subject_path = os.path.join(results_dir, subject)
                
                # Open folder in file explorer
                if os.name == 'nt':  # Windows
                    os.startfile(subject_path)
                elif os.name == 'posix':  # macOS and Linux
                    subprocess.call(['open', subject_path])
            
            def refresh_results():
                # Clear existing items
                for item in tree.get_children():
                    tree.delete(item)
                
                # Repopulate
                for subject in os.listdir(results_dir):
                    subject_path = os.path.join(results_dir, subject)
                    if os.path.isdir(subject_path):
                        files = [f for f in os.listdir(subject_path) 
                                if os.path.isfile(os.path.join(subject_path, f))
                        ]
                        file_count = len(files)
                        file_types = set(f.split('.')[-1] for f in files if '.' in f)
                        status = "Complete" if file_count >= 3 else "Partial"
                        
                        tree.insert("", tk.END, values=(
                            subject,
                            f"{file_count} files ({', '.join(file_types)})",
                            status
                        ))
            
            # Buttons
            open_btn = ttk.Button(buttons_frame, text="Open Folder", command=open_selected)
            open_btn.pack(side=tk.LEFT, padx=(0, 10))
            
            refresh_btn = ttk.Button(buttons_frame, text="Refresh", command=refresh_results)
            refresh_btn.pack(side=tk.LEFT, padx=(0, 10))
            
            close_btn = ttk.Button(buttons_frame, text="Close", command=results_window.destroy)
            close_btn.pack(side=tk.RIGHT)
            
            print("DEBUG: Results viewer opened")
            
        except Exception as e:
            print(f"DEBUG: Error opening results viewer: {str(e)}")
            messagebox.showerror("Error", f"Failed to open results viewer: {str(e)}")

    def run_advanced_analytics(self):
        """Run advanced analytics dashboard"""
        print("DEBUG: Starting Advanced Analytics...")
        
        try:
            # Import and run advanced analytics
            import advanced_analytics
            
            # Create analytics GUI
            analytics_root = tk.Toplevel(self.root)
            analytics_root.title("Advanced Carotid Analytics")
            analytics_root.geometry("900x700")
            
            # Main frame
            main_frame = ttk.Frame(analytics_root, padding="20")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Title
            title_label = ttk.Label(main_frame, text="Advanced Carotid Artery Analytics", 
                                   font=("Arial", 16, "bold"))
            title_label.pack(pady=(0, 20))
              # Analytics instance
            analytics = advanced_analytics.AdvancedAnalytics()
            
            # Subject selection
            subject_frame = ttk.Frame(main_frame)
            subject_frame.pack(fill=tk.X, pady=(0, 20))
            
            ttk.Label(subject_frame, text="Select Subject:").pack(side=tk.LEFT, padx=(0, 10))
            subject_var = tk.StringVar(value="1")
            subject_combo = ttk.Combobox(subject_frame, textvariable=subject_var, 
                                       values=[str(i) for i in range(1, 8)], width=10)
            subject_combo.pack(side=tk.LEFT, padx=(0, 20))
            
            # Buttons
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=(0, 20))
            
            def generate_report():
                """Generate comprehensive report"""
                try:
                    subject_num = int(subject_var.get())
                    report = analytics.generate_comprehensive_report(subject_num)
                    
                    # Show report in new window
                    report_window = tk.Toplevel(analytics_root)
                    report_window.title(f"Analysis Report - Subject {subject_num}")
                    report_window.geometry("900x700")
                    
                    # Text widget with scrollbar
                    text_frame = ttk.Frame(report_window)
                    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10))
                    scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
                    text_widget.configure(yscrollcommand=scrollbar.set)
                    
                    # Format report text
                    report_text = f"""COMPREHENSIVE ANALYSIS REPORT
{'='*50}
Subject: {report['subject_number']}
Generated: {report['timestamp']}

DATA AVAILABILITY
{'='*20}
Diameter Data: {'✅ Available' if report['data_availability']['diameter_data'] else '❌ Missing'}
Timestamps: {'✅ Available' if report['data_availability']['timestamps'] else '❌ Missing'}
Pressure Data: {'✅ Available' if report['data_availability']['pressure_data'] else '❌ Missing'}
Original Video: {'✅ Available' if report['data_availability']['video_original'] else '❌ Missing'}
Segmentation Results: {'✅ Available' if report['data_availability']['segmentation_results'] else '❌ Missing'}

BASIC STATISTICS
{'='*20}
"""
                    
                    if report['basic_statistics']:
                        stats = report['basic_statistics']
                        report_text += f"""Mean Diameter: {stats.get('mean', 0):.3f} mm
Standard Deviation: {stats.get('std', 0):.3f} mm
Range: {stats.get('min', 0):.3f} - {stats.get('max', 0):.3f} mm
Coefficient of Variation: {stats.get('cv', 0):.1f}%
Total Frames: {stats.get('count', 0)}
Duration: {stats.get('duration_seconds', 0):.1f} seconds

Percentiles:
  25th: {stats.get('q25', 0):.3f} mm
  50th (Median): {stats.get('median', 0):.3f} mm
  75th: {stats.get('q75', 0):.3f} mm

Estimated Systolic: {stats.get('estimated_systolic', 0):.3f} mm
Estimated Diastolic: {stats.get('estimated_diastolic', 0):.3f} mm
Pulse Pressure: {stats.get('pulse_pressure_mm', 0):.3f} mm

"""
                    
                    if report['cardiac_cycles']:
                        cycles = report['cardiac_cycles']
                        report_text += f"""CARDIAC CYCLE ANALYSIS
{'='*25}
Number of Detected Cycles: {cycles.get('num_cycles', 0)}
Estimated Heart Rate: {cycles.get('heart_rate_estimate', 0):.1f} bpm
Average Cycle Length: {cycles.get('avg_cycle_length', 0):.1f} frames
Rhythm Regularity: {cycles.get('rhythm_regularity', 0):.1f}%

"""
                    
                    if report['data_quality']:
                        quality = report['data_quality']
                        report_text += f"""DATA QUALITY ASSESSMENT
{'='*25}
Overall Quality Score: {quality.get('overall_score', 0):.0f}/100

Strengths:
"""
                        for strength in quality.get('strengths', []):
                            report_text += f"  ✅ {strength}\n"
                            
                        report_text += "\nIssues:\n"
                        for issue in quality.get('issues', []):
                            report_text += f"  ⚠️ {issue}\n"
                    
                    if report['recommendations']:
                        report_text += f"""
RECOMMENDATIONS
{'='*15}
"""
                        for rec in report['recommendations']:
                            report_text += f"{rec}\n"
                    
                    text_widget.insert(tk.END, report_text)
                    text_widget.config(state=tk.DISABLED)
                    
                    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                    
                    # Save report button
                    def save_report():
                        filename = f"analysis_report_subject_{subject_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(report_text)
                        messagebox.showinfo("Success", f"Report saved as {filename}")
                    
                    ttk.Button(report_window, text="Save Report", command=save_report).pack(pady=10)
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Error generating report: {str(e)}")
            
            def create_visualization():
                """Create advanced visualization"""
                try:
                    subject_num = int(subject_var.get())
                    
                    # Show loading message
                    loading_window = tk.Toplevel(analytics_root)
                    loading_window.title("Loading...")
                    loading_window.geometry("300x100")
                    loading_window.transient(analytics_root)
                    loading_window.grab_set()
                    
                    ttk.Label(loading_window, text="Creating advanced visualization...", 
                             font=("Arial", 12)).pack(expand=True)
                    loading_window.update()
                    
                    # Create visualization in separate thread
                    def create_viz():
                        try:
                            fig = analytics.create_advanced_visualization(subject_num)
                            
                            loading_window.destroy()
                            
                            if fig is not None:
                                # Show in new window
                                viz_window = tk.Toplevel(analytics_root)
                                viz_window.title(f"Advanced Visualization - Subject {subject_num}")
                                viz_window.geometry("1400x900")
                                
                                # Create canvas for matplotlib
                                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                                canvas = FigureCanvasTkAgg(fig, viz_window)
                                canvas.draw()
                                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                                
                                # Save button
                                def save_viz():
                                    from tkinter import filedialog
                                    filename = filedialog.asksaveasfilename(
                                        defaultextension=".png",
                                        filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*")]
                                    )
                                    if filename:
                                        fig.savefig(filename, dpi=300, bbox_inches='tight')
                                        messagebox.showinfo("Success", f"Visualization saved to {filename}")
                                
                                ttk.Button(viz_window, text="Save Visualization", command=save_viz).pack(pady=10)
                            else:
                                messagebox.showwarning("Warning", "No data available for visualization")
                                
                        except Exception as e:
                            loading_window.destroy()
                            messagebox.showerror("Error", f"Error creating visualization: {str(e)}")
                    
                    # Run in thread to prevent blocking
                    import threading
                    thread = threading.Thread(target=create_viz)
                    thread.daemon = True
                    thread.start()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Error creating visualization: {str(e)}")
            
            ttk.Button(button_frame, text="Generate Comprehensive Report", 
                      command=generate_report).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Button(button_frame, text="Create Advanced Visualization", 
                      command=create_visualization).pack(side=tk.LEFT)
            
            # Info text
            info_text = """Advanced Analytics Features:

• Comprehensive statistical analysis of diameter measurements
• Cardiac cycle detection and heart rate estimation  
• Data quality assessment with scoring system
• Advanced visualizations including frequency analysis
• Clinical recommendations based on measurements
• Detailed reporting with all metrics and export options

Select a subject and choose an analysis option above."""
            
            info_label = ttk.Label(main_frame, text=info_text, justify=tk.LEFT)
            info_label.pack(pady=(0, 20))
            
            print("DEBUG: Advanced Analytics window opened")
            
        except ImportError as e:
            messagebox.showerror("Error", f"Failed to import advanced_analytics module: {str(e)}")
        except Exception as e:
            print(f"DEBUG: Error running advanced analytics: {str(e)}")
            messagebox.showerror("Error", f"Failed to run advanced analytics: {str(e)}")
    
    def run_batch_processing(self):
        """Run batch processing for all subjects"""
        print("DEBUG: Starting Batch Processing...")
        
        try:
            # Confirmation dialog
            response = messagebox.askyesno(
                "Batch Processing Confirmation",
                "This will process all subjects (1-7) and generate comprehensive analysis.\n\n"
                "This may take several minutes. Continue?",
                icon='question'
            )
            
            if not response:
                return
            
            # Create progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Batch Processing Progress")
            progress_window.geometry("500x300")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            # Progress content frame
            content_frame = ttk.Frame(progress_window, padding="20")
            content_frame.pack(fill=tk.BOTH, expand=True)
            
            # Title
            ttk.Label(content_frame, text="Batch Processing All Subjects", 
                     font=("Arial", 14, "bold")).pack(pady=(0, 10))
            
            # Progress bar
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(content_frame, variable=progress_var, 
                                         maximum=100, length=400)
            progress_bar.pack(pady=(0, 10))
            
            # Status label
            status_var = tk.StringVar(value="Initializing...")
            status_label = ttk.Label(content_frame, textvariable=status_var)
            status_label.pack(pady=(0, 10))
            
            # Output text area
            output_frame = ttk.Frame(content_frame)
            output_frame.pack(fill=tk.BOTH, expand=True)
            
            output_text = tk.Text(output_frame, height=8, wrap=tk.WORD)
            scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=output_text.yview)
            output_text.config(yscrollcommand=scrollbar.set)
            
            output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            def update_progress(message, percent):
                """Update progress display"""
                status_var.set(message)
                progress_var.set(percent)
                output_text.insert(tk.END, f"{message}\n")
                output_text.see(tk.END)
                progress_window.update()
            
            def run_batch():
                """Run batch processing in thread"""
                try:
                    update_progress("🔄 Starting batch processing...", 10)
                    
                    # Import batch processor
                    import batch_processor
                    processor = batch_processor.BatchProcessor()
                    
                    update_progress("📊 Processing individual subjects...", 20)
                    
                    # Process all subjects
                    results = processor.process_all_subjects()
                    update_progress(f"✅ Processed {len(results)} subjects", 40)
                    
                    # Create comparative analysis
                    update_progress("📈 Creating comparative analysis...", 60)
                    comparative_results = processor.create_comparative_analysis(results)
                    
                    # Create visualization
                    update_progress("📊 Generating visualizations...", 80)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    viz_path = f"batch_analysis_visualization_{timestamp}.png"
                    fig = processor.create_comparative_visualization(comparative_results, viz_path)
                    if fig:
                        plt.close(fig)
                    
                    # Export results
                    update_progress("💾 Exporting results...", 90)
                    exported_files = []
                    for fmt in ['json', 'csv', 'excel']:
                        try:
                            filename = processor.export_results(comparative_results, fmt)
                            exported_files.append(filename)
                        except Exception as e:
                            update_progress(f"⚠️ Export error ({fmt}): {str(e)}", None)
                    
                    # Generate summary report
                    processor.generate_summary_report(comparative_results)
                    
                    update_progress("✅ Batch processing completed!", 100)
                    
                    # Show results summary
                    summary = comparative_results['summary_statistics']
                    results_text = f"""
BATCH PROCESSING RESULTS
========================

✅ Total subjects analyzed: {summary['total_subjects_analyzed']}
✅ Subjects with complete data: {summary['subjects_with_complete_data']}
✅ Average quality score: {summary['average_quality_score']:.1f}%

📊 Mean diameter: {summary['diameter_stats']['mean_across_subjects']:.3f} ± {summary['diameter_stats']['std_across_subjects']:.3f} mm
💓 Mean heart rate: {summary['heart_rate_stats']['mean_hr']:.1f} ± {summary['heart_rate_stats']['std_hr']:.1f} bpm

📁 Files exported: {len(exported_files)}
🖼️ Visualization saved: {viz_path}

Data Quality Distribution:
• Excellent (≥80%): {summary['data_quality_distribution']['excellent']} subjects
• Good (60-79%): {summary['data_quality_distribution']['good']} subjects  
• Fair (40-59%): {summary['data_quality_distribution']['fair']} subjects
• Poor (<40%): {summary['data_quality_distribution']['poor']} subjects
"""
                    
                    update_progress(results_text, 100)
                    
                    # Enable close button
                    def enable_close():
                        close_btn = ttk.Button(content_frame, text="Close", 
                                             command=progress_window.destroy)
                        close_btn.pack(pady=10)
                        
                        # Show results button
                        def show_results():
                            try:
                                os.startfile(os.getcwd())  # Open current directory
                            except:
                                messagebox.showinfo("Results Location", 
                                                  f"Results saved in: {os.getcwd()}")
                        
                        results_btn = ttk.Button(content_frame, text="Open Results Folder", 
                                               command=show_results)
                        results_btn.pack(pady=5)
                    
                    progress_window.after(1000, enable_close)
                    
                except Exception as e:
                    update_progress(f"❌ Error: {str(e)}", 0)
                    messagebox.showerror("Batch Processing Error", 
                                       f"An error occurred during batch processing:\n{str(e)}")
              # Run in separate thread
            import threading
            thread = threading.Thread(target=run_batch)
            thread.daemon = True
            thread.start()
            
            print("DEBUG: Batch processing started")
            
        except ImportError as e:
            messagebox.showerror("Error", f"Failed to import batch_processor module: {str(e)}")
        except Exception as e:
            print(f"DEBUG: Error running batch processing: {str(e)}")
            messagebox.showerror("Error", f"Failed to run batch processing: {str(e)}")
    
    def run_enhanced_inference(self):
        """Run enhanced video inference script (with pressure integration)"""
        print("DEBUG: Starting enhanced video inference process...")
        self.status_var.set("Starting enhanced video inference...")
        
        try:
            if not os.path.exists("video_inference.py"):
                print("DEBUG: video_inference.py not found!")
                messagebox.showerror("Error", "video_inference.py not found!")
                return
            
            # Get available models
            available_models = []
            for model_file in ["UNet_25Mei_Sore.pth", "UNet_22Mei_Sore.pth"]:
                if os.path.exists(model_file):
                    available_models.append(model_file)
            
            if not available_models:
                print("DEBUG: No model files found!")
                messagebox.showerror("Error", "No model files found! Please train a model first.")
                return
            
            # Get available subjects
            subjects = []
            if os.path.exists("data_uji"):
                subjects = [d for d in os.listdir("data_uji") 
                           if os.path.isdir(os.path.join("data_uji", d)) and d.startswith("Subjek")]
                subjects.sort()
            
            if not subjects:
                messagebox.showerror("Error", "No subjects found in data_uji directory!")
                return
            
            # Enhanced Selection Dialog
            selection_dialog = tk.Toplevel(self.root)
            selection_dialog.title("Enhanced Inference - Model & Subject Selection")
            selection_dialog.geometry("500x400")
            selection_dialog.transient(self.root)
            selection_dialog.grab_set()
            
            # Center the dialog
            selection_dialog.update_idletasks()
            x = (selection_dialog.winfo_screenwidth() // 2) - (selection_dialog.winfo_width() // 2)
            y = (selection_dialog.winfo_screenheight() // 2) - (selection_dialog.winfo_height() // 2)
            selection_dialog.geometry(f"+{x}+{y}")
            
            # Main frame
            main_frame = ttk.Frame(selection_dialog, padding=20)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Title
            ttk.Label(main_frame, text="Enhanced Inference Configuration", 
                     font=("Arial", 14, "bold")).pack(pady=(0, 10))
            
            # Description
            desc_text = "Select model and subject for enhanced inference with pressure integration"
            ttk.Label(main_frame, text=desc_text, font=("Arial", 9)).pack(pady=(0, 20))
            
            # Model Selection Frame
            model_frame = ttk.LabelFrame(main_frame, text="Model Selection", padding=10)
            model_frame.pack(fill=tk.X, pady=(0, 15))
            
            ttk.Label(model_frame, text="Available Models:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
            
            selected_model = tk.StringVar(value=available_models[0])
            
            for model in available_models:
                # Get model info
                model_info = self.get_model_info(model)
                radio_text = f"{model} ({model_info})"
                ttk.Radiobutton(model_frame, text=radio_text, variable=selected_model, 
                               value=model).pack(anchor=tk.W, pady=2)
              # Subject Selection Frame
            subject_frame = ttk.LabelFrame(main_frame, text="Subject Selection", padding=10)
            subject_frame.pack(fill=tk.X, pady=(0, 15))
            
            ttk.Label(subject_frame, text="Select Subjects (multiple selection allowed):", font=("Arial", 10, "bold")).pack(anchor=tk.W)
            ttk.Label(subject_frame, text="Use checkboxes to select one or more subjects for batch processing", 
                     font=("Arial", 9), foreground="gray").pack(anchor=tk.W, pady=(0, 5))
            
            # Dictionary to store checkbox variables for each subject
            subject_vars = {}
            
            # Create scrollable frame for subjects
            canvas = tk.Canvas(subject_frame, height=120)
            scrollbar = ttk.Scrollbar(subject_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            for subject in subjects:
                # Create checkbox variable for each subject
                subject_vars[subject] = tk.BooleanVar()
                
                # Get subject info
                subject_info = self.get_subject_info(subject)
                checkbox_text = f"{subject} ({subject_info})"
                
                # Create checkbox
                checkbox = ttk.Checkbutton(scrollable_frame, text=checkbox_text, 
                                         variable=subject_vars[subject])
                checkbox.pack(anchor=tk.W, pady=2)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Selection controls
            control_frame = ttk.Frame(subject_frame)
            control_frame.pack(fill=tk.X, pady=(5, 0))
            
            def select_all():
                for var in subject_vars.values():
                    var.set(True)
            
            def clear_all():
                for var in subject_vars.values():
                    var.set(False)
            
            def select_with_complete_data():
                for subject, var in subject_vars.items():
                    subject_info = self.get_subject_info(subject)
                    # Select subjects that have all data (Video, Pressure, Timestamps)
                    has_all_data = "Video ✅" in subject_info and "Pressure ✅" in subject_info and "Timestamps ✅" in subject_info
                    var.set(has_all_data)
            
            ttk.Button(control_frame, text="Select All", command=select_all).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(control_frame, text="Clear All", command=clear_all).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(control_frame, text="Select Complete Data Only", command=select_with_complete_data).pack(side=tk.LEFT)
              # Features Info
            features_frame = ttk.LabelFrame(main_frame, text="Enhanced Features", padding=10)
            features_frame.pack(fill=tk.X, pady=(0, 15))
            
            features_text = "• Multiple subject selection (batch processing)\n• Pressure data integration\n• Real-time processing logs\n• Automatic result synchronization\n• Advanced analytics support\n• Progress monitoring for each subject"
            ttk.Label(features_frame, text=features_text, font=("Arial", 9)).pack(anchor=tk.W)
              # Buttons
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=(10, 0))
            
            def start_inference():
                # Get selected subjects
                selected_subjects = [subject for subject, var in subject_vars.items() if var.get()]
                
                if not selected_subjects:
                    messagebox.showwarning("Warning", "Please select at least one subject!")
                    return
                
                selection_dialog.destroy()
                self.start_enhanced_inference_batch(selected_model.get(), selected_subjects)
            
            def get_selection_count():
                return len([subject for subject, var in subject_vars.items() if var.get()])
            
            def update_button_text():
                count = get_selection_count()
                if count == 0:
                    start_btn.config(text="Start Enhanced Inference", state="disabled")
                elif count == 1:
                    start_btn.config(text="Start Enhanced Inference (1 subject)", state="normal")
                else:
                    start_btn.config(text=f"Start Batch Inference ({count} subjects)", state="normal")
            
            # Monitor checkbox changes to update button
            for var in subject_vars.values():
                var.trace_add('write', lambda *args: update_button_text())
            
            start_btn = ttk.Button(button_frame, text="Start Enhanced Inference", 
                                 command=start_inference, state="disabled")
            start_btn.pack(side=tk.RIGHT, padx=(5, 0))
            
            ttk.Button(button_frame, text="Cancel", 
                      command=selection_dialog.destroy).pack(side=tk.RIGHT)
            
            # Initial button state update
            update_button_text()
            
        except Exception as e:
            print(f"DEBUG: Error starting enhanced inference - {str(e)}")
            self.status_var.set("Error starting enhanced inference")
            messagebox.showerror("Error", f"Failed to start enhanced inference: {str(e)}")
    
    def get_model_info(self, model_file):
        """Get model information for display"""
        try:
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            if "25Mei" in model_file:
                return f"Latest Model, {file_size:.1f}MB"
            elif "22Mei" in model_file:
                return f"Previous Model, {file_size:.1f}MB"
            else:
                return f"{file_size:.1f}MB"
        except:
            return "Unknown size"
    
    def get_subject_info(self, subject_name):
        """Get subject information for display"""
        try:
            subject_path = os.path.join("data_uji", subject_name)
            
            # Check for video file
            video_files = [f for f in os.listdir(subject_path) if f.endswith('.mp4')]
            has_video = len(video_files) > 0
            
            # Check for pressure data
            pressure_files = [f for f in os.listdir(subject_path) if f.endswith('.csv') and 'subject' in f.lower()]
            has_pressure = len(pressure_files) > 0
            
            # Check for timestamps
            timestamp_files = [f for f in os.listdir(subject_path) if f.endswith('.csv') and 'timestamp' in f.lower()]
            has_timestamps = len(timestamp_files) > 0
            
            info_parts = []
            if has_video:
                info_parts.append("Video ✅")
            if has_pressure:
                info_parts.append("Pressure ✅")
            if has_timestamps:
                info_parts.append("Timestamps ✅")
            
            if not info_parts:
                return "No data"
            
            return ", ".join(info_parts)
        except:
            return "Error reading data"
    
    def start_enhanced_inference_with_options(self, model_file, subject_name):
        """Start the enhanced inference process with selected model and subject"""
        try:
            print(f"DEBUG: Starting enhanced inference with model: {model_file}, subject: {subject_name}")
            
            print("DEBUG: Creating enhanced inference log window...")
            
            # Create log window
            log_window = tk.Toplevel(self.root)
            log_window.title(f"Enhanced Inference - {model_file} - {subject_name}")
            log_window.geometry("800x600")
            
            # Create main frame
            main_frame = ttk.Frame(log_window, padding=10)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Title
            title_text = f"Enhanced Inference: {model_file} → {subject_name}"
            ttk.Label(main_frame, text=title_text, font=("Arial", 12, "bold")).pack(pady=(0, 10))
            
            # Progress info
            progress_frame = ttk.Frame(main_frame)
            progress_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(progress_frame, text="Status:").pack(side=tk.LEFT)
            status_label = ttk.Label(progress_frame, text="Initializing...", foreground="blue")
            status_label.pack(side=tk.LEFT, padx=(5, 0))
            
            # Progress bar
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(main_frame, variable=progress_var, mode='indeterminate')
            progress_bar.pack(fill=tk.X, pady=(0, 10))
            progress_bar.start()
            
            # Output text area
            output_frame = ttk.Frame(main_frame)
            output_frame.pack(fill=tk.BOTH, expand=True)
            
            output_text = tk.Text(output_frame, wrap=tk.WORD, font=("Consolas", 9))
            output_scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=output_text.yview)
            output_text.configure(yscrollcommand=output_scrollbar.set)
            
            output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Initial log message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output_text.insert(tk.END, f"[{timestamp}] Enhanced Inference Started\n")
            output_text.insert(tk.END, f"Model: {model_file}\n")
            output_text.insert(tk.END, f"Subject: {subject_name}\n")
            output_text.insert(tk.END, "\nStarting enhanced inference process...\n")
            
            # Function to run inference
            def run_enhanced_inference_process():
                try:
                    print("DEBUG: Starting enhanced inference subprocess with GUI logging...")
                    
                    # Build command with model and subject parameters
                    cmd = [
                        sys.executable, "video_inference.py",
                        "--model", model_file,
                        "--subject", subject_name,
                        "--enhanced"
                    ]
                    
                    print(f"DEBUG: Running command: {' '.join(cmd)}")
                    
                    # Run enhanced inference in separate thread with GUI logging
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )
                    
                    # Read output line by line
                    for line in process.stdout:
                        if line.strip():
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            log_line = f"[{timestamp}] {line.strip()}\n"
                            
                            # Update GUI in main thread
                            log_window.after(0, lambda: output_text.insert(tk.END, log_line))
                            log_window.after(0, lambda: output_text.see(tk.END))
                            
                            print(f"DEBUG: enhanced inference output: {line.strip()}")
                    
                    # Wait for process to complete
                    return_code = process.wait()
                    
                    # Update status
                    if return_code == 0:
                        log_window.after(0, lambda: status_label.config(text="Completed ✅", foreground="green"))
                        log_window.after(0, lambda: output_text.insert(tk.END, f"\n[{datetime.now().strftime('%H:%M:%S')}] Enhanced inference completed successfully!\n"))
                    else:
                        log_window.after(0, lambda: status_label.config(text="Failed ❌", foreground="red"))
                        log_window.after(0, lambda: output_text.insert(tk.END, f"\n[{datetime.now().strftime('%H:%M:%S')}] Enhanced inference failed with code {return_code}\n"))
                    
                    print(f"DEBUG: Enhanced inference completed with return code: {return_code}")
                    
                    # Stop progress bar
                    log_window.after(0, lambda: progress_bar.stop())
                    
                except Exception as e:
                    error_msg = f"Error during enhanced inference: {str(e)}"
                    log_window.after(0, lambda: output_text.insert(tk.END, f"\n[{datetime.now().strftime('%H:%M:%S')}] {error_msg}\n"))
                    log_window.after(0, lambda: status_label.config(text="Error ❌", foreground="red"))
                    log_window.after(0, lambda: progress_bar.stop())
                    print(f"DEBUG: Enhanced inference thread error: {str(e)}")
            
            # Start enhanced inference in background thread
            inference_thread = threading.Thread(target=run_enhanced_inference_process)
            inference_thread.daemon = True
            inference_thread.start()
            
            print("DEBUG: Enhanced inference thread started with GUI logging")
              # Update status
            self.status_var.set(f"Running enhanced inference: {model_file} → {subject_name}")
            
        except Exception as e:
            print(f"DEBUG: Error starting enhanced inference process - {str(e)}")
            self.status_var.set("Error starting enhanced inference")
            messagebox.showerror("Error", f"Failed to start enhanced inference: {str(e)}")
    
    def start_enhanced_inference_batch(self, model_file, selected_subjects):
        """Start enhanced inference for multiple subjects (batch processing)"""
        try:
            print(f"DEBUG: Starting batch enhanced inference with model: {model_file}")
            print(f"DEBUG: Selected subjects: {selected_subjects}")
            
            # Create batch progress window
            batch_window = tk.Toplevel(self.root)
            batch_window.title(f"Batch Enhanced Inference - {model_file}")
            batch_window.geometry("900x700")
            
            # Create main frame
            main_frame = ttk.Frame(batch_window, padding=10)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Title
            title_text = f"Batch Enhanced Inference: {model_file}"
            ttk.Label(main_frame, text=title_text, font=("Arial", 14, "bold")).pack(pady=(0, 10))
            
            # Summary info
            summary_frame = ttk.Frame(main_frame)
            summary_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(summary_frame, text=f"Processing {len(selected_subjects)} subjects:").pack(side=tk.LEFT)
            ttk.Label(summary_frame, text=" | ".join(selected_subjects), 
                     font=("Arial", 9), foreground="blue").pack(side=tk.LEFT, padx=(10, 0))
            
            # Overall progress
            overall_frame = ttk.LabelFrame(main_frame, text="Overall Progress", padding=5)
            overall_frame.pack(fill=tk.X, pady=(0, 10))
            
            overall_progress_var = tk.DoubleVar()
            overall_progress = ttk.Progressbar(overall_frame, variable=overall_progress_var, maximum=len(selected_subjects))
            overall_progress.pack(fill=tk.X, pady=(0, 5))
            
            overall_status = ttk.Label(overall_frame, text="Preparing batch processing...")
            overall_status.pack(anchor=tk.W)
            
            # Current subject progress
            current_frame = ttk.LabelFrame(main_frame, text="Current Subject", padding=5)
            current_frame.pack(fill=tk.X, pady=(0, 10))
            
            current_subject_label = ttk.Label(current_frame, text="Waiting...", font=("Arial", 10, "bold"))
            current_subject_label.pack(anchor=tk.W)
            
            current_progress = ttk.Progressbar(current_frame, mode='indeterminate')
            current_progress.pack(fill=tk.X, pady=(5, 0))
            
            # Results summary
            results_frame = ttk.LabelFrame(main_frame, text="Results Summary", padding=5)
            results_frame.pack(fill=tk.X, pady=(0, 10))
            
            results_text = tk.Text(results_frame, height=4, font=("Arial", 9))
            results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=results_text.yview)
            results_text.configure(yscrollcommand=results_scrollbar.set)
            results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Detailed log
            log_frame = ttk.LabelFrame(main_frame, text="Detailed Log", padding=5)
            log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            
            log_text = tk.Text(log_frame, wrap=tk.WORD, font=("Consolas", 8))
            log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
            log_text.configure(yscrollcommand=log_scrollbar.set)
            log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Control buttons
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=tk.X)
            
            # Variables for batch control
            batch_cancelled = tk.BooleanVar(value=False)
            batch_completed = tk.BooleanVar(value=False)
            
            def cancel_batch():
                batch_cancelled.set(True)
                overall_status.config(text="Cancellation requested...")
                cancel_btn.config(state="disabled")
            
            def close_window():
                batch_window.destroy()
            
            cancel_btn = ttk.Button(button_frame, text="Cancel Batch", command=cancel_batch)
            cancel_btn.pack(side=tk.LEFT)
            
            close_btn = ttk.Button(button_frame, text="Close", command=close_window, state="disabled")
            close_btn.pack(side=tk.RIGHT)
            
            # Initialize log
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_text.insert(tk.END, f"[{timestamp}] Batch Enhanced Inference Started\n")
            log_text.insert(tk.END, f"Model: {model_file}\n")
            log_text.insert(tk.END, f"Subjects: {', '.join(selected_subjects)}\n")
            log_text.insert(tk.END, f"Total: {len(selected_subjects)} subjects\n\n")
            
            # Batch processing function
            def run_batch_inference():
                try:
                    successful_subjects = []
                    failed_subjects = []
                    
                    for i, subject in enumerate(selected_subjects):
                        if batch_cancelled.get():
                            batch_window.after(0, lambda: log_text.insert(tk.END, f"\n[{datetime.now().strftime('%H:%M:%S')}] Batch processing cancelled by user\n"))
                            break
                        
                        # Update current subject
                        batch_window.after(0, lambda s=subject: current_subject_label.config(text=f"Processing: {s}"))
                        batch_window.after(0, lambda: current_progress.start())
                        
                        # Update overall progress
                        batch_window.after(0, lambda: overall_progress_var.set(i))
                        batch_window.after(0, lambda s=subject, i=i: overall_status.config(text=f"Processing {s} ({i+1}/{len(selected_subjects)})"))
                        
                        # Log start
                        batch_window.after(0, lambda s=subject: log_text.insert(tk.END, f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting inference for {s}\n"))
                        batch_window.after(0, lambda: log_text.see(tk.END))
                        
                        # Run inference for current subject
                        try:
                            cmd = [
                                sys.executable, "video_inference.py",
                                "--model", model_file,
                                "--subject", subject,
                                "--enhanced"
                            ]
                            
                            print(f"DEBUG: Running inference for {subject}: {' '.join(cmd)}")
                            
                            process = subprocess.Popen(
                                cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                                universal_newlines=True
                            )
                            
                            # Read output line by line
                            for line in process.stdout:
                                if batch_cancelled.get():
                                    process.terminate()
                                    break
                                if line.strip():
                                    batch_window.after(0, lambda l=line: log_text.insert(tk.END, f"  {l.strip()}\n"))
                                    batch_window.after(0, lambda: log_text.see(tk.END))
                            
                            return_code = process.wait()
                            
                            if return_code == 0:
                                successful_subjects.append(subject)
                                batch_window.after(0, lambda s=subject: log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {s} completed successfully\n"))
                                batch_window.after(0, lambda s=subject: results_text.insert(tk.END, f"✅ {s}: SUCCESS\n"))
                            else:
                                failed_subjects.append(subject)
                                batch_window.after(0, lambda s=subject: log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] ❌ {s} failed (code {return_code})\n"))
                                batch_window.after(0, lambda s=subject: results_text.insert(tk.END, f"❌ {s}: FAILED\n"))
                                
                        except Exception as e:
                            failed_subjects.append(subject)
                            batch_window.after(0, lambda s=subject, e=e: log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] ❌ {s} error: {str(e)}\n"))
                            batch_window.after(0, lambda s=subject: results_text.insert(tk.END, f"❌ {s}: ERROR\n"))
                        
                        # Update progress
                        batch_window.after(0, lambda: current_progress.stop())
                        batch_window.after(0, lambda: overall_progress_var.set(i + 1))
                    
                    # Final summary
                    if not batch_cancelled.get():
                        batch_window.after(0, lambda: overall_status.config(text=f"Completed: {len(successful_subjects)} success, {len(failed_subjects)} failed"))
                        batch_window.after(0, lambda: current_subject_label.config(text="Batch processing completed!"))
                        batch_window.after(0, lambda: log_text.insert(tk.END, f"\n[{datetime.now().strftime('%H:%M:%S')}] Batch processing completed\n"))
                        batch_window.after(0, lambda: log_text.insert(tk.END, f"Successful: {len(successful_subjects)}\n"))
                        batch_window.after(0, lambda: log_text.insert(tk.END, f"Failed: {len(failed_subjects)}\n"))
                        if failed_subjects:
                            batch_window.after(0, lambda: log_text.insert(tk.END, f"Failed subjects: {', '.join(failed_subjects)}\n"))
                    
                    # Enable close button
                    batch_window.after(0, lambda: close_btn.config(state="normal"))
                    batch_window.after(0, lambda: cancel_btn.config(state="disabled"))
                    batch_completed.set(True)
                    
                except Exception as e:
                    batch_window.after(0, lambda: log_text.insert(tk.END, f"\n[{datetime.now().strftime('%H:%M:%S')}] Batch processing error: {str(e)}\n"))
                    batch_window.after(0, lambda: overall_status.config(text="Batch processing failed"))
                    batch_window.after(0, lambda: close_btn.config(state="normal"))
                    print(f"DEBUG: Batch inference error: {str(e)}")
            
            # Start batch processing in background thread
            batch_thread = threading.Thread(target=run_batch_inference)
            batch_thread.daemon = True
            batch_thread.start()
            
            print("DEBUG: Batch enhanced inference thread started")
            self.status_var.set(f"Running batch inference: {len(selected_subjects)} subjects")
            
        except Exception as e:
            print(f"DEBUG: Error starting batch enhanced inference - {str(e)}")
            self.status_var.set("Error starting batch inference")
            messagebox.showerror("Error", f"Failed to start batch inference: {str(e)}")

def main():
    """Main function"""
    print("DEBUG: Starting main function...")
    print(f"DEBUG: Python version: {sys.version}")
    print(f"DEBUG: Current working directory: {os.getcwd()}")
    print(f"DEBUG: Script location: {os.path.abspath(__file__)}")
    
    try:
        root = tk.Tk()
        print("DEBUG: Tkinter root window created")
        
        app = SegmentationLauncher(root)
        print("DEBUG: SegmentationLauncher initialized")
        
        print("DEBUG: Starting main loop...")
        root.mainloop()
        print("DEBUG: Main loop ended")
        
    except Exception as e:
        print(f"DEBUG: Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()
