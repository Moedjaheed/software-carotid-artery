# Carotid Artery Segmentation Analysis Software

## Project Overview

A comprehensive software solution for carotid artery segmentation analysis using deep learning models. This application provides advanced tools for medical image processing, data analysis, and visualization of carotid artery diameter measurements with integrated pressure analysis.

## Key Features

### 1. Tabbed Launcher Interface
- Modern browser-like interface with 5 main tabs
- Compact design optimized for workflow efficiency
- Real-time logging and progress monitoring
- Quick access to all major functions

### 2. Enhanced Video Inference Engine
- Support for multiple UNet model selection
- Batch processing capabilities for multiple subjects
- Real-time progress tracking with detailed logs
- Automatic result organization and export

### 3. Advanced Data Viewer
- Interactive CSV data visualization
- Correlation analysis between diameter and pressure measurements
- Multiple plot generation (time series, correlation plots)
- Data export functionality with customizable formats

### 4. Batch Processing System
- Simultaneous processing of multiple video subjects
- Progress monitoring with real-time updates
- Error handling and recovery mechanisms
- Automatic result consolidation

### 5. Analytics Dashboard
- Statistical analysis of segmentation results
- Comparative analysis across multiple subjects
- Data synchronization tools
- Advanced plotting capabilities

## System Requirements

### Software Dependencies
- Python 3.8 or higher
- OpenCV 4.5+
- PyTorch 1.8+
- NumPy, Pandas, Matplotlib
- Tkinter (included with Python)

### Hardware Requirements
- GPU with CUDA support (recommended)
- Minimum 8GB RAM
- 10GB free disk space for models and results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Moedjaheed/software-carotid-artery.git
cd software-carotid-artery
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Setup Data Directories**: 
   - Create your data structure according to the format below
   - Place video files and CSV data in appropriate directories
   - Models and large data files are excluded from repository

4. **Add Model Files**:
   - Download or place your .pth model files in the root directory
   - Supported models: UNet_22Mei_Sore.pth, UNet_25Mei_Sore.pth

5. Verify installation:
```bash
python launcher_with_inference_log.py
```

## Usage Guide

### Quick Start
1. **Using Batch File**: Double-click `run_launcher.bat`
2. **Manual Start**: Run `python launcher_with_inference_log.py`
3. **Direct Data Viewer**: Run `python data_viewer.py`

### Main Application Workflow

#### 1. Video Inference Processing
- Select model: UNet_22Mei_Sore.pth or UNet_25Mei_Sore.pth
- Choose subjects for batch processing using checkboxes
- Monitor progress in real-time log window
- Review batch summary upon completion

#### 2. Data Analysis
- Load processed CSV files through Data Viewer
- Analyze diameter measurements over time
- Generate correlation plots with pressure data
- Export results in multiple formats

#### 3. Advanced Analytics
- Access statistical analysis tools
- Compare results across multiple subjects
- Generate comprehensive reports
- Synchronize data across different measurements

## File Structure

### Core Application Files
```
├── main.py                              # Primary analysis script
├── launcher_with_inference_log.py       # Tabbed launcher interface
├── data_viewer.py                       # Data visualization interface
├── video_inference.py                   # Video processing engine
├── run_launcher.bat                     # Quick launcher batch file
```

### Processing Modules
```
├── batch_processor.py                   # Batch processing utilities
├── training_model.py                    # Model training utilities
├── advanced_analytics.py               # Advanced analysis tools
├── data_sync.py                        # Data synchronization tools
├── config.py                           # Configuration management
```

### Data Directories (Not included in repository)
**Note: Data files are excluded from the repository. You need to create the following structure:**

```
├── data_uji/                           # Test data (Subjek1-7) - USER PROVIDED
│   ├── Subjek1/
│   │   ├── subject1.csv
│   │   ├── Subjek1.mp4
│   │   ├── timestamps.csv
│   │   └── pictures/
│   └── ...
├── inference_results/                  # Processing results - GENERATED
├── models/                            # Additional model storage - USER PROVIDED
└── results/                           # Analysis outputs - GENERATED
    ├── csv_data/
    ├── plots/
    ├── synced_data/
    └── videos/
```

### Model Files (Not included in repository)
**Note: Model files are excluded due to large size. Download separately:**

```
├── UNet_22Mei_Sore.pth                # Primary UNet model
└── UNet_25Mei_Sore.pth                # Alternative UNet model
```

## Recent Updates and Improvements

### Version 2.0 - Major Interface Redesign
- **New Tabbed Interface**: Modern browser-like design with 5 main tabs
- **Enhanced Batch Processing**: Multiple subject selection with real-time progress
- **Model Selection**: Support for multiple UNet models via radio buttons
- **Improved User Experience**: Compact 800x600 window with intuitive navigation

### Performance Optimizations
- **Faster Processing**: Optimized inference pipeline for batch operations
- **Memory Management**: Improved memory usage for large video files
- **Error Handling**: Comprehensive error recovery and logging system
- **Progress Tracking**: Real-time progress updates with detailed status

### Data Analysis Enhancements
- **Advanced Visualization**: Enhanced plotting capabilities with correlation analysis
- **Export Options**: Multiple format support for data export
- **Statistical Tools**: Comprehensive statistical analysis features
- **Data Validation**: Automated data integrity checks

### Code Quality Improvements
- **Clean Architecture**: Modular design with clear separation of concerns
- **Documentation**: Comprehensive inline documentation and user guides
- **Testing**: Validated functionality across all major features
- **Version Control**: Proper Git integration with comprehensive .gitignore

## Configuration

### Model Configuration
Models are automatically detected from the main directory. Place additional .pth files in the root directory for automatic recognition.

### Data Input Format
- Video files: MP4 format recommended
- CSV files: Standard comma-separated format
- Timestamp files: CSV format with time column

### Output Configuration
Results are automatically organized in the following structure:
- Segmented videos: `inference_results/{subject}/`
- Diameter data: CSV format with timestamp correlation
- Plots: PNG format in `results/plots/`

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or use CPU processing
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Model Loading Errors**: Verify .pth files are not corrupted
4. **GUI Issues**: Ensure tkinter is properly installed

### Performance Tips
- Use GPU acceleration when available
- Process smaller batches for limited memory systems
- Close other applications during intensive processing
- Ensure sufficient disk space for output files

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make changes with proper documentation
4. Test thoroughly before submitting
5. Submit pull request with detailed description

### Code Standards
- Follow PEP 8 Python style guidelines
- Include comprehensive docstrings
- Add appropriate error handling
- Write unit tests for new features

## License

This project is developed for academic and research purposes. Please cite appropriately when using in research publications.

## Support

For issues, questions, or contributions, please contact the development team or submit an issue through the GitHub repository.

## Acknowledgments

This software was developed as part of advanced medical image processing research, with special thanks to the medical imaging community for their valuable insights and feedback.
