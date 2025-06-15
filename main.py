"""
Main Entry Point for Carotid Segmentation Project
File utama untuk menjalankan seluruh pipeline
"""

import os
import sys
import argparse
from datetime import datetime

def print_banner():
    """Print banner aplikasi"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║        🫀 CAROTID ARTERY SEGMENTATION SYSTEM 🫀              ║
    ║                                                               ║
    ║        Segmentasi Otomatis Arteri Karotis                    ║
    ║        Menggunakan Deep Learning (U-Net)                     ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"    📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("    " + "="*60)

def run_training():
    """Jalankan training model"""
    print("\n🎯 STARTING MODEL TRAINING...")
    print("-" * 40)
    
    try:
        import training_model
        training_model.main()
        print("✅ Training completed successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed (pip install -r requirements.txt)")
        return False
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def run_inference():
    """Jalankan video inference"""
    print("\n🎬 STARTING VIDEO INFERENCE...")
    print("-" * 40)
    
    try:
        import video_inference
        video_inference.main()
        print("✅ Video inference completed successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed (pip install -r requirements.txt)")
        return False
    except Exception as e:
        print(f"❌ Video inference failed: {e}")
        return False

def run_data_sync():
    """Jalankan data synchronization"""
    print("\n🔄 STARTING DATA SYNCHRONIZATION...")
    print("-" * 40)
    
    try:
        import data_sync
        data_sync.main()
        print("✅ Data synchronization completed successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Data synchronization failed: {e}")
        return False

def run_data_viewer():
    """Jalankan data viewer"""
    print("\n📊 OPENING DATA VIEWER...")
    print("-" * 40)
    
    try:
        import data_viewer
        data_viewer.main()
        print("✅ Data viewer opened successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Data viewer failed: {e}")
        return False

def run_launcher():
    """Jalankan GUI launcher"""
    print("\n🚀 OPENING LAUNCHER...")
    print("-" * 40)
    
    try:
        import launcher_with_inference_log
        launcher_with_inference_log.main()
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Launcher failed: {e}")
        return False

def run_advanced_analytics():
    """Jalankan advanced analytics"""
    print("\n📊 OPENING ADVANCED ANALYTICS...")
    print("-" * 40)
    
    try:
        import advanced_analytics
        advanced_analytics.main()
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Advanced analytics failed: {e}")
        return False

def run_full_pipeline():
    """Jalankan pipeline lengkap"""
    print("\n🏭 RUNNING FULL PIPELINE...")
    print("=" * 50)
    
    steps = [
        ("Training Model", run_training),
        ("Video Inference", run_inference),
        ("Data Synchronization", run_data_sync)
    ]
    
    results = []
    for step_name, step_func in steps:
        print(f"\n🔄 Step: {step_name}")
        success = step_func()
        results.append((step_name, success))
        
        if not success:
            print(f"❌ Pipeline stopped at: {step_name}")
            break
        else:
            print(f"✅ {step_name} completed")
    
    # Summary
    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    
    for step_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{step_name:<25} : {status}")
    
    all_success = all(success for _, success in results)
    if all_success:
        print("\n🎉 Full pipeline completed successfully!")
        print("📊 You can now run the data viewer to see results")
    else:
        print("\n⚠️  Pipeline completed with some failures")
    
    return all_success

def check_setup():
    """Check sistem setup"""
    print("\n🔍 CHECKING SYSTEM SETUP...")
    print("-" * 40)
    
    # Check Python version
    python_version = sys.version
    print(f"🐍 Python Version: {python_version.split()[0]}")
    
    # Check if files exist
    required_files = [
        "training_model.py",
        "video_inference.py", 
        "data_viewer.py",
        "data_sync.py",
        "launcher_with_inference_log.py",
        "advanced_analytics.py",
        "requirements.txt",
        "config.py"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    # Try to import config
    try:
        import config
        print("✅ Configuration loaded")
        
        # Check requirements
        success, missing = config.check_requirements()
        if not success:
            print("⚠️  Missing requirements:")
            for item in missing:
                print(f"   - {item}")
        
        return success
    except ImportError:
        print("❌ Configuration could not be loaded")
        return False

def show_menu():
    """Tampilkan menu interaktif"""
    while True:
        print("\n" + "="*60)
        print("🫀 CAROTID SEGMENTATION - MAIN MENU")
        print("="*60)
        print("1. 📊 Advanced Analytics")
        print("2. 🔍 Check Setup")
        print("3. 🎯 Train Model")
        print("4. 🎬 Run Video Inference")
        print("5. 🔄 Synchronize Data")
        print("6. 📊 Open Data Viewer")
        print("7. 🚀 Open GUI Launcher")
        print("8. 🏭 Run Full Pipeline")
        print("9. ❌ Exit")
        print("-"*60)
        
        try:
            choice = input("Select option (1-9): ").strip()
            
            if choice == '1':
                run_advanced_analytics()
            elif choice == '2':
                check_setup()
            elif choice == '3':
                run_training()
            elif choice == '4':
                run_inference()
            elif choice == '5':
                run_data_sync()
            elif choice == '6':
                run_data_viewer()
            elif choice == '7':
                run_launcher()
            elif choice == '8':
                run_full_pipeline()
            elif choice == '9':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-9.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Carotid Artery Segmentation System")
    parser.add_argument('--mode', choices=['train', 'inference', 'sync', 'viewer', 'launcher', 'analytics', 'pipeline', 'menu'],
                       default='menu', help='Mode to run')
    parser.add_argument('--gui', action='store_true', help='Use GUI launcher')
    parser.add_argument('--no-banner', action='store_true', help='Skip banner')
    
    args = parser.parse_args()
    
    if not args.no_banner:
        print_banner()
    
    if args.gui or args.mode == 'launcher':
        return run_launcher()
    elif args.mode == 'train':
        return run_training()
    elif args.mode == 'inference':
        return run_inference()
    elif args.mode == 'sync':
        return run_data_sync()
    elif args.mode == 'viewer':
        return run_data_viewer()
    elif args.mode == 'analytics':
        return run_advanced_analytics()
    elif args.mode == 'pipeline':
        return run_full_pipeline()
    elif args.mode == 'menu':
        show_menu()
        return True
    else:
        print(f"❌ Unknown mode: {args.mode}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
