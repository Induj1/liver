#!/usr/bin/env python3
"""
Quick Setup Script for Liver Disease Classification System

This script helps you quickly set up the environment and verify everything is working.
Run this after installing the requirements.txt file.
"""

import os
import sys
from pathlib import Path
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", f"{version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            print(f"✅ {package_name} - installed")
            return True
        else:
            print(f"❌ {package_name} - not found")
            return False
    except ImportError:
        print(f"❌ {package_name} - import error")
        return False

def check_tensorflow():
    """Special check for TensorFlow with GPU support"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow - version {tf.__version__}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 GPU support available: {len(gpus)} GPU(s) detected")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("⚠️ No GPU detected - training will use CPU (slower)")
        
        return True
    except ImportError:
        print("❌ TensorFlow - not installed")
        return False

def create_directory_structure():
    """Create the required directory structure"""
    print("\n📁 Creating directory structure...")
    
    # Base directories
    directories = [
        "data/train/normal",
        "data/train/cirrhosis", 
        "data/train/liver_cancer",
        "data/train/fatty_liver",
        "data/train/hepatitis",
        "data/val/normal",
        "data/val/cirrhosis",
        "data/val/liver_cancer", 
        "data/val/fatty_liver",
        "data/val/hepatitis",
        "data/test/normal",
        "data/test/cirrhosis",
        "data/test/liver_cancer",
        "data/test/fatty_liver", 
        "data/test/hepatitis",
        "models",
        "results",
        "logs"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
    print("✅ Directory structure created successfully!")
    
    # Create .gitkeep files for empty directories
    for directory in directories:
        gitkeep_path = Path(directory) / ".gitkeep"
        if not gitkeep_path.exists():
            gitkeep_path.touch()

def create_demo_images():
    """Create sample demo images for testing"""
    print("\n🖼️ Creating demo images for testing...")
    
    try:
        import numpy as np
        from PIL import Image
        
        classes = ["normal", "cirrhosis", "liver_cancer", "fatty_liver", "hepatitis"]
        splits = ["train", "val", "test"]
        
        for split in splits:
            num_images = {"train": 5, "val": 2, "test": 2}[split]
            
            for class_name in classes:
                class_dir = Path(f"data/{split}/{class_name}")
                
                for i in range(num_images):
                    # Create a simple synthetic image (224x224x3)
                    # Different patterns for different classes
                    if class_name == "normal":
                        img_array = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
                    elif class_name == "cirrhosis":
                        img_array = np.random.randint(80, 150, (224, 224, 3), dtype=np.uint8)
                    elif class_name == "liver_cancer":
                        img_array = np.random.randint(60, 120, (224, 224, 3), dtype=np.uint8)
                    elif class_name == "fatty_liver":
                        img_array = np.random.randint(120, 220, (224, 224, 3), dtype=np.uint8)
                    else:  # hepatitis
                        img_array = np.random.randint(90, 160, (224, 224, 3), dtype=np.uint8)
                    
                    # Add some pattern to make it more realistic
                    center_x, center_y = 112, 112
                    y, x = np.ogrid[:224, :224]
                    mask = (x - center_x)**2 + (y - center_y)**2 <= 80**2
                    img_array[mask] = img_array[mask] * 0.8
                    
                    img = Image.fromarray(img_array.astype(np.uint8))
                    img.save(class_dir / f"demo_{class_name}_{i:03d}.jpg")
        
        print("✅ Demo images created successfully!")
        print("   📝 Note: These are synthetic images for testing only")
        print("   📝 Replace with real medical images for actual training")
        
    except ImportError as e:
        print(f"❌ Could not create demo images: {e}")
        print("   Install PIL/Pillow and numpy to create demo images")

def check_jupyter():
    """Check if Jupyter is available"""
    try:
        result = subprocess.run(['jupyter', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Jupyter - available")
            return True
        else:
            print("❌ Jupyter - not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Jupyter - not found")
        return False

def main():
    """Main setup function"""
    print("🚀 Liver Disease Classification System - Quick Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    print("\n📦 Checking required packages...")
    
    # Critical packages
    required_packages = [
        ("tensorflow", "tensorflow"),
        ("numpy", "numpy"), 
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scikit-learn", "sklearn"),
        ("opencv-python", "cv2"),
        ("Pillow", "PIL"),
        ("gradio", "gradio"),
        ("streamlit", "streamlit")
    ]
    
    missing_packages = []
    
    # Special check for TensorFlow
    if not check_tensorflow():
        missing_packages.append("tensorflow")
    
    # Check other packages
    for package_name, import_name in required_packages[1:]:  # Skip tensorflow, already checked
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    # Check Jupyter
    check_jupyter()
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("\n💡 To install missing packages:")
        print("   pip install -r requirements.txt")
        print("   or")
        print(f"   pip install {' '.join(missing_packages)}")
    else:
        print("\n✅ All required packages are installed!")
    
    # Create directory structure
    create_directory_structure()
    
    # Ask user if they want demo images
    print("\n🤔 Would you like to create demo images for testing?")
    print("   (These are synthetic images - replace with real data for training)")
    
    try:
        choice = input("Create demo images? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            create_demo_images()
    except KeyboardInterrupt:
        print("\n⏹️ Setup interrupted by user")
        sys.exit(0)
    
    # Final instructions
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("1. 📊 Add your medical images to data/ directories")
    print("2. 🚀 Open Jupyter notebook:")
    print("   jupyter lab notebooks/liver_disease_classification.ipynb")
    print("3. ▶️ Run all cells to train your model")
    print("4. 🌐 Launch web interface:")
    print("   python app/gradio_app.py")
    print("   streamlit run app/streamlit_app.py")
    
    print("\n⚠️ IMPORTANT:")
    print("   This system is for research/educational use only!")
    print("   Always consult medical professionals for diagnosis.")
    
    print("\n📚 Documentation:")
    print("   • GETTING_STARTED.md - Detailed setup guide")
    print("   • README.md - Project overview")
    print("   • requirements.txt - Package dependencies")
    
    print("\n🔗 Useful Links:")
    print("   • TensorFlow: https://tensorflow.org")
    print("   • Medical Datasets: See GETTING_STARTED.md")
    print("   • Gradio Docs: https://gradio.app")
    
    print("\n✨ Happy AI development! 🏥🤖")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Setup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        print("Please check the error message and try again.")
        sys.exit(1)
