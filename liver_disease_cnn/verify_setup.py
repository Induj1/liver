#!/usr/bin/env python3
"""
Verify Setup - Check if everything is ready for liver disease classification
"""

import os
import sys
from pathlib import Path

def check_python_packages():
    """Check if required Python packages are installed"""
    print("🐍 Checking Python packages...")
    
    required_packages = [
        'tensorflow', 'numpy', 'pandas', 'matplotlib', 
        'seaborn', 'PIL', 'cv2', 'sklearn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)
    
    return missing
            
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    # Test other critical imports
    packages_to_test = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib.pyplot'),
        ('cv2', 'opencv-python'),
        ('PIL', 'Pillow'),
        ('sklearn', 'scikit-learn'),
        ('seaborn', 'seaborn'),
        ('gradio', 'gradio'),
        ('streamlit', 'streamlit')
    ]
    
    all_good = True
    for import_name, package_name in packages_to_test:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - not installed")
            all_good = False
    
    return all_good

def test_directories():
    """Test if directory structure exists"""
    print("\n📁 Testing directory structure...")
    
    from pathlib import Path
    
    required_dirs = [
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
        "notebooks",
        "app",
        "gradcam"
    ]
    
    all_dirs_exist = True
    for directory in required_dirs:
        path = Path(directory)
        if path.exists():
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ - missing")
            all_dirs_exist = False
    
    return all_dirs_exist

def test_notebook():
    """Test if notebook exists and is readable"""
    print("\n📓 Testing notebook...")
    
    from pathlib import Path
    
    notebook_path = Path("notebooks/liver_disease_classification.ipynb")
    if notebook_path.exists():
        print("✅ Main notebook exists")
        return True
    else:
        print("❌ Main notebook missing")
        return False

def main():
    """Run all tests"""
    print("🏥 Liver Disease Classification System - Verification")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test directories
    dirs_ok = test_directories()
    
    # Test notebook
    notebook_ok = test_notebook()
    
    # Overall status
    print("\n" + "=" * 60)
    if imports_ok and dirs_ok and notebook_ok:
        print("🎉 SUCCESS! Your system is ready to use!")
        print("\n📋 Next Steps:")
        print("1. 📊 Add liver images to data/ directories")
        print("2. 🚀 Open notebook: jupyter lab notebooks/liver_disease_classification.ipynb")
        print("3. ▶️ Run all cells to train your model")
        print("4. 🌐 Launch interface: python app/gradio_app.py")
        
    else:
        print("❌ ISSUES DETECTED - Please fix the errors above")
        
        if not imports_ok:
            print("\n💡 To fix import issues:")
            print("   pip install -r requirements.txt")
        
        if not dirs_ok:
            print("\n💡 To fix directory issues:")
            print("   Re-run: python setup.py")
            
        if not notebook_ok:
            print("\n💡 Notebook missing - check project files")
    
    print("\n⚠️ Remember: This is for research/education only!")
    print("Always consult medical professionals for diagnosis.")

if __name__ == "__main__":
    main()
