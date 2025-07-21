#!/usr/bin/env python3
"""
Alternative Dataset Downloader
Downloads real medical datasets using various methods
"""

import requests
import zipfile
import os
from pathlib import Path
import shutil

def download_medical_dataset():
    print("🏥 Downloading Real Medical Liver Dataset...")
    
    # Create downloads directory
    downloads_dir = Path("downloads")
    downloads_dir.mkdir(exist_ok=True)
    
    # Try to download from public sources
    datasets = [
        {
            "name": "Sample Liver CT Images",
            "url": "https://github.com/sfikas/medical-imaging-datasets/archive/refs/heads/master.zip",
            "filename": "medical_images.zip"
        },
        {
            "name": "Medical Image Sample",
            "url": "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/medical.zip",
            "filename": "medical_sample.zip"
        }
    ]
    
    for dataset in datasets:
        try:
            print(f"📥 Trying to download: {dataset['name']}")
            
            response = requests.get(dataset['url'], timeout=30)
            if response.status_code == 200:
                file_path = downloads_dir / dataset['filename']
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"✅ Downloaded: {file_path}")
                
                # Try to extract
                if file_path.suffix == '.zip':
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(downloads_dir)
                    print(f"✅ Extracted: {file_path}")
                
                return True
                
        except Exception as e:
            print(f"⚠️  Failed to download {dataset['name']}: {e}")
            continue
    
    print("❌ Could not download from public sources")
    return False

def create_better_synthetic_dataset():
    """Create an improved synthetic dataset while waiting for real data"""
    print("🎨 Creating improved synthetic dataset...")
    
    # Remove old synthetic images first
    print("🧹 Cleaning old synthetic images...")
    for split in ['train', 'val', 'test']:
        for class_name in ['normal', 'cirrhosis', 'liver_cancer', 'fatty_liver', 'hepatitis']:
            class_dir = Path('data') / split / class_name
            if class_dir.exists():
                for img_file in class_dir.glob('medical_*.jpg'):
                    img_file.unlink()
    
    # Use the realistic liver generator
    print("🏥 Generating new realistic medical images...")
    
    try:
        from realistic_liver_generator import create_realistic_liver_dataset
        create_realistic_liver_dataset()
        return True
    except Exception as e:
        print(f"❌ Error creating synthetic dataset: {e}")
        return False

def show_manual_options():
    """Show manual download options"""
    print("\n🌐 Manual Download Options:")
    print("=" * 50)
    
    print("\n1. 🏥 Medical Imaging Repositories:")
    print("   • Cancer Imaging Archive: https://www.cancerimagingarchive.net/")
    print("   • Medical Segmentation Decathlon: http://medicaldecathlon.com/")
    print("   • LiTS Challenge: https://competitions.codalab.org/competitions/17094")
    
    print("\n2. 📊 Kaggle Datasets (if API issues):")
    print("   • Manual download from: https://www.kaggle.com/datasets")
    print("   • Search for: 'liver tumor', 'liver segmentation', 'medical imaging'")
    
    print("\n3. 🧠 Research Datasets:")
    print("   • CHAOS Challenge: https://chaos.grand-challenge.org/")
    print("   • NIH Clinical Center: https://nihcc.app.box.com/")
    
    print(f"\n💡 For now, using synthetic but realistic medical images")

def main():
    print("🏥 Alternative Medical Dataset Setup")
    print("=" * 50)
    
    # Try to download real data
    if not download_medical_dataset():
        print("\n🎨 Falling back to improved synthetic dataset...")
        if create_better_synthetic_dataset():
            print("\n✅ Synthetic medical dataset ready!")
        else:
            print("❌ Could not create dataset")
            return
    
    # Show manual options
    show_manual_options()
    
    print(f"\n🚀 Your dataset is ready!")
    print("📊 Current options:")
    print("   1. Train with current synthetic data: jupyter lab notebooks/liver_disease_classification.ipynb")
    print("   2. Replace with real data when available")
    print("   3. Use web interface: streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    main()
