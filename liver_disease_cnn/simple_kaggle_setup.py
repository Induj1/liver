#!/usr/bin/env python3
"""
Simple Kaggle Setup Guide
Easy steps to download real liver disease datasets
"""

import os
from pathlib import Path

def main():
    print("🏥 Real Liver Disease Dataset Setup")
    print("=" * 50)
    
    print("\n🔑 STEP 1: Get Kaggle API Token")
    print("1. Go to: https://www.kaggle.com/account")
    print("2. Scroll to 'API' section") 
    print("3. Click 'Create New Token'")
    print("4. Download kaggle.json file")
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_file = kaggle_dir / "kaggle.json"
    
    print(f"5. Place kaggle.json in: {kaggle_dir}")
    
    # Create the directory if it doesn't exist
    kaggle_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Kaggle directory created: {kaggle_dir}")
    
    if kaggle_file.exists():
        print("✅ kaggle.json found!")
        show_datasets()
    else:
        print("❌ kaggle.json not found")
        print(f"\n💡 After downloading kaggle.json:")
        print(f"   - Place it in: {kaggle_dir}")
        print(f"   - Run this script again")
        
        # Ask if they want to enter credentials manually
        manual = input("\n❓ Enter Kaggle credentials manually now? (y/n): ").lower()
        if manual == 'y':
            setup_manual_kaggle(kaggle_dir)

def setup_manual_kaggle(kaggle_dir):
    """Setup Kaggle credentials manually"""
    print("\n🔑 Manual Kaggle Setup:")
    print("(Get these from: https://www.kaggle.com/account -> API section)")
    
    username = input("Kaggle Username: ").strip()
    api_key = input("Kaggle API Key: ").strip()
    
    if username and api_key:
        import json
        
        kaggle_config = {
            "username": username,
            "key": api_key
        }
        
        kaggle_file = kaggle_dir / "kaggle.json"
        with open(kaggle_file, 'w') as f:
            json.dump(kaggle_config, f, indent=2)
        
        print("✅ Kaggle credentials saved!")
        show_datasets()
    else:
        print("❌ Invalid credentials")

def show_datasets():
    """Show available datasets and download commands"""
    print("\n🏥 STEP 2: Choose a Dataset")
    print("=" * 40)
    
    datasets = [
        {
            "name": "Liver Tumor Segmentation (BEST for Deep Learning)",
            "command": "kaggle datasets download -d andrewmvd/liver-tumor-segmentation -p downloads --unzip",
            "size": "~2GB",
            "description": "CT images with liver tumors"
        },
        {
            "name": "CT Medical Images",
            "command": "kaggle datasets download -d mohamedhanyyy/ct-medical-images -p downloads --unzip", 
            "size": "~1.5GB",
            "description": "Various CT scan images"
        },
        {
            "name": "Indian Liver Patient Records",
            "command": "kaggle datasets download -d uciml/indian-liver-patient-records -p downloads --unzip",
            "size": "~1MB", 
            "description": "Clinical data (not images)"
        }
    ]
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n{i}. {dataset['name']}")
        print(f"   Size: {dataset['size']}")
        print(f"   Description: {dataset['description']}")
        print(f"   Command: {dataset['command']}")
    
    print("\n🚀 STEP 3: Download Dataset")
    print("Copy and run one of the commands above in your terminal")
    
    print("\n🎯 STEP 4: Organize Data")
    print("After download, run: python setup_dataset.py")
    
    print("\n💡 RECOMMENDED: Download dataset #1 (Liver Tumor Segmentation)")
    print("It's the best for training image classification models!")

if __name__ == "__main__":
    main()
