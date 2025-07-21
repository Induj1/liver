#!/usr/bin/env python3
"""
Simple Dataset Creator for Liver Disease Classification
Creates a demo dataset for testing the liver disease classification system
"""

import os
import random
from pathlib import Path

def create_demo_dataset():
    """Create a demo dataset with synthetic images"""
    print("🎨 Creating demo liver disease dataset...")
    
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFilter
    except ImportError:
        print("📦 Installing required packages...")
        os.system("pip install pillow numpy")
        import numpy as np
        from PIL import Image, ImageDraw, ImageFilter
    
    # Dataset configuration
    classes = ["normal", "cirrhosis", "liver_cancer", "fatty_liver", "hepatitis"]
    
    # Number of images per split
    split_counts = {
        "train": 25,  # 25 images per class for training
        "val": 8,     # 8 images per class for validation
        "test": 8     # 8 images per class for testing
    }
    
    data_dir = Path("data")
    total_created = 0
    
    # Create directory structure and images
    for split, count in split_counts.items():
        print(f"\n📁 Creating {split} set ({count} images per class)...")
        
        for class_name in classes:
            class_dir = data_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Remove existing demo files
            for existing_file in class_dir.glob("demo_*.jpg"):
                existing_file.unlink()
            for existing_file in class_dir.glob("synthetic_*.jpg"):
                existing_file.unlink()
            
            # Create images for this class
            for i in range(count):
                img = create_liver_image(class_name, i)
                img_path = class_dir / f"demo_{class_name}_{i:03d}.jpg"
                img.save(img_path, "JPEG", quality=85)
                total_created += 1
    
    print(f"\n✅ Created {total_created} demo images!")
    show_dataset_summary()
    return True

def create_liver_image(condition, seed):
    """Create a synthetic liver image"""
    random.seed(seed + hash(condition))
    
    # Image parameters
    width, height = 256, 256
    
    # Create base image
    img = Image.new('RGB', (width, height), color=(30, 30, 40))
    draw = ImageDraw.Draw(img)
    
    # Get condition-specific colors
    base_color = get_liver_color(condition)
    
    # Draw liver shape
    liver_outline = [
        (50, 100), (100, 80), (180, 90), (200, 130),
        (190, 180), (150, 200), (100, 190), (60, 160)
    ]
    
    draw.polygon(liver_outline, fill=base_color, outline=(100, 100, 100))
    
    # Add condition-specific features
    add_condition_features(draw, condition, width, height)
    
    # Add some medical imaging texture
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    
    return img

def get_liver_color(condition):
    """Get base color for each liver condition"""
    colors = {
        "normal": (80, 60, 45),       # Healthy liver brown
        "cirrhosis": (65, 50, 35),    # Darker, scarred
        "liver_cancer": (70, 55, 65), # Grayish with abnormal tint
        "fatty_liver": (95, 75, 50),  # Yellowish fatty deposits
        "hepatitis": (85, 50, 35)     # Reddish inflammation
    }
    return colors.get(condition, (75, 60, 45))

def add_condition_features(draw, condition, width, height):
    """Add visual markers for each condition"""
    if condition == "cirrhosis":
        # Add scarring/fibrosis patterns
        for _ in range(15):
            x = random.randint(80, 180)
            y = random.randint(100, 180)
            draw.ellipse([x-2, y-2, x+2, y+2], fill=(45, 35, 25))
    
    elif condition == "liver_cancer":
        # Add tumor masses
        for _ in range(3):
            x = random.randint(80, 180)
            y = random.randint(100, 180)
            r = random.randint(8, 15)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(90, 70, 85))
    
    elif condition == "fatty_liver":
        # Add fatty deposits (brighter spots)
        for _ in range(20):
            x = random.randint(80, 180)
            y = random.randint(100, 180)
            r = random.randint(2, 6)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(110, 90, 60))
    
    elif condition == "hepatitis":
        # Add inflammation patterns
        for _ in range(12):
            x = random.randint(80, 180)
            y = random.randint(100, 180)
            r = random.randint(3, 8)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(100, 60, 40))

def show_dataset_summary():
    """Display summary of created dataset"""
    print("\n" + "="*50)
    print("📊 DATASET SUMMARY")
    print("="*50)
    
    data_dir = Path("data")
    total_images = 0
    
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            print(f"\n{split.upper()}:")
            split_total = 0
            
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    count = len([f for f in class_dir.iterdir() 
                               if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                    print(f"  {class_dir.name:15}: {count:4} images")
                    split_total += count
            
            print(f"  {'TOTAL':15}: {split_total:4} images")
            total_images += split_total
    
    print(f"\n📊 GRAND TOTAL: {total_images} images")
    print("="*50)

def download_real_datasets():
    """Information about downloading real medical datasets"""
    print("\n🏥 REAL MEDICAL DATASETS")
    print("="*50)
    
    datasets = [
        {
            "name": "LiTS (Liver Tumor Segmentation)",
            "url": "https://competitions.codalab.org/competitions/17094",
            "description": "CT scans with liver tumors",
            "access": "Registration required"
        },
        {
            "name": "Medical Segmentation Decathlon",
            "url": "http://medicaldecathlon.com/",
            "description": "Multi-organ segmentation including liver",
            "access": "Free download"
        },
        {
            "name": "CHAOS Challenge",
            "url": "https://chaos.grand-challenge.org/",
            "description": "Liver segmentation from CT and MR",
            "access": "Registration required"
        },
        {
            "name": "Kaggle Liver Datasets",
            "url": "https://www.kaggle.com/search?q=liver+disease",
            "description": "Various liver-related datasets",
            "access": "Kaggle account required"
        }
    ]
    
    print("📋 Recommended real datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"\n{i}. {dataset['name']}")
        print(f"   URL: {dataset['url']}")
        print(f"   Description: {dataset['description']}")
        print(f"   Access: {dataset['access']}")
    
    print(f"\n💡 To use real datasets:")
    print("1. Download from one of the sources above")
    print("2. Extract and organize images by diagnosis")
    print("3. Run: python setup_dataset.py")

def main():
    print("🏥 Liver Disease Dataset Setup")
    print("="*50)
    
    print("\nChoose an option:")
    print("1. Create demo dataset (synthetic images for testing)")
    print("2. Show real dataset sources")
    print("3. Show current dataset info")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        success = create_demo_dataset()
        if success:
            print("\n🎉 Demo dataset created successfully!")
            print("\n🚀 Next steps:")
            print("1. Open the Jupyter notebook: jupyter lab notebooks/liver_disease_classification.ipynb")
            print("2. Run all cells to train the model")
            print("3. Replace demo images with real medical data when available")
    
    elif choice == "2":
        download_real_datasets()
    
    elif choice == "3":
        show_dataset_summary()
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
