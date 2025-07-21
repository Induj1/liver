#!/usr/bin/env python3
"""
Simple Interactive Dataset Setup
Quick and easy way to organize your liver disease images
"""

import os
import shutil
from pathlib import Path

def main():
    print("🏥 Liver Disease Dataset Setup")
    print("=" * 50)
    
    # Get source directory
    while True:
        source_dir = input("\n📁 Enter the path to your image folder: ").strip().strip('"')
        if os.path.exists(source_dir):
            break
        print("❌ Directory not found. Please check the path.")
    
    source_path = Path(source_dir)
    
    # Check what we have
    print(f"\n🔍 Scanning {source_path}...")
    
    # Look for subdirectories (organized by class)
    subdirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    # Count image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm'}
    all_files = [f for f in source_path.rglob("*") 
                if f.suffix.lower() in image_extensions]
    
    print(f"📊 Found {len(all_files)} images")
    
    if subdirs:
        print(f"📁 Found {len(subdirs)} subdirectories:")
        for d in subdirs[:10]:  # Show first 10
            count = len([f for f in d.rglob("*") if f.suffix.lower() in image_extensions])
            print(f"   - {d.name}: {count} images")
        
        organize_mode = input("\n❓ Do your images already organized in folders by diagnosis? (y/n): ").lower()
    else:
        organize_mode = 'n'
        print("📄 All images appear to be in a single folder")
    
    if organize_mode == 'y':
        print("\n🎯 Great! I'll organize from your existing folders.")
        print("\n📋 Please map your folder names to our classes:")
        print("   Our classes: normal, cirrhosis, liver_cancer, fatty_liver, hepatitis")
        
        class_mapping = {}
        for subdir in subdirs:
            if any(f.suffix.lower() in image_extensions for f in subdir.rglob("*")):
                suggested_class = suggest_class(subdir.name)
                user_input = input(f"   '{subdir.name}' → {suggested_class} (or type different class): ").strip()
                if user_input:
                    class_mapping[subdir.name] = user_input
                else:
                    class_mapping[subdir.name] = suggested_class
        
        organize_from_folders(source_path, class_mapping)
    
    else:
        print("\n🎯 I'll help you organize mixed images by filename patterns.")
        print("\n💡 I'll look for these keywords in filenames:")
        print("   - normal, healthy → normal")
        print("   - cirrhosis, cirrho → cirrhosis") 
        print("   - cancer, tumor → liver_cancer")
        print("   - fatty, steat → fatty_liver")
        print("   - hepatitis, inflam → hepatitis")
        
        proceed = input("\n❓ Does this look good? (y/n): ").lower()
        if proceed == 'y':
            organize_from_patterns(source_path)
        else:
            print("💡 Please manually organize your images into folders first, then run this script again.")
            return
    
    print("\n🎉 Dataset organization complete!")
    print("📊 Run this to see your dataset summary:")
    print("   python organize_dataset.py --source dummy --mode folders")

def suggest_class(folder_name):
    """Suggest a class based on folder name"""
    name = folder_name.lower()
    
    if any(word in name for word in ['normal', 'healthy', 'control']):
        return 'normal'
    elif any(word in name for word in ['cirrhosis', 'cirrho', 'fibros']):
        return 'cirrhosis'
    elif any(word in name for word in ['cancer', 'tumor', 'malign', 'hcc']):
        return 'liver_cancer'
    elif any(word in name for word in ['fatty', 'steat', 'nafld']):
        return 'fatty_liver'
    elif any(word in name for word in ['hepatitis', 'inflam']):
        return 'hepatitis'
    else:
        return 'normal'  # default

def organize_from_folders(source_path, class_mapping):
    """Organize images from existing class folders"""
    import random
    
    classes = ["normal", "cirrhosis", "liver_cancer", "fatty_liver", "hepatitis"]
    
    # Create directories
    for split in ["train", "val", "test"]:
        for class_name in classes:
            dir_path = Path("data") / split / class_name
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Process each mapped folder
    for folder_name, target_class in class_mapping.items():
        source_folder = source_path / folder_name
        
        # Find images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm'}
        images = [f for f in source_folder.rglob("*") if f.suffix.lower() in image_extensions]
        
        if not images:
            continue
            
        random.shuffle(images)
        
        # Split 70/15/15
        total = len(images)
        train_end = int(0.7 * total)
        val_end = int(0.85 * total)
        
        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end], 
            "test": images[val_end:]
        }
        
        # Copy files
        for split, image_list in splits.items():
            target_dir = Path("data") / split / target_class
            for i, image_path in enumerate(image_list):
                new_name = f"{target_class}_{i:03d}{image_path.suffix}"
                target_path = target_dir / new_name
                shutil.copy2(image_path, target_path)
        
        print(f"✅ {target_class}: {total} images (train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])})")

def organize_from_patterns(source_path):
    """Organize images using filename patterns"""
    import random
    
    patterns = {
        "normal": ["normal", "healthy", "control"],
        "cirrhosis": ["cirrhosis", "cirrho", "fibros"],
        "liver_cancer": ["cancer", "tumor", "malign", "hcc"],
        "fatty_liver": ["fatty", "steat", "nafld"],
        "hepatitis": ["hepatitis", "inflam"]
    }
    
    # Create directories
    for split in ["train", "val", "test"]:
        for class_name in patterns.keys():
            dir_path = Path("data") / split / class_name
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Find and classify images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm'}
    all_images = [f for f in source_path.rglob("*") if f.suffix.lower() in image_extensions]
    
    classified = {class_name: [] for class_name in patterns.keys()}
    unclassified = []
    
    for image_path in all_images:
        filename = image_path.name.lower()
        matched = False
        
        for class_name, keywords in patterns.items():
            if any(keyword in filename for keyword in keywords):
                classified[class_name].append(image_path)
                matched = True
                break
        
        if not matched:
            unclassified.append(image_path)
    
    # Process each class
    for class_name, images in classified.items():
        if not images:
            continue
            
        random.shuffle(images)
        
        total = len(images)
        train_end = int(0.7 * total)
        val_end = int(0.85 * total)
        
        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }
        
        for split, image_list in splits.items():
            target_dir = Path("data") / split / class_name
            for i, image_path in enumerate(image_list):
                new_name = f"{class_name}_{i:03d}{image_path.suffix}"
                target_path = target_dir / new_name
                shutil.copy2(image_path, target_path)
        
        print(f"✅ {class_name}: {total} images")
    
    if unclassified:
        print(f"⚠️  {len(unclassified)} images couldn't be classified by filename")

if __name__ == "__main__":
    main()
