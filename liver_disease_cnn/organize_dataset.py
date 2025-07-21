#!/usr/bin/env python3
"""
Dataset Organization Script for Liver Disease Classification
This script helps organize your medical images into the proper directory structure
"""

import os
import shutil
import random
from pathlib import Path
from typing import Dict, List
import argparse

class DatasetOrganizer:
    def __init__(self, source_dir: str, target_dir: str = "data"):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.classes = ["normal", "cirrhosis", "liver_cancer", "fatty_liver", "hepatitis"]
        
        # Create directory structure
        self.create_directories()
    
    def create_directories(self):
        """Create the required directory structure"""
        for split in ["train", "val", "test"]:
            for class_name in self.classes:
                dir_path = self.target_dir / split / class_name
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created: {dir_path}")
    
    def organize_from_folders(self, class_mapping: Dict[str, str] = None):
        """
        Organize images from source folders
        
        Args:
            class_mapping: Dictionary mapping source folder names to class names
                          e.g., {"healthy": "normal", "cancer": "liver_cancer"}
        """
        if class_mapping is None:
            class_mapping = {
                "normal": "normal",
                "healthy": "normal", 
                "cirrhosis": "cirrhosis",
                "cancer": "liver_cancer",
                "liver_cancer": "liver_cancer",
                "fatty": "fatty_liver",
                "fatty_liver": "fatty_liver",
                "hepatitis": "hepatitis"
            }
        
        for source_folder, target_class in class_mapping.items():
            source_path = self.source_dir / source_folder
            if source_path.exists():
                self.process_class_folder(source_path, target_class)
            else:
                print(f"⚠️  Warning: Source folder '{source_path}' not found")
    
    def process_class_folder(self, source_path: Path, class_name: str):
        """Process all images in a class folder"""
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm']
        
        # Find all images
        images = []
        for ext in extensions:
            images.extend(list(source_path.glob(f"*{ext}")))
            images.extend(list(source_path.glob(f"*{ext.upper()}")))
        
        if not images:
            print(f"⚠️  No images found in {source_path}")
            return
        
        # Shuffle images for random distribution
        random.shuffle(images)
        
        # Split ratios: 70% train, 15% val, 15% test
        total = len(images)
        train_end = int(0.7 * total)
        val_end = int(0.85 * total)
        
        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }
        
        # Copy images to appropriate folders
        for split, image_list in splits.items():
            target_dir = self.target_dir / split / class_name
            for i, image_path in enumerate(image_list):
                # Create new filename with consistent naming
                new_name = f"{class_name}_{i:03d}{image_path.suffix}"
                target_path = target_dir / new_name
                
                shutil.copy2(image_path, target_path)
                
            print(f"✅ Copied {len(image_list)} {class_name} images to {split}/")
        
        print(f"📊 {class_name}: {total} total images "
              f"(Train: {len(splits['train'])}, "
              f"Val: {len(splits['val'])}, "
              f"Test: {len(splits['test'])})")
    
    def organize_from_mixed_folder(self, naming_patterns: Dict[str, List[str]]):
        """
        Organize images from a single folder using filename patterns
        
        Args:
            naming_patterns: Dictionary with class names and their filename patterns
                           e.g., {"normal": ["normal", "healthy"], 
                                  "cirrhosis": ["cirrhosis", "cirrho"]}
        """
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm']
        
        # Find all images
        all_images = []
        for ext in extensions:
            all_images.extend(list(self.source_dir.glob(f"*{ext}")))
            all_images.extend(list(self.source_dir.glob(f"*{ext.upper()}")))
        
        # Classify images based on filename patterns
        classified_images = {class_name: [] for class_name in self.classes}
        unclassified = []
        
        for image_path in all_images:
            filename = image_path.name.lower()
            classified = False
            
            for class_name, patterns in naming_patterns.items():
                if any(pattern.lower() in filename for pattern in patterns):
                    classified_images[class_name].append(image_path)
                    classified = True
                    break
            
            if not classified:
                unclassified.append(image_path)
        
        # Process each class
        for class_name, images in classified_images.items():
            if images:
                self.process_image_list(images, class_name)
        
        if unclassified:
            print(f"⚠️  {len(unclassified)} unclassified images found:")
            for img in unclassified[:5]:  # Show first 5
                print(f"   - {img.name}")
            if len(unclassified) > 5:
                print(f"   ... and {len(unclassified) - 5} more")
    
    def process_image_list(self, images: List[Path], class_name: str):
        """Process a list of images for a specific class"""
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
            target_dir = self.target_dir / split / class_name
            for i, image_path in enumerate(image_list):
                new_name = f"{class_name}_{i:03d}{image_path.suffix}"
                target_path = target_dir / new_name
                shutil.copy2(image_path, target_path)
        
        print(f"📊 {class_name}: {total} total images "
              f"(Train: {len(splits['train'])}, "
              f"Val: {len(splits['val'])}, "
              f"Test: {len(splits['test'])})")
    
    def show_dataset_summary(self):
        """Display summary of organized dataset"""
        print("\n" + "="*60)
        print("📊 DATASET SUMMARY")
        print("="*60)
        
        total_images = 0
        for split in ["train", "val", "test"]:
            print(f"\n{split.upper()}:")
            split_total = 0
            for class_name in self.classes:
                class_dir = self.target_dir / split / class_name
                count = len([f for f in class_dir.iterdir() 
                           if f.is_file() and not f.name.startswith('.')])
                print(f"  {class_name:15}: {count:4} images")
                split_total += count
            print(f"  {'TOTAL':15}: {split_total:4} images")
            total_images += split_total
        
        print(f"\n📊 GRAND TOTAL: {total_images} images")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Organize liver disease dataset")
    parser.add_argument("--source", "-s", required=True, 
                       help="Source directory containing your images")
    parser.add_argument("--mode", "-m", choices=["folders", "mixed"], default="folders",
                       help="Organization mode: 'folders' or 'mixed'")
    
    args = parser.parse_args()
    
    # Initialize organizer
    organizer = DatasetOrganizer(args.source)
    
    if args.mode == "folders":
        print("📁 Organizing from separate class folders...")
        print("Expected folder structure: source_dir/normal/, source_dir/cirrhosis/, etc.")
        
        # Customize this mapping based on your folder names
        class_mapping = {
            "normal": "normal",
            "healthy": "normal",
            "cirrhosis": "cirrhosis", 
            "cancer": "liver_cancer",
            "liver_cancer": "liver_cancer",
            "fatty": "fatty_liver",
            "fatty_liver": "fatty_liver", 
            "hepatitis": "hepatitis"
        }
        
        organizer.organize_from_folders(class_mapping)
        
    elif args.mode == "mixed":
        print("📄 Organizing from mixed folder using filename patterns...")
        
        # Customize these patterns based on your filenames
        naming_patterns = {
            "normal": ["normal", "healthy", "control"],
            "cirrhosis": ["cirrhosis", "cirrho", "fibros"],
            "liver_cancer": ["cancer", "tumor", "malign", "hcc"],
            "fatty_liver": ["fatty", "steat", "nafld"],
            "hepatitis": ["hepatitis", "inflam"]
        }
        
        organizer.organize_from_mixed_folder(naming_patterns)
    
    # Show summary
    organizer.show_dataset_summary()
    
    print("\n🎉 Dataset organization complete!")
    print("💡 Next step: Run the Jupyter notebook to start training!")

if __name__ == "__main__":
    main()
