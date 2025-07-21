#!/usr/bin/env python3
"""
Liver Disease Dataset Downloader
Downloads publicly available liver disease datasets for classification
"""

import os
import requests
import zipfile
import tarfile
import urllib.request
from pathlib import Path
import shutil
from typing import Dict, List
import json
import random

class DatasetDownloader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.downloads_dir = Path("downloads")
        self.downloads_dir.mkdir(exist_ok=True)
        
        # Available datasets
        self.datasets = {
            "liver_tumor_segmentation": {
                "name": "Liver Tumor Segmentation Dataset",
                "description": "CT images with liver tumors for classification",
                "url": "https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation",
                "type": "kaggle",
                "size": "~2GB",
                "classes": ["normal", "tumor"],
                "format": "nii.gz"
            },
            "liver_classification": {
                "name": "Liver Disease Classification Dataset",
                "description": "Mixed liver condition images",
                "url": "https://www.kaggle.com/datasets/mpwolke/cusersmarilyndesktopliver-disorders",
                "type": "kaggle", 
                "size": "~500MB",
                "classes": ["normal", "cirrhosis", "hepatitis"],
                "format": "jpg"
            },
            "medical_liver_dataset": {
                "name": "Medical Liver Dataset",
                "description": "Comprehensive liver disease dataset",
                "url": "https://drive.google.com/file/d/1ABC123XYZ/view?usp=sharing",
                "type": "gdrive",
                "size": "~1GB", 
                "classes": ["normal", "cirrhosis", "liver_cancer", "fatty_liver", "hepatitis"],
                "format": "jpg"
            },
            "demo_dataset": {
                "name": "Demo Liver Dataset (Small)",
                "description": "Small demo dataset for testing",
                "url": "synthetic",
                "type": "synthetic",
                "size": "~50MB",
                "classes": ["normal", "cirrhosis", "liver_cancer", "fatty_liver", "hepatitis"],
                "format": "jpg"
            }
        }
    
    def list_available_datasets(self):
        """List all available datasets"""
        print("🏥 Available Liver Disease Datasets:")
        print("=" * 60)
        
        for key, dataset in self.datasets.items():
            print(f"\n📊 {dataset['name']}")
            print(f"   Description: {dataset['description']}")
            print(f"   Size: {dataset['size']}")
            print(f"   Classes: {', '.join(dataset['classes'])}")
            print(f"   Format: {dataset['format']}")
            print(f"   Type: {dataset['type']}")
    
    def download_kaggle_dataset(self, dataset_name: str):
        """Download dataset from Kaggle"""
        try:
            import kaggle
            print(f"📥 Downloading {dataset_name} from Kaggle...")
            
            # This requires Kaggle API setup
            print("⚠️  Note: Kaggle datasets require API key setup")
            print("   1. Create account at kaggle.com")
            print("   2. Go to Account -> API -> Create New Token")
            print("   3. Place kaggle.json in ~/.kaggle/ or C:\\Users\\{username}\\.kaggle\\")
            
            return False
            
        except ImportError:
            print("❌ Kaggle package not installed. Installing...")
            os.system("pip install kaggle")
            return False
    
    def download_gdrive_dataset(self, file_id: str, output_path: str):
        """Download dataset from Google Drive"""
        try:
            print(f"📥 Downloading from Google Drive...")
            # Try to import and use gdown
            try:
                import gdown
                gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
            except ImportError:
                print("❌ gdown not available. Please install: pip install gdown")
                return False
            return True
        except Exception as e:
            print(f"❌ Error downloading from Google Drive: {e}")
            return False
    
    def create_synthetic_dataset(self):
        """Create a synthetic dataset for testing"""
        print("🎨 Creating synthetic demo dataset...")
        
        try:
            import numpy as np
            from PIL import Image, ImageDraw, ImageFilter
        except ImportError:
            print("❌ Required packages not available. Installing...")
            os.system("pip install pillow numpy")
            try:
                import numpy as np
                from PIL import Image, ImageDraw, ImageFilter
            except ImportError:
                return False
        
        # Create synthetic liver images
        classes = ["normal", "cirrhosis", "liver_cancer", "fatty_liver", "hepatitis"]
        images_per_class = 20
        
        for split in ["train", "val", "test"]:
            split_images = {
                "train": images_per_class,
                "val": images_per_class // 4,
                "test": images_per_class // 4
            }
            
            for class_name in classes:
                class_dir = self.data_dir / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Remove existing demo files
                for existing_file in class_dir.glob("demo_*.jpg"):
                    existing_file.unlink()
                
                for i in range(split_images[split]):
                    # Create synthetic liver image
                    img = self.generate_liver_image(class_name, i)
                    img_path = class_dir / f"synthetic_{class_name}_{i:03d}.jpg"
                    img.save(img_path, "JPEG", quality=85)
        
        print("✅ Synthetic dataset created successfully!")
        return True
    
    def generate_liver_image(self, condition: str, seed: int):
        """Generate a synthetic liver image based on condition"""
        random.seed(seed)
        np.random.seed(seed)
        
        # Base image parameters
        width, height = 512, 512
        
        # Create base liver shape
        img = Image.new('RGB', (width, height), color=(20, 20, 30))
        draw = ImageDraw.Draw(img)
        
        # Draw liver-like shape
        liver_color = self.get_liver_color(condition)
        
        # Main liver outline
        liver_points = [
            (100, 200), (200, 150), (350, 180), (400, 250),
            (380, 350), (300, 400), (200, 380), (120, 320)
        ]
        draw.polygon(liver_points, fill=liver_color, outline=(100, 100, 100))
        
        # Add condition-specific features
        self.add_condition_features(draw, condition, width, height)
        
        # Add some noise and texture
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Convert to numpy for noise
        img_array = np.array(img)
        noise = np.random.normal(0, 10, img_array.shape).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def get_liver_color(self, condition: str):
        """Get base color for liver based on condition"""
        colors = {
            "normal": (120, 80, 60),      # Healthy brown
            "cirrhosis": (90, 70, 50),    # Darker, scarred
            "liver_cancer": (80, 60, 80), # Grayish with purple tint
            "fatty_liver": (140, 100, 70), # Yellowish
            "hepatitis": (100, 60, 40)    # Reddish inflammation
        }
        return colors.get(condition, (100, 80, 60))
    
    def add_condition_features(self, draw, condition: str, width: int, height: int):
        """Add visual features specific to each condition"""
        if condition == "cirrhosis":
            # Add scarring patterns
            for _ in range(20):
                x = random.randint(150, 350)
                y = random.randint(200, 350)
                draw.ellipse([x-3, y-3, x+3, y+3], fill=(60, 40, 30))
        
        elif condition == "liver_cancer":
            # Add tumor-like masses
            for _ in range(5):
                x = random.randint(150, 350)
                y = random.randint(200, 350)
                r = random.randint(10, 25)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=(120, 80, 120))
        
        elif condition == "fatty_liver":
            # Add fatty deposits (lighter spots)
            for _ in range(30):
                x = random.randint(150, 350)
                y = random.randint(200, 350)
                r = random.randint(3, 8)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=(160, 120, 80))
        
        elif condition == "hepatitis":
            # Add inflammation indicators (reddish areas)
            for _ in range(15):
                x = random.randint(150, 350)
                y = random.randint(200, 350)
                r = random.randint(5, 15)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=(140, 70, 50))
    
    def download_public_dataset(self):
        """Download a publicly available liver dataset"""
        print("🔍 Searching for public liver disease datasets...")
        
        # Try to download from various public sources
        public_urls = [
            {
                "name": "Sample Liver Images",
                "url": "https://github.com/sfikas/medical-imaging-datasets/raw/master/liver-lesion-segmentation.zip",
                "filename": "liver_sample.zip"
            }
        ]
        
        for dataset in public_urls:
            try:
                print(f"📥 Trying to download {dataset['name']}...")
                
                response = requests.get(dataset['url'], timeout=30)
                if response.status_code == 200:
                    file_path = self.downloads_dir / dataset['filename']
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"✅ Downloaded {dataset['filename']}")
                    return self.extract_dataset(file_path)
                
            except Exception as e:
                print(f"⚠️  Could not download {dataset['name']}: {e}")
                continue
        
        return False
    
    def extract_dataset(self, file_path: Path):
        """Extract downloaded dataset"""
        try:
            if file_path.suffix == '.zip':
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(self.downloads_dir)
            elif file_path.suffix in ['.tar', '.gz']:
                with tarfile.open(file_path, 'r:*') as tar_ref:
                    tar_ref.extractall(self.downloads_dir)
            
            print(f"✅ Extracted {file_path.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error extracting {file_path}: {e}")
            return False
    
    def show_dataset_info(self):
        """Show information about current dataset"""
        print("\n📊 Current Dataset Summary:")
        print("=" * 50)
        
        total_images = 0
        for split in ["train", "val", "test"]:
            split_dir = self.data_dir / split
            if split_dir.exists():
                print(f"\n{split.upper()}:")
                split_total = 0
                
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        count = len([f for f in class_dir.iterdir() 
                                   if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                        print(f"  {class_dir.name:15}: {count:4} images")
                        split_total += count
                
                print(f"  {'TOTAL':15}: {split_total:4} images")
                total_images += split_total
        
        print(f"\n📊 GRAND TOTAL: {total_images} images")
        
        if total_images == 0:
            print("\n💡 No images found. Try downloading a dataset!")

def main():
    downloader = DatasetDownloader()
    
    print("🏥 Liver Disease Dataset Downloader")
    print("=" * 50)
    
    # Show available options
    downloader.list_available_datasets()
    
    print(f"\n🎯 Choose a dataset option:")
    print("1. Create synthetic demo dataset (recommended for testing)")
    print("2. Try downloading public datasets")
    print("3. Show current dataset info")
    print("4. Manual setup instructions")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        success = downloader.create_synthetic_dataset()
        if success:
            downloader.show_dataset_info()
            print("\n🎉 Demo dataset ready! You can now:")
            print("   - Run the Jupyter notebook to train a model")
            print("   - Replace with real medical images when available")
    
    elif choice == "2":
        success = downloader.download_public_dataset()
        if not success:
            print("⚠️  Public download failed. Creating synthetic dataset instead...")
            downloader.create_synthetic_dataset()
        downloader.show_dataset_info()
    
    elif choice == "3":
        downloader.show_dataset_info()
    
    elif choice == "4":
        print("\n📋 Manual Dataset Setup:")
        print("=" * 40)
        print("1. Download liver disease images from:")
        print("   - Medical imaging repositories")
        print("   - Kaggle competitions")
        print("   - Hospital collaborations (with proper permissions)")
        print("   - Research datasets")
        print("\n2. Organize images into folders:")
        print("   data/train/normal/")
        print("   data/train/cirrhosis/")
        print("   data/train/liver_cancer/")
        print("   data/train/fatty_liver/")
        print("   data/train/hepatitis/")
        print("\n3. Run: python setup_dataset.py")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
