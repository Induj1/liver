#!/usr/bin/env python3
"""
Kaggle Liver Disease Dataset Downloader
Downloads real medical datasets from Kaggle for liver disease classification
"""

import os
import zipfile
import shutil
import pandas as pd
from pathlib import Path
import subprocess
import json

class KaggleDatasetDownloader:
    def __init__(self):
        self.data_dir = Path("data")
        self.downloads_dir = Path("kaggle_downloads")
        self.downloads_dir.mkdir(exist_ok=True)
        
        # Popular liver disease datasets on Kaggle
        self.available_datasets = {
            "1": {
                "name": "Liver Patient Dataset",
                "dataset": "uciml/indian-liver-patient-records",
                "description": "Clinical data for liver patient classification",
                "type": "tabular",
                "size": "Small (~1MB)"
            },
            "2": {
                "name": "Liver Tumor Segmentation", 
                "dataset": "andrewmvd/liver-tumor-segmentation",
                "description": "CT images with liver tumors",
                "type": "medical_images",
                "size": "Large (~2GB)"
            },
            "3": {
                "name": "Medical Image Datasets",
                "dataset": "kmader/medical-image-datasets",
                "description": "Various medical imaging datasets",
                "type": "medical_images", 
                "size": "Medium (~500MB)"
            },
            "4": {
                "name": "CT Medical Images",
                "dataset": "mohamedhanyyy/ct-medical-images",
                "description": "CT scan images for classification",
                "type": "medical_images",
                "size": "Large (~1.5GB)"
            },
            "5": {
                "name": "Liver Disease Prediction",
                "dataset": "jeevannagaraj/liver-disease-prediction",
                "description": "Clinical liver disease data",
                "type": "tabular",
                "size": "Small (~500KB)"
            }
        }
    
    def setup_kaggle_api(self):
        """Setup Kaggle API credentials"""
        print("🔑 Setting up Kaggle API...")
        
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_file = kaggle_dir / "kaggle.json"
        
        if not kaggle_file.exists():
            print("\n❌ Kaggle API credentials not found!")
            print("\n📋 To download Kaggle datasets, please:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Scroll to 'API' section")
            print("3. Click 'Create New Token'")
            print("4. Download kaggle.json file")
            print(f"5. Place it in: {kaggle_dir}")
            print("6. Run this script again")
            
            # Create the directory
            kaggle_dir.mkdir(exist_ok=True)
            
            # Ask if user wants to enter credentials manually
            manual = input("\n❓ Enter Kaggle credentials manually? (y/n): ").lower()
            if manual == 'y':
                return self.setup_manual_credentials(kaggle_dir)
            else:
                return False
        
        # Set proper permissions on Windows/Unix
        try:
            if os.name != 'nt':  # Unix/Linux
                os.chmod(kaggle_file, 0o600)
        except:
            pass
        
        print("✅ Kaggle API credentials found!")
        return True
    
    def setup_manual_credentials(self, kaggle_dir):
        """Setup Kaggle credentials manually"""
        print("\n🔑 Manual Kaggle Setup:")
        username = input("Kaggle Username: ").strip()
        api_key = input("Kaggle API Key: ").strip()
        
        if username and api_key:
            kaggle_config = {
                "username": username,
                "key": api_key
            }
            
            kaggle_file = kaggle_dir / "kaggle.json"
            with open(kaggle_file, 'w') as f:
                json.dump(kaggle_config, f)
            
            print("✅ Kaggle credentials saved!")
            return True
        else:
            print("❌ Invalid credentials")
            return False
    
    def install_kaggle(self):
        """Install Kaggle package if not available"""
        try:
            import kaggle
            return True
        except ImportError:
            print("📦 Installing Kaggle package...")
            subprocess.run([
                "pip", "install", "kaggle"
            ], check=True)
            return True
    
    def list_datasets(self):
        """List available datasets"""
        print("\n🏥 Available Liver Disease Datasets from Kaggle:")
        print("=" * 60)
        
        for key, dataset in self.available_datasets.items():
            print(f"\n{key}. {dataset['name']}")
            print(f"   Dataset: {dataset['dataset']}")
            print(f"   Description: {dataset['description']}")
            print(f"   Type: {dataset['type']}")
            print(f"   Size: {dataset['size']}")
    
    def download_dataset(self, choice):
        """Download selected dataset"""
        if choice not in self.available_datasets:
            print("❌ Invalid choice")
            return False
        
        dataset_info = self.available_datasets[choice]
        dataset_name = dataset_info["dataset"]
        
        print(f"\n📥 Downloading: {dataset_info['name']}")
        print(f"Dataset: {dataset_name}")
        
        try:
            # Download using Kaggle API
            download_path = self.downloads_dir / choice
            download_path.mkdir(exist_ok=True)
            
            subprocess.run([
                "kaggle", "datasets", "download", 
                "-d", dataset_name,
                "-p", str(download_path),
                "--unzip"
            ], check=True)
            
            print(f"✅ Downloaded to: {download_path}")
            return self.organize_downloaded_data(choice, download_path)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Download failed: {e}")
            print("\n💡 Common issues:")
            print("- Dataset might be private or require competition acceptance")
            print("- Check internet connection")
            print("- Verify Kaggle credentials")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def organize_downloaded_data(self, choice, download_path):
        """Organize downloaded data into proper structure"""
        dataset_info = self.available_datasets[choice]
        
        if dataset_info["type"] == "medical_images":
            return self.organize_medical_images(download_path)
        else:
            return self.organize_tabular_data(download_path)
    
    def organize_medical_images(self, download_path):
        """Organize medical image data"""
        print("🏥 Organizing medical images...")
        
        # Look for image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.dcm', '.nii', '.nii.gz']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(download_path.rglob(f"*{ext}")))
        
        if not image_files:
            print("⚠️  No image files found in downloaded data")
            return False
        
        print(f"📊 Found {len(image_files)} image files")
        
        # Try to organize by existing folder structure
        organized = self.organize_by_folders(image_files)
        
        if not organized:
            # If no clear organization, try by filename patterns
            organized = self.organize_by_filenames(image_files)
        
        if organized:
            print("✅ Images organized successfully!")
            self.show_dataset_summary()
            return True
        else:
            print("⚠️  Could not automatically organize images")
            print("💡 Please manually organize images by diagnosis")
            return False
    
    def organize_by_folders(self, image_files):
        """Organize images based on existing folder structure"""
        folder_mapping = {
            'normal': ['normal', 'healthy', 'control', 'neg'],
            'cirrhosis': ['cirrhosis', 'cirrho', 'fibrosis'],
            'liver_cancer': ['cancer', 'tumor', 'malignant', 'hcc', 'tumor'],
            'fatty_liver': ['fatty', 'steatosis', 'nafld'],
            'hepatitis': ['hepatitis', 'inflammation', 'acute']
        }
        
        organized_count = 0
        
        for img_file in image_files:
            # Check parent folder names
            folder_parts = [p.lower() for p in img_file.parts]
            
            target_class = None
            for class_name, keywords in folder_mapping.items():
                for keyword in keywords:
                    if any(keyword in part for part in folder_parts):
                        target_class = class_name
                        break
                if target_class:
                    break
            
            if target_class:
                # Copy to appropriate directory
                target_dir = self.data_dir / "train" / target_class
                target_dir.mkdir(parents=True, exist_ok=True)
                
                new_name = f"kaggle_{target_class}_{organized_count:03d}{img_file.suffix}"
                target_path = target_dir / new_name
                
                shutil.copy2(img_file, target_path)
                organized_count += 1
        
        return organized_count > 0
    
    def organize_by_filenames(self, image_files):
        """Organize images based on filename patterns"""
        filename_mapping = {
            'normal': ['normal', 'healthy', 'control', 'neg'],
            'cirrhosis': ['cirrhosis', 'cirrho', 'fibrosis'],
            'liver_cancer': ['cancer', 'tumor', 'malignant', 'hcc'],
            'fatty_liver': ['fatty', 'steatosis', 'nafld'],
            'hepatitis': ['hepatitis', 'inflammation', 'acute']
        }
        
        organized_count = 0
        
        for img_file in image_files:
            filename = img_file.name.lower()
            
            target_class = None
            for class_name, keywords in filename_mapping.items():
                if any(keyword in filename for keyword in keywords):
                    target_class = class_name
                    break
            
            if not target_class:
                # Default to normal if no pattern matches
                target_class = 'normal'
            
            # Copy to appropriate directory
            target_dir = self.data_dir / "train" / target_class
            target_dir.mkdir(parents=True, exist_ok=True)
            
            new_name = f"kaggle_{target_class}_{organized_count:03d}{img_file.suffix}"
            target_path = target_dir / new_name
            
            shutil.copy2(img_file, target_path)
            organized_count += 1
        
        return organized_count > 0
    
    def organize_tabular_data(self, download_path):
        """Handle tabular data (convert to note for user)"""
        print("📊 Found tabular data (not images)")
        print("💡 This dataset contains clinical data, not images")
        print("📋 You might want to choose an image dataset instead")
        
        # List CSV files found
        csv_files = list(download_path.rglob("*.csv"))
        if csv_files:
            print(f"\n📄 CSV files found:")
            for csv_file in csv_files:
                print(f"   - {csv_file.name}")
        
        return False
    
    def show_dataset_summary(self):
        """Show summary of organized dataset"""
        print("\n📊 DATASET SUMMARY")
        print("=" * 40)
        
        total_images = 0
        for split in ["train", "val", "test"]:
            split_dir = self.data_dir / split
            if split_dir.exists():
                split_total = 0
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        count = len([f for f in class_dir.iterdir() 
                                   if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.dcm']])
                        if count > 0:
                            print(f"{class_dir.name:15}: {count:4} images")
                            split_total += count
                total_images += split_total
        
        print(f"\nTotal images: {total_images}")

def main():
    downloader = KaggleDatasetDownloader()
    
    print("🏥 Kaggle Liver Disease Dataset Downloader")
    print("=" * 50)
    
    # Check/setup Kaggle
    if not downloader.install_kaggle():
        print("❌ Failed to install Kaggle package")
        return
    
    if not downloader.setup_kaggle_api():
        print("❌ Kaggle API setup failed")
        return
    
    # Show available datasets
    downloader.list_datasets()
    
    print(f"\n🎯 Choose a dataset to download:")
    choice = input("Enter choice (1-5): ").strip()
    
    if choice in downloader.available_datasets:
        success = downloader.download_dataset(choice)
        if success:
            print("\n🎉 Dataset downloaded and organized!")
            print("\n🚀 Next step: jupyter lab notebooks/liver_disease_classification.ipynb")
        else:
            print("\n⚠️  Download completed but organization may need manual work")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
