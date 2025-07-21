#!/usr/bin/env python3
"""
Automated Liver Disease Dataset Downloader
Downloads comprehensive datasets for ALL liver conditions
"""

import subprocess
import os
from pathlib import Path

def download_all_liver_diseases():
    """Download datasets for all liver disease types"""
    print("🏥 COMPREHENSIVE LIVER DISEASE DATASET DOWNLOADER")
    print("=" * 60)
    print("Downloading datasets for: Normal, Cirrhosis, Fatty Liver, Hepatitis, Liver Cancer")
    
    # Organize datasets by condition
    liver_datasets = {
        "cirrhosis": [
            "rishianand/liver-cirrhosis-stage-dataset",
            "fedesoriano/cirrhosis-prediction-dataset"
        ],
        "fatty_liver": [
            "kmader/liver-disease-datasets",
            "humansintheloop/medical-images-liver"
        ],
        "hepatitis": [
            "rahul121/hepatitis-data",
            "uciml/hepatitis", 
            "fedesoriano/hepatitis-c-dataset"
        ],
        "liver_cancer": [
            "andrewmvd/liver-tumor-segmentation",
            "xhlulu/ct-images-with-liver-tumors",
            "nazmulislam/liver-tumor-segmentation-part-1"
        ],
        "general": [
            "joebeachcapital/liver-disease",
            "uciml/indian-liver-patient-records"
        ]
    }
    
    downloads_dir = Path("kaggle_downloads")
    downloads_dir.mkdir(exist_ok=True)
    
    total_downloaded = 0
    successful_downloads = []
    failed_downloads = []
    
    # Download each category
    for condition, datasets in liver_datasets.items():
        print(f"\n🔍 Downloading {condition.upper()} datasets...")
        print("-" * 40)
        
        condition_dir = downloads_dir / condition
        condition_dir.mkdir(exist_ok=True)
        
        for dataset in datasets:
            try:
                print(f"📥 Downloading: {dataset}")
                
                # Use kaggle CLI to download
                result = subprocess.run([
                    "kaggle", "datasets", "download", "-d", dataset, 
                    "--path", str(condition_dir), "--unzip"
                ], capture_output=True, text=True, cwd=os.getcwd())
                
                if result.returncode == 0:
                    print(f"✅ Success: {dataset}")
                    successful_downloads.append(f"{condition}: {dataset}")
                    total_downloaded += 1
                else:
                    print(f"❌ Failed: {dataset}")
                    print(f"   Error: {result.stderr.strip()}")
                    failed_downloads.append(f"{condition}: {dataset} - {result.stderr.strip()}")
                    
            except Exception as e:
                print(f"❌ Error downloading {dataset}: {e}")
                failed_downloads.append(f"{condition}: {dataset} - {str(e)}")
    
    # Summary
    print(f"\n📊 DOWNLOAD SUMMARY")
    print("=" * 40)
    print(f"✅ Successfully downloaded: {total_downloaded} datasets")
    print(f"❌ Failed downloads: {len(failed_downloads)}")
    
    if successful_downloads:
        print(f"\n🎉 SUCCESSFUL DOWNLOADS:")
        for download in successful_downloads:
            print(f"  • {download}")
    
    if failed_downloads:
        print(f"\n⚠️  FAILED DOWNLOADS:")
        for failure in failed_downloads:
            print(f"  • {failure}")
        
        print(f"\n💡 Common solutions for failed downloads:")
        print("1. Check Kaggle API setup: kaggle auth status")
        print("2. Accept dataset terms on Kaggle website")
        print("3. Check dataset availability")
    
    # Check what we got
    check_downloaded_data()
    
    return total_downloaded > 0

def check_downloaded_data():
    """Check what data was actually downloaded"""
    print(f"\n📁 CHECKING DOWNLOADED DATA")
    print("=" * 35)
    
    downloads_dir = Path("kaggle_downloads")
    if not downloads_dir.exists():
        print("❌ No downloads directory found")
        return
    
    total_files = 0
    
    for condition_dir in downloads_dir.iterdir():
        if condition_dir.is_dir():
            print(f"\n{condition_dir.name.upper()}:")
            
            # Count different file types
            csv_files = list(condition_dir.rglob("*.csv"))
            image_files = list(condition_dir.rglob("*.jpg")) + list(condition_dir.rglob("*.png")) + list(condition_dir.rglob("*.dcm"))
            nifti_files = list(condition_dir.rglob("*.nii*"))
            
            if csv_files:
                print(f"  📊 CSV files: {len(csv_files)}")
                for csv_file in csv_files[:3]:  # Show first 3
                    print(f"    • {csv_file.name}")
                if len(csv_files) > 3:
                    print(f"    ... and {len(csv_files) - 3} more")
            
            if image_files:
                print(f"  🖼️  Image files: {len(image_files)}")
                for img_file in image_files[:3]:  # Show first 3
                    print(f"    • {img_file.name}")
                if len(image_files) > 3:
                    print(f"    ... and {len(image_files) - 3} more")
            
            if nifti_files:
                print(f"  🧠 Medical images: {len(nifti_files)}")
                for nifti_file in nifti_files[:3]:  # Show first 3
                    print(f"    • {nifti_file.name}")
                if len(nifti_files) > 3:
                    print(f"    ... and {len(nifti_files) - 3} more")
            
            condition_total = len(csv_files) + len(image_files) + len(nifti_files)
            print(f"  📦 Total files: {condition_total}")
            total_files += condition_total
    
    print(f"\n🎯 GRAND TOTAL: {total_files} files across all liver conditions")
    
    if total_files > 0:
        print(f"\n🚀 Next steps:")
        print("1. Run: python organize_real_data.py")
        print("2. Process downloaded data into training format")
        print("3. Train model with real medical data")
    else:
        print(f"\n⚠️  No data downloaded. Check Kaggle API setup.")

def main():
    print("Starting comprehensive liver disease dataset download...")
    
    # Check if kaggle is available
    try:
        result = subprocess.run(["kaggle", "--version"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Kaggle CLI not found. Run: pip install kaggle")
            return
    except:
        print("❌ Kaggle CLI not found. Run: pip install kaggle")
        return
    
    # Download all datasets
    success = download_all_liver_diseases()
    
    if success:
        print(f"\n🎉 SUCCESS! Downloaded comprehensive liver disease datasets")
        print("Your model will now have much better accuracy with real medical data!")
    else:
        print(f"\n⚠️  No datasets downloaded. Check Kaggle setup and try again.")

if __name__ == "__main__":
    main()
