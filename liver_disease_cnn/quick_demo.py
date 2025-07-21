#!/usr/bin/env python3
"""
Quick Demo Dataset Creator
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw
import random

def main():
    print("🎨 Creating demo liver disease dataset...")
    
    # Configuration
    classes = ['normal', 'cirrhosis', 'liver_cancer', 'fatty_liver', 'hepatitis']
    splits = {'train': 15, 'val': 5, 'test': 5}
    
    colors = {
        'normal': (120, 100, 80),
        'cirrhosis': (100, 80, 60),
        'liver_cancer': (110, 90, 100),
        'fatty_liver': (140, 120, 90),
        'hepatitis': (130, 90, 70)
    }
    
    # Create directories and images
    total = 0
    for split, count in splits.items():
        print(f"📁 Creating {split} set ({count} images per class)...")
        
        for class_name in classes:
            class_dir = Path('data') / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Remove old demo files
            for old_file in class_dir.glob('demo_*.jpg'):
                old_file.unlink()
            
            # Create images
            for i in range(count):
                # Create simple colored image with medical-like appearance
                img = Image.new('RGB', (224, 224), color=colors[class_name])
                draw = ImageDraw.Draw(img)
                
                # Add liver-like shape
                center_x, center_y = 112, 112
                liver_points = []
                for angle in range(0, 360, 30):
                    import math
                    radius = 60 + random.randint(-10, 10)
                    x = center_x + radius * math.cos(math.radians(angle))
                    y = center_y + radius * math.sin(math.radians(angle))
                    liver_points.append((x, y))
                
                # Draw liver outline
                base_color = colors[class_name]
                liver_color = tuple(max(0, c + random.randint(-20, 20)) for c in base_color)
                draw.polygon(liver_points, fill=liver_color, outline=(80, 80, 80))
                
                # Add condition-specific features
                feature_count = random.randint(5, 15)
                for j in range(feature_count):
                    x = random.randint(50, 174)
                    y = random.randint(50, 174)
                    r = random.randint(2, 8)
                    
                    if class_name == 'normal':
                        feature_color = tuple(c + random.randint(-10, 10) for c in base_color)
                    elif class_name == 'cirrhosis':
                        feature_color = (80, 60, 40)  # Darker spots for scarring
                    elif class_name == 'liver_cancer':
                        feature_color = (140, 100, 120)  # Tumor-like masses
                    elif class_name == 'fatty_liver':
                        feature_color = (160, 140, 100)  # Fatty deposits
                    elif class_name == 'hepatitis':
                        feature_color = (150, 80, 60)  # Inflammation
                    
                    draw.ellipse([x-r, y-r, x+r, y+r], fill=feature_color)
                
                # Save image
                img_path = class_dir / f'demo_{class_name}_{i:03d}.jpg'
                img.save(img_path, 'JPEG', quality=85)
                total += 1
    
    print(f"✅ Created {total} demo images successfully!")
    
    # Show summary
    print("\n📊 Dataset Summary:")
    print("="*40)
    for split in ['train', 'val', 'test']:
        print(f"{split.upper()}:")
        for class_name in classes:
            class_dir = Path('data') / split / class_name
            count = len(list(class_dir.glob('demo_*.jpg')))
            print(f"  {class_name:15}: {count:2} images")
        print()
    
    print("🎉 Demo dataset is ready!")
    print("\n🚀 Next steps:")
    print("1. Open: jupyter lab notebooks/liver_disease_classification.ipynb")
    print("2. Run all cells to train your model")
    print("3. Replace demo images with real medical data when available")

if __name__ == "__main__":
    main()
