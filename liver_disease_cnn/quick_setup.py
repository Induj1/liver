#!/usr/bin/env python3
"""
Simple Dataset Creator - Creates minimal medical-style images
"""

import os
from pathlib import Path

try:
    from PIL import Image, ImageDraw
    import random
    import numpy as np
    
    print("🏥 Creating Enhanced Medical Dataset")
    print("=" * 40)
    
    # Define classes and splits
    classes = ['normal', 'cirrhosis', 'liver_cancer', 'fatty_liver', 'hepatitis']
    splits = {'train': 20, 'val': 8, 'test': 7}
    
    # Color schemes for different conditions
    color_schemes = {
        'normal': [(120, 120, 120), (130, 130, 130), (110, 110, 110)],
        'cirrhosis': [(100, 90, 80), (110, 100, 90), (90, 80, 70)],
        'liver_cancer': [(140, 100, 110), (150, 110, 120), (130, 90, 100)],
        'fatty_liver': [(150, 140, 100), (160, 150, 110), (140, 130, 90)],
        'hepatitis': [(130, 110, 90), (140, 120, 100), (120, 100, 80)]
    }
    
    total_created = 0
    
    for split, count in splits.items():
        print(f"\n📁 Creating {split} set ({count} images per class)...")
        
        for class_name in classes:
            class_dir = Path('data') / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Remove old files
            for old_file in class_dir.glob('*.jpg'):
                if any(prefix in old_file.name for prefix in ['demo_', 'synthetic_', 'enhanced_']):
                    old_file.unlink()
            
            # Create new images
            for i in range(count):
                # Create image with medical-like appearance
                img = Image.new('RGB', (256, 256), (40, 40, 40))
                draw = ImageDraw.Draw(img)
                
                # Base color for condition
                base_colors = color_schemes[class_name]
                base_color = random.choice(base_colors)
                
                # Create liver-like shape
                center_x, center_y = 128, 128
                
                # Draw main organ area
                points = []
                for angle in range(0, 360, 15):
                    radius = 60 + random.randint(-15, 15)
                    x = center_x + radius * np.cos(np.radians(angle))
                    y = center_y + radius * np.sin(np.radians(angle))
                    points.append((x, y))
                
                draw.polygon(points, fill=base_color)
                
                # Add some texture/patterns based on condition
                if class_name == 'cirrhosis':
                    # Add nodular pattern
                    for _ in range(8):
                        x = center_x + random.randint(-40, 40)
                        y = center_y + random.randint(-40, 40)
                        r = random.randint(3, 8)
                        color = (base_color[0] + 20, base_color[1] + 15, base_color[2] + 10)
                        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
                
                elif class_name == 'liver_cancer':
                    # Add tumor-like spots
                    for _ in range(3):
                        x = center_x + random.randint(-30, 30)
                        y = center_y + random.randint(-30, 30)
                        r = random.randint(8, 15)
                        color = (base_color[0] + 30, base_color[1] - 10, base_color[2] + 20)
                        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
                
                elif class_name == 'fatty_liver':
                    # Add fatty deposits
                    for _ in range(12):
                        x = center_x + random.randint(-50, 50)
                        y = center_y + random.randint(-50, 50)
                        r = random.randint(2, 5)
                        color = (base_color[0] + 25, base_color[1] + 25, base_color[2] + 5)
                        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
                
                elif class_name == 'hepatitis':
                    # Add inflammatory changes
                    for _ in range(15):
                        x = center_x + random.randint(-45, 45)
                        y = center_y + random.randint(-45, 45)
                        r = random.randint(1, 3)
                        color = (base_color[0] + 15, base_color[1] + 5, base_color[2] - 5)
                        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
                
                # Add vessels
                for _ in range(5):
                    x1 = center_x + random.randint(-40, 40)
                    y1 = center_y + random.randint(-40, 40)
                    x2 = x1 + random.randint(-20, 20)
                    y2 = y1 + random.randint(-20, 20)
                    vessel_color = (max(0, base_color[0] - 30), 
                                  max(0, base_color[1] - 30), 
                                  max(0, base_color[2] - 30))
                    draw.line([(x1, y1), (x2, y2)], fill=vessel_color, width=2)
                
                # Save image
                img_path = class_dir / f'medical_{class_name}_{i:03d}.jpg'
                img.save(img_path, 'JPEG', quality=90)
                total_created += 1
    
    print(f"\n✅ Created {total_created} medical-style images!")
    
    # Show final status
    print(f"\n📊 Final Dataset Status:")
    print("=" * 30)
    
    grand_total = 0
    for split in splits.keys():
        split_total = 0
        print(f"\n{split.upper()}:")
        for class_name in classes:
            class_dir = Path('data') / split / class_name
            if class_dir.exists():
                count = len([f for f in class_dir.iterdir() if f.suffix.lower() == '.jpg'])
                print(f"  {class_name:15}: {count:3} images")
                split_total += count
        print(f"  {'TOTAL':15}: {split_total:3} images")
        grand_total += split_total
    
    print(f"\n🎯 TOTAL DATASET: {grand_total} images")
    print("\n🚀 Ready for training!")
    print("Next: Open notebooks/liver_disease_classification.ipynb")

except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Run: pip install pillow numpy")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
