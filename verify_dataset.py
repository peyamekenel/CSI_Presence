import os
import csv
import sys
from pathlib import Path

def verify_dataset(dataset_name):
    print(f"\nVerifying dataset: {dataset_name}")
    data_dir = Path(f"data/{dataset_name}/{dataset_name}")
    
    if not data_dir.exists():
        print(f"Dataset directory not found: {data_dir}")
        return False
    
    # Check required files
    required_files = ['trainLabels.csv', 'validationLabels.csv', 'testLabels.csv', 'meanStd.csv']
    for file in required_files:
        if not (data_dir / file).exists():
            print(f"Missing required file: {file}")
            return False
    
    # Verify all images exist
    for label_file in ['trainLabels.csv', 'validationLabels.csv', 'testLabels.csv']:
        with open(data_dir / label_file) as f:
            reader = csv.reader(f)
            missing = []
            for row in reader:
                img_path = data_dir / f"{row[0]}.png"
                if not img_path.exists():
                    missing.append(row[0])
            
            if missing:
                print(f"Missing images in {label_file}:")
                for img_id in missing:
                    print(f"  {img_id}.png")
            else:
                print(f"All images exist for {label_file}")
    
    # Verify class distribution
    class_counts = {}
    with open(data_dir / 'trainLabels.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            label = row[1]
            class_counts[label] = class_counts.get(label, 0) + 1
    
    print("\nClass distribution in training set:")
    for label in sorted(class_counts.keys()):
        print(f"Class {label}: {class_counts[label]} samples")
    
    return True

if __name__ == '__main__':
    datasets = ['DP_LOS', 'DP_NLOS']
    for dataset in datasets:
        verify_dataset(dataset)
