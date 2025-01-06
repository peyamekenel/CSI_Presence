import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Function to load and display spectrogram
def show_spectrogram(filepath, label):
    img = Image.open(filepath)
    plt.figure(figsize=(6, 4))
    plt.imshow(np.array(img))
    plt.title(f"Class {label}")
    plt.axis("off")
    plt.savefig(f"sample_class_{label}.png")
    plt.close()

# Get one sample from each class for DP_LOS
with open("DP_LOS/DP_LOS/trainLabels.csv", "r") as f:
    lines = f.readlines()

samples = {}
for line in lines:
    if line.strip():
        img_idx, label = map(int, line.strip().split(","))
        if label not in samples:
            samples[label] = img_idx

# Display one sample from each class
for label, img_idx in samples.items():
    filepath = f"DP_LOS/DP_LOS/{img_idx}.png"
    if os.path.exists(filepath):
        show_spectrogram(filepath, label)

# Print class distribution
class_dist = {}
for line in lines:
    if line.strip():
        _, label = map(int, line.strip().split(","))
        class_dist[label] = class_dist.get(label, 0) + 1

print("\nClass distribution in training set:")
for label in sorted(class_dist.keys()):
    print(f"Class {label}: {class_dist[label]} samples")
