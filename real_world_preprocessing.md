# Real-World CSI Data Collection and Preprocessing Guide

## Overview
This guide explains how to collect and preprocess WiFi Channel State Information (CSI) data for use with our presence detection model. The model expects CSI data in the form of 256×256 grayscale spectrograms, saved as PNG files.

## Data Collection Requirements

### Hardware Setup
- ESP32 or similar WiFi device capable of CSI extraction
- WiFi access point (AP) for transmission
- Computer for data collection and processing

### CSI Collection Process
1. Configure ESP32 to collect CSI data:
   - Sampling rate: 100Hz recommended
   - Collect both amplitude and phase information
   - Record timestamps for each measurement
   - Save raw CSI matrices in CSV format

2. Data Collection Protocol:
   - Record baseline (unoccupied) measurements first
   - Collect data for each room/location separately
   - Minimum 50 samples per class recommended
   - Include various environmental conditions

## Data Preprocessing Pipeline

### 1. Raw CSI Processing
```python
import numpy as np
from scipy import signal

def process_raw_csi(csi_data):
    """
    Process raw CSI data from ESP32
    Args:
        csi_data: Raw CSI measurements (complex numbers)
    Returns:
        Processed CSI data
    """
    # Extract amplitude
    amplitude = np.abs(csi_data)
    
    # Optional: Phase sanitization
    phase = np.angle(csi_data)
    
    # Noise reduction (if needed)
    amplitude = signal.medfilt(amplitude, kernel_size=3)
    
    return amplitude
```

### 2. Spectrogram Generation
```python
def generate_spectrogram(csi_data, fs=100):
    """
    Convert CSI data to spectrogram
    Args:
        csi_data: Processed CSI amplitude data
        fs: Sampling frequency (Hz)
    Returns:
        Spectrogram array (256x256)
    """
    # Generate spectrogram
    frequencies, times, Sxx = signal.spectrogram(
        csi_data,
        fs=fs,
        window='hamming',
        nperseg=256,
        noverlap=128
    )
    
    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx)
    
    # Resize to 256x256
    from scipy.ndimage import zoom
    zoom_factor = (256.0/Sxx_db.shape[0], 256.0/Sxx_db.shape[1])
    spectrogram = zoom(Sxx_db, zoom_factor)
    
    # Normalize
    spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
    
    return spectrogram
```

### 3. Save Spectrograms
```python
from PIL import Image

def save_spectrogram(spectrogram, output_path):
    """
    Save spectrogram as PNG file
    Args:
        spectrogram: 256x256 normalized spectrogram array
        output_path: Output PNG file path
    """
    # Scale to 0-255 range
    img_data = ((spectrogram - spectrogram.min()) * (255.0 / (spectrogram.max() - spectrogram.min()))).astype(np.uint8)
    
    # Create and save image
    img = Image.fromarray(img_data)
    img.save(output_path)
```

## File Organization

### Directory Structure
```
data/
├── raw_csi/
│   ├── unoccupied/
│   ├── room1/
│   └── ...
├── spectrograms/
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── labels.csv
```

### Label File Format
Create a CSV file with two columns:
```csv
image_id,class_label
0,0        # unoccupied
1,1        # room 1
2,1        # room 1
...
```

- `image_id`: Unique identifier for each spectrogram (0 to N-1)
- `class_label`: 0 for unoccupied, 1-5 for different rooms

## Integration with Training Pipeline

1. Split your data into train/validation/test sets
2. Create corresponding label files:
   - trainLabels.csv
   - validationLabels.csv
   - testLabels.csv

3. Calculate dataset statistics:
```python
def calculate_dataset_stats(spectrogram_dir):
    """
    Calculate mean and std of spectrograms
    """
    all_specs = []
    for file in os.listdir(spectrogram_dir):
        if file.endswith('.png'):
            img = Image.open(os.path.join(spectrogram_dir, file))
            all_specs.append(np.array(img))
    
    all_specs = np.array(all_specs)
    mean = np.mean(all_specs) / 255.0
    std = np.std(all_specs) / 255.0
    
    return mean, std
```

4. Save statistics in meanStd.csv:
```csv
0.447    # mean
0.162    # std
```

## Model Requirements

The CSINet model expects:
- Input shape: (1, 256, 256) - single channel grayscale
- Pixel values: Normalized using dataset mean/std
- File format: PNG images
- Label format: Integer classes (0-5)

## Best Practices

1. Data Collection:
   - Maintain consistent device positions
   - Record environmental conditions
   - Collect data at different times of day

2. Preprocessing:
   - Handle missing packets in raw CSI data
   - Remove outliers before generating spectrograms
   - Verify spectrogram quality visually

3. Quality Control:
   - Check spectrogram dimensions (256×256)
   - Verify normalization
   - Validate label assignments

4. Documentation:
   - Record data collection parameters
   - Document any filtering/processing steps
   - Note environmental conditions

## Troubleshooting

Common issues and solutions:
1. Missing packets in CSI data
   - Interpolate missing values
   - Filter out corrupted measurements

2. Noisy spectrograms
   - Adjust window size and overlap
   - Apply additional filtering

3. Inconsistent dimensions
   - Verify resize operations
   - Check input data shape

4. Poor model performance
   - Verify normalization
   - Check label consistency
   - Validate data quality
