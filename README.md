# WiFi CSI-Based Presence Detection

This project implements a machine learning system for detecting human presence in indoor environments using WiFi Channel State Information (CSI) data. The system processes CSI amplitude spectrograms to determine whether a space is occupied or not.

## Features

- Support for both Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) scenarios
- High accuracy presence detection (92% for LOS, 97% for NLOS)
- Room-level localization capabilities
- Robust data preprocessing pipeline
- Parameterized training configuration
- Comprehensive model evaluation tools

## Project Structure

```
.
 data/                    # Dataset directory
   ├── DP_LOS/             # Line-of-sight presence detection dataset
   └── DP_NLOS/            # Non-line-of-sight presence detection dataset
 train.py                # Main training script
 train_parameterized.py  # Configurable training script
 verify_dataset.py       # Dataset integrity verification
 analyze_dataset.py      # Dataset analysis utilities
```

## Dataset

The project uses the WiFi CSI dataset containing:
- DP_LOS: Line-of-sight presence detection dataset (392 CSI amplitude spectrograms)
- DP_NLOS: Non-line-of-sight presence detection dataset (384 CSI amplitude spectrograms)

Each spectrogram is represented as a 256x256 pixel grayscale image, capturing the CSI amplitude patterns that indicate human presence or absence.

## Usage

1. Install dependencies:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

2. Verify dataset integrity:
```bash
python verify_dataset.py
```

3. Train the model:
```bash
python train.py
```

Or use the parameterized version:
```bash
python train_parameterized.py --dataset DP_LOS --batch_size 32 --epochs 100
```

## Model Architecture

The system uses a Convolutional Neural Network (CSINet) specifically designed for processing CSI spectrograms. The architecture includes:
- Multiple convolutional layers for feature extraction
- Batch normalization for training stability
- ReLU activation functions
- Dropout for regularization
- Fully connected layers for classification

## Performance

- LOS Scenario: 92% accuracy
- NLOS Scenario: 97% accuracy
- Room-level localization capability

## Real-world Deployment

For real-world applications:
1. Set up ESP32 devices for CSI data collection
2. Configure WiFi parameters and sampling rate
3. Process raw CSI data into spectrograms
4. Use trained model for real-time presence detection

## License

MIT License
