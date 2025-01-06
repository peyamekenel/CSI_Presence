# WiFi CSI-Based Presence Detection

This project implements a machine learning system for detecting human presence in indoor environments using WiFi Channel State Information (CSI) data. The system can determine both the presence/absence of people and their location within different rooms.

## Features

- Human presence detection using WiFi CSI data
- Support for both Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) scenarios
- High accuracy (92-97%) in presence detection
- Room-level localization capabilities
- Comprehensive data preprocessing pipeline
- PyTorch-based CNN model implementation

## Project Structure

```
├── data/
│   ├── DP_LOS/        # Line-of-Sight dataset
│   └── DP_NLOS/       # Non-Line-of-Sight dataset
├── train.py           # Basic training script
├── train_parameterized.py  # Configurable training script
├── verify_dataset.py  # Dataset verification utility
└── real_world_preprocessing.md  # Guide for real-world deployment
```

## Dataset Format

The project uses CSI amplitude spectrograms (256×256 grayscale images) organized in the following structure:
- Class 0: Unoccupied state
- Classes 1-5: Presence in different rooms

Each dataset variant (DP_LOS, DP_NLOS) includes:
- trainLabels.csv
- validationLabels.csv
- testLabels.csv
- meanStd.csv (normalization parameters)

## Usage

1. Train the model using the parameterized script:
```bash
python train_parameterized.py --dataset DP_LOS --epochs 20
```

2. Verify dataset integrity:
```bash
python verify_dataset.py
```

## Model Performance

- DP_LOS Dataset: 92% accuracy
- DP_NLOS Dataset: 97% accuracy
- Perfect/near-perfect unoccupied state detection
- Strong room-specific presence detection

## Real-World Deployment

See `real_world_preprocessing.md` for detailed instructions on:
- Raw CSI data collection
- Data preprocessing
- Spectrogram generation
- Model deployment

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib
