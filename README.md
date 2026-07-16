# TOW-IDS: Intrusion Detection System Based on Three Overlapped Wavelets for Automotive Ethernet

> Reproduction status (honest notes): [REPRODUCTION_NOTES.md](./REPRODUCTION_NOTES.md)  
> Table-II local results (tables + figures, no retraining): [results/summary/table2_summary.md](./results/summary/table2_summary.md)

## Project Overview
This project is dedicated to developing an effective Intrusion Detection System (IDS) aimed at identifying Abnormal behaviors in a network traffic which is collected from CAN, AVB, and gPTP
protocols in Automotive Ethernet. Utilizing Deep Learning algorithms like ResNet50 and EfficientNetB0 and a customized DCNN model, this system is designed to detect abnormal network traffic. refer to the  [TOW-IDS - IEEE paper link](https://ieeexplore.ieee.org/document/9947068/algorithms?tabFilter=dataset#algorithms)


## Installation
This repository can now run as a pure Python project (without Jupyter).

1. **Prerequisites**: Ensure you have [Python 3.8+](https://www.python.org/downloads/) and [pip](https://pip.pypa.io/en/stable/installation/) installed.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Python Workflow (No Notebook Required)

### 1) Training pipeline (preprocess + wavelet + custom TOW-IDS model)
```bash
python run_pipeline.py --mode train
```

### 2) Generate test-side transformed data
```bash
python run_pipeline.py --mode test
```

### 3) Train all models (custom CNN + no-wavelet CNN + ResNet50 + EfficientNetB0)
```bash
python run_pipeline.py --mode train_all
```

## Individual Script Entry Points
- `preprocessing.py`: Convert training PCAP payloads to normalized N x N groups.
- `wavelet_transformation.py`: Apply three overlapped wavelets and save layered arrays.
- `tow_ids_model.py`: Train the custom TOW-IDS CNN model.
- `tow_ids_no_wavelet.py`: Train custom model directly on non-wavelet normalized data.
- `resnet_model.py`: Train ResNet50 transfer-learning model.
- `EfficientNetB0.py`: Train EfficientNetB0 transfer-learning model.
- `plot_table2_results.py`: Rebuild Table-II summary tables/figures from `results/table2_results.csv` (no training).
