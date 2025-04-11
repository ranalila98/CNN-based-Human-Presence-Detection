# CSI-Based Human Presence Detection (1D CNN)

This project uses WiFi Channel State Information (CSI) and a 1D Convolutional Neural Network (CNN) to detect human presence.

## Requirements
- Conda
- Python 3.8.5
- CUDA 10.2 compatible GPU 

## Setup Instructions

1. **Install Conda** (if not already): https://docs.conda.io/en/latest/miniconda.html

2. **Clone or unzip the project**, then open terminal in the project root.

3. **Create and activate the environment:**
   ```bash
   conda env create -f environment.yml
   conda activate sdrcsi_env
   ```
   
4. **Run the project:**
   ```bash
   python main.py
   ```

## Folder Structure
- `dataset_SDR/` — Raw CSI `.npy` files.
- `training/` — Model training logic.
- `testing/` — Model evaluation and confusion matrix.
- `models/` — CNN architecture.
- `utils/` — Dataset loaders and helpers.
- `plots/` — Accuracy and confusion matrix plots.
- `checkpoints/` — Saved model (`best_model.pth`).
- `main.py` — Entry point.
- `environment.yml` — Conda environment config.

##  Outputs
-  Accuracy plot: `plots/accuracy_plot.png`
- Confusion matrix: `plots/confusion_matrix.png`
- Model weights: `checkpoints/best_model.pth`