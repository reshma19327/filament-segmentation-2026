# AI-Assisted Structural Analysis of Solar Filaments

## Abstract

This project investigates deep learning methods for segmenting solar filaments in H-alpha (Hα) images. It explores U-Net-based architectures and spine-guided training supervision combined with topology-aware learning using clDice loss. The project also examines segmentation quality and structural properties such as filament area, length, width, orientation, connectivity, and fragmentation.

## Objectives

- Segment solar filaments from H-alpha images.
- Compare deep learning segmentation models.
- Investigate spine-based supervision and topology-aware learning.
- Evaluate segmentation using Dice, IoU, precision, recall, and other relevant metrics.
- Analyse the structural and morphological properties of segmented filaments.

## Technologies Used

- Python
- PyTorch
- NumPy
- OpenCV
- scikit-image
- Matplotlib
- pytest
- Git and GitHub Actions

## Dataset

The project uses the MAGFiLO solar filament dataset. Dataset files and trained model weights are excluded from this repository because of their size. Obtain the dataset separately and configure the appropriate local paths before running the scripts.

## Project Workflow

1. Prepare and preprocess H-alpha images and filament annotations.
2. Train the selected segmentation models.
3. Generate predicted filament masks.
4. Evaluate segmentation performance using quantitative metrics.
5. Analyse structural and morphological properties.
6. Run automated tests through GitHub Actions.

## Installation

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies after creating `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running the Project

Run the relevant training, inference, evaluation, or morphology analysis script after configuring the dataset paths and required dependencies. Refer to the individual scripts for their command-line arguments and configuration requirements.

## Testing

Run the automated tests using:

```bash
pytest -v
```

## Team Members

- Add team member names and roles here.

## License

Add the project license or institutional requirements before public distribution.