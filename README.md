# iMAP Analyzer: Digital Vascular Profiling Software

## Overview
This repository contains the source code for the iMAP Analyzer, a Python-based software tool developed for the quantitative analysis of vascular networks from fluorescence microscopy images. This software enables the extraction of topological features such as vessel area, length, thickness, and network connectivity.

This tool was developed and validated as part of the following research article:
"High-Throughput Digital Decoding of Vascular Heterogeneity in Patient-Specific Tumor Microenvironments" (Advanced Healthcare Materials, 2026).

## Key Features
- Interactive Segmentation: User-guided thresholding with real-time visual feedback to ensure accurate vessel detection.
- Topological Skeletonization: Automated extraction of vascular centerlines based on the Zhang-Suen thinning algorithm.
- Quantitative Profiling: Calculation of key morphological metrics including vessel area, total length, mean thickness, weighted thickness uniformity (WTU), and mesh fraction.
- Batch Processing: Support for high-throughput analysis of multiple regions of interest (ROIs).

## System Requirements
- Operating System: Windows 10/11, macOS, or Linux
- Python Version: Python 3.8 or higher

## Installation
1. Clone this repository to your local machine:
```bash
git clone [https://github.com/JungseubLee/iMAP-Analyzer.git](https://github.com/JungseubLee/iMAP-Analyzer.git)
```
2. Navigate to the project directory:
```bash
cd iMAP-Analyzer
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
To launch the graphical user interface (GUI), run the main script from the terminal:
```bash
python src/main.py
```
1. Click "Load Image" to select a fluorescence image file (TIF, PNG, JPG).
2. Adjust the threshold slider to define the vascular area.
3. Click "Analyze" to generate the skeleton and calculate metrics.
4. The results will be displayed on the screen and can be exported as CSV files.

## Repository Structure
iMAP-Analyzer/

├── src/

│   ├── core.py       # Core algorithms for image processing

│   ├── gui.py        # Graphical User Interface (PyQt5)

│   └── main.py       # Application entry point

├── sample_data/      # Sample images for testing

├── requirements.txt  # Python dependencies

└── LICENSE           # MIT License

## Citation
If you use this software in your research, please cite the following paper:

> Lee, J., et al. (2026). High-Throughput Digital Decoding of Vascular Heterogeneity in Patient-Specific Tumor Microenvironments. *Advanced Healthcare Materials*. (Under revision)

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions regarding the code or the research, please contact the corresponding author or open an issue in this repository.
