
# **S.I.F.T.E.R. — Seismic Investigation and Frequency Tracking for Extraterrestrial Research**

## Project Overview

**S.I.F.T.E.R.** is a comprehensive project designed to analyze seismic data from planetary missions, particularly the Apollo and Mars InSight Lander. The primary goal is to detect seismic events (e.g., quakes) and distinguish them from noise using machine learning models and advanced seismic processing techniques. This project integrates data from lunar, Martian, and Earth-based seismic sources, optimizing the system for deployment on non-terrestrial seismometers.

The system is initially developed in Python for rapid prototyping and machine learning, then finalized in C++ for deployment in low-power environments on planetary missions. A terrestrial front-end is included for real-time analysis and visualization, leveraging tools such as Streamlit, Plotly, or Dash.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Setup Instructions](#setup-instructions)
- [Data Sources](#data-sources)
- [System Architecture](#system-architecture)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [C++ Deployment](#cpp-deployment)
- [Frontend (Visualization)](#frontend-visualization)
- [Testing](#testing)
- [License](#license)

---

## Folder Structure

```bash
SIFTER/
│
├── server/                   # Backend or server-side components, if any
├── src/                      # Source code for Python and C++
│   ├── python/               # Python preprocessing and modeling scripts
│   ├── tests/                # Unit tests for Python and C++ code
│   └── cpp/                  # C++ deployment code
│
├── notebooks/                # Jupyter notebooks for data exploration and modeling
│   ├── lunar_analysis/       # Analysis of lunar seismic data
│   │   └── lunar_plots/      # Visualizations and plots related to lunar analysis
│   └── marsquake_analysis/   # Analysis of Marsquake data
│       └── mars_plots/       # Visualizations and plots related to Marsquake analysis
│
├── model/                    # Machine learning models (Python and converted C++)
│   ├── model_output/         # Model outputs, predictions, and evaluations
│
├── extra/                    # Extra files or scripts (miscellaneous)
│
├── docs/                     # Documentation (architecture, setup, API)
││
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker setup for reproducible builds
├── README.md                 # This file
└── LICENSE                   # License for the project
```

---

## Setup Instructions

### Prerequisites

- **Python 3.8+**
- **CMake** for building the C++ components
- **Docker** (optional, for containerized development)
- **Jupyter** for notebooks
- **Git** for version control
- **Git LFS** for handling large seismic data files
- **Streamlit**, **Plotly**, or **Dash** for frontend visualizations

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Bespoke-Robot-Society/SIFTER
   cd SIFTER_Project
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install C++ dependencies and build the project:

   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

4. (Optional) Build the Docker environment:

   ```bash
   docker build -t sifter-project .
   ```

---

## Data Sources

The project integrates seismic data from multiple sources:

- **Apollo 12 Moonquake Data**: Provided in `data/apollo12_catalogs/`. The dataset includes seismic event catalogs with timestamped events and waveforms.
- **Mars InSight Lander Data**: Marsquake data from the InSight mission, found in `data/marsquake_data/`.
- **Earthquake Data**: Earth-based seismic data accessed via **IRIS** or **PyWeed** for training models or comparative studies (`data/external/`).
- **Synthetic Seismic Wavefields**: Simulated seismic data generated using reflectivity methods (`data/seismic_wavefields/`).

---

## System Architecture

The architecture consists of the following components:

1. **Data Collection and Preprocessing**:

   - Continuous logging of raw seismic data (MiniSEED format).
   - Filtering and event detection using **STA/LTA** algorithms.
   - Data segmentation for machine learning model input.

2. **Machine Learning Classification**:

   - Python-based models such as **Random Forest**, **SVM**, and **Decision Trees** are used to classify seismic events.
   - Trained models are converted into **C++** for real-time classification on non-terrestrial seismometers.

3. **Real-Time Event Classification**:

   - The final classification occurs in **C++**, with incoming seismic data classified as various types of seismic events.
   - Only relevant data is compressed and transmitted back during solar-powered periods.

4. **Frontend Visualization**:

   - Visualization of seismic data and classification results using tools like **Streamlit**, **Plotly**, or **Dash**.

---

## Machine Learning Pipeline

- **Data Preprocessing**:

  - Implements STA/LTA filtering and segmentation of seismic data.
  
- **Model Training**:

  - Models are trained in Python using **Random Forest**, **SVM**, and **Decision Trees**.
  - Evaluation metrics (accuracy, precision, recall, F1-score) are computed using cross-validation.

- **Model Deployment**:

  - Trained models are converted to formats like **ONNX** and deployed in C++ for low-power, real-time applications.

---

## C++ Deployment

The final system is implemented in C++ for use on non-terrestrial seismometers, ensuring low power consumption and real-time data processing.

- **MiniSEED Data Processing**: C++ handles loading and filtering raw seismic data.
- **Real-Time Classification**: Optimized machine learning models classify seismic events in real time.
- **Data Compression and Transmission**: Seismic data is compressed and transmitted only when necessary (e.g., during daylight hours).

---

## Frontend (Visualization)

A terrestrial front-end for real-time seismic data analysis and visualization:

- **Streamlit** Dashboard: Provides an interactive UI for seismic event visualization.
- **Plotly** Visualizations: Real-time waveform plotting and event timelines.
- **Dash** (optional): Advanced data visualization in a web interface.

---

## Testing

Run unit tests to ensure the correctness of the Python and C++ components:

- **Python Tests**:

  ```bash
  pytest src/tests/test_preprocessing.py
  pytest src/tests/test_models.py
  ```

- **C++ Tests**:

  ```bash
  cd build
  make test
  ```

CI/CD pipelines are integrated with **GitHub Actions** to ensure continuous testing and deployment.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

### **Updated Dependencies** (from `requirements.txt`)

```txt
# Core Python libraries for data handling and scientific computing
numpy==1.21.0
pandas==1.5.2
scipy==1.9.3

# Seismic data processing
obspy==1.3.0  # For MiniSEED handling and seismic data analysis

# Machine learning models
scikit-learn==1.1.2  # For Random Forest, SVM, and Decision Trees
xgboost==1.6.2  # Optional for gradient boosting classifier
pytorch==1.12.1  # PyTorch for deep learning models

# Data visualization
matplotlib==3.6.2
plotly==5.10.0
dash==2.6.1  # Optional for Dash-based web applications

# Model conversion for C++ deployment
onnx==1.18.0  # For converting models to ONNX format
onnx2c  # Optional for converting ONNX models to C code

# CI/CD and testing
pytest==7.2.0  # Unit testing framework
black==22.10.0  # Code formatter for linting

# Data version control (DVC)
dvc==2.17.0  # Versioning for large seismic datasets

# Jupyter notebook support
jupyterlab==3.5.0

# Additional tools for streamlining data management
pyyaml==6.0  # Configuration file handling

# Docker environment (optional for Docker builds)
docker==5.0.3

# For HTTP requests and API integration
requests==2.28.1
```
