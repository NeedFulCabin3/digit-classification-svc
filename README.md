# digit-classification-svc

> A streamlined command-line utility for training and evaluating Support Vector Machine classifiers on high-dimensional handwritten digit image datasets.

## Overview

Machine learning practitioners and developers evaluating multi-class classification workflows often encounter friction when setting up clean data pipelines, managing feature scaling, and interpreting model performance metrics without heavy visualization overhead. `digit-classification-svc` solves this by providing a self-contained, reproducible Python execution script that loads standard digit arrays, normalizes feature ranges, trains a non-linear Support Vector Machine estimator, and outputs granular validation reports directly to the console.

## How It Works

The execution pipeline operates through a structured sequence managed entirely in memory:

1. **Dataset Ingestion:** Loads 8x8 pixel handwritten digit samples and discrete integer labels from `scikit-learn`'s internal dataset registry.
2. **Feature Normalization:** Passes raw pixel intensity values through `MinMaxScaler` to rescale the feature space from `[0, 16]` to a bounded interval of `[0, 1]`, preventing feature dominance during kernel optimization.
3. **Stratified Splitting:** Divides the normalized feature matrix into training (`80%`) and testing (`20%`) partitions using deterministic pseudo-random seeding for reproducibility.
4. **Estimator Training:** Initializes an `SVC` instance configured with a radial basis function (RBF) kernel and fits the model against the training partitions.
5. **Evaluation & Reporting:** Computes overall accuracy via `accuracy_score` and generates a detailed classification report displaying precision, recall, and F1-scores across all ten digit classes, concluding with concrete sample predictions.

## Key Features

* Automated dataset downloading and verification via `scikit-learn`.
* Bounded feature scaling preventing gradient imbalance in distance-based estimators.
* Deterministic train-test splitting with configurable random seeds.
* Detailed multi-class evaluation breakdown including precision, recall, and support metrics.
* Robust exception handling with descriptive standard error logging.

## Tech Stack & Core Dependencies Breakdown

* **Python (`>=3.9`):** Core runtime environment providing modern syntax and standard library support.
* **`scikit-learn` (`>=1.0`):** Provides the digit dataset repository, `MinMaxScaler` preprocessing utilities, `SVC` classifier implementation, and evaluation metrics.
* **`numpy`:** Underlying array manipulation backend utilized by scikit-learn estimators.
* **`sys`:** Standard library module utilized for exit code management and error redirection.

## Environment & Web-Based Quick Start

You can run this project instantly using GitHub Codespaces or a local virtual environment without touching complex container configurations.

### Option A: GitHub Codespaces (Browser-Only)
1. Click **Code** -> **Codespaces** -> **Create codespace on main** within your GitHub repository interface.
2. Once the terminal loads, install the required dependencies:
   ```bash
   pip install scikit-learn numpy
   ```
3. Execute the script:
    ```bash
   python main.py
   ```

### Option B: Local Setup (venv)
1. Clone or download the repository, then navigate to the root directory.

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

3. Install dependencies and run the pipeline:

```bash
pip install --upgrade pip
pip install scikit-learn numpy
python main.py
```

## Repository Structure

```bash
digit-classification-svc/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── README.md
└── main.py
```

## Roadmap

**[ ] External Configuration Parsing:** Introduce YAML or JSON configuration files to allow dynamic tuning of SVM hyper-parameters (C, gamma, kernel type) without modifying source code.

**[ ] Structured Logging Migration:** Replace print statements with Python's standard logging module to support log level filtering and file output sinks.

**[ ] Type Annotation Stubs:** Add strict Python type hints (mypy-compliant) across all functions to improve editor auto-completion and static analysis checks.
