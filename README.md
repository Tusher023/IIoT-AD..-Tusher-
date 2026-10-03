# Unsupervised Anomaly Detection for Predictive Maintenance in Industrial IoT

## Research Project

**Working Title:** *Unsupervised Anomaly Detection for Predictive Maintenance in Industrial IoT Using Autoencoder and LSTM-Based Models*

**Type:** BTech Final Year Project + Research Paper

---

## Abstract

This project investigates unsupervised and semi-unsupervised anomaly detection methods for predictive maintenance in Industrial IoT systems. We systematically compare classical isolation-based and deep-learning-based approaches for detecting machine degradation and providing early failure warnings in multivariate industrial sensor data, when labeled failure examples are limited.

## Primary Research Question

> How effectively can unsupervised anomaly-detection methods identify machine degradation and provide early warnings of impending failure in multivariate industrial sensor data when labeled failure examples are limited?

## Methods Under Investigation

| Model | Type | Category |
|-------|------|----------|
| Statistical Threshold | Z-score / Percentile | Baseline |
| Isolation Forest | Ensemble tree-based | Classical ML |
| One-Class SVM | Kernel-based | Classical ML |
| Fully Connected Autoencoder | Reconstruction-based | Deep Learning |
| LSTM Autoencoder | Temporal reconstruction-based | Deep Learning |

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset**

| Subset | Engines (Train) | Engines (Test) | Operating Conditions | Fault Modes |
|--------|----------------|----------------|---------------------|-------------|
| FD001  | 100            | 100            | 1                   | 1 (HPC)     |
| FD002  | 260            | 259            | 6                   | 1 (HPC)     |
| FD003  | 100            | 100            | 1                   | 2 (HPC+Fan) |
| FD004  | 249            | 248            | 6                   | 2 (HPC+Fan) |

Source: [NASA Prognostics Center](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6)

## Project Structure

```
IIoT-AD/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── configs/                   # Experiment configuration files
│   ├── baseline.yaml
│   ├── autoencoder.yaml
│   └── lstm_autoencoder.yaml
├── data/
│   ├── raw/                   # Original unmodified datasets
│   └── processed/             # Preprocessed data (generated)
├── notebooks/                 # Jupyter notebooks for EDA & visualization
├── src/                       # Source code
│   ├── data/                  # Data loading and splitting
│   ├── preprocessing/         # Feature engineering and normalization
│   ├── models/                # Model architectures
│   ├── training/              # Training loops and procedures
│   ├── detection/             # Anomaly scoring and thresholding
│   ├── evaluation/            # Metrics and evaluation framework
│   ├── visualization/         # Plotting utilities
│   └── utils/                 # Reproducibility, config, logging
├── experiments/               # Saved models and experiment artifacts
├── results/
│   ├── tables/                # Result tables (CSV/JSON)
│   ├── figures/               # Publication-quality figures
│   └── logs/                  # Experiment logs
├── tests/                     # Unit tests
├── docs/                      # Research documentation
│   ├── project_roadmap.md
│   ├── research_questions.md
│   ├── experiment_protocol.md
│   ├── literature_review.md
│   └── research_gap.md
└── paper/                     # Research paper and thesis
    ├── outline.md
    ├── figures/
    └── tables/
```

## Environment

- **OS:** Windows 11 (x64)
- **CPU:** Intel Core i7-1165G7 @ 2.80 GHz (4C/8T)
- **RAM:** ~16 GB
- **GPU:** None (CPU-only training)
- **Python:** 3.10.7
- **Framework:** PyTorch 2.3.0

## Setup

```bash
# Clone the repository
git clone <repository-url>
cd IIoT-AD

# Install dependencies
pip install -r requirements.txt

# Download C-MAPSS dataset
# Place raw data files in data/raw/
# See data/raw/README.md for details

# Run tests
python -m pytest tests/ -v
```

## Key Design Principles

1. **Scientific validity over metric optimization** — Results are reported honestly.
2. **No data leakage** — Engine-level splits, train-only normalization, validation-only thresholds.
3. **Reproducibility** — Configuration-driven experiments with fixed seeds.
4. **Evidence-driven conclusions** — If simpler models outperform complex ones, we report that.
5. **Modular code** — Reusable components with clear interfaces.

## Research Integrity

- All citations are verified against actual publications.
- No experimental results are fabricated.
- Thresholds are never tuned using test labels.
- Limitations are explicitly documented.
- Statistical significance is tested before claiming differences.

## License

This project is for academic research purposes.

## Author

BTech CSE Final Year Project — 2026

