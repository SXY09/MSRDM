# MSRDM
This is the relevant code for the paper 《Multi-Stage Reasoning Framework for Biomedical Document-Level Relation Extraction with Dynamic Memory Mechanism》.
# Requirements
Environment Setup
Please make sure you have Python 3.9+ installed. Install the required dependencies using:
```bash
pip install -r requirements.txt
```
# Setup
This project uses Hydra for hierarchical configuration management. All configurations for datasets, models, and training hyperparameters are located in configs/train.yaml.

To start training from scratch:
```bash
python train.py
```
# Datasets
Our framework is evaluated on three widely used benchmark datasets: CDR, GDA and CHR
