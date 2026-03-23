# MSRDM
This is the relevant code for the paper 《Multi-Stage Reasoning Framework for Biomedical Document-Level Relation Extraction with Dynamic Memory Mechanism》.
# Requirements
Environment Setup
Please make sure you have Python 3.9+ installed. Install the required dependencies using:
```bash
pip install -r requirements.txt
```
# Setup
1.Data Augmentation: To generate high-quality synthetic data using our LLM-based generative prompt strategy:
```bash
python Generative-based Data Augmentation.py
```

2.Training the DMER Model: To train the model on the benchmark datasets, configure your paths in the config folder and run:
```bash
python train.py
```
# Datasets
Our framework is evaluated on three widely used benchmark datasets: CDR, GDA and CHR
