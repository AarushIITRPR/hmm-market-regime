# HMM for Market Regime Detection

A research-oriented project for detecting market regimes using Hidden Markov Models (HMMs) and alternative machine learning methods on financial data.

## Overview

This repository explores how different probabilistic and clustering-based approaches can be used to identify market regimes such as low-volatility, high-volatility, and transitional states.

The project includes:
- a from-scratch HMM implementation
- library-based HMM experiments
- alternative unsupervised and supervised methods
- a final model comparison notebook

## Repository Structure

```text
HMM-for-Market-Regime-Detection/
├── README.md
├── requirements.txt
├── utils/
│   ├── data_utils.py
│   ├── viz_utils.py
│   └── metrics.py
├── 01_hmm_from_scratch/
│   ├── hmm_core.py
│   └── 01_hmm_from_scratch.ipynb
├── 02_hmm_libraries/
│   └── 02_hmm_libraries.ipynb
├── 03_alternative_methods/
│   └── 03_alternative_methods.ipynb
└── 04_model_comparison/
    └── 04_model_comparison.ipynb
```

## What’s Included

### 01. HMM From Scratch
Implements the core HMM algorithms manually, including:
- Forward algorithm
- Backward algorithm
- Viterbi decoding
- Baum-Welch training

This notebook is useful for understanding the mathematical foundation of HMMs in detail.

### 02. HMM Libraries
Recreates the regime detection workflow using popular HMM libraries such as:
- hmmlearn
- pomegranate

This section helps compare a custom implementation against standard tools.

### 03. Alternative Methods
Benchmarks non-HMM approaches such as:
- Gaussian Mixture Models
- K-Means clustering
- DBSCAN
- Agglomerative clustering
- Isolation Forest
- Random Forest
- Wasserstein-distance based clustering

### 04. Model Comparison
Summarizes and compares the methods using:
- log-likelihood
- AIC and BIC
- regime persistence
- regime purity
- qualitative comparison plots

## Features

- Modular utility functions for data, visualization, and metrics
- Notebook-based research workflow
- Interactive Plotly visualizations
- Side-by-side comparison of probabilistic and non-probabilistic methods
- Focus on interpretability for financial regime analysis

## Mathematical Core

The HMM formulation used in this project follows the standard setup:

- hidden states represent latent market regimes
- observations are financial market features
- transitions model regime persistence over time
- emissions model the distribution of observed data within each regime

The project covers:
- evaluation with Forward and Backward algorithms
- decoding with Viterbi
- parameter learning with Baum-Welch EM

## Installation

```bash
git clone https://github.com/arjunaggarwaliit/HMM-for-Market-Regime-Detection.git
cd HMM-for-Market-Regime-Detection
pip install -r requirements.txt
```

## Usage

Run the notebooks in this order:

1. 01_hmm_from_scratch  
2. 02_hmm_libraries  
3. 03_alternative_methods  
4. 04_model_comparison  

Start Jupyter Notebook from the project root:

```bash
jupyter notebook
```

## References

- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition  
- Nguyen, N., & Nguyen, D. (2015). Hidden Markov Model for Stock Selection  
- McGreevy, J. (2021). Hidden Markov Models in Finance  
- Additional quantitative finance and market regime detection literature  

## Acknowledgements

This project builds on standard HMM theory and market regime detection methods from the quantitative finance and machine learning literature under the faculty of AI111 (Mathematical Foundations of AI & DE), [Dr. Puneet Kumar](https://puneetkumar.com/).
