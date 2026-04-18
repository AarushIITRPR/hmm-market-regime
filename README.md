# HMM for Market Regime Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange.svg)](https://jupyter.org/)

A research-oriented framework for identifying financial market regimes using **Hidden Markov Models (HMMs)** and a suite of alternative machine learning methods. This project provides a unique educational resource by offering both a fully transparent, from-scratch implementation of core HMM algorithms alongside practical library-based applications, enabling a deep conceptual understanding of probabilistic regime detection.

---

## Overview

Market regimes, distinct periods characterized by persistent levels of volatility, trend, or mean reversion, pose a significant challenge to static investment strategies. This repository demonstrates how HMMs, as unsupervised probabilistic models, can effectively uncover these latent states from observable market data like returns and volatility. The project emphasizes a comparative analysis, benchmarking HMMs against both probabilistic (Gaussian Mixture Models) and deterministic (K-Means, Hierarchical Clustering) alternatives, as well as supervised and anomaly detection methods, to provide a holistic view of regime classification in quantitative finance.

---

## Key Features

- **Dual HMM Implementation:** Includes a **from-scratch, NumPy-based HMM** (Forward-Backward, Viterbi, Baum-Welch) for educational clarity, alongside optimized implementations using industry-standard libraries like `hmmlearn` and `pomegranate`.
- **Comprehensive Benchmarking:** Compares HMMs with a wide array of alternative methods including GMMs, K-Means, DBSCAN, Agglomerative Clustering, Isolation Forest, Random Forest, and Wasserstein-distance based clustering.
- **Rigorous Quantitative Evaluation:** Employs a robust set of financial and statistical metrics such as **Log-Likelihood, AIC/BIC, Regime Persistence, Regime Purity, Conditional Sharpe Ratio, and Wasserstein Distances** to objectively assess and compare model performance.
- **Modular and Reproducible Design:** Built with a clear separation of concerns: utility modules for data fetching, preprocessing, visualization, and metrics; and a logical progression of Jupyter notebooks for a structured research workflow.
- **Interactive Visualizations:** Features rich, interactive Plotly charts that overlay detected regimes on price charts, display state transition matrices, and visualize distributional characteristics of each regime.

---

## Project Structure

```
HMM-for-Market-Regime-Detection/
├── README.md
├── requirements.txt
├── utils/                          # Shared utility modules
│   ├── data_utils.py               # Data fetching (yfinance) and feature engineering
│   ├── viz_utils.py                # Plotly-based visualization helpers
│   └── metrics.py                  # Statistical and financial evaluation metrics
├── 01_hmm_from_scratch/            # Educational implementation of core HMM algorithms
│   ├── hmm_core.py                 # GaussianHMM class (Forward, Backward, Viterbi, Baum-Welch)
│   └── 01_hmm_from_scratch.ipynb  # Notebook applying the custom HMM to financial data
├── 02_hmm_libraries/               # HMM implementation using popular Python libraries
│   └── 02_hmm_libraries.ipynb     # Comparison of hmmlearn and pomegranate with the from-scratch model
├── 03_alternative_methods/         # Benchmarking alternative clustering and ML approaches
│   └── 03_alternative_methods.ipynb  # Implementation of GMM, K-Means, DBSCAN, etc.
└── 04_model_comparison/            # Comprehensive quantitative comparison of all methods
    └── 04_model_comparison.ipynb  # Dashboard for evaluating and ranking all models
```

---

## Mathematical Core

The project's foundation is the **Gaussian Hidden Markov Model**, a doubly stochastic process where a sequence of unobserved (hidden) states generates a sequence of observable, continuous emissions.

- **Hidden States ($S_t \in \{1, \ldots, N\}$):** Represent the latent market regimes (e.g., low-volatility bull, high-volatility bear, transitional). These states evolve according to a first-order Markov process defined by the transition probability matrix $\mathbf{A}$.
- **Observations ($O_t$):** These are the financial time series features we can measure, such as daily log-returns, realized volatility, or momentum. The probability of observing $O_t$ given the current state $S_t$ is modeled as a multivariate Gaussian distribution, parameterized by mean vectors $\boldsymbol{\mu}_i$ and covariance matrices $\boldsymbol{\Sigma}_i$ for each state $i$.

**Core Algorithms:**

- **Forward-Backward Algorithm:** Computes the posterior probabilities of being in a particular state at each time point:
$$\gamma_t(i) = P(S_t = i \mid O_1, \ldots, O_T, \lambda)$$

- **Viterbi Algorithm:** Finds the most likely sequence:
$\hat{S}_1, \ldots, \hat{S}_T = \arg\max_{S_1,\ldots,S_T} P(S_1,\ldots,S_T \mid O_1,\ldots,O_T, \lambda)$

- **Baum-Welch Algorithm (EM):** An Expectation-Maximization procedure that iteratively re-estimates the model parameters $\lambda = (\mathbf{A}, \boldsymbol{\mu}, \boldsymbol{\Sigma})$ to maximize the likelihood of the observed data.

---

## Installation

**Prerequisites:** Python 3.9 or higher.

**1. Clone the repository:**
```bash
git clone https://github.com/arjunaggarwaliit/HMM-for-Market-Regime-Detection.git
cd HMM-for-Market-Regime-Detection
```

**2. Install dependencies:**

It is highly recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

The project is designed to be explored sequentially through Jupyter notebooks.

Start a Jupyter server from the project's root directory:

```bash
jupyter lab
```

Run the notebooks in the recommended order to follow the research narrative:

1. `01_hmm_from_scratch/01_hmm_from_scratch.ipynb` - Understand the inner workings of an HMM by using a custom implementation.
2. `02_hmm_libraries/02_hmm_libraries.ipynb` - See how the same problem is solved efficiently using `hmmlearn` and `pomegranate`.
3. `03_alternative_methods/03_alternative_methods.ipynb` - Explore how other unsupervised and supervised methods perform on the regime detection task.
4. `04_model_comparison/04_model_comparison.ipynb` - Examine the final comparative analysis across all models to draw conclusions.

---

## Methodology & Results

The project follows a rigorous, multi-stage methodology:

1. **Data Acquisition & Feature Engineering:** Fetches OHLCV data for a specified ticker (default `SPY`) via `yfinance`. Constructs a rich feature set including log-returns, realized volatility over multiple windows, momentum, and higher-order moments to serve as the observation sequence.

2. **Model Training & Decoding:** Each model is trained on a training set. For HMMs, the Baum-Welch algorithm learns the optimal transition and emission parameters. The Viterbi algorithm is then used to decode the most probable sequence of hidden regimes for the entire time series.

3. **Regime Interpretation:** Decoded states are analyzed post-hoc. Statistical summaries of returns and volatility are computed for each state to assign meaningful labels (e.g., *"Low Volatility Bull"*, *"High Volatility Bear"*, *"Transitional"*).

4. **Comparative Evaluation:** A dashboard-style notebook (`04_model_comparison`) presents a unified comparison of all models using the defined metrics, providing a clear, quantitative basis for assessing model suitability.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Log-Likelihood** | Measures how well the model fits the observed data. Higher values (closer to zero) indicate a better fit. |
| **AIC / BIC** | Penalized measures of model fit, trading off goodness-of-fit against model complexity. Lower values are preferred for model selection. |
| **Regime Persistence** | The average duration (in days) of a regime. Higher persistence often corresponds to more interpretable and stable regimes. |
| **Regime Purity** | Compares predicted regimes against a simple, threshold-based volatility regime ("Low", "Med", "High") to measure alignment with an intuitive baseline. |
| **Conditional Sharpe Ratio** | Calculates the risk-adjusted return within each identified regime, providing a direct financial interpretation of the regime's characteristics. |
| **Wasserstein Distance** | Measures the "earth mover's distance" between the empirical return distributions of different regimes. Larger distances indicate better separation and distinctiveness of the identified states. |

---

## Implementation Details

### `01_hmm_from_scratch`

This module contains a fully self-contained `GaussianHMM` class written in Python with NumPy. It implements all core algorithms:

- `forward_pass()` - Computes scaled forward probabilities.
- `backward_pass()` - Computes scaled backward probabilities.
- `viterbi()` - Implements the Viterbi algorithm in log-space for numerical stability.
- `fit()` - Performs parameter learning via the Expectation-Maximization (Baum-Welch) algorithm.

### `02_hmm_libraries`

This notebook demonstrates the practical application of HMMs using two popular libraries:

- **`hmmlearn`** - A scikit-learn compatible library offering `GaussianHMM` and `GMMHMM` models.
- **`pomegranate`** - A modern probabilistic modeling library that provides a flexible `DenseHMM` implementation.

The notebook verifies the numerical consistency between these libraries and the custom implementation, comparing execution speed and ease of use.

### `utils/`

The utility modules provide a solid foundation for the entire project:

- **`data_utils.py`** - Handles all data acquisition and feature engineering. Includes functions to fetch data from Yahoo Finance, compute log-returns, realized volatility, and other technical indicators, and to create feature matrices for modeling.
- **`viz_utils.py`** - Provides a suite of functions for creating interactive Plotly charts. Includes tools for overlaying regime bands on price charts, plotting transition matrices as heatmaps, and visualizing distributional characteristics of regimes.
- **`metrics.py`** - Implements all the quantitative evaluation metrics used across the project, ensuring consistency and reproducibility in the analysis.

---

## References

1. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257–286.
2. Nguyen, N., & Nguyen, D. (2015). Hidden Markov Model for Stock Selection. *Risks*, 3(4), 455–473.
3. McGreevy, J. (2021). Hidden Markov Models in Finance. Imperial College London, MSc Thesis.
4. Tsang, E. (2021). Market Regime Detection using Hidden Markov Models in QSTrader. *QuantStart*.
5. Chen, X. (2025). HMM-based market regime detection with reinforcement learning. *IDS*.
6. Hikmath Technologies: Market Regime Detection: From HMMs to Wasserstein Clustering.

---

## Acknowledgements

This project was developed as part of the coursework for **AI111 (Mathematical Foundations of AI & Data Engineering)** under the guidance and mentorship of [**Dr. Puneet Kumar**](https://puneetkumar.com/). It builds upon the theoretical foundations and practical applications of Hidden Markov Models in quantitative finance, drawing inspiration from both academic research and industry practices in market regime detection.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
