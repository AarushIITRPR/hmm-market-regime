# Market Regime Detection

A comprehensive, research-grade implementation of market regime detection using Hidden Markov Models and alternative machine learning methods, applied to real financial data.

---

## Project Structure

```
market-regime-detection/
├── README.md
├── requirements.txt
│
├── utils/
│   ├── data_utils.py          # Data fetching, preprocessing, feature engineering
│   ├── viz_utils.py           # Reusable Plotly visualisation helpers
│   └── metrics.py             # Regime evaluation metrics (AIC, BIC, persistence, etc.)
│
├── 01_hmm_from_scratch/
│   ├── hmm_core.py            # GaussianHMM — Forward, Backward, Viterbi, Baum-Welch
│   └── 01_hmm_from_scratch.ipynb
│
├── 02_hmm_libraries/
│   └── 02_hmm_libraries.ipynb # hmmlearn, pomegranate — comparison with scratch impl.
│
├── 03_alternative_methods/
│   └── 03_alternative_methods.ipynb  # GMM, K-Means, DBSCAN, Random Forest, Wasserstein
│
└── 04_model_comparison/
    └── 04_model_comparison.ipynb     # Head-to-head evaluation & summary dashboard
```

---

## Notebooks at a Glance

| Notebook | Contents |
|---|---|
| `01_hmm_from_scratch` | Forward/Backward algorithms, Viterbi decoding, Baum-Welch EM — all hand-coded with full mathematical commentary matching the reference PDF notation |
| `02_hmm_libraries` | Same regimes reproduced via `hmmlearn` and `pomegranate`; numerical agreement with scratch impl. verified |
| `03_alternative_methods` | GMM, K-Means, DBSCAN, Agglomerative Clustering, Isolation Forest (anomaly), Random Forest (supervised labelling), Wasserstein-distance clustering |
| `04_model_comparison` | AIC/BIC, regime persistence, log-likelihood, regime purity, qualitative comparison table |

All charts are **interactive Plotly figures**.

---

## Algorithms Implemented from Scratch (`hmm_core.py`)

### Problem 1 — Evaluation: Forward Algorithm
$$\alpha_t(i) = P(O_1,\dots,O_t,\, Q_t=S_i \mid \lambda)$$

Initialisation: $\alpha_1(i) = \pi_i \, b_i(O_1)$

Recursion: $\alpha_{t+1}(j) = \left[\sum_{i=1}^{N} \alpha_t(i)\, a_{ij}\right] b_j(O_{t+1})$

### Problem 1 — Evaluation: Backward Algorithm
$$\beta_t(i) = P(O_{t+1},\dots,O_T \mid Q_t=S_i,\lambda)$$

### Problem 2 — Decoding: Viterbi Algorithm
$$\delta_t(i) = \max_{Q_1,\dots,Q_{t-1}} P(Q_1,\dots,Q_t=S_i,O_1,\dots,O_t \mid \lambda)$$

Back-pointer: $\psi_t(j) = \arg\max_i \delta_{t-1}(i)\, a_{ij}$

### Problem 3 — Learning: Baum-Welch (EM)

**E-step** — state occupancy $\gamma_t(i)$ and transition occupancy $\xi_t(i,j)$

**M-step** — closed-form Gaussian re-estimation:

$$\mu_j^{\text{new}} = \frac{\sum_t \gamma_t(j)\, O_t}{\sum_t \gamma_t(j)}, \qquad \sigma_j^{2,\text{new}} = \frac{\sum_t \gamma_t(j)\,(O_t - \mu_j^{\text{new}})^2}{\sum_t \gamma_t(j)}$$

---

## Quick Start

```bash
git clone https://github.com/<your-username>/market-regime-detection
cd market-regime-detection
pip install -r requirements.txt
jupyter notebook
```

Run notebooks in order: `01` → `02` → `03` → `04`.

---

## References

1. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257–286.

2. Nguyen, N., & Nguyen, D. (2015). Hidden Markov Model for Stock Selection. *Risks*, 3(4), 455–473. https://doi.org/10.3390/risks3040455

3. McGreevy, J. (2021). *Hidden Markov Models in Finance*. Imperial College London MSc dissertation. https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/212236006---James-Mc-Greevy---MCGREEVY_JAMES_01075416.pdf

4. Paolucci, R. (2025). *Hidden Markov Models for Quantitative Finance*. Quant Guild. (Reference PDF — notation used throughout this project.)

5. Chen, X. (2025). HMM-based market regime detection with RL for portfolio management. *IDS*.

6. Nguyen, T. et al. (2024). Market Regime Detection: From Hidden Markov Models to Wasserstein Clustering. *Hikmath Technologies Publication*. https://publication.hikmahtechnologies.com/market-regime-detection-from-hidden-markov-models-to-wasserstein-clustering-6ba0a09559dc

7. Hassan, R. et al. (2005). A combination of hidden Markov model and fuzzy model for stock market forecasting. *Neurocomputing*.

8. LSEG API Samples. Market Regime Detection Using Statistical and ML-Based Approaches. https://github.com/LSEG-API-Samples/Article.RD.Python.MarketRegimeDetectionUsingStatisticalAndMLBasedApproaches

9. Liechty, J. C. et al. (2008). Portfolio selection with higher moments. *Quantitative Finance*. (SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3406068)

10. QuantStart. Market Regime Detection Using Hidden Markov Models in QSTrader. https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/

---

## License

MIT License — free to use, modify, and distribute.
