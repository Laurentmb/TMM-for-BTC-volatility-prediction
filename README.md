
# Bitcoin Volatility Forecasting with Temporal Mixture Models

An exploratory study of Bitcoin volatility prediction using historical volatility,
order-book features and market sentiment. The project compares temporal mixture
models with econometric baselines, gradient-boosted trees and recurrent neural
networks.

**Main question:** can order-book information and sentiment improve volatility
prediction beyond what is explained by past volatility alone?

## Data

The repository includes processed hourly observations from June to September 2018:

- `Data/hourly_data_ff.csv`: volatility (`vol`) and order-book features, including
  spreads, bid/ask depths, volumes and slopes.
- `Data/hourly_data_ff_with_returns.csv`: additional return data used by GARCH.
- `Data/sentiment.csv`: sentiment indicators aligned with the hourly observations.

`Data/SPP.py` documents a sentiment-processing step using Fear & Greed and CBBI
indicators from the [LLMs Sentiment-Augmented Bitcoin Dataset](https://huggingface.co/datasets/danilocorsi/LLMs-Sentiment-Augmented-Bitcoin-Dataset).
It constructs an aggregate indicator and an additional regression-residual feature.
The script works from daily source data; the repository also includes the processed
hourly sentiment file used in the experiments.

## Analysis workflow

1. **Examine stationarity.** Apply a KPSS test to the volatility series. The saved
   output rejects level stationarity at the 5% threshold, motivating attention to
   changes in market conditions.
2. **Explore predictive relationships.** Run Granger tests over lags of 1–30 hours
   to screen order-book variables and sentiment. These tests assess incremental
   predictive information; they do not establish economic causation.
3. **Compare feature sets.** Investigate all order-book variables, a reduced set
   selected by the exploratory tests, and sentiment-augmented variants.
4. **Fit the models.** Use successive temporal blocks or rolling windows, with
   model-specific validation and hyperparameter searches.
5. **Evaluate predictions.** Compute RMSE and MAE, plot predicted versus observed
   volatility, and inspect the mixture weights or hybrid-model weights over time.

The saved Granger tests identify five candidate order-book variables:
`ask_depth`, `bid_depth`, `bid_volume`, `spread` and `volume_difference`.
The sentiment test reports significant results at lags 1 and 2. These are
exploratory findings from the supplied sample, rather than validated feature
selection for an independent future dataset.

## Models

| Model | Role in the study |
| --- | --- |
| **Gaussian temporal mixture (`TM_G`)** | Combines a historical-volatility component and an order-book component using a learned, time-varying gate. |
| **Sentiment temporal mixture (`TM_SG`)** | Adds a third component based on sentiment history and learns its relative weight. |
| **Inverse-Gamma temporal mixture (`TM_IG`)** | Additional experimental implementation using positive-support component distributions; not included in the main notebook's comparison sections. |
| **HAR / HARX** | Linear baselines using volatility averages over multiple horizons, optionally augmented with order-book features. |
| **GARCH** | Econometric baseline forecasting conditional volatility from return history. |
| **Dual LSTM** | Processes volatility history and order-book sequences in separate recurrent branches before combining them. |
| **LSTM–GARCH** | Learns a dynamic weight between a neural prediction and a GARCH forecast. |
| **XGBoost** | Nonlinear tree-based baseline tested with different combinations of market and sentiment features. |

The temporal mixture models make the relative use of each input source visible
through their gating weights. These weights describe the model's combination of
components; they are not causal explanations.

## Repository structure

| Location | Contents |
| --- | --- |
| `main.ipynb` | Main experiment notebook: model fitting, feature comparisons and prediction plots. |
| `Data/GCT.ipynb` | Exploratory Granger tests and saved outputs. |
| `Data/KPSS.ipynb` | Stationarity test and saved output. |
| `Data/SPP.py` | Sentiment-processing script; requires an output path. |
| `Models/` | Model implementations and hyperparameter-search helpers. |
| `Model_eval/` | Evaluation functions for temporal mixture and LSTM models. |

## Exploring the project

Start with the diagnostic notebooks, then read the corresponding sections of
`main.ipynb` alongside the implementations in `Models/`.

The code uses Python, Jupyter, NumPy, pandas, Matplotlib, PyTorch,
scikit-learn, statsmodels, `arch`, XGBoost and tqdm. Run `main.ipynb` from the
repository root; the diagnostic notebooks use paths relative to `Data/`.


## Attribution

This repository is a fork of
[AugusteDP-git/bitcoin-volatility-forecasting](https://github.com/AugusteDP-git/bitcoin-volatility-forecasting).
The original implementation and commit history are retained, with credit to the
project's contributors. This README describes the code currently available in
the repository.
