# Walmart Retail Demand Forecasting

### Time-Series Machine Learning for Inventory Planning and High-Demand Alerts

An end-to-end data science project using the **M5 Walmart Forecasting dataset** to predict daily retail demand, flag unusually high-demand periods, and translate model outputs into interactive inventory-planning insights using **Tableau** and **Streamlit**.

**[GitHub repository](https://github.com/shirleyamponsah5-maker/M5-Retail-Demand-Forecasting)** · **[Tableau Public Story](https://public.tableau.com/shared/CSS9575KQ?:display_count=n&:origin=viz_share_link)** · **[Live Streamlit app](https://m5-retail-demand-forecasting-shirley.streamlit.app/)**

> **Project scope:** Historical backtesting and decision-support demonstration. The interactive Streamlit prediction selects observations from the held-out historical test period; it is not a production forecast for arbitrary future dates.

## 1. Business Problem

Retailers need to anticipate customer demand to make effective replenishment decisions. Underestimating demand can lead to stockouts and lost sales; overestimating it can lead to excess inventory and avoidable holding costs.

This project addresses two complementary questions:

1. **How many units will customers demand?** — regression-based demand forecasting.
2. **When is demand likely to be unusually high?** — classification-based high-demand alerts.

The outputs are presented in dashboards designed to support store- and category-level inventory monitoring.

## 2. Dataset and Scope

- **Source:** [M5 Forecasting dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data), based on Walmart sales data.
- **Inputs:** Daily unit sales, calendar information/events, and selling prices.
- **Coverage:** 10 stores across California (CA), Texas (TX), and Wisconsin (WI).
- **Analysis level:** 30 store–product-category series (10 stores × 3 categories: FOODS, HOBBIES, HOUSEHOLD).
- **History:** 1,913 daily observations per series.
- **Held-out test window:** March 28–April 24, 2016 (28 days; 840 store-category-day observations).

Large raw data files are not included in this repository. Download the source dataset separately if you wish to reproduce the full pipeline.

## 3. Workflow

1. **Data preparation:** Clean and join sales, calendar, and price information; aggregate daily demand to store–category level.
2. **Exploratory analysis:** Investigate sales trends, category mix, weekday seasonality, and store-level differences.
3. **Feature engineering:** Create historical demand lags, rolling demand statistics, calendar and seasonal indicators, pricing, and store/category features.
4. **Time-series splitting:** Use chronological training, validation, and test periods rather than a random split.
5. **Regression benchmarking:** Compare a seasonal-naive baseline, linear regression, and tree-based ensemble models.
6. **Model tuning:** Tune and evaluate XGBoost, using WAPE as the principal forecasting selection metric.
7. **High-demand classification:** Train a classifier, inspect the confusion matrix and ROC curve, and explore the precision–recall trade-off through a configurable threshold.
8. **Decision-support delivery:** Present results through Tableau and an interactive Streamlit application.

## 4. Model Results

### Demand forecasting

| Metric | Tuned XGBoost — final test |
|---|---:|
| WAPE | **8.61%** |
| MAE | **121.36 units** |
| RMSE | **182.67 units** |
| R² | **0.976** |

The validation WAPE improved from **12.39%** for the seasonal-naive baseline to **9.14%** for the original XGBoost model and **9.12%** after tuning. The tuned model was then assessed on the untouched 28-day test window.

**Interpretation:** A test WAPE of 8.61% means that the sum of absolute forecast errors was about 8.61% of total observed demand over the evaluated test observations. It does not mean that every individual prediction was within 8.61% of actual demand.

### High-demand classification

| Metric | Final test result at threshold 0.25 |
|---|---:|
| Accuracy | **80.24%** |
| Precision | **72.64%** |
| Recall | **87.05%** |
| ROC-AUC | **0.9095** |
| True positives | 316 |
| False negatives | 47 |
| False positives | 119 |
| True negatives | 358 |

The **0.25 probability threshold** prioritizes recall to reduce missed high-demand observations. Raising or lowering the threshold changes the trade-off between missed surges and false alerts; the Streamlit app allows users to explore this interactively.

## 5. Key Findings

- **Historical demand matters most:** Demand lags and rolling statistics together account for approximately **80%** of the forecasting model's reported feature importance; the 28-day lag is the strongest individual feature (**42.92%**).
- **Category mix differs:** FOODS accounts for approximately **68.63%** of observed demand in the analysis.
- **Weekly seasonality is meaningful:** Saturday demand was approximately **38.4% higher than Wednesday** in the exploratory analysis.
- **Inventory decisions need two perspectives:** Unit forecasts help estimate replenishment quantities; high-demand alerts help identify periods requiring closer monitoring.

Feature importance indicates predictive contribution within the fitted model; it should not be interpreted as proof of causation.

## 6. Interactive Dashboards

### Tableau — Inventory Planning and Story

The Tableau workbook presents the project narrative and interactive inventory-planning views, including:

- Forecasted demand by store and product category.
- Forecasting risk / WAPE by store.
- Predicted high-demand alert rates.
- State and category filters for exploring local inventory priorities.

**Workbook:** [`tableau/Walmart Retail Demand Forecasting - Inventory Planning.twbx`](tableau/Walmart%20Retail%20Demand%20Forecasting%20-%20Inventory%20Planning.twbx)

**Tableau Public Story:** [Open interactive story](https://public.tableau.com/shared/CSS9575KQ?:display_count=n&:origin=viz_share_link)

### Streamlit — Model Exploration

The Streamlit application provides:

- Project overview and forecasting results.
- An **XGBoost live prediction demo** for a selected historical test observation.
- High-demand alert exploration.
- An **interactive classification threshold slider** (default: 0.25).
- A **ROC curve** and model-performance views.

**App source:** [`streamlit_app/app.py`](streamlit_app/app.py)

**Live application:** [Open Streamlit dashboard](https://m5-retail-demand-forecasting-shirley.streamlit.app/)

## 7. Repository Guide

| Location | Purpose |
|---|---|
| `notebooks/` | Analysis and machine-learning notebooks |
| `streamlit_app/app.py` | Interactive Streamlit application |
| `tableau/` | Tableau packaged workbook (`.twbx`) |
| `dashboard_data/` | Prepared dashboard inputs, where available locally |
| `models/` | Saved model artifacts, where available |
| `requirements.txt` | Python dependencies |

**Note:** Some datasets and generated artifacts may be intentionally excluded from GitHub because of file size or reproducibility considerations. The repository may not contain every local input used during development.

## 8. Running the Streamlit App Locally

Clone the repository and create a Python environment:

```bash
git clone https://github.com/shirleyamponsah5-maker/M5-Retail-Demand-Forecasting.git
cd M5-Retail-Demand-Forecasting
python -m venv .venv
```

Activate the environment (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies and start the app:

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

**Important:** The app may require prepared CSV files and trained model artifacts from the project pipeline. Ensure the expected files are available at the paths referenced by `streamlit_app/app.py`; cloning the repository alone may not be sufficient if those assets are excluded.

## 9. Limitations and Future Work

- Predictions are evaluated at **store–category** level, not individual SKU level.
- The current analysis does not incorporate actual on-hand inventory, replenishment lead times, or supplier constraints.
- Forecast-time availability of engineered inputs must be validated before genuine future-date deployment.
- Results are based on historical backtesting and may change as retail patterns shift.

**Next steps:** Strengthen forecast-origin feature validation, test rolling-origin backtesting, integrate inventory and supplier data, automate monitoring/retraining, and evaluate inventory decisions using operational cost metrics.

## 10. Tools and Technologies

**Python · pandas · NumPy · scikit-learn · XGBoost · Jupyter Notebook · Tableau · Streamlit · Plotly · Git · GitHub**

---

**Project:** Walmart Retail Demand Forecasting — End-to-End Data Science and Inventory Decision Support.

