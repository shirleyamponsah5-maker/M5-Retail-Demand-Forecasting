
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import r2_score, roc_curve, roc_auc_score
import joblib

# --------------------------------------------------
# 1. Configure the Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="Walmart Retail Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dashboard_data"

# Load the final trained XGBoost forecasting model
@st.cache_resource
def load_forecasting_model():
    model_path = PROJECT_ROOT / "models" / "final_xgboost_forecasting.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    return joblib.load(model_path)

forecast_model_bundle = load_forecasting_model()
forecast_model = forecast_model_bundle["model"]
forecast_feature_columns = forecast_model_bundle["feature_columns"]


# Load prepared historical test inputs for live model inference
@st.cache_data
def load_live_prediction_inputs():
    file_path = DATA_DIR / "live_prediction_inputs.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Live prediction inputs not found: {file_path}"
        )

    data = pd.read_csv(file_path)
    data["date"] = pd.to_datetime(data["date"])

    return data


live_prediction_inputs = load_live_prediction_inputs()

# Check that all 78 required model features are available
missing_features = [
    col for col in forecast_feature_columns
    if col not in live_prediction_inputs.columns
]

if missing_features:
    st.error(f"Missing model features: {missing_features}")
    st.stop()


# --------------------------------------------------
# 3. Load the exported dashboard datasets
# --------------------------------------------------

@st.cache_data
def load_data():

    required_files = {
        "forecast": DATA_DIR / "forecast_predictions.csv",
        "classification": DATA_DIR / "high_demand_predictions.csv",
        "metrics": DATA_DIR / "model_performance_metrics.csv"
    }

    # Check that all required files exist before loading.
    for name, file_path in required_files.items():
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required dashboard dataset not found: {file_path}"
            )

    # Load forecasting predictions.
    forecast = pd.read_csv(
        required_files["forecast"],
        parse_dates=["date"]
    )

    # Load high-demand classification predictions.
    classification = pd.read_csv(
        required_files["classification"],
        parse_dates=["date"]
    )

    # Load final model performance metrics.
    metrics = pd.read_csv(
        required_files["metrics"]
    )

    return forecast, classification, metrics

try:
    forecast_df, classification_df, metrics_df = load_data()

except (FileNotFoundError, OSError, ValueError) as error:
    st.error("The dashboard could not load its required datasets.")
    st.error(str(error))
    st.stop()

# --------------------------------------------------
# 4. Create sidebar navigation
# --------------------------------------------------

st.sidebar.title("Walmart Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Demand Forecasting",
        "High-Demand Alerts",
        "Model Performance",
        "Business Insights"
    ]
)

# --------------------------------------------------
# 5. Dashboard title
# --------------------------------------------------

st.title("Walmart Retail Demand Forecasting")

st.caption(
    "Machine Learning | Time Series Forecasting | "
    "Inventory Decision Support"
)

# --------------------------------------------------
# 6. Project overview
# --------------------------------------------------

if page == "Project Overview":

    st.header("Project Overview")

    st.write(
        "Explore the performance of machine learning models "
        "developed to forecast Walmart retail demand and "
        "identify unusually high-demand observations."
    )

    st.caption(
        "Historical test period: March 28 – April 24, 2016 | "
        "30 store-category combinations"
    )

    # ------------------------------------------
    # 6.1 Interactive filters
    # ------------------------------------------

    st.subheader("Dashboard Filters")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        selected_stores = st.multiselect(
            "Select Walmart Stores",
            options=sorted(forecast_df["store_id"].unique()),
            default=sorted(forecast_df["store_id"].unique())
        )

    with filter_col2:
        selected_categories = st.multiselect(
            "Select Product Categories",
            options=sorted(forecast_df["category"].unique()),
            default=sorted(forecast_df["category"].unique())
        )

    filtered_forecast = forecast_df[
        forecast_df["store_id"].isin(selected_stores)
        & forecast_df["category"].isin(selected_categories)
    ].copy()

    filtered_classification = classification_df[
        classification_df["store_id"].isin(selected_stores)
        & classification_df["category"].isin(selected_categories)
    ].copy()

    if filtered_forecast.empty:
        st.warning(
            "No observations match the selected filters. "
            "Please select at least one store and category."
        )
        st.stop()

    # ------------------------------------------
    # 6.2 Calculate filtered KPI metrics
    # ------------------------------------------

    actual = filtered_forecast["actual_demand"]
    predicted = filtered_forecast["predicted_demand"]

    mae = (actual - predicted).abs().mean()

    total_actual = actual.abs().sum()

    wape = (
        (actual - predicted).abs().sum() / total_actual * 100
        if total_actual > 0
        else np.nan
    )

    high_demand_recall = (
        filtered_classification.loc[
            (filtered_classification["actual_high_demand"] == 1)
            & (filtered_classification["predicted_high_demand"] == 1)
        ].shape[0]
        /
        filtered_classification.loc[
            filtered_classification["actual_high_demand"] == 1
        ].shape[0]
        if (
            filtered_classification["actual_high_demand"] == 1
        ).any()
        else None
    )

    # ------------------------------------------
    # 6.3 Display KPI cards
    # ------------------------------------------

    st.subheader("Model Performance — Selected Test Data")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
      st.metric(
        "Forecast WAPE",
        f"{wape:.2f}%" if np.isfinite(wape) else "N/A"
    )

    with kpi2:
        st.metric(
            "Forecast MAE",
            f"{mae:,.2f}"
        )

    with kpi3:
        st.metric(
            "High-Demand Recall",
            (
                f"{high_demand_recall:.2%}"
                if high_demand_recall is not None
                else "N/A"
            )
        )

    with kpi4:
        st.metric(
            "Test Observations",
            f"{len(filtered_forecast):,}"
        )

    st.divider()

    # ------------------------------------------
    # 6.4 Actual vs Predicted Demand Chart
    # ------------------------------------------

    st.subheader("Actual vs Predicted Retail Demand")

    daily_forecast = (
        filtered_forecast
        .groupby("date", as_index=False)[
            ["actual_demand", "predicted_demand"]
        ]
        .sum()
    )

    daily_forecast_long = daily_forecast.melt(
        id_vars="date",
        value_vars=[
            "actual_demand",
            "predicted_demand"
        ],
        var_name="Demand Type",
        value_name="Demand Units"
    )

    daily_forecast_long["Demand Type"] = (
        daily_forecast_long["Demand Type"].replace({
            "actual_demand": "Actual Demand",
            "predicted_demand": "Predicted Demand"
        })
    )

    fig = px.line(
        daily_forecast_long,
        x="date",
        y="Demand Units",
        color="Demand Type",
        markers=True,
        title="Daily Actual vs Predicted Demand"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Total Demand Units",
        legend_title="",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "The chart compares actual and predicted demand "
        "across the selected Walmart stores and categories. "
        "Use the filters above to investigate individual "
        "stores or product categories."
    )

    # ------------------------------------------
    # 6.5 Demand Distribution by Category
    # ------------------------------------------

    st.subheader("Actual Demand by Product Category")

    category_demand = (
        filtered_forecast
        .groupby("category", as_index=False)["actual_demand"]
        .sum()
    )

    category_fig = px.bar(
        category_demand,
        x="category",
        y="actual_demand",
        color="category",
        title="Total Actual Demand by Category",
        labels={
            "category": "Product Category",
            "actual_demand": "Total Demand Units"
        }
    )

    category_fig.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        category_fig,
        use_container_width=True
    )

    # ------------------------------------------
    # 6.6 Business Interpretation
    # ------------------------------------------

    st.subheader("Business Interpretation")

    st.write(
        "This dashboard provides an overview of the "
        "final machine learning results. "
        "The demand forecasting model estimates daily "
        "sales quantities, while the classification model "
        "identifies observations with unusually high demand."
    )

    st.write(
        "Together, these predictions could support "
        "inventory monitoring, replenishment planning, "
        "and more informed retail decision-making."
    )

    st.caption(
        "All results represent historical test predictions, "
        "not live sales or real-time inventory alerts."
    )


# --------------------------------------------------
# 7. Detailed Demand Forecasting Dashboard
# --------------------------------------------------

elif page == "Demand Forecasting":

    st.header("Demand Forecasting Analytics")

    st.write(
        "Evaluate the performance of the final Tuned XGBoost "
        "Regressor across Walmart stores, product categories, "
        "and the 28-day historical test period."
    )

    st.caption(
        "Test period: March 28 – April 24, 2016 | "
        "Model: Tuned XGBoost Regressor"
    )

    # ------------------------------------------
    # 7.1 Interactive filters
    # ------------------------------------------

    st.subheader("Forecast Filters")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        forecast_stores = st.multiselect(
            "Select Stores",
            options=sorted(forecast_df["store_id"].unique()),
            default=sorted(forecast_df["store_id"].unique()),
            key="forecast_stores"
        )

    with filter_col2:
        forecast_categories = st.multiselect(
            "Select Categories",
            options=sorted(forecast_df["category"].unique()),
            default=sorted(forecast_df["category"].unique()),
            key="forecast_categories"
        )

    # Date filter
    min_date = forecast_df["date"].min().date()
    max_date = forecast_df["date"].max().date()

    selected_dates = st.date_input(
        "Select Forecast Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="forecast_dates"
    )

    # Streamlit may temporarily return one date
    # while the user is selecting a range.
    if not isinstance(selected_dates, (tuple, list)):
        st.info("Please select a start and end date.")
        st.stop()

    if len(selected_dates) != 2:
        st.info("Please select both dates.")
        st.stop()

    start_date, end_date = selected_dates

    # ------------------------------------------
    # 7.2 Apply filters
    # ------------------------------------------

    filtered = forecast_df[
        forecast_df["store_id"].isin(forecast_stores)
        & forecast_df["category"].isin(forecast_categories)
        & (
            forecast_df["date"].dt.date >= start_date
        )
        & (
            forecast_df["date"].dt.date <= end_date
        )
    ].copy()

    if filtered.empty:
        st.warning(
            "No observations match the selected filters. "
            "Please adjust your selection."
        )
        st.stop()

    # ------------------------------------------
    # 7.3 Calculate forecast performance
    # ------------------------------------------

    actual = filtered["actual_demand"].to_numpy()
    predicted = filtered["predicted_demand"].to_numpy()

    errors = actual - predicted

    mae = np.mean(np.abs(errors))

    rmse = np.sqrt(np.mean(errors ** 2))

    actual_total = np.abs(actual).sum()

    wape = (
        np.abs(errors).sum() / actual_total * 100
        if actual_total > 0
        else np.nan
    )

    r2 = (
        r2_score(actual, predicted)
        if len(actual) >= 2
        else np.nan
    )

    # ------------------------------------------
    # 7.4 Performance KPI cards
    # ------------------------------------------

    st.subheader("Forecast Performance")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric("MAE", f"{mae:,.2f}")

    with kpi2:
        st.metric("RMSE", f"{rmse:,.2f}")

    with kpi3:
        st.metric(
            "WAPE",
            f"{wape:.2f}%" if np.isfinite(wape) else "N/A"
        )

    with kpi4:
        st.metric(
            "R²",
            f"{r2:.3f}" if np.isfinite(r2) else "N/A"
        )

    st.caption(
        f"Metrics calculated from {len(filtered):,} "
        "selected store-category-day observations. "
        "They change when filters are applied."
    )

    st.divider()

    # ------------------------------------------
    # 7.5 Actual vs Predicted Demand
    # ------------------------------------------

    st.subheader("Actual vs Predicted Demand")

    daily = (
        filtered.groupby("date", as_index=False)[
            ["actual_demand", "predicted_demand"]
        ].sum()
    )

    daily_long = daily.melt(
        id_vars="date",
        value_vars=["actual_demand", "predicted_demand"],
        var_name="Demand Type",
        value_name="Demand Units"
    )

    daily_long["Demand Type"] = (
        daily_long["Demand Type"].replace({
            "actual_demand": "Actual Demand",
            "predicted_demand": "Predicted Demand"
        })
    )

    fig_forecast = px.line(
        daily_long,
        x="date",
        y="Demand Units",
        color="Demand Type",
        markers=True,
        title="Daily Actual vs Predicted Demand"
    )

    fig_forecast.update_layout(
        xaxis_title="Date",
        yaxis_title="Demand Units",
        hovermode="x unified",
        legend_title=""
    )

    st.plotly_chart(
        fig_forecast,
        use_container_width=True
    )

    st.write(
        "This chart shows how closely predicted demand "
        "follows observed demand over the selected dates. "
        "Smaller gaps between the lines indicate more "
        "accurate forecasts."
    )

    # ------------------------------------------
    # 7.6 Forecast Error Over Time
    # ------------------------------------------

    st.subheader("Forecast Error Analysis")

    daily["forecast_error"] = (
        daily["actual_demand"]
        - daily["predicted_demand"]
    )

    fig_error = px.bar(
        daily,
        x="date",
        y="forecast_error",
        title="Daily Forecast Error",
        labels={
            "date": "Date",
            "forecast_error": "Forecast Error (Units)"
        }
    )

    fig_error.add_hline(
        y=0,
        line_dash="dash"
    )

    st.plotly_chart(
        fig_error,
        use_container_width=True
    )

    st.info(
        "Positive forecast errors indicate underprediction "
        "(actual demand exceeded predicted demand). "
        "Negative errors indicate overprediction."
    )

    # ------------------------------------------
    # 7.7 Forecast Performance by Category
    # ------------------------------------------

    st.subheader("Forecast Accuracy by Product Category")

    category_performance = (
        filtered.groupby("category")
        .agg(
            total_actual=("actual_demand", "sum"),
            total_absolute_error=("absolute_error", "sum")
        )
        .reset_index()
    )

    category_performance["WAPE"] = np.where(
        category_performance["total_actual"] > 0,
        (
            category_performance["total_absolute_error"]
            / category_performance["total_actual"]
        ) * 100,
        np.nan
    )

    fig_category = px.bar(
        category_performance,
        x="category",
        y="WAPE",
        color="category",
        title="Forecast WAPE by Product Category",
        labels={
            "category": "Product Category",
            "WAPE": "WAPE (%)"
        }
    )

    fig_category.update_layout(showlegend=False)

    st.plotly_chart(
        fig_category,
        use_container_width=True
    )

    st.write(
        "Lower WAPE values indicate better relative "
        "forecast accuracy. Comparing categories helps "
        "identify where forecasting performance may "
        "require further improvement."
    )

    # ------------------------------------------
    # 7.8 Detailed Forecast Table
    # ------------------------------------------

    st.subheader("Detailed Forecast Predictions")

    display_columns = [
        "date",
        "store_id",
        "category",
        "actual_demand",
        "predicted_demand",
        "forecast_error",
        "absolute_error"
    ]

    st.dataframe(
        filtered[display_columns].sort_values(
            ["date", "store_id", "category"]
        ),
        use_container_width=True,
        hide_index=True
    )

    # ------------------------------------------
    # 7.9 Download Filtered Predictions
    # ------------------------------------------

    csv_data = filtered[display_columns].to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Forecast Predictions (CSV)",
        data=csv_data,
        file_name="filtered_forecast_predictions.csv",
        mime="text/csv"
    )

    # ------------------------------------------
    # 7.10 Business Interpretation
    # ------------------------------------------

    st.subheader("Business Interpretation")

    st.write(
        "The forecasting dashboard helps identify "
        "differences between expected and observed "
        "retail demand. These insights could support "
        "inventory planning, replenishment assessments, "
        "and targeted monitoring of stores or categories "
        "with higher forecast errors."
    )

    st.caption(
        "Results are based on historical test data. "
        "This page does not generate live forecasts."
    )

    # ---------------------------------------------
    # LIVE PREDICTION DEMO
    # ---------------------------------------------
    st.divider()
    st.header("Live XGBoost Prediction Demo")

    st.info(
        "Generate a new model prediction using a selected "
        "historical test observation (March 28–April 24, 2016)."
    )

    # Select a store
    selected_store = st.selectbox(
        "Select Walmart Store",
        sorted(live_prediction_inputs["store_id"].unique()),
        key="demo_store"
    )

    # Select a product category
    selected_category = st.selectbox(
        "Select Product Category",
        sorted(live_prediction_inputs["category"].unique()),
        key="demo_category"
    )

    # Filter available historical observations
    demo_filtered = live_prediction_inputs[
        (live_prediction_inputs["store_id"] == selected_store)
        & (live_prediction_inputs["category"] == selected_category)
    ].copy()

    # Select a date
    available_dates = sorted(demo_filtered["date"].dt.date.unique())

    selected_date = st.selectbox(
        "Select Historical Test Date",
        available_dates,
        key="demo_date"
    )

    # Generate a fresh prediction when the button is clicked
    if st.button("Generate Prediction", type="primary"):

        selected_row = demo_filtered[
            demo_filtered["date"].dt.date == selected_date
        ].iloc[[0]]

        # Use the exact 78 features and original training order
        model_input = selected_row[forecast_feature_columns].copy()

        # Generate prediction using the saved trained model
        prediction = float(forecast_model.predict(model_input)[0])

        actual = float(selected_row["demand"].iloc[0])
        error = actual - prediction

        st.success("Prediction generated successfully!")

        col1, col2, col3 = st.columns(3)

        col1.metric("Predicted Demand", f"{prediction:,.0f} units")
        col2.metric("Actual Demand", f"{actual:,.0f} units")
        col3.metric("Forecast Error", f"{error:+,.0f} units")

        if error > 0:
            st.warning(
                "The model underpredicted demand. "
                "This could indicate a potential stockout risk."
            )
        elif error < 0:
            st.info(
                "The model overpredicted demand. "
                "This could indicate potential excess inventory."
            )
        else:
            st.success("Predicted demand matches actual demand.")


# --------------------------------------------------
# 8. High-Demand Alerts Dashboard
# --------------------------------------------------

elif page == "High-Demand Alerts":

    st.header("High-Demand Alert Analytics")

    st.write(
        "Explore historical high-demand predictions generated "
        "by the final XGBoost Classifier. Identify store-category "
        "combinations associated with elevated demand risk."
    )

    st.caption(
        "Test period: March 28 – April 24, 2016 | "
        "Final classification threshold: 0.25"
    )

    # ------------------------------------------
    # 8.1 Interactive filters
    # ------------------------------------------

    st.subheader("High-Demand Filters")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        alert_stores = st.multiselect(
            "Select Stores",
            options=sorted(classification_df["store_id"].unique()),
            default=sorted(classification_df["store_id"].unique()),
            key="alert_stores"
        )

    with filter_col2:
        alert_categories = st.multiselect(
            "Select Categories",
            options=sorted(classification_df["category"].unique()),
            default=sorted(classification_df["category"].unique()),
            key="alert_categories"
        )

    min_date = classification_df["date"].min().date()
    max_date = classification_df["date"].max().date()

    alert_dates = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="alert_dates"
    )

    if (
        not isinstance(alert_dates, (tuple, list))
        or len(alert_dates) != 2
    ):
        st.info("Please select a start and end date.")
        st.stop()

    start_date, end_date = alert_dates

    filtered_alerts = classification_df[
        classification_df["store_id"].isin(alert_stores)
        & classification_df["category"].isin(alert_categories)
        & (classification_df["date"].dt.date >= start_date)
        & (classification_df["date"].dt.date <= end_date)
    ].copy()

    if filtered_alerts.empty:
        st.warning(
            "No observations match your filters. "
            "Please select at least one store and category."
        )
        st.stop()

    # ------------------------------------------
    # 8.2 Calculate classification metrics
    # ------------------------------------------

    actual = filtered_alerts["actual_high_demand"].astype(int)
    predicted = filtered_alerts["predicted_high_demand"].astype(int)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())

    total = len(filtered_alerts)

    accuracy = (tp + tn) / total

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else None
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else None
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and (precision + recall) > 0
        else None
    )

    # ------------------------------------------
    # 8.3 KPI cards
    # ------------------------------------------

    st.subheader("Classification Performance")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric("Accuracy", f"{accuracy:.2%}")

    with kpi2:
        st.metric(
            "Precision",
            f"{precision:.2%}" if precision is not None else "N/A"
        )

    with kpi3:
        st.metric(
            "Recall",
            f"{recall:.2%}" if recall is not None else "N/A"
        )

    with kpi4:
        st.metric(
            "F1-Score",
            f"{f1:.2%}" if f1 is not None else "N/A"
        )

    st.caption(
        f"Metrics calculated from {total:,} selected observations. "
        "The classifier's final decision threshold is fixed at 0.25."
    )

    st.divider()

    # ------------------------------------------
    # 8.4 Confusion matrix
    # ------------------------------------------

    st.subheader("Confusion Matrix")

    confusion_matrix = np.array([
        [tn, fp],
        [fn, tp]
    ])

    fig_cm = px.imshow(
        confusion_matrix,
        x=["Predicted Normal", "Predicted High"],
        y=["Actual Normal", "Actual High"],
        text_auto=True,
        color_continuous_scale="Blues",
        title="High-Demand Classification Confusion Matrix",
        aspect="auto"
    )

    fig_cm.update_layout(
        xaxis_title="Predicted Demand Class",
        yaxis_title="Actual Demand Class"
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )

    st.write(
        "**True Positive:** A high-demand observation correctly "
        "identified by the model."
    )

    st.write(
        "**False Positive:** The model predicted high demand, "
        "but actual demand was normal."
    )

    st.write(
        "**False Negative:** Actual demand was high, "
        "but the model did not flag it."
    )

    st.write(
        "**True Negative:** Normal demand correctly identified."
    )

    # ------------------------------------------
    # 8.5 High-demand probability distribution
    # ------------------------------------------

    st.subheader("High-Demand Probability Distribution")

    fig_probability = px.histogram(
        filtered_alerts,
        x="high_demand_probability",
        color="actual_high_demand",
        nbins=30,
        barmode="overlay",
        opacity=0.7,
        title="Predicted High-Demand Probabilities",
        labels={
            "high_demand_probability": "High-Demand Probability",
            "actual_high_demand": "Actual High Demand"
        }
    )

    fig_probability.add_vline(
        x=0.25,
        line_dash="dash",
        line_color="red",
        annotation_text="Decision Threshold = 0.25"
    )

    st.plotly_chart(
        fig_probability,
        use_container_width=True
    )

    st.info(
        "The model classifies an observation as high demand "
        "when its predicted probability is at least 0.25. "
        "This threshold was selected during model validation "
        "to prioritize identifying high-demand observations."
    )

    # ------------------------------------------
    # 8.5A Interactive threshold and ROC analysis
    # ------------------------------------------
    st.divider()
    st.subheader("Interactive Classification Threshold Analysis")
    st.write(
        "Explore how changing the probability cutoff affects historical "
        "high-demand detection. The deployed project decision threshold "
        "remains 0.25; changing this slider is for analysis only."
    )
    threshold = st.slider(
        "High-demand probability threshold", min_value=0.05,
        max_value=0.95, value=0.25, step=0.05,
        key="interactive_alert_threshold"
    )
    y_true = filtered_alerts["actual_high_demand"].astype(int).to_numpy()
    y_prob = pd.to_numeric(
        filtered_alerts["high_demand_probability"], errors="coerce"
    ).to_numpy()
    valid = np.isfinite(y_prob)
    if not valid.all():
        st.warning(f"Excluded {(~valid).sum()} rows with missing probability values.")
    y_true, y_prob = y_true[valid], y_prob[valid]
    if len(y_true) == 0:
        st.warning("No valid prediction probabilities in this selection.")
    else:
        y_adjusted = (y_prob >= threshold).astype(int)
        tp_new = int(((y_true == 1) & (y_adjusted == 1)).sum())
        fp_new = int(((y_true == 0) & (y_adjusted == 1)).sum())
        fn_new = int(((y_true == 1) & (y_adjusted == 0)).sum())
        tn_new = int(((y_true == 0) & (y_adjusted == 0)).sum())
        precision_new = tp_new / (tp_new + fp_new) if tp_new + fp_new else 0.0
        recall_new = tp_new / (tp_new + fn_new) if tp_new + fn_new else 0.0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision at threshold", f"{precision_new:.2%}")
        c2.metric("Recall at threshold", f"{recall_new:.2%}")
        c3.metric("Alerts", f"{tp_new + fp_new:,}")
        c4.metric("Missed high-demand", f"{fn_new:,}")
        st.caption(
            "Threshold changes recalculate results for the selected historical "
            "observations only; they do not retrain the model or alter exported alerts."
        )
        st.subheader("ROC Curve — High-Demand Classifier")
        if np.unique(y_true).size < 2:
            st.info("ROC requires both normal-demand and high-demand examples. "
                    "Broaden the store, category, or date filters.")
        else:
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
            auc_value = roc_auc_score(y_true, y_prob)
            fpr_selected = fp_new / (fp_new + tn_new) if fp_new + tn_new else 0.0
            tpr_selected = recall_new
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc_value:.4f})"
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random classifier",
                line=dict(dash="dash", color="gray")
            ))
            fig_roc.add_trace(go.Scatter(
                x=[fpr_selected], y=[tpr_selected], mode="markers",
                marker=dict(size=13, color="orange"),
                name=f"Threshold = {threshold:.2f}"
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate (Recall)",
                xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
                title="ROC Curve on Selected Historical Test Observations"
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.caption(
                "ROC-AUC is calculated from the currently selected test rows. "
                "It can differ from the full-test ROC-AUC of 0.9095 when filters are applied."
            )

    # ------------------------------------------
    # 8.6 High-demand alerts over time
    # ------------------------------------------

    st.subheader("Predicted High-Demand Alerts Over Time")

    daily_alerts = (
        filtered_alerts.groupby("date", as_index=False)
        ["predicted_high_demand"]
        .sum()
    )

    fig_alerts = px.bar(
        daily_alerts,
        x="date",
        y="predicted_high_demand",
        title="Number of Predicted High-Demand Alerts by Date",
        labels={
            "date": "Date",
            "predicted_high_demand": "Number of Alerts"
        }
    )

    st.plotly_chart(
        fig_alerts,
        use_container_width=True
    )

    st.caption(
        "Each alert represents one store-category-day "
        "observation classified as high demand."
    )

    # ------------------------------------------
    # 8.7 Alert summary by store
    # ------------------------------------------

    st.subheader("High-Demand Alerts by Store")

    store_alerts = (
        filtered_alerts.groupby("store_id", as_index=False)
        ["predicted_high_demand"]
        .sum()
        .sort_values("predicted_high_demand", ascending=False)
    )

    fig_store = px.bar(
        store_alerts,
        x="store_id",
        y="predicted_high_demand",
        color="store_id",
        title="Predicted High-Demand Alerts by Store",
        labels={
            "store_id": "Walmart Store",
            "predicted_high_demand": "Number of Alerts"
        }
    )

    fig_store.update_layout(showlegend=False)

    st.plotly_chart(
        fig_store,
        use_container_width=True
    )

    # ------------------------------------------
    # 8.8 Detailed alert table
    # ------------------------------------------

    st.subheader("Detailed High-Demand Alerts")

    alert_columns = [
        "date",
        "store_id",
        "state_id",
        "category",
        "demand",
        "high_demand_threshold",
        "high_demand_probability",
        "actual_high_demand",
        "predicted_high_demand",
        "classification_outcome",
        "demand_alert"
    ]

    high_risk = filtered_alerts[
        filtered_alerts["predicted_high_demand"] == 1
    ].copy()

    high_risk = high_risk.sort_values(
        "high_demand_probability",
        ascending=False
    )

    st.write(
        f"**{len(high_risk):,} predicted high-demand alerts** "
        "match the selected filters."
    )

    st.dataframe(
        high_risk[alert_columns],
        use_container_width=True,
        hide_index=True
    )

    # ------------------------------------------
    # 8.9 Download alerts
    # ------------------------------------------

    alert_csv = high_risk[alert_columns].to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download High-Demand Alerts (CSV)",
        data=alert_csv,
        file_name="high_demand_alerts_filtered.csv",
        mime="text/csv"
    )

    # ------------------------------------------
    # 8.10 Business interpretation
    # ------------------------------------------

    st.subheader("Business Interpretation")

    st.write(
        "High-demand classification can help retail teams "
        "identify store-category combinations that may "
        "require closer inventory monitoring. "
        "Prioritizing recall reduces the risk of overlooking "
        "actual high-demand observations, although it can "
        "also increase false-positive alerts."
    )

    st.caption(
        "This dashboard evaluates historical predictions. "
        "It does not provide real-time inventory alerts "
        "or independently validate future demand."
    )


elif page == "Model Performance":

    st.header("Model Performance Comparison")

    st.write(
        "This dashboard presents the final test performance "
        "of the Tuned XGBoost Regressor and XGBoost Classifier "
        "over March 28 – April 24, 2016."
    )

        # Reuse the model metrics already loaded and cached.
    performance_df = metrics_df.copy()

    required_columns = {"Model", "Task", "Metric", "Value"}

    if not required_columns.issubset(performance_df.columns):
        st.error("The model performance CSV has missing columns.")
        st.stop()

    performance_df["Value"] = pd.to_numeric(
        performance_df["Value"], errors="coerce"
    )

    st.divider()

    # -----------------------------------------------------
    # REGRESSION MODEL PERFORMANCE
    # -----------------------------------------------------

    st.subheader("1. Demand Forecasting Model")

    st.write("**Final Model: Tuned XGBoost Regressor**")

    regression_metrics = performance_df[
        performance_df["Task"] == "Demand Forecasting"
    ].copy()

    reg_values = dict(
        zip(regression_metrics["Metric"], regression_metrics["Value"])
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "MAE",
        f"{reg_values['MAE']:.2f}"
        if pd.notna(reg_values.get("MAE")) else "N/A"
    )

    col2.metric(
        "RMSE",
        f"{reg_values['RMSE']:.2f}"
        if pd.notna(reg_values.get("RMSE")) else "N/A"
    )

    col3.metric(
        "WAPE",
        f"{reg_values['WAPE']:.2f}%"
        if pd.notna(reg_values.get("WAPE")) else "N/A"
    )

    col4.metric(
        "R²",
        f"{reg_values['R2']:.3f}"
        if pd.notna(reg_values.get("R2")) else "N/A"
    )

    reg_chart = regression_metrics[
        regression_metrics["Metric"].isin(["MAE", "RMSE"])
    ]

    fig_reg = px.bar(
        reg_chart,
        x="Metric",
        y="Value",
        color="Metric",
        text="Value",
        title="Forecast Error Metrics",
        labels={"Value": "Demand Units"},
    )

    fig_reg.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig_reg.update_layout(
        showlegend=False,
        yaxis_range=[0, reg_chart["Value"].max() * 1.2]
    )

    st.plotly_chart(fig_reg, use_container_width=True)

    st.info(
        "MAE and RMSE measure forecast error in demand units. "
        "Lower values indicate better forecasting performance. "
        "WAPE expresses absolute forecast error relative to "
        "total actual demand, while R² measures explained variation."
    )

    st.divider()

    # -----------------------------------------------------
    # CLASSIFICATION MODEL PERFORMANCE
    # -----------------------------------------------------

    st.subheader("2. High-Demand Classification Model")

    st.write("**Final Model: XGBoost Classifier**")
    st.caption("Final classification decision threshold: 0.25")

    classification_metrics = performance_df[
        performance_df["Task"] == "High-Demand Classification"
    ].copy()

    clf_values = dict(
        zip(
            classification_metrics["Metric"],
            classification_metrics["Value"]
        )
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{clf_values['Accuracy']:.2%}"
        if pd.notna(clf_values.get("Accuracy")) else "N/A"
    )

    c2.metric(
        "Precision",
        f"{clf_values['Precision']:.2%}"
        if pd.notna(clf_values.get("Precision")) else "N/A"
    )

    c3.metric(
        "Recall",
        f"{clf_values['Recall']:.2%}"
        if pd.notna(clf_values.get("Recall")) else "N/A"
    )

    c4.metric(
        "F1-Score",
        f"{clf_values['F1']:.2%}"
        if pd.notna(clf_values.get("F1")) else "N/A"
    )

    st.metric(
        "ROC-AUC",
        f"{clf_values['ROC_AUC']:.4f}"
        if pd.notna(clf_values.get("ROC_AUC")) else "N/A"
    )

    clf_chart = classification_metrics.copy()
    clf_chart["Percentage"] = clf_chart["Value"] * 100

    fig_clf = px.bar(
        clf_chart,
        x="Metric",
        y="Percentage",
        color="Metric",
        text="Percentage",
        title="Classification Evaluation Metrics",
        labels={"Percentage": "Score (%)"},
    )

    fig_clf.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_clf.update_layout(
        showlegend=False,
        yaxis_range=[0, 110]
    )

    st.plotly_chart(fig_clf, use_container_width=True)

    st.info(
        "Recall is especially important for high-demand alerts "
        "because missed high-demand periods could contribute "
        "to inventory shortages. Higher recall can also increase "
        "false-positive alerts."
    )

    st.divider()

    # -----------------------------------------------------
    # DETAILED METRICS
    # -----------------------------------------------------

    st.subheader("3. Detailed Model Performance Results")

    st.dataframe(
        performance_df,
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        label="Download Model Performance Metrics (CSV)",
        data=performance_df.to_csv(index=False).encode("utf-8"),
        file_name="model_performance_metrics.csv",
        mime="text/csv"
    )

    st.divider()

    # -----------------------------------------------------
    # BUSINESS INTERPRETATION
    # -----------------------------------------------------

    st.subheader("4. Business Interpretation")

    st.write(
        "The forecasting model estimates daily demand quantities "
        "for each Walmart store-category combination. These "
        "estimates could support replenishment and inventory planning."
    )

    st.write(
        "The classification model identifies observations that "
        "may experience unusually high demand. Its recall-oriented "
        "threshold prioritizes detecting potential demand spikes, "
        "while accepting some additional false-positive alerts."
    )

    st.caption(
        "All metrics are based on historical test observations. "
        "This dashboard does not represent live operational performance."
    )

    # --------------------------------------------------
    # 21.6A — Historical Forecasting Model Comparison
    # --------------------------------------------------

    st.divider()
    st.subheader("5. Historical Forecasting Model Comparison")

    st.write(
        "Compare the eight forecasting configurations evaluated "
        "during model development on the same 28-day validation "
        "period: February 29 – March 27, 2016."
    )

    comparison_path = (
        Path(__file__).resolve().parent.parent
        / "dashboard_data"
        / "forecasting_model_comparison.csv"
    )

    if comparison_path.exists():

        model_comparison = pd.read_csv(comparison_path)

        # Show models in order of lowest validation MAE
        model_comparison = (
            model_comparison
            .sort_values("MAE")
            .reset_index(drop=True)
        )

        # Select the performance metric to visualize
        selected_metric = st.selectbox(
            "Select Forecasting Evaluation Metric",
            ["MAE", "RMSE", "WAPE (%)", "R2"],
            key="historical_model_metric"
        )

        # Lower is better for error metrics;
        # higher is better for R2.
        ascending = selected_metric != "R2"

        chart_data = model_comparison.sort_values(
            selected_metric,
            ascending=ascending
        )

        fig_models = px.bar(
            chart_data,
            x="Model",
            y=selected_metric,
            color="Model",
            text=selected_metric,
            title=f"Validation Model Comparison — {selected_metric}",
        )

        fig_models.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )

        fig_models.update_layout(
            xaxis_title="Forecasting Model",
            yaxis_title=selected_metric,
            showlegend=False,
            xaxis_tickangle=-35
        )

        st.plotly_chart(
            fig_models,
            use_container_width=True
        )

        st.markdown("#### Complete Validation Results")

        st.dataframe(
            model_comparison,
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            label="Download Forecasting Model Comparison (CSV)",
            data=model_comparison.to_csv(index=False),
            file_name="forecasting_model_comparison.csv",
            mime="text/csv",
            key="download_forecasting_comparison"
        )

        st.markdown("#### Model Selection Interpretation")

        st.write(
            "The Tuned XGBoost Regressor achieved the lowest "
            "validation MAE (127.629) and WAPE (9.116%). "
            "It was therefore selected for final test evaluation. "
            "However, the original XGBoost model had a slightly "
            "lower validation RMSE, illustrating that tuning "
            "did not improve every evaluation metric."
        )

        st.info(
            "These results come from the validation period, "
            "not the final test period. Prophet is excluded "
            "from this ranking because it was evaluated on "
            "aggregated daily demand rather than individual "
            "store-category series."
        )

    else:
        st.warning(
            "Forecasting comparison CSV not found. "
            "Please export forecasting_model_comparison.csv "
            "from notebook Step 13.8."
        )

# ============================================================
# STEP 21.7 — BUSINESS INSIGHTS DASHBOARD
# ============================================================

elif page == "Business Insights":

    st.header("Business Insights and Recommendations")

    st.write(
        "This dashboard translates the results of the Walmart "
        "retail demand forecasting project into actionable "
        "business insights. It highlights how machine learning "
        "can support inventory planning, replenishment decisions, "
        "and early identification of high-demand periods."
    )

    st.info(
        "The analysis is based on historical Walmart M5 data "
        "and the final 28-day test period "
        "(March 28 – April 24, 2016). "
        "These insights are not live operational forecasts."
    )

    st.subheader("Final Machine Learning Models")

        # Retrieve final model metrics from the cached dataset.
    regression_values = dict(
        zip(
            metrics_df.loc[
                metrics_df["Task"] == "Demand Forecasting", "Metric"
            ],
            metrics_df.loc[
                metrics_df["Task"] == "Demand Forecasting", "Value"
            ]
        )
    )

    classification_values = dict(
        zip(
            metrics_df.loc[
                metrics_df["Task"] == "High-Demand Classification", "Metric"
            ],
            metrics_df.loc[
                metrics_df["Task"] == "High-Demand Classification", "Value"
            ]
        )
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Demand Forecasting")
        st.write("**Model:** Tuned XGBoost Regressor")
        st.metric(
    "Test WAPE",
    f"{float(regression_values['WAPE']):.2f}%"
)

    st.metric(
    "Test R²",
    f"{float(regression_values['R2']):.3f}"
)

    st.write(
            "Estimates daily product-category demand across "
            "Walmart stores to support inventory and "
            "replenishment planning."
        )

    with col2:
        st.markdown("#### High-Demand Detection")
        st.write("**Model:** XGBoost Classifier")
        st.metric(
    "Test Recall",
    f"{float(classification_values['Recall']):.2%}"
)

    st.metric(
    "Test F1-Score",
    f"{float(classification_values['F1']):.2%}"
)

    st.write(
            "Identifies potential high-demand periods "
            "to help retail teams prioritize monitoring "
            "and prepare for possible demand spikes."
        )

    st.divider()


    # ============================================================
    # STEP 21.7.2 — STORE AND PRODUCT CATEGORY BUSINESS INSIGHTS
    # ============================================================

    st.header("Store and Product Category Business Insights")

    st.write(
        "Explore historical demand patterns and forecasting "
        "performance across Walmart stores and product categories. "
        "These insights can help identify areas that may require "
        "closer inventory monitoring."
    )

    # Use a copy of the forecasting dataset for business insights.
    insights_df = forecast_df.copy()

    insights_df["date"] = pd.to_datetime(insights_df["date"])

    # ------------------------------------------------------------
    # 1. Interactive filters
    # ------------------------------------------------------------

    st.subheader("Business Insights Filters")

    col1, col2 = st.columns(2)

    with col1:
        selected_insight_stores = st.multiselect(
            "Select Stores",
            options=sorted(insights_df["store_id"].unique()),
            default=sorted(insights_df["store_id"].unique()),
            key="insight_stores"
        )

    with col2:
        selected_insight_categories = st.multiselect(
            "Select Product Categories",
            options=sorted(insights_df["category"].unique()),
            default=sorted(insights_df["category"].unique()),
            key="insight_categories"
        )

    insights_filtered = insights_df[
        insights_df["store_id"].isin(selected_insight_stores)
        & insights_df["category"].isin(selected_insight_categories)
    ].copy()

    if insights_filtered.empty:
        st.warning(
            "No observations match the selected filters. "
            "Please select at least one store and category."
        )
        st.stop()

    # ------------------------------------------------------------
    # 2. Business KPIs
    # ------------------------------------------------------------

    total_actual_demand = insights_filtered["actual_demand"].sum()

    total_predicted_demand = insights_filtered["predicted_demand"].sum()

    total_absolute_error = (
        insights_filtered["actual_demand"]
        - insights_filtered["predicted_demand"]
    ).abs().sum()

    insights_wape = (
        100 * total_absolute_error / total_actual_demand
        if total_actual_demand > 0
        else float("nan")
    )

    st.subheader("Demand Overview")

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.metric(
            "Total Actual Demand",
            f"{total_actual_demand:,.0f}"
        )

    with kpi2:
        st.metric(
            "Total Predicted Demand",
            f"{total_predicted_demand:,.0f}"
        )

    with kpi3:
        st.metric(
            "Forecast WAPE",
            f"{insights_wape:.2f}%"
        )

    st.caption(
        "KPIs are calculated from the selected historical "
        "store-category-day observations."
    )

    st.divider()

    # ------------------------------------------------------------
    # 3. Actual vs predicted demand by category
    # ------------------------------------------------------------

    st.subheader("Actual vs Predicted Demand by Product Category")

    category_summary = (
        insights_filtered.groupby("category", as_index=False)
        .agg(
            Actual_Demand=("actual_demand", "sum"),
            Predicted_Demand=("predicted_demand", "sum")
        )
    )

    category_chart_df = category_summary.melt(
        id_vars="category",
        value_vars=["Actual_Demand", "Predicted_Demand"],
        var_name="Demand Type",
        value_name="Demand Units"
    )

    fig_category = px.bar(
        category_chart_df,
        x="category",
        y="Demand Units",
        color="Demand Type",
        barmode="group",
        text_auto=".0f",
        title="Actual vs Predicted Demand by Category"
    )

    fig_category.update_layout(
        xaxis_title="Product Category",
        yaxis_title="Total Demand Units",
        legend_title="Demand Type"
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True
    )

    st.write(
        "Comparing actual and predicted demand helps assess "
        "how closely the forecasting model estimates demand "
        "across product categories."
    )

    st.divider()

    # ------------------------------------------------------------
    # 4. Store-category performance table
    # ------------------------------------------------------------

    st.subheader("Store and Category Forecast Performance")

    insights_filtered["absolute_error"] = (
        insights_filtered["actual_demand"]
        - insights_filtered["predicted_demand"]
    ).abs()

    store_category_summary = (
        insights_filtered.groupby(
            ["store_id", "category"],
            as_index=False
        )
        .agg(
            Actual_Demand=("actual_demand", "sum"),
            Predicted_Demand=("predicted_demand", "sum"),
            Total_Absolute_Error=("absolute_error", "sum"),
            Observations=("actual_demand", "size")
        )
    )

    store_category_summary["WAPE (%)"] = (
        100
        * store_category_summary["Total_Absolute_Error"]
        / store_category_summary["Actual_Demand"].replace(0, float("nan"))
    )

    store_category_summary = store_category_summary.sort_values(
        "WAPE (%)",
        ascending=False
    )

    st.dataframe(
        store_category_summary,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Higher WAPE indicates greater relative forecasting error. "
        "The table is ranked from highest to lowest WAPE."
    )

    st.divider()

    # ============================================================
    # STEP 21.7.3 — HIGH-DEMAND RISK AND INVENTORY RECOMMENDATIONS
    # ============================================================

    st.header("High-Demand Risk and Inventory Recommendations")

    st.write(
        "Identify store-category combinations with elevated "
        "predicted high-demand risk and explore practical "
        "inventory monitoring recommendations."
    )

    # Reuse the classification dataset loaded at application startup.
    # Apply the same store/category selections from Step 21.7.2.
    risk_filtered = classification_df[
        classification_df["store_id"].isin(selected_insight_stores)
        & classification_df["category"].isin(selected_insight_categories)
    ].copy()

    if risk_filtered.empty:
        st.warning(
            "No high-demand observations match the selected filters."
        )
    else:
        # --------------------------------------------------------
        # 1. High-demand risk KPIs
        # --------------------------------------------------------

        total_alerts = int(
            risk_filtered["predicted_high_demand"].sum()
        )

        actual_high_demand = int(
            risk_filtered["actual_high_demand"].sum()
        )

        average_probability = (
            risk_filtered["high_demand_probability"].mean()
        )

        alert_rate = (
            100 * total_alerts / len(risk_filtered)
        )

        st.subheader("High-Demand Risk Overview")

        r1, r2, r3, r4 = st.columns(4)

        r1.metric("Predicted High-Demand Alerts", f"{total_alerts:,}")
        r2.metric("Actual High-Demand Observations", f"{actual_high_demand:,}")
        r3.metric("Alert Rate", f"{alert_rate:.2f}%")
        r4.metric(
            "Average Predicted Probability",
            f"{average_probability:.2%}"
        )

        st.caption(
            "Each observation represents one store-category-day. "
            "Alerts use the final classification threshold of 0.25."
        )

        st.divider()

        # --------------------------------------------------------
        # 2. High-demand alerts by store and category
        # --------------------------------------------------------

        st.subheader("High-Demand Alerts by Store and Category")

        risk_summary = (
            risk_filtered.groupby(
                ["store_id", "category"],
                as_index=False
            )
            .agg(
                Predicted_Alerts=("predicted_high_demand", "sum"),
                Actual_High_Demand=("actual_high_demand", "sum"),
                Average_Probability=("high_demand_probability", "mean"),
                Observations=("predicted_high_demand", "size")
            )
        )

        risk_summary["Alert_Rate (%)"] = (
            100
            * risk_summary["Predicted_Alerts"]
            / risk_summary["Observations"]
        )

        risk_summary = risk_summary.sort_values(
            ["Predicted_Alerts", "Average_Probability"],
            ascending=[False, False]
        )

        risk_summary["Store_Category"] = (
            risk_summary["store_id"]
            + " / "
            + risk_summary["category"]
        )

        fig_risk = px.bar(
            risk_summary,
            x="Store_Category",
            y="Predicted_Alerts",
            color="category",
            text="Predicted_Alerts",
            title="Predicted High-Demand Alerts by Store-Category",
            labels={
                "Store_Category": "Store / Product Category",
                "Predicted_Alerts": "Number of Alerts",
                "category": "Product Category"
            }
        )

        fig_risk.update_layout(
            xaxis_tickangle=-45,
            yaxis_title="Number of Predicted Alerts"
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True
        )

        st.caption(
            "Higher alert counts indicate more historical days "
            "flagged by the classifier, not necessarily greater "
            "inventory shortages."
        )

        # --------------------------------------------------------
        # 3. Detailed risk summary
        # --------------------------------------------------------

        st.subheader("Store-Category High-Demand Risk Summary")

        st.dataframe(
            risk_summary[
                [
                    "store_id",
                    "category",
                    "Predicted_Alerts",
                    "Actual_High_Demand",
                    "Average_Probability",
                    "Alert_Rate (%)",
                    "Observations"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------------
        # 4. Inventory recommendations
        # --------------------------------------------------------

        st.subheader("Inventory Monitoring Recommendations")

        st.write(
            "The following recommendations use historical "
            "classification alerts to highlight store-category "
            "combinations that may deserve closer review."
        )

        risk_summary["Monitoring_Priority"] = np.select(
            [
                risk_summary["Alert_Rate (%)"] >= 50,
                risk_summary["Alert_Rate (%)"] >= 25
            ],
            [
                "High",
                "Moderate"
            ],
            default="Routine"
        )

        recommendation_map = {
            "High": (
                "Prioritize stock availability reviews, "
                "replenishment planning, and investigation "
                "of recurring demand spikes."
            ),
            "Moderate": (
                "Monitor demand trends and review replenishment "
                "schedules for potentially elevated demand."
            ),
            "Routine": (
                "Maintain routine inventory monitoring and "
                "reassess if demand patterns change."
            )
        }

        risk_summary["Recommendation"] = (
            risk_summary["Monitoring_Priority"].map(
                recommendation_map
            )
        )

        priority_order = {
            "High": 0,
            "Moderate": 1,
            "Routine": 2
        }

        risk_summary["Priority_Order"] = (
            risk_summary["Monitoring_Priority"].map(
                priority_order
            )
        )

        recommendations_df = risk_summary.sort_values(
            ["Priority_Order", "Predicted_Alerts"],
            ascending=[True, False]
        )[
            [
                "store_id",
                "category",
                "Predicted_Alerts",
                "Alert_Rate (%)",
                "Monitoring_Priority",
                "Recommendation"
            ]
        ]

        st.dataframe(
    recommendations_df,
    use_container_width=True,
    hide_index=True,
    row_height=90,
    column_config={
        "store_id": st.column_config.TextColumn(
            "Store",
            width="small"
        ),
        "category": st.column_config.TextColumn(
            "Category",
            width="small"
        ),
        "Predicted_Alerts": st.column_config.NumberColumn(
            "Alerts",
            format="%d"
        ),
        "Alert_Rate (%)": st.column_config.NumberColumn(
            "Alert Rate (%)",
            format="%.2f"
        ),
        "Monitoring_Priority": st.column_config.TextColumn(
            "Priority",
            width="small"
        ),
        "Recommendation": st.column_config.TextColumn(
            "Recommended Action",
            width="large",
            help="Suggested inventory monitoring and replenishment actions"
        )
    }
)

        st.caption(
            "Monitoring priority bands are illustrative dashboard "
            "rules based on historical alert frequency. They are "
            "not optimized inventory policies or validated "
            "stockout-risk thresholds."
        )

        st.download_button(
            label="Download Inventory Recommendations (CSV)",
            data=recommendations_df.to_csv(index=False).encode("utf-8"),
            file_name="inventory_recommendations.csv",
            mime="text/csv",
            key="download_inventory_recommendations"
        )

        st.divider()

        st.subheader("Business Interpretation")

        st.write(
            "The high-demand classifier complements the demand "
            "forecasting model by highlighting store-category "
            "observations that may require additional attention. "
            "Retail teams could use these historical patterns "
            "to prioritize inventory reviews and investigate "
            "replenishment requirements."
        )

        st.info(
            "These recommendations are decision-support examples "
            "based on historical test predictions. They do not "
            "represent real-time inventory levels, confirmed "
            "stockout events, or automatically generated "
            "purchase orders."
        )

    
