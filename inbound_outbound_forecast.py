import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict_forecast_for_a_category(category, df_real_data, start_month, end_month, start_week=1, end_week=5, year=pd.Timestamp.now().year, time_frame=60, threshold=0.1):
    """
        category: The category model to forecast.
        df_real_data: The actual historical data containing 'Dispatch_Date', 'Dispatch_Quantity', and 'Inbound_Quantity'.
        start_month: The starting month (1-12).
        end_month: The ending month (1-12).
        start_week: The starting week of the month (1-5). Defaults to 1 optional.
        end_week: The ending week of the month (1-5). Defaults to 5 optional.
        year: The year for prediction. Defaults to the current year.
        time_frame: Number of weeks to predict. Defaults to 60.
        threshold: Percentage threshold for deviation detection. Default is 0.1 (10%).
    """
    directory = "Forecast_Models"

    # Load saved Prophet models
    model_outbound_loaded = joblib.load(f"{directory}/{category}_outbound.pkl")
    model_inbound_loaded = joblib.load(f"{directory}/{category}_inbound.pkl")

    # Generate future dates for forecasting (weekly)
    future_dates = model_outbound_loaded.make_future_dataframe(periods=time_frame, freq="W")

    # Make predictions using the loaded models
    forecast_outbound = model_outbound_loaded.predict(future_dates)
    forecast_outbound["yhat"] = np.expm1(forecast_outbound["yhat"])  # Reverse log transformation

    forecast_inbound = model_inbound_loaded.predict(future_dates)
    forecast_inbound["yhat"] = np.expm1(forecast_inbound["yhat"])

    # Merge predictions into a single DataFrame
    forecast_combined = future_dates.copy()
    forecast_combined["Outbound_Forecast"] = forecast_outbound["yhat"]
    forecast_combined["Inbound_Forecast"] = forecast_inbound["yhat"]

    # Apply rolling average to smooth the forecasted data
    forecast_combined["Outbound_Forecast"] = forecast_combined["Outbound_Forecast"].rolling(window=4, min_periods=1).mean()
    forecast_combined["Inbound_Forecast"] = forecast_combined["Inbound_Forecast"].rolling(window=4, min_periods=1).mean()

    forecast_combined["ds"] = pd.to_datetime(forecast_combined["ds"])
    forecast_combined["week_of_month"] = forecast_combined["ds"].apply(lambda x: (x.day - 1) // 7 + 1)

    # Filter predictions for the specified month, week range, and year
    selected_range_forecast = forecast_combined[
        (forecast_combined["ds"].dt.month >= start_month) &
        (forecast_combined["ds"].dt.month <= end_month) &
        (forecast_combined["ds"].dt.year == year) &
        (forecast_combined["week_of_month"] >= start_week) &
        (forecast_combined["week_of_month"] <= end_week)
    ]

    # Load real inbound and outbound data for the same category
    df_real_data["Dispatch_Date"] = pd.to_datetime(df_real_data["Dispatch_Date"])

    real_data_filtered = df_real_data[
        (df_real_data["Dispatch_Date"].dt.month >= start_month) &
        (df_real_data["Dispatch_Date"].dt.month <= end_month) &
        (df_real_data["Dispatch_Date"].dt.year == year) &
        (df_real_data["Dispatch_Date"].apply(lambda x: (x.day - 1) // 7 + 1) >= start_week) &
        (df_real_data["Dispatch_Date"].apply(lambda x: (x.day - 1) // 7 + 1) <= end_week) &
        (df_real_data["Category"] == category)
    ]

    # Aggregate real data by week
    real_data_grouped = real_data_filtered.groupby("Dispatch_Date")[["Dispatch_Quantity", "Inbound_Quantity"]].sum().reset_index()

    # Apply rolling average to smooth the real data
    real_data_grouped["Inbound_Quantity"] = real_data_grouped["Inbound_Quantity"].rolling(window=4, min_periods=1).mean()
    real_data_grouped["Dispatch_Quantity"] = real_data_grouped["Dispatch_Quantity"].rolling(window=4, min_periods=1).mean()

    # Detect deviations
    # executes sql like query to the dataframe
    df_comparison = selected_range_forecast.merge(real_data_grouped, left_on="ds", right_on="Dispatch_Date", how="left")

    df_comparison["Outbound_Above"] = df_comparison["Dispatch_Quantity"] > (df_comparison["Outbound_Forecast"] * (1 + threshold))
    df_comparison["Outbound_Below"] = df_comparison["Dispatch_Quantity"] < (df_comparison["Outbound_Forecast"] * (1 - threshold))

    df_comparison["Inbound_Above"] = df_comparison["Inbound_Quantity"] > (df_comparison["Inbound_Forecast"] * (1 + threshold))
    df_comparison["Inbound_Below"] = df_comparison["Inbound_Quantity"] < (df_comparison["Inbound_Forecast"] * (1 - threshold))

    # Extract deviation points
    outbound_above_points = df_comparison[df_comparison["Outbound_Above"]]
    outbound_below_points = df_comparison[df_comparison["Outbound_Below"]]
    inbound_above_points = df_comparison[df_comparison["Inbound_Above"]]
    inbound_below_points = df_comparison[df_comparison["Inbound_Below"]]

    # Print forecasted vs actual data
    print(f"\nComparison of Real vs Forecasted Inbound & Outbound Quantities for {category}:")
    print(df_comparison[["ds", "Inbound_Forecast", "Outbound_Forecast", "Dispatch_Quantity", "Inbound_Quantity"]])

    # Plot Forecast vs Real Data
    plt.figure(figsize=(12, 6))

    # Plot Forecasted Data (Dashed Lines)
    plt.plot(df_comparison["ds"], df_comparison["Outbound_Forecast"], label="Forecasted Outbound", color='red', linestyle="--", linewidth=2)
    plt.plot(df_comparison["ds"], df_comparison["Inbound_Forecast"], label="Forecasted Inbound", color='blue', linestyle="--", linewidth=2)

    # Plot Real Data (Solid Lines)
    plt.plot(real_data_grouped["Dispatch_Date"], real_data_grouped["Dispatch_Quantity"], label="Real Outbound", color='brown', linewidth=2)
    plt.plot(real_data_grouped["Dispatch_Date"], real_data_grouped["Inbound_Quantity"], label="Real Inbound", color='green', linewidth=2)

    # Highlight Deviations
    plt.scatter(outbound_above_points["ds"], outbound_above_points["Dispatch_Quantity"], color='red', marker="^", label="Outbound Above")
    plt.scatter(outbound_below_points["ds"], outbound_below_points["Dispatch_Quantity"], color='red', marker="v", label="Outbound Below")

    plt.scatter(inbound_above_points["ds"], inbound_above_points["Inbound_Quantity"], color='blue', marker="^", label="Inbound Above")
    plt.scatter(inbound_below_points["ds"], inbound_below_points["Inbound_Quantity"], color='blue', marker="v", label="Inbound Below")

    plt.xlabel("Date")
    plt.ylabel("Quantity")
    plt.title(f"Real vs Forecasted Data for {category} (Deviation Detection)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    return df_comparison[["ds", "Dispatch_Quantity", "Outbound_Forecast", "Outbound_Above", "Outbound_Below",
                          "Inbound_Quantity", "Inbound_Forecast", "Inbound_Above", "Inbound_Below"]]


if __name__ == "__main__":
    df_realistic = pd.read_csv("WareHouseDataset.csv")
    predict_forecast_for_a_category("explosive",df_realistic,start_month=3,end_month=7,time_frame=5,year=2024)