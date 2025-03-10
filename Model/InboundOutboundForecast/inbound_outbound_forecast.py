import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "Forecast_Models")

def _weekly_real(df, category):
    df = df[df["Category"] == category].copy()
    df["Dispatch_Date"] = pd.to_datetime(df["Dispatch_Date"])
    weekly = (
        df.set_index("Dispatch_Date")
          .resample("W")
          .agg({"Dispatch_Quantity": "sum", "Inbound_Quantity": "sum"})
          .reset_index()
          .rename(columns={"Dispatch_Date": "ds"})
    )
    return weekly

def _weeks_needed(last_hist, year, end_month, buffer_weeks=8):
    target_end = pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
    need = max(0, int(np.ceil((target_end - last_hist).days / 7)))
    return need + buffer_weeks

def predict_forecast_for_a_category(
    category,
    start_month,
    end_month,
    start_week=1,
    end_week=5,
    year=pd.Timestamp.now().year,
    time_frame=60,
    threshold=0.10,
):
    # ----- Load data & models (logistic Prophet models saved as dicts {model, cap, floor}) -----
    df_real = pd.read_csv("../WareHouseDataset.csv")
    real_weekly_raw = _weekly_real(df_real, category)

    ob = joblib.load(os.path.join(MODEL_DIR, f"{category}_outbound.pkl"))
    ib = joblib.load(os.path.join(MODEL_DIR, f"{category}_inbound.pkl"))

    model_out, cap_out, floor_out = ob["model"], ob["cap"], ob["floor"]
    model_in,  cap_in,  floor_in  = ib["model"], ib["cap"], ib["floor"]

    last_hist = pd.to_datetime(model_out.history["ds"]).max()
    weeks_needed = _weeks_needed(last_hist, year, end_month)

    # ----- Make future frames with caps/floors -----
    future = model_out.make_future_dataframe(periods=weeks_needed, freq="W", include_history=True)
    future_out = future.copy(); future_out["cap"] = cap_out; future_out["floor"] = floor_out
    future_in  = future.copy(); future_in["cap"]  = cap_in;  future_in["floor"]  = floor_in

    # ----- Predict (already on original scale) -----
    fc_out = model_out.predict(future_out)[["ds", "yhat"]].rename(columns={"yhat": "Outbound_Forecast"})
    fc_in  = model_in.predict(future_in)[["ds", "yhat"]].rename(columns={"yhat": "Inbound_Forecast"})
    fc_out["Outbound_Forecast"] = fc_out["Outbound_Forecast"].clip(lower=0, upper=cap_out)
    fc_in["Inbound_Forecast"]   = fc_in["Inbound_Forecast"].clip(lower=0, upper=cap_in)

    forecast_df = fc_out.merge(fc_in, on="ds", how="inner")
    # Smoothed forecasts for plotting
    forecast_df["Outbound_Forecast"] = forecast_df["Outbound_Forecast"].rolling(4, min_periods=1).mean()
    forecast_df["Inbound_Forecast"]  = forecast_df["Inbound_Forecast"].rolling(4, min_periods=1).mean()

    # ----- Build selected date window -----
    start_date = pd.Timestamp(year=year, month=start_month, day=1)
    end_date   = pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
    selected = forecast_df[(forecast_df["ds"] >= start_date) & (forecast_df["ds"] <= end_date)].copy()

    # Optional: keep week-of-month filter
    selected["week_of_month"] = ((selected["ds"].dt.day - 1) // 7) + 1
    selected = selected[(selected["week_of_month"] >= start_week) & (selected["week_of_month"] <= end_week)]
    selected = selected.drop(columns=["week_of_month"])
    if selected.empty:
        raise ValueError(
            f"No forecasted points in {year}-{start_month:02d}..{end_month:02d} "
            f"(weeks {start_week}..{end_week}). History ends at {last_hist.date()}."
        )

    # ====== ROBUST DEVIATION DETECTION ======
    # 1) Align real data to forecast calendar to minimize NaNs
    calendar = forecast_df[["ds"]].copy()
    real_aligned = calendar.merge(real_weekly_raw, on="ds", how="left")

    # 2) Smoothed real series for plotting (keep raw for deviation math)
    real_smooth = real_aligned.copy()
    real_smooth[["Dispatch_Quantity", "Inbound_Quantity"]] = (
        real_smooth[["Dispatch_Quantity", "Inbound_Quantity"]].rolling(4, min_periods=1).mean()
    )

    # 3) Join selected forecast window with aligned raw real data
    comparison = selected.merge(real_aligned, on="ds", how="left")

    # 4) Percent error with epsilon, computed on RAW values
    EPS = 1e-6
    valid_out = comparison["Dispatch_Quantity"].notna() & comparison["Outbound_Forecast"].notna()
    valid_in  = comparison["Inbound_Quantity"].notna()  & comparison["Inbound_Forecast"].notna()

    pe_out = pd.Series(np.nan, index=comparison.index)
    pe_in  = pd.Series(np.nan, index=comparison.index)

    pe_out[valid_out] = (
        (comparison.loc[valid_out, "Dispatch_Quantity"] - comparison.loc[valid_out, "Outbound_Forecast"]) /
        np.maximum(comparison.loc[valid_out, "Outbound_Forecast"].abs(), EPS)
    )
    pe_in[valid_in] = (
        (comparison.loc[valid_in, "Inbound_Quantity"] - comparison.loc[valid_in, "Inbound_Forecast"]) /
        np.maximum(comparison.loc[valid_in, "Inbound_Forecast"].abs(), EPS)
    )

    comparison["Outbound_Above"] = False
    comparison.loc[valid_out, "Outbound_Above"] = pe_out[valid_out] > threshold
    comparison["Outbound_Below"] = False
    comparison.loc[valid_out, "Outbound_Below"] = pe_out[valid_out] < -threshold

    comparison["Inbound_Above"] = False
    comparison.loc[valid_in, "Inbound_Above"] = pe_in[valid_in] > threshold
    comparison["Inbound_Below"] = False
    comparison.loc[valid_in, "Inbound_Below"] = pe_in[valid_in] < -threshold

    # ----- Plot -----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Real (smoothed) vs Forecast (smoothed) over the selected window
    axes[0].plot(comparison["ds"], comparison["Outbound_Forecast"], label="Forecasted Outbound", linestyle="--")
    axes[0].plot(comparison["ds"], comparison["Inbound_Forecast"],  label="Forecasted Inbound",  linestyle="--")

    axes[0].plot(real_smooth["ds"], real_smooth["Dispatch_Quantity"], label="Real Outbound")
    axes[0].plot(real_smooth["ds"], real_smooth["Inbound_Quantity"],  label="Real Inbound")

    # Triangles from RAW deviation flags
    oa = comparison["Outbound_Above"]; ob = comparison["Outbound_Below"]
    ia = comparison["Inbound_Above"];  ib = comparison["Inbound_Below"]

    if oa.any():
        axes[0].scatter(comparison.loc[oa, "ds"], comparison.loc[oa, "Dispatch_Quantity"],
                        marker="^", label="Outbound Above")
    if ob.any():
        axes[0].scatter(comparison.loc[ob, "ds"], comparison.loc[ob, "Dispatch_Quantity"],
                        marker="v", label="Outbound Below")
    if ia.any():
        axes[0].scatter(comparison.loc[ia, "ds"], comparison.loc[ia, "Inbound_Quantity"],
                        marker="^", label="Inbound Above")
    if ib.any():
        axes[0].scatter(comparison.loc[ib, "ds"], comparison.loc[ib, "Inbound_Quantity"],
                        marker="v", label="Inbound Below")

    axes[0].set_title(f"Real vs Forecasted for {category}")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Quantity")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Right: Future-only panel (last `time_frame` weeks of forecast)
    future_only = forecast_df.tail(time_frame)
    axes[1].plot(future_only["ds"], future_only["Outbound_Forecast"], label="Forecasted Outbound")
    axes[1].plot(future_only["ds"], future_only["Inbound_Forecast"],  label="Forecasted Inbound")
    axes[1].set_title(f"Future Forecast (Next {time_frame} Weeks)")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Forecasted Quantity")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    output_path = os.path.join("static", f"forecast_plot_{category}.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    # Helpful debug
    print("History ends at:", last_hist)
    print("Forecast covers:", forecast_df["ds"].min(), "→", forecast_df["ds"].max())
    print("Valid Outbound points:", int(valid_out.sum()),
          "| Above:", int(comparison["Outbound_Above"].sum()),
          "Below:", int(comparison["Outbound_Below"].sum()))
    print("Valid Inbound points:", int(valid_in.sum()),
          "| Above:", int(comparison["Inbound_Above"].sum()),
          "Below:", int(comparison["Inbound_Below"].sum()))

    return forecast_df, output_path
