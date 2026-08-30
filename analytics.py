import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# -------------------------------
# Demand Classification
# -------------------------------
def demand_level(value):
    if value > 30000:
        return "🔥 High Demand"
    elif value > 15000:
        return "⚡ Medium Demand"
    return "❄ Low Demand"


def get_recommendation(demand):
    if "High" in demand:
        return "📦 Increase inventory and ensure stock availability."
    elif "Medium" in demand:
        return "⚖ Maintain balanced inventory levels."
    return "📉 Consider promotions or reduce stock."


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def predict_dashboard(model, le, features, df, dept_name, year, month, day):

    # -------------------------------
    # Date Handling
    # -------------------------------
    try:
        date_input = pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
    except:
        return "Invalid Date", "", "Invalid date", "", None, None

    temp_df = df[df['Dept_Name'] == dept_name]
    if temp_df.empty:
        return "No Data", "", "Department not found", "", None, None

    row = temp_df.iloc[-1]

    # Encode dept
    if dept_name in le.classes_:
        dept_encoded = le.transform([dept_name])[0]
    else:
        dept_encoded = -1

    week = int(date_input.isocalendar()[1])

    input_data = pd.DataFrame([{
        "Store": row["Store"],
        "Dept_Encoded": dept_encoded,
        "Temperature": row["Temperature"],
        "Fuel_Price": row["Fuel_Price"],
        "CPI": row["CPI"],
        "Unemployment": row["Unemployment"],
        "month": date_input.month,
        "week": week
    }])

    input_data = input_data[features]

    pred = max(0, float(model.predict(input_data)[0]))

    demand = demand_level(pred)
    recommendation = get_recommendation(demand)

    # =========================================================
    # 📈 TREND GRAPH (COMPARISON)
    # =========================================================
    temp_df = temp_df.copy()
    temp_df["month"] = temp_df["Date"].dt.month

    best_dept = df.groupby('Dept_Name')['Weekly_Sales'].mean().idxmax()
    best_df = df[df['Dept_Name'] == best_dept]

    months = range(1, 13)

    selected_monthly = (
        temp_df.groupby("month")["Weekly_Sales"]
        .mean().reindex(months, fill_value=0)
    )

    best_monthly = (
        best_df.groupby(best_df["Date"].dt.month)["Weekly_Sales"]
        .mean().reindex(months, fill_value=0)
    )

    selected_k = selected_monthly / 1000
    best_k = best_monthly / 1000

    fig1, ax1 = plt.subplots(figsize=(9, 5))

    ax1.grid(True, linestyle='--', alpha=0.5)

    ax1.plot(months, best_k, marker='o', linewidth=2.5,
             label=f"{best_dept} (Best)")

    ax1.plot(months, selected_k, marker='o', linewidth=2.5,
             label=f"{dept_name} (Selected)")

    ax1.set_title("📈 Monthly Sales Comparison", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Avg Weekly Sales (₹ thousands)")
    ax1.set_xticks(list(months))
    ax1.legend()

    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}")
    )

    plt.tight_layout()
    trend_plot = fig1
    plt.close(fig1)

    # =========================================================
    # 📊 BAR GRAPH (TOP DEPARTMENTS)
    # =========================================================
    dept_avg = (
        df.groupby("Dept_Name")["Weekly_Sales"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    colors = [
        "orange" if d == dept_name else "steelblue"
        for d in dept_avg.index
    ]

    fig2, ax2 = plt.subplots(figsize=(9, 5))

    bars = ax2.bar(dept_avg.index, dept_avg.values, color=colors)

    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"₹{height:,.0f}",
            ha='center',
            va='bottom',
            fontsize=8
        )

    ax2.set_title("📊 Top 10 Departments by Avg Sales", fontweight='bold')
    ax2.set_ylabel("Avg Weekly Sales (₹)")
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    bar_plot = fig2
    plt.close(fig2)

    # =========================================================
    # INSIGHT TEXT
    # =========================================================
    insight = f"""
📊 Analysis Summary

Department: {dept_name}
Date: {date_input.date()}

💰 Predicted Sales: ₹{round(pred, 0):,}
📈 Demand Level: {demand}

💡 Recommendation:
{recommendation}
"""

    return (
        f"₹ {round(pred, 0):,}",
        demand,
        recommendation,
        insight,
        trend_plot,
        bar_plot
    )