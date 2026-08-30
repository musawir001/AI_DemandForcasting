import gradio as gr
import pandas as pd
import calendar

from model import load_artifacts
from analytics import predict_dashboard

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("walmart_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Department mapping
dept_map = {
    1: "Electronics", 2: "Clothing", 3: "Grocery", 4: "Furniture",
    5: "Sports", 6: "Toys", 7: "Beauty", 8: "Pharmacy",
    9: "Automotive", 10: "Home Decor", 11: "Footwear",
    12: "Accessories", 13: "Stationery", 14: "Appliances",
    15: "Health Care", 16: "Baby Products", 17: "Pet Supplies",
    18: "Books", 19: "Music", 20: "Garden"
}
df["Dept_Name"] = df["Dept"].map(dept_map).fillna("Other")

# -------------------------------
# Load model artifacts
# -------------------------------
model, le, features = load_artifacts()

# -------------------------------
# Dynamic UI Logic
# -------------------------------
years = list(range(df['Date'].dt.year.min(), 2036))
months = list(range(1, 13))

def update_days(year, month):
    max_day = calendar.monthrange(int(year), int(month))[1]
    return gr.Dropdown(choices=list(range(1, max_day + 1)), value=1)

# -------------------------------
# Dashboard Function
# -------------------------------
def full_dashboard(dept, year, month, day):

    sales_val, demand_val, rec_val, insight_val, trend_fig, bar_fig = (
        predict_dashboard(
            model, le, features, df,
            dept,
            int(year),
            int(month),
            int(day)
        )
    )

    best_dept = (
        df.groupby("Dept_Name")["Weekly_Sales"]
        .mean()
        .idxmax()
    )

    return (
        sales_val,
        demand_val,
        rec_val,
        insight_val,
        trend_fig,
        bar_fig,
        f"🏆 {best_dept}"
    )

# -------------------------------
# UI DESIGN (YOUR STYLE)
# -------------------------------
with gr.Blocks(
    title="AI Demand Forecasting",
    css="""
    .gr-button-primary {
        background: #6b7280 !important;
        border: 1px solid #4b5563 !important;
        color: white !important;
        font-weight: 600 !important;
    }

    .gr-button-primary:hover {
        background: #4b5563 !important;
        border: 1px solid #374151 !important;
    }
    """
) as demo:

    # Header
    gr.Markdown("""
    # 🛒 AI Demand Forecasting
    ### Predict Future Sales • Analyze Trends • Smart Recommendations
    """)

    # Inputs
    with gr.Row():
        dept_input = gr.Dropdown(
            choices=sorted(df['Dept_Name'].unique()),
            label="📦 Select Department",
            value=sorted(df['Dept_Name'].unique())[0]
        )

    with gr.Row():
        year_input = gr.Dropdown(choices=years, value=2027, label="📅 Year")
        month_input = gr.Dropdown(choices=months, value=1, label="📅 Month")
        day_input = gr.Dropdown(choices=list(range(1, 32)), value=1, label="📅 Day")

    year_input.change(update_days, [year_input, month_input], day_input)
    month_input.change(update_days, [year_input, month_input], day_input)

    # Button
    analyze_btn = gr.Button("🚀 Analyze Demand", variant="primary")

    # Results
    gr.Markdown("## 📊 Prediction Results")

    with gr.Row():
        sales_out = gr.Textbox(label="💰 Predicted Weekly Sales")
        demand_out = gr.Textbox(label="📈 Demand Level")

    recommendation_out = gr.Textbox(
        label="💡 Business Recommendation",
        lines=6
    )

    # Tabs
    with gr.Tabs():

        with gr.Tab("📊 Business Insights"):
            insight_out = gr.Textbox(label="AI Insights", lines=8)

        with gr.Tab("📈 Monthly Trend Analysis"):
            trend_chart = gr.Plot()

        with gr.Tab("📊 Top Departments"):
            bar_chart = gr.Plot()

        with gr.Tab("🏆 Best Performer"):
            best_dept_out = gr.Textbox()

    # Button Action
    analyze_btn.click(
        fn=full_dashboard,
        inputs=[dept_input, year_input, month_input, day_input],
        outputs=[
            sales_out,
            demand_out,
            recommendation_out,
            insight_out,
            trend_chart,
            bar_chart,
            best_dept_out
        ]
    )

    # Footer
    gr.Markdown("""
    ---
    ⚡ **Features**
    - Future Weekly Sales Prediction  
    - Demand Classification  
    - Smart Business Recommendations  
    - Monthly Trend Analysis  
    - Top Department Comparison  
    - Best Performing Department  
    """)

demo.launch()