import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf

# ==============================
# CONFIG
# ==============================
SEQ_LEN = 12
FEATURES = 3  # value, month, year

st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📈",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = load_model("gru_model.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    scaler = joblib.load('scaler1.pkl')
    return model, scaler

model, scaler = load_artifacts()


def forecast_next_n_weeks(model, scaler, last_sequence, n_weeks):
    future_preds = []
    current_seq = last_sequence.copy()

    for _ in range(n_weeks):
        pred = model.predict(
            current_seq.reshape(1, SEQ_LEN, FEATURES),
            verbose=0
        )[0, 0]

        future_preds.append(pred)

        next_row = current_seq[-1].copy()
        next_row[0] = pred  # update sales only

        current_seq = np.vstack([current_seq[1:], next_row])

    temp = np.zeros((n_weeks, FEATURES))
    temp[:, 0] = future_preds

    return scaler.inverse_transform(temp)[:, 0]

# ==============================
# UI HEADER
# ==============================
st.title("📈 Sales Demand Forecasting")
st.write(
    "Forecast **future weekly sales (1–10 weeks)** using a **GRU deep learning model**"
)

st.divider()

# ==============================
# USER INPUT
# ==============================
st.subheader("Enter Last 12 Weeks Data")

sales, months, years = [], [], []

for i in range(SEQ_LEN):
    st.markdown(f"**Week {i+1}**")
    c1, c2, c3 = st.columns(3)

    with c1:
        sales.append(
            st.number_input(
                "Weekly Sales",
                min_value=0.0,
                value=100.0,
                key=f"s_{i}"
            )
        )

    with c2:
        months.append(
            st.number_input(
                "Month",
                min_value=1,
                max_value=12,
                value=6,
                key=f"m_{i}"
            )
        )

    with c3:
        years.append(
            st.number_input(
                "Year",
                min_value=2011,
                max_value=2035,
                value=2016,
                key=f"y_{i}"
            )
        )

st.divider()

# FORECAST CONTROL
weeks_ahead = st.slider(
    "Select number of weeks to forecast",
    min_value=1,
    max_value=10,
    value=10
)

# PREDICTION
if st.button("🔮 Forecast Sales"):
    input_data = np.column_stack([sales, months, years])
    input_scaled = scaler.transform(input_data)
    last_sequence = input_scaled.reshape(SEQ_LEN, FEATURES)

    future_sales = forecast_next_n_weeks(
        model,
        scaler,
        last_sequence,
        weeks_ahead
    )

    # --------------------------
    # TABLE
    # --------------------------
    st.subheader("📊 Forecast Results")
    result_table = {
        "Week": [f"Week {i+1}" for i in range(weeks_ahead)],
        "Predicted Sales": [int(v) for v in future_sales]
    }
    st.table(result_table)

    # BEST PRACTICE PLOT

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(
        range(1, weeks_ahead + 1),
        future_sales,
        marker="o",
        linewidth=2
    )

    ax.set_title("📈 Sales Forecast (Next Weeks)", fontsize=14)
    ax.set_xlabel("Future Weeks")
    ax.set_ylabel("Sales Units")
    ax.grid(True, linestyle="--", alpha=0.6)

    st.pyplot(fig)

st.divider()

st.caption(
    "Model: GRU | Data: M5 Weekly Sales | UI: Streamlit"
)
