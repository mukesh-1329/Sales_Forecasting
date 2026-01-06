📈 Sales Demand Forecasting using GRU
This project is an end-to-end sales demand forecasting application that predicts weekly and multi-week sales using a GRU (Gated Recurrent Unit) deep learning model, deployed through an interactive Streamlit web interface.

🚀 Project Description
Accurate sales forecasting is critical for inventory management and business planning.
This application leverages time-series deep learning to learn historical sales patterns and forecast future demand while capturing trend and seasonality.
The model is trained on weekly aggregated sales data and deployed as a user-friendly web app that allows real-time forecasting.

🧠 Model Details

Model Type: GRU (Gated Recurrent Unit)
Forecasting Type: Time Series
Input Window: Last 12 weeks
Forecast Horizon: 1 to 10 weeks
Loss Function: Mean Squared Error (MSE)
Framework: TensorFlow / Keras

📊 Evaluation Metrics
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
The model achieves stable performance with low single-digit percentage error relative to average weekly sales.

🖥️ Deployed Web Application

The Streamlit UI enables users to:
Enter recent weekly sales data
Select the number of future weeks to forecast
View predictions in a table
Visualize forecasts using a clean line plot

🗂 Dataset & Features
Weekly sales values
Month and year as time-based features
Data scaled using MinMaxScaler for deep learning

🔄 Workflow
Data Cleaning → Feature Scaling → Sequence Creation →
GRU Training → Model Evaluation → Multi-week Forecasting → Deployment

🧪 Tech Stack
Python
TensorFlow / Keras
Pandas, NumPy
Scikit-learn
Matplotlib
Streamlit

📁 Project Structure
Sales Forecast.ipynb – Model training and evaluation
app.py – Streamlit application
gru_model.keras – Trained GRU model
scaler.pkl – Saved scaler for inference

▶️ How to Run Locally
pip install -r requirements.txt
streamlit run app.py

📌 Future Enhancements
Add confidence intervals
Include price and promotion features
CSV upload support
Cloud deployment

⭐ Acknowledgment

If you find this project useful, consider giving it a ⭐ on GitHub.
