# 📈 Stock Trend Prediction App

![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-%23F7931E.svg?style=for-the-badge)

## 🚀 Overview
The **Stock Trend Prediction App** is a machine learning-powered web application that allows users to visualize stock trends and predict future prices based on historical data. Built using **Streamlit**, **Yahoo Finance API**, and a **pre-trained deep learning model**, this app provides insightful visualizations and predictions for various stock attributes such as **Open, Close, High, and Low prices**.

## ✨ Features
✔️ **User-friendly Interface** – Modern and interactive UI for smooth navigation  
✔️ **Real-time Stock Data** – Fetches live stock data using Yahoo Finance  
✔️ **Customizable Date Range** – Users can select a date range for analysis  
✔️ **Multiple Stock Attributes** – View Open, Close, High, and Low prices  
✔️ **Moving Averages** – 100-day and 200-day moving averages for trend analysis  
✔️ **Deep Learning Predictions** – Uses LSTM model for future price predictions  
✔️ **Interactive Plots** – Beautiful visualizations with Matplotlib  

---

## 🛠️ Installation & Setup

### Prerequisites
Ensure you have **Python 3.8+** installed along with the following dependencies:
```bash
pip install streamlit pandas numpy yfinance keras matplotlib scikit-learn gdown
```

### Clone Repository
```bash
git clone https://github.com/yourusername/stock-trend-predictor.git
cd stock-trend-predictor
```

### Download Pre-trained Model
Since the model (`keras_model.h5`) is stored on Google Drive, download it using:
```python
import gdown
url = "https://drive.google.com/file/d/1-wRBK0Wnbtw5E0kV2A_DHcvAOFAJeIiq/view?usp=sharing"
gdown.download(url, "keras_model.h5", quiet=False)
```

### Run the App
```bash
streamlit run app.py
```

---

## 📊 How It Works
1️⃣ **Enter a stock ticker symbol** (e.g., AAPL, TSLA)  
2️⃣ **Select the stock attribute** (Close, Open, High, or Low)  
3️⃣ **Choose a date range** for historical data  
4️⃣ **View trend graphs & moving averages**  
5️⃣ **Get AI-powered stock price predictions**  

---

## 🔬 Model Details
- **Algorithm:** LSTM (Long Short-Term Memory)
- **Trained On:** Historical stock data from Yahoo Finance (2015-2025)
- **Scaling:** MinMaxScaler for normalization
- **Data Split:** 70% Training, 30% Testing
- **Evaluation Metric:** RMSE (Root Mean Square Error)

---

## 🖥️ Screenshots
### **📌 Main Dashboard**
![App Screenshot](https://your-screenshot-link.com/main.png)

### **📈 Stock Price Trends**
![Trends](https://your-screenshot-link.com/trends.png)

---

## 🚀 Deployment Guide
This app can be deployed on **Streamlit Cloud**, **Heroku**, or **AWS**.

### Deploy on Streamlit Cloud
1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub repo and deploy 🚀

---

## 🤝 Contributing
We welcome contributions! Feel free to **fork the repo**, submit a **pull request**, or suggest improvements via **issues**.

---

## 📞 Contact
📧 Email: lakumvipul6351@gmail.com    
📂 GitHub: [yourusername](https://github.com/Vipullakum007)  

---

💡 *If you found this project useful, don't forget to ⭐ star the repo!* 🚀

