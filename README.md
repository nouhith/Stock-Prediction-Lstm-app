# 📈 LSTM Stock Price Predictor

A Streamlit web app that uses an LSTM (Long Short-Term Memory) neural network to predict future stock prices based on historical data from Alpha Vantage.

---

## 🚀 Features

- Choose from popular stocks (Apple, Microsoft, Tesla, etc.)
- Fetch real-time daily stock data using the Alpha Vantage API
- Normalize and prepare data for LSTM input
- Train or load an existing LSTM model
- Predict and visualize:
  - Actual vs. Predicted historical prices
  - Future stock prices (up to 30 days)
- Download forecast results as CSV

---

## 🧰 Tools Used

- Python
- Streamlit
- Alpha Vantage API
- Keras (TensorFlow backend)
- Scikit-learn
- Pandas, Numpy, Matplotlib

---

## 🗂️ Project Structure

```

.
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
└── README.md           # This file

````

---

## 🔑 Get Alpha Vantage API Key

1. Visit [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
2. Sign up and copy your free API key
3. Paste it into the app sidebar when prompted

---

## 🧪 Run Locally

1. Clone this repository:

```bash
git clone https://github.com/yourusername/stock-lstm-app.git
cd stock-lstm-app
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

---

## 🌐 Deploy to Streamlit Cloud

1. Push this project to a GitHub repository
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **“New App”**, connect your GitHub repo
4. Set `app.py` as the main file and deploy

---

## 📸 Screenshots

### 📊 Actual Stock Price

![Actual Stock Price](./ActualStockprice.png)

### 🔮 Predicted Stock Price using LSTM

![Predicted Stock Price](./PredictedStockPrice.png)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

Developed by **Nouhith**.
Feel free to open an issue or contact me if you need help.

```
