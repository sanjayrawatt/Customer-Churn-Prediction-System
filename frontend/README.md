# 🎨 Customer Churn Prediction – Streamlit Frontend

This folder contains the **Streamlit-based frontend** for the Customer Churn Prediction System built by **Sanjay Singh Rawat**.

The frontend interacts with the FastAPI backend and provides a modern, clean user interface.

---

## 🚀 Features

- 🧾 **Single Prediction Form** — Predict churn for individual users  
- 📂 **Batch Prediction** — Upload CSV for bulk predictions  
- 📊 **Analytics Visualizations** — Probability charts, risk distribution  
- 🎚️ **Risk Classification** — Low / Medium / High churn risk  
- 💾 **Downloadable CSV Output**  
- 🎨 **Custom UI Styling**  

---

## 🔗 Backend Connection  
The frontend communicates with this backend API:

```
API_URL = "https://customer-churn-prediction-system-yoa6.onrender.com"
```

---

## 🛠️ Run Locally  

### Start backend first:
```
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Run Streamlit app:
```
cd frontend
streamlit run app.py
```

The app will open at:

```
http://localhost:8501
```

---

## 📦 Requirements  
Install dependencies:

```
pip install -r ../requirements.txt
```

or

```
pip install streamlit plotly pandas requests
```

---

## 👤 Author  
Created by **Sanjay Singh Rawat**.

---

Enjoy the clean UI and smooth predictions! 🚀
