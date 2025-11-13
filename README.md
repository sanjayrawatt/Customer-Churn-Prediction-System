# ⭐ Customer Churn Prediction System  
An end‑to‑end machine learning application that predicts telecom customer churn using a fully deployed **FastAPI backend** and **Streamlit frontend**.

This project was built and documented by **Sanjay Singh Rawat**.  
It includes a complete ML pipeline, interactive UI, and production-grade API.

---

## 🚀 Live Demo

### 🔹 Frontend (Streamlit App)  
**https://customer-churn-prediction-system-q4wjmmfuyc4wukkffro2pz.streamlit.app/**

### 🔹 Backend API (FastAPI - Render)  
**https://customer-churn-prediction-system-yoa6.onrender.com/docs**

---

## 📌 Overview  
This project solves the business problem of **customer churn prediction** using:

- A trained XGBoost model  
- A preprocessing pipeline (encoding, scaling, feature handling)  
- A REST API built with FastAPI  
- A user‑friendly interface built in Streamlit  
- Fully deployed cloud services  

---

## ✨ Features  

### 🔹 Machine Learning  
- XGBoost classification model  
- Preprocessing (Label Encoding, Standard Scaling)  
- Feature metadata + model artifacts  
- Robust batch processing

### 🔹 Backend (FastAPI)  
- `/predict` — Single customer prediction  
- `/predict/batch` — CSV batch prediction  
- `/model/info` — Returns model metadata  
- `/health` — API health check  
- Automatic Swagger UI documentation  
- Modular and clean architecture  

### 🔹 Frontend (Streamlit)  
- Interactive form for single predictions  
- CSV upload for batch predictions  
- Probability gauges  
- Risk classification  
- Analytics visualization  
- Downloadable results  

### 🔹 Deployment  
- Backend deployed using **Render (Docker)**  
- Frontend deployed using **Streamlit Cloud**  
- Production-ready environment  

---

## 🧱 Tech Stack  
- **XGBoost, Pandas, NumPy**  
- **FastAPI, Uvicorn**  
- **Streamlit, Plotly**  
- **Docker, Render, Streamlit Cloud**

---

## 📂 Folder Structure  

```
Customer-Churn-Prediction-System/
│
├── backend/
│   ├── api/
│   ├── core/
│   ├── services/
│   └── main.py
│
├── frontend/
│   └── app.py
│
├── models/
├── data/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🛠️ Running Locally  

### 1️⃣ Clone the repository  
```
git clone https://github.com/sanjayrawatt/Customer-Churn-Prediction-System.git
```

### 2️⃣ Start Backend  
```
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3️⃣ Start Frontend  
```
streamlit run frontend/app.py
```

---

## 👤 Author  
Built and maintained by **Sanjay Singh Rawat**.

---

## ⭐ Support  
If you like this project, consider giving it a **⭐ star on GitHub**!
