# 🎨 Customer Churn Prediction – Frontend (Streamlit App)

Beautiful, modern, and interactive Streamlit web application for predicting **customer churn**.  
This frontend communicates with a FastAPI backend to deliver real-time and batch predictions.

---

## ✨ Features

### 🔮 Single Prediction
- Predict churn for **one customer**
- Clean and interactive form
- Probability gauge visualization
- Color‑coded risk level (🟢 Low • 🟡 Medium • 🔴 High)

### 📊 Batch Prediction
- Upload CSV files
- Batch churn prediction for **hundreds of customers**
- Pie chart, bar chart & histogram analytics
- Downloadable prediction results

### 📈 Model Info Page
- Model metadata
- Performance metrics
- Radar chart visualization
- Model comparison table

### 🎨 Beautiful UI
- Gradient themed UI
- Custom CSS for enhanced visuals
- Fully responsive layouts
- Smooth animations & modern aesthetics

---

## 🚀 Installation

### Install dependencies:

```bash
pip install streamlit plotly requests pandas
```

Or install everything from project root:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Frontend App

### **Step 1 — Make sure the backend API is running**

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### **Step 2 — Run the Streamlit app**

From project root:

```bash
streamlit run frontend/app.py
```

Or from inside the frontend folder:

```bash
cd frontend
streamlit run app.py
```

App opens at:

📍 **http://localhost:8501**

---

## ⚙️ API Configuration

The app connects to this backend API:

```python
API_URL = "http://localhost:8000"
```

To modify it, edit the `API_URL` variable inside **app.py**.

---

## 📁 Project Structure (Frontend Only)

```
frontend/
│
├── app.py         # Main Streamlit application
├── README.md      # This documentation
└── (assets)       # Future images, styles, etc.
```

---

## 🏠 Pages Included

### **Home Page**
- Overview of system
- Key insights section
- Project highlights

### **Single Prediction**
- Interactive form for input
- Model prediction visualization
- Risk classification

### **Batch Prediction**
- CSV uploader
- Rich analytics dashboard
- Downloadable results

### **Model Info**
- Metrics cards
- Radar chart
- Comparison table

---

## 📉 Visual Features

- 🔵 Probability gauge meter  
- 🟠 Interactive Plotly visuals  
- 🟣 Pie charts  
- 🟩 Bar charts  
- 🟦 Histograms  
- 🟪 Styled metric cards  

---

## 🛠️ Troubleshooting

### ❌ API Connection Error
- Backend not running
- Wrong API_URL
- Firewall blocking 8000 port

### ❌ Import Errors
Run:

```bash
pip install streamlit plotly requests pandas
```

### ❌ Port Already In Use
Run Streamlit on alternate port:

```bash
streamlit run app.py --server.port 8502
```

---

## 🔧 Customization

### 🎨 Change Colors & Theme
Modify CSS block at the top of `app.py`:

```python
st.markdown("""<style> ... </style>""", unsafe_allow_html=True)
```

### 🚦 Update Risk Thresholds
Edit the `get_risk_color()` function to adjust categories.

### 📄 Add New Pages
Create new functions and add to the sidebar navigation.

---

## 📦 Dependencies

- Streamlit  
- Plotly  
- Requests  
- Pandas  

---

## 🚀 Performance
- Real-time prediction  
- Smooth UI rendering  
- Efficient large CSV processing  
- Optimized charts  

---

## ⭐ Author

Made by **Crystal Jain**

🔗 GitHub: https://github.com/crystaljain27  
🔗 LinkedIn: https://www.linkedin.com/in/crystal-jain-b10025264  

---

## ⭐ Support  
If you found this useful, please ⭐ the repo!

