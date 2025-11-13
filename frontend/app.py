"""
Customer Churn Prediction - Streamlit Frontend
Beautiful and user-friendly interface for churn prediction
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
        padding-bottom: 1rem;
    }
    h2 {
        color: #2c3e50;
        font-weight: 600;
        padding-top: 1rem;
    }
    h3 {
        color: #34495e;
        font-weight: 500;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .high-risk {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .medium-risk {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #333;
    }
    .low-risk {
        background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
        color: white;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
</style>
""",
    unsafe_allow_html=True,
)

# API Configuration
API_URL = "https://customer-churn-prediction-system-yoa6.onrender.com"



# Helper Functions
def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def predict_single(customer_data):
    """Make single prediction"""
    try:
        response = requests.post(f"{API_URL}/predict", json=customer_data, timeout=10)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Connection error: {str(e)}"


def predict_batch(customers_data):
    """Make batch prediction"""
    try:
        response = requests.post(
            f"{API_URL}/predict/batch", json={"customers": customers_data}, timeout=30
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Connection error: {str(e)}"


def get_risk_color(risk_level):
    """Get color for risk level"""
    colors = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
    return colors.get(risk_level, "#6b7280")


def create_probability_gauge(probability):
    """Create a gauge chart for churn probability"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": "Churn Probability (%)",
                "font": {"size": 24, "color": "#1e293b"},
            },
            delta={"reference": 50, "increasing": {"color": "red"}},
            number={"font": {"size": 40, "color": "#1e293b"}},
            gauge={
                "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "#1e293b"},
                "bar": {"color": "#3b82f6"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#94a3b8",
                "steps": [
                    {"range": [0, 30], "color": "#d1fae5"},
                    {"range": [30, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )

    fig.update_layout(
        height=300,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1e293b", "family": "Arial"},
    )

    return fig


# Main App
def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<h1 style='text-align: center;'>📊 Customer Churn Prediction System</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align: center; color: #666; font-size: 1.1em;'>Predict customer churn using advanced ML models</p>",
            unsafe_allow_html=True,
        )

    # Check API health
    api_status = check_api_health()

    if not api_status:
        st.error(
            "⚠️ Cannot connect to the API. Please make sure the backend is running on http://localhost:8000"
        )
        st.info("Run: `uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`")
        return

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/analytics.png", width=150)
        st.markdown("## 🎯 Navigation")

        page = st.radio(
            "Select a page:",
            ["🏠 Home", "🔮 Single Prediction", "📊 Batch Prediction", "📈 Model Info"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 🎨 About")
        st.info(
            "This application uses XGBoost ML model to predict customer churn with 93.18% ROC-AUC score."
        )

        st.markdown("---")
        st.markdown("### 📡 API Status")
        if api_status:
            st.success("✅ Connected")
        else:
            st.error("❌ Disconnected")

    # Main content
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Single Prediction":
        show_single_prediction_page()
    elif page == "📊 Batch Prediction":
        show_batch_prediction_page()
    elif page == "📈 Model Info":
        show_model_info_page()


def show_home_page():
    """Home page with overview"""
    st.markdown("## Welcome to Customer Churn Prediction System! 👋")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; color: white;'>
            <h3 style='margin: 0; color: white;'>93.18%</h3>
            <p style='margin: 0;'>ROC-AUC Score</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; color: white;'>
            <h3 style='margin: 0; color: white;'>85.41%</h3>
            <p style='margin: 0;'>Accuracy</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; color: white;'>
            <h3 style='margin: 0; color: white;'>30</h3>
            <p style='margin: 0;'>Features Used</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
        <div style='background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; color: white;'>
            <h3 style='margin: 0; color: white;'>XGBoost</h3>
            <p style='margin: 0;'>Best Model</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Features
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Key Features")
        st.markdown(
            """
        - ✅ **Real-time Predictions**: Get instant churn predictions
        - ✅ **Batch Processing**: Analyze multiple customers at once
        - ✅ **Risk Classification**: Low, Medium, and High risk levels
        - ✅ **Advanced ML Model**: XGBoost with 93.18% ROC-AUC
        - ✅ **Interactive Dashboard**: Beautiful and user-friendly interface
        - ✅ **Detailed Analytics**: Comprehensive model performance metrics
        """
        )

    with col2:
        st.markdown("### 📊 How It Works")
        st.markdown(
            """
        1. **Input Customer Data**: Enter customer details or upload CSV
        2. **AI Analysis**: XGBoost model analyzes 30+ features
        3. **Get Prediction**: Receive churn probability and risk level
        4. **Take Action**: Use insights for retention strategies
        """
        )

        st.info("💡 **Tip**: Start with a single prediction to see how it works!")

    # Sample insights
    st.markdown("### 🔍 Key Insights from Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div style='background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3b82f6;'>
            <h4 style='color: #1e293b;'>📅 Contract Type</h4>
            <p style='color: #334155;'>Month-to-month contracts have <strong>42.71% churn</strong> vs 2.83% for two-year contracts</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div style='background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #8b5cf6;'>
            <h4 style='color: #1e293b;'>⏱️ Tenure Effect</h4>
            <p style='color: #334155;'>Churned customers average <strong>18 months</strong> tenure vs 37.6 months for retained</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div style='background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ec4899;'>
            <h4 style='color: #1e293b;'>👥 Family Status</h4>
            <p style='color: #334155;'>Customers with partners/dependents have <strong>19.82% churn</strong> vs 34.24% without</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


def show_single_prediction_page():
    """Single prediction page"""
    st.markdown("## 🔮 Single Customer Prediction")
    st.markdown("Enter customer details below to predict churn probability")

    # Create form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 👤 Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])

        with col2:
            st.markdown("#### 📞 Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["Yes", "No", "No phone service"]
            )
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )

        with col3:
            st.markdown("#### 🛡️ Additional Services")
            device_protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📋 Account Info")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox(
                "Contract Type", ["Month-to-month", "One year", "Two year"]
            )

        with col2:
            st.markdown("#### 💳 Billing")
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        with col3:
            st.markdown("#### 💰 Charges")
            monthly_charges = st.number_input(
                "Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0
            )
            total_charges = st.number_input(
                "Total Charges ($)", 0.0, 10000.0, 840.0, step=50.0
            )

        # Submit button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_button = st.form_submit_button("🚀 Predict Churn", width="stretch")

    if submit_button:
        # Prepare data
        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        # Make prediction
        with st.spinner("🔄 Analyzing customer data..."):
            result, error = predict_single(customer_data)

        if error:
            st.error(f"❌ {error}")
        else:
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")

            col1, col2 = st.columns([1, 1])

            with col1:
                # Probability gauge
                fig = create_probability_gauge(result["churn_probability"])
                st.plotly_chart(fig, config={"displayModeBar": False})

            with col2:
                st.markdown("<br><br>", unsafe_allow_html=True)

                # Risk level
                risk_color = get_risk_color(result["risk_level"])
                risk_icon = (
                    "🔴"
                    if result["risk_level"] == "High"
                    else "🟡" if result["risk_level"] == "Medium" else "🟢"
                )

                st.markdown(
                    f"""
                <div style='background: {risk_color}; padding: 2rem; border-radius: 1rem; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h2 style='color: white; margin: 0;'>{risk_icon} {result['risk_level']} Risk</h2>
                    <h3 style='color: white; margin: 0.5rem 0;'>{result['churn_label']}</h3>
                    <p style='margin: 0; font-size: 1.2em;'>Probability: {result['churn_probability']:.1%}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # Recommendations
                if result["risk_level"] == "High":
                    st.warning(
                        """
                    **⚠️ Recommended Actions:**
                    - Immediate contact with retention team
                    - Offer personalized incentives
                    - Review service satisfaction
                    - Consider contract upgrade options
                    """
                    )
                elif result["risk_level"] == "Medium":
                    st.info(
                        """
                    **💡 Recommended Actions:**
                    - Proactive engagement
                    - Monitor account activity
                    - Offer value-added services
                    - Check customer satisfaction
                    """
                    )
                else:
                    st.success(
                        """
                    **✅ Recommended Actions:**
                    - Continue standard engagement
                    - Maintain service quality
                    - Regular satisfaction checks
                    - Loyalty program enrollment
                    """
                    )


def show_batch_prediction_page():
    """Batch prediction page"""
    st.markdown("## 📊 Batch Customer Prediction")
    st.markdown("Upload a CSV file with multiple customer records for batch prediction")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Upload a CSV file with customer data. Make sure it has all required columns.",
    )

    # Sample data button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📥 Download Sample CSV Template", width="stretch"):
            # Create sample CSV
            sample_data = {
                "gender": ["Male", "Female"],
                "SeniorCitizen": [0, 1],
                "Partner": ["Yes", "No"],
                "Dependents": ["No", "Yes"],
                "tenure": [12, 48],
                "PhoneService": ["Yes", "Yes"],
                "MultipleLines": ["No", "Yes"],
                "InternetService": ["Fiber optic", "DSL"],
                "OnlineSecurity": ["No", "Yes"],
                "OnlineBackup": ["Yes", "Yes"],
                "DeviceProtection": ["No", "Yes"],
                "TechSupport": ["No", "Yes"],
                "StreamingTV": ["Yes", "No"],
                "StreamingMovies": ["Yes", "No"],
                "Contract": ["Month-to-month", "Two year"],
                "PaperlessBilling": ["Yes", "No"],
                "PaymentMethod": ["Electronic check", "Bank transfer (automatic)"],
                "MonthlyCharges": [85.50, 65.00],
                "TotalCharges": [1026.00, 3120.00],
            }
            df_sample = pd.DataFrame(sample_data)
            csv = df_sample.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Template",
                data=csv,
                file_name="customer_churn_template.csv",
                mime="text/csv",
                width="stretch",
            )

    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)

            st.success(f"✅ Successfully loaded {len(df)} customer records")

            # Show preview
            with st.expander("👀 Preview Data"):
                st.dataframe(df.head(10), width="stretch")

            # Predict button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button(
                    "🚀 Run Batch Prediction", width="stretch", type="primary"
                ):
                    # Convert to list of dicts
                    customers_data = df.to_dict("records")

                    # Make batch prediction
                    with st.spinner(f"🔄 Analyzing {len(df)} customers..."):
                        result, error = predict_batch(customers_data)

                    if error:
                        st.error(f"❌ {error}")
                    else:
                        # Display results
                        st.markdown("---")
                        st.markdown("## 📊 Batch Prediction Results")

                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Total Customers", result["total_customers"])
                        with col2:
                            st.metric(
                                "Predicted Churners", result["predicted_churners"]
                            )
                        with col3:
                            st.metric("Churn Rate", f"{result['churn_rate']*100:.1f}%")
                        with col4:
                            retention_rate = (1 - result["churn_rate"]) * 100
                            st.metric("Retention Rate", f"{retention_rate:.1f}%")

                        # Create results dataframe
                        predictions_df = pd.DataFrame(result["predictions"])
                        results_df = pd.concat([df, predictions_df], axis=1)

                        # Risk distribution
                        st.markdown("### 📈 Risk Distribution")

                        col1, col2 = st.columns([1, 1])

                        with col1:
                            # Pie chart
                            risk_counts = predictions_df["risk_level"].value_counts()
                            fig = px.pie(
                                values=risk_counts.values,
                                names=risk_counts.index,
                                title="Customers by Risk Level",
                                color=risk_counts.index,
                                color_discrete_map={
                                    "Low": "#10b981",
                                    "Medium": "#f59e0b",
                                    "High": "#ef4444",
                                },
                            )
                            fig.update_traces(
                                textposition="inside", textinfo="percent+label"
                            )
                            st.plotly_chart(fig, config={"displayModeBar": False})

                        with col2:
                            # Bar chart
                            fig = px.bar(
                                x=risk_counts.index,
                                y=risk_counts.values,
                                title="Risk Level Distribution",
                                labels={"x": "Risk Level", "y": "Number of Customers"},
                                color=risk_counts.index,
                                color_discrete_map={
                                    "Low": "#10b981",
                                    "Medium": "#f59e0b",
                                    "High": "#ef4444",
                                },
                            )
                            st.plotly_chart(fig, config={"displayModeBar": False})

                        # Probability distribution
                        st.markdown("### 📊 Churn Probability Distribution")
                        fig = px.histogram(
                            predictions_df,
                            x="churn_probability",
                            nbins=30,
                            title="Distribution of Churn Probabilities",
                            labels={"churn_probability": "Churn Probability"},
                            color_discrete_sequence=["#667eea"],
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, config={"displayModeBar": False})

                        # Detailed results table
                        st.markdown("### 📋 Detailed Predictions")

                        # Add filters
                        col1, col2 = st.columns(2)
                        with col1:
                            risk_filter = st.multiselect(
                                "Filter by Risk Level",
                                options=["Low", "Medium", "High"],
                                default=["Low", "Medium", "High"],
                            )
                        with col2:
                            churn_filter = st.multiselect(
                                "Filter by Churn Prediction",
                                options=["No Churn", "Churn"],
                                default=["No Churn", "Churn"],
                            )

                        # Apply filters
                        filtered_df = results_df[
                            (results_df["risk_level"].isin(risk_filter))
                            & (results_df["churn_label"].isin(churn_filter))
                        ]

                        # Display filtered results
                        st.dataframe(filtered_df, width="stretch", height=400)

                        # Download results
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col2:
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv,
                                file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                width="stretch",
                            )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info(
                "Please make sure your CSV has all required columns. Download the template for reference."
            )


def show_model_info_page():
    """Model information page"""
    st.markdown("## 📈 Model Performance & Information")

    # Get model info
    model_info = get_model_info()

    if model_info:
        # Model overview
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 🤖 Model Details")
            st.markdown(
                f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 1rem; color: white;'>
                <h2 style='color: white; margin: 0;'>{model_info['model_name']}</h2>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.1em;'>Gradient Boosting Classifier</p>
                <p style='margin: 0.5rem 0 0 0;'>Features: {model_info['n_features']}</p>
                <p style='margin: 0.5rem 0 0 0;'>Training Date: {model_info['training_date']}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown("### 🎯 Best Metric")
            st.markdown(
                f"""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 2rem; border-radius: 1rem; text-align: center; color: white;'>
                <h1 style='color: white; margin: 0; font-size: 3em;'>{model_info['roc_auc']:.1%}</h1>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.2em;'>ROC-AUC Score</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Performance metrics
        st.markdown("### 📊 Performance Metrics")

        col1, col2, col3, col4, col5 = st.columns(5)

        metrics = [
            ("Accuracy", model_info["accuracy"], "🎯"),
            ("Precision", model_info["precision"], "🔍"),
            ("Recall", model_info["recall"], "📍"),
            ("F1-Score", model_info["f1_score"], "⚖️"),
            ("ROC-AUC", model_info["roc_auc"], "📈"),
        ]

        for col, (name, value, icon) in zip([col1, col2, col3, col4, col5], metrics):
            with col:
                st.markdown(
                    f"""
                <div style='background: #f8fafc; padding: 1.5rem; border-radius: 0.5rem; text-align: center; border: 2px solid #e2e8f0;'>
                    <div style='font-size: 2em;'>{icon}</div>
                    <h3 style='margin: 0.5rem 0; color: #1e40af;'>{value:.4f}</h3>
                    <p style='margin: 0; color: #64748b;'>{name}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Metrics visualization
        st.markdown("<br>", unsafe_allow_html=True)

        # Create radar chart
        categories = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        values = [
            model_info["accuracy"],
            model_info["precision"],
            model_info["recall"],
            model_info["f1_score"],
            model_info["roc_auc"],
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name="Model Performance",
                line_color="#667eea",
                fillcolor="rgba(102, 126, 234, 0.3)",
            )
        )

        fig.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
            showlegend=False,
            title="Model Performance Radar Chart",
            height=500,
        )

        st.plotly_chart(fig, config={"displayModeBar": False})

        # Model comparison (from training)
        st.markdown("### 🏆 Model Comparison")
        st.markdown("Performance comparison of different models during training:")

        comparison_data = {
            "Model": ["XGBoost", "Gradient Boosting", "Random Forest", "Decision Tree"],
            "Accuracy": [0.8541, 0.8556, 0.8493, 0.8159],
            "F1-Score": [0.8561, 0.8559, 0.8538, 0.8247],
            "ROC-AUC": [0.9318, 0.9304, 0.9291, 0.8785],
            "Training Time (s)": [15.88, 738.79, 28.98, 4.86],
        }

        df_comparison = pd.DataFrame(comparison_data)

        # Bar chart comparison
        fig = make_subplots(
            rows=1, cols=3, subplot_titles=("Accuracy", "F1-Score", "ROC-AUC")
        )

        fig.add_trace(
            go.Bar(
                x=df_comparison["Model"],
                y=df_comparison["Accuracy"],
                marker_color="#667eea",
                name="Accuracy",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=df_comparison["Model"],
                y=df_comparison["F1-Score"],
                marker_color="#f093fb",
                name="F1-Score",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Bar(
                x=df_comparison["Model"],
                y=df_comparison["ROC-AUC"],
                marker_color="#30cfd0",
                name="ROC-AUC",
            ),
            row=1,
            col=3,
        )

        fig.update_layout(
            height=400, showlegend=False, title_text="Model Metrics Comparison"
        )
        st.plotly_chart(fig, config={"displayModeBar": False})

        # Show comparison table
        st.dataframe(df_comparison, width="stretch")

    else:
        st.error("❌ Could not fetch model information from API")


if __name__ == "__main__":
    main()
