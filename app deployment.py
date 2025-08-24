import streamlit as st
import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from cleaning_preprocessing import preprocess_data, clean_data
from feature_engineering import perform_feature_engineering
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile

# Visualization function for CSV upload only
def plot_visualizations(data):
    if data is not None and 'Churn Prediction' in data.columns:
        # Histograms after Cleaning
        st.subheader("📊 Feature Distributions After Cleaning")
        important_numeric_cols = ['age', 'points_in_wallet', 'avg_frequency_login_days']
        set2_colors = sns.color_palette("Set2", len(important_numeric_cols))
        fig, axes = plt.subplots(1, len(important_numeric_cols), figsize=(15, 5), facecolor='black')
        for i, col in enumerate(important_numeric_cols):
            axes[i].set_facecolor('black')
            sns.histplot(data[col], kde=True, bins=30, ax=axes[i], color=set2_colors[i])
            axes[i].set_title(f"Distribution of {col}", color='white')
            axes[i].tick_params(colors='white')
            axes[i].set_xlabel(col, color='white')
            axes[i].set_ylabel('Count', color='white')
        st.pyplot(fig)

        # Category Distributions (Pie + Bar)
        st.subheader("📊 Category Distributions")
        categorical_columns = ['past_complaint', 'membership_category', 'complaint_status']
        include_bar_for = 'feedback'
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor='black')
        axes = axes.flatten()
        for ax in axes:
            ax.set_facecolor('black')

        # Pie charts
        for idx, col in enumerate(categorical_columns):
            counts = data[col].value_counts()
            wedges, texts, autotexts = axes[idx].pie(
                counts.values,
                labels=counts.index,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'color': 'white', 'fontsize': 8}
            )
            axes[idx].set_title(f"{col} Distribution", color='white')
            axes[idx].axis('equal')

        # Feedback bar plot
        fb_counts = data[include_bar_for].value_counts()
        ax_bar = axes[len(categorical_columns)]
        sns.barplot(x=fb_counts.values, y=fb_counts.index, ax=ax_bar, palette="Set2")
        ax_bar.set_title("Feedback Distribution", color='white')
        ax_bar.tick_params(colors='white')
        ax_bar.set_xlabel('Count', color='white')
        for i, v in enumerate(fb_counts.values):
            ax_bar.text(v + 0.3, i, str(v), va='center', color='white')

        plt.tight_layout()
        st.pyplot(fig)

        # Churn overview
        st.subheader("Churn Distribution Overview")
        churn_counts = data['Churn Prediction'].value_counts().sort_index()
        churn_labels = ["Not Churn", "Churn"] if len(churn_counts) == 2 else churn_counts.index.astype(str)
        churn_colors = sns.color_palette("Set2", len(churn_labels))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='black')
        ax1.set_facecolor('black'); ax2.set_facecolor('black')
        sns.barplot(x=churn_labels, y=churn_counts.values, ax=ax1, palette=churn_colors)
        ax1.set_title("Churn Distribution", color='white'); ax1.set_ylabel('Count', color='white'); ax1.tick_params(colors='white')
        for i, v in enumerate(churn_counts.values): ax1.text(i, v + 0.5, str(v), ha='center', color='white')
        ax2.pie(
            churn_counts.values,
            labels=churn_labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=churn_colors,
            textprops={'color': 'white'}
        )
        ax2.set_title("Churn Pie Chart", color='white'); ax2.axis('equal')
        st.pyplot(fig)

# Helper functions
def get_manual_input():
    return {
        'age': st.sidebar.number_input('Age', 18, 100, 30),
        'gender': st.sidebar.selectbox('Gender', ['Male', 'Female']),
        'region_category': st.sidebar.selectbox('Region Category', ['City', 'Town', 'Village']),
        'membership_category': st.sidebar.selectbox('Membership Category', [
            'No Membership', 'Basic Membership', 'Silver Membership',
            'Gold Membership', 'Platinum Membership', 'Premium Membership']),
        'medium_of_operation': st.sidebar.selectbox('Medium of Operation', ['Desktop', 'Smartphone']),
        'internet_option': st.sidebar.selectbox('Internet Option', ['Wi-Fi', 'Mobile Data', 'Fiber Optic']),
        'days_since_last_login': st.sidebar.slider('Days Since Last Login', 0, 60, 10),
        'avg_time_spent': st.sidebar.slider('Average Time Spent', 0.0, 1000.0, 300.0),
        'avg_transaction_value': st.sidebar.slider('Average Transaction Value', 0.0, 100000.0, 20000.0),
        'avg_frequency_login_days': st.sidebar.selectbox('Average Login Frequency (days)', [10, 15, 22, 6, 17, 20]),
        'points_in_wallet': st.sidebar.slider('Points in Wallet', 0.0, 1000.0, 500.0),
        'used_special_discount': st.sidebar.selectbox('Used Special Discount', ['Yes', 'No']),
        'offer_application_preference': st.sidebar.selectbox('Offer Application Preference', ['Yes', 'No']),
        'preferred_offer_types': st.sidebar.selectbox('Preferred Offer Types', [
            'Gift Vouchers/Coupons', 'Credit/Debit Card Offers', 'Without Offers']),
        'past_complaint': st.sidebar.selectbox('Past Complaint', ['Yes', 'No']),
        'complaint_status': st.sidebar.selectbox('Complaint Status', ['Solved', 'Unsolved', 'Solved in Follow-up']),
        'feedback': st.sidebar.selectbox('Feedback', [
            'Poor Product Quality', 'No reason specified', 'Poor Website', 'Poor Customer Service',
            'Reasonable Price', 'Too many ads', 'User Friendly Website',
            'Products always in Stock', 'Quality Customer Care']),
        'joining_date': st.sidebar.date_input('Joining Date', datetime.today())
    }

def process_data(data):

    preprocessed = preprocess_data(data)
    engineered = perform_feature_engineering(preprocessed)
    required_features = [
        'membership_category(Basic Membership)',
        'feedback(Products always in Stock)',
        'membership_category(No Membership)',
        'log_customer_tenure',
        'feedback(Quality Customer Care)',
        'feedback(Reasonable Price)',
        'log_points_in_wallet',
        'membership_category(Silver Membership)',
        'feedback(User Friendly Website)',
        'membership_category(Gold Membership)',
        'membership_category(Platinum Membership)',
        'membership_category(Premium Membership)'
    ]
    for feat in required_features:
        if feat not in engineered.columns:
            engineered[feat] = 0
    engineered = engineered[required_features]
    data['Churn Prediction'] = model.predict(engineered)
    return data

def display_prediction(data):
    pred = data['Churn Prediction'].iloc[0]
    if pred == 1:
        st.error("⚠️ Warning: This customer is at risk of churning!")
    else:
        st.success("✅ Good news: This customer is likely to stay!")
    st.subheader("Prediction Details")
    st.write(data)

# Streamlit UI setup
st.set_page_config(page_title="Customer Churn Prediction App", layout="wide")
st.title("Customer Churn Prediction App")

# Initialize MLflow
try:
    tmp = tempfile.mkdtemp()
    mlflow.set_tracking_uri(f"file:{tmp}")
    st.success("MLflow initialized successfully")
except Exception as e:
    st.error(f"Error initializing MLflow: {e}")

# Load model
MODEL_PATH = "best_lgb_model.pkl"
try:
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    if model:
        st.success("Prediction model loaded successfully")
    else:
        st.warning("Model file not found. Please ensure it's in the correct path.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Input method
st.sidebar.header("Data Input Method")
input_method = st.sidebar.radio("Choose input method", ["Upload CSV", "Manual Input"])

data = None
if input_method == "Upload CSV":
    file = st.sidebar.file_uploader("Choose CSV file", type=["csv"])
    if file:
        try:
            data = pd.read_csv(file)
            st.subheader("Uploaded Data")
            st.dataframe(data.head())
            if model:
                processed = process_data(data.copy())
                st.subheader("Prediction Results")
                st.write(processed['Churn Prediction'].value_counts())
                plot_visualizations(processed)
        except Exception as e:
            st.error(f"Error processing data: {e}")
else:
    manual = get_manual_input()
    data = pd.DataFrame([manual])
    if st.sidebar.button('Predict'):
        if model:
            try:
                result = process_data(data.copy())
                display_prediction(result)
            except Exception as e:
                st.error(f"Prediction error: {e}")
