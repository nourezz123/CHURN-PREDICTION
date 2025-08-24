import streamlit as st
import pandas as pd
import joblib
import os
from cleaning_preprocessing import preprocess_data, clean_data
from feature_engineering import perform_feature_engineering
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, datetime
# Streamlit setup
st.set_page_config(page_title="Customer Churn Prediction App", layout="wide")
st.title("Customer Churn Prediction App")
# Load the model
MODEL_PATH = 'best_lgb_model.pkl'
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

# Choose input method (Upload dataset or enter manually)
st.sidebar.header("Upload or Input Data")
input_method = st.sidebar.radio("Choose input method", ["Upload CSV", "Enter Data Manually"])

data = None
manual_mode = False

# Input data logic
if input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Dataset")
        st.dataframe(data.head())

        # Prediction for dataset
        if model:
            # Preprocessing and feature engineering
            cleaned = clean_data(data.copy())
            st.subheader("After Cleaning")
            st.dataframe(cleaned.head())
            # Histograms after Cleaning
            st.subheader("📊 Feature Distributions After Cleaning")
            important_numeric_cols = ['age', 'points_in_wallet', 'avg_frequency_login_days']
            fig, axes = plt.subplots(1, len(important_numeric_cols), figsize=(15, 5), facecolor='black')

            for i, col in enumerate(important_numeric_cols):
                if col in cleaned.columns:
                    sns.histplot(cleaned[col], kde=True, bins=30, palette='set2', ax=axes[i])
                    axes[i].set_title(f"Distribution of {col}", color='white')
                    axes[i].set_xlabel(col, color='white')

            st.pyplot(fig)
            st.subheader("📊 Category Distributions (Pie + Bar Plots)")
            categorical_columns = ['past_complaint', 'membership_category', 'complaint_status']
            include_bar_for = 'feedback'

            n_cols = 2
            n_rows = 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows), facecolor='black')
            axes = axes.flatten()

            set2_colors = sns.color_palette("Set2", 10)

            # Pie Charts
            for idx, col in enumerate(categorical_columns):
                if col in cleaned.columns:
                    counts = cleaned[col].value_counts()
                    wedges, texts, autotexts = axes[idx].pie(
                        counts.values,
                        labels=counts.index,
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=set2_colors[:len(counts)],
                        textprops={'fontsize': 8, 'color': 'white'}
                    )
                    axes[idx].set_title(f"{col} Distribution", fontsize=10, color='white')
                    axes[idx].axis('equal')
                    axes[idx].tick_params(colors='white')

            # Bar plot for feedback
            if include_bar_for in cleaned.columns:
                fb_counts = cleaned[include_bar_for].value_counts()
                ax_bar = axes[len(categorical_columns)]
                sns.barplot(x=fb_counts.values, y=fb_counts.index, palette="Set2", ax=ax_bar)
                ax_bar.set_title("Feedback Distribution", fontsize=10, color='white')
                ax_bar.set_xlabel("Count", fontsize=9, color='white')
                ax_bar.tick_params(colors='white')
                ax_bar.set_xlim(0, max(fb_counts.values) * 1.1)
                for i, v in enumerate(fb_counts.values):
                    ax_bar.text(v + 0.3, i, str(v), va='center', fontsize=7, color='white')

            plt.tight_layout()
            st.pyplot(fig)


            preprocessed = preprocess_data(cleaned)
            st.subheader("After Preprocessing")
            st.dataframe(preprocessed.head())

            engineered = perform_feature_engineering(preprocessed)
            st.subheader("After Feature Engineering")
            st.dataframe(engineered.head())

        
            # Common Features from Training
            common_features = [
                'membership_category(Basic Membership)', 'feedback(Products always in Stock)',
                'membership_category(No Membership)', 'log_customer_tenure',
                'feedback(Quality Customer Care)', 'feedback(Reasonable Price)',
                'log_points_in_wallet', 'membership_category(Silver Membership)',
                'feedback(User Friendly Website)', 'membership_category(Gold Membership)',
                'membership_category(Platinum Membership)', 'membership_category(Premium Membership)'
            ]

            # Ensure all expected features exist
            for feature in common_features:
                if feature not in engineered.columns:
                    engineered[feature] = 0

            # Arrange features in the same order
            engineered = engineered[common_features]

            # ✅ Diagnostics
            st.write("Model expects these features: ")
            st.write(common_features)

            # Display Common Features with zero values
            all_zero_columns = engineered.columns[(engineered == 0).all()].tolist()
            if all_zero_columns:
                st.warning(f"⚠️ Features with all-zero values: {all_zero_columns}")

            # Check for missing values in engineered data
            missing_values = engineered.isnull().sum()
            missing_values = missing_values[missing_values > 0]
            if not missing_values.empty:
                st.error("⚠️ There are missing values in the features!")
                st.dataframe(missing_values)

            # Show the final features sent to the model
            st.subheader("Final Features Sent to Model")
            st.dataframe(engineered)

            # Display prediction results
            predictions = model.predict(engineered)
              
            data['Churn Prediction'] = predictions
            st.subheader("Prediction Results")
            st.write(data[['Churn Prediction']].value_counts().rename_axis('Churn').reset_index(name='Count'))

            # Visualizations
            st.subheader("Churn Bar Chart")
            churn_counts = data['Churn Prediction'].value_counts().sort_index()
            churn_labels = ["Not Churn", "Churn"] if len(churn_counts) == 2 else churn_counts.index.astype(str)
            
            # Create one figure with 2 subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='black')
            
            # Barplot
            sns.barplot(x=churn_labels, y=churn_counts.values, palette="Set2", ax=ax1)
            ax1.set_ylabel("Count", color='white')
            ax1.set_title("Churn Distribution", color='white')
            ax1.tick_params(colors='white')  # Color tick labels
            for i, v in enumerate(churn_counts.values):
                ax1.text(i, v + 0.5, str(v), ha='center', color='white')

            # Pie Chart
            ax2.pie(
                churn_counts.values,
                labels=churn_labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=["#66c2a5", "#fc8d62"],
                textprops={'color': 'white'}
            )
            ax2.axis('equal')  # Make the circle equal in dimensions
            ax2.set_title("Churn Pie Chart", color='white')

            # Check if Churn count is higher than No Churn count and show warning
            if churn_counts[1] > churn_counts[0]:
                st.warning("⚠️ Warning: More customers are churning than staying. The current retention strategy may not be effective!")

            # Show combined figure in Streamlit
            st.subheader("Churn Distribution Overview")
            st.pyplot(fig)

elif input_method == "Enter Data Manually":
    manual_mode = True
    st.sidebar.write("## Prediction")
    manual_input = {
    'age': st.sidebar.number_input('Age', 18, 100, 30),
    'gender': st.sidebar.selectbox('Gender', ['M', 'F']),
    'region_category': st.sidebar.selectbox('Region', ['City', 'Town', 'Village']),
    'membership_category': st.sidebar.selectbox('Membership', [
        'No Membership', 'Basic Membership', 'Silver Membership',
        'Gold Membership', 'Platinum Membership', 'Premium Membership']),
    'medium_of_operation': st.sidebar.selectbox('Medium of Operation', ['Desktop', 'Smartphone']),
    'internet_option': st.sidebar.selectbox('Internet Option', ['Wi-Fi', 'Mobile_Data', 'Fiber_Optic']),
    'days_since_last_login': st.sidebar.slider('Days Since Last Login', 0, 60, 10),
    'avg_time_spent': st.sidebar.slider('Average Time Spent', 0.0, 1000.0, 300.0),
    'avg_transaction_value': st.sidebar.slider('Average Transaction Value', 0.0, 100000.0, 20000.0),
    'avg_frequency_login_days': st.sidebar.selectbox('Login Frequency (days)', ['10', '15', '22', '6', '17', '20+']),
    'points_in_wallet': st.sidebar.slider('Points in Wallet', 0.0, 1000.0, 500.0),
    'used_special_discount': st.sidebar.selectbox('Used Special Discount', ['Yes', 'No']),
    'offer_application_preference': st.sidebar.selectbox('Offer Application Preference', ['Yes', 'No']),
    'preferred_offer_types': st.sidebar.selectbox('Preferred Offer Type', [
        'Gift Vouchers/Coupons', 'Credit/Debit Card Offers', 'Without Offers']),
    'past_complaint': st.sidebar.selectbox('Past Complaint', ['Yes', 'No']),
    'complaint_status': st.sidebar.selectbox('Complaint Status', ['Solved', 'Unsolved', 'Solved in Follow-up']),
    'feedback': st.sidebar.selectbox('Feedback', [
        'Poor Product Quality', 'No reason specified', 'Poor Website', 'Poor Customer Service',
        'Reasonable Price', 'Too many ads', 'User Friendly Website',
        'Products always in Stock', 'Quality Customer Care']),
    'joining_date': st.sidebar.date_input('Joining Date', datetime.today())
}

    data = pd.DataFrame([manual_input])
    st.subheader("Entered Data")
    st.dataframe(data)

# Rif st.sidebar.button('Run Prediction'):
    if model is not None and data is not None:
        try:
            data = data.copy()
            # Preprocessing
            preprocessed = preprocess_data(data)
            st.subheader("After Preprocessing")
            st.dataframe(preprocessed)

            # Feature Engineering
            engineered = perform_feature_engineering(preprocessed)
            st.subheader("After Feature Engineering")
            st.dataframe(engineered)

            # Common Features from Training
            common_features = [
                'membership_category(Basic Membership)', 'feedback(Products always in Stock)',
                'membership_category(No Membership)', 'log_customer_tenure',
                'feedback(Quality Customer Care)', 'feedback(Reasonable Price)',
                'log_points_in_wallet', 'membership_category(Silver Membership)',
                'feedback(User Friendly Website)', 'membership_category(Gold Membership)',
                'membership_category(Platinum Membership)', 'membership_category(Premium Membership)'
            ]

            # Ensure all expected features exist
            for feature in common_features:
                if feature not in engineered.columns:
                    engineered[feature] = 0

            # Arrange features in the same order
            engineered = engineered[common_features]

            st.write("Model expects these features: ")
            st.write(common_features)


            # Final Data to Predict
            st.subheader("Final Features Sent to Model")
            st.dataframe(engineered)

            # Prediction
            prediction = model.predict(engineered)

            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error("⚠️ Warning: This customer is at risk of leaving!")
            else:
                st.success("✅ Good news: This customer is likely to stay!")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please upload data or enter data manually to make a prediction.")
