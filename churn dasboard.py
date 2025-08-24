import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from datetime import datetime

# Page setup
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>📊 Customer Churn Dashboard</h1>
    <hr style='border: 1px solid #ccc;'/>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("churn_processed.csv")
    return df

data_fe = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filter Options")

churn_map = {0: "Not Churned", 1: "Churned"}
inv_churn_map = {v: k for k, v in churn_map.items()}
churn_filter = st.sidebar.multiselect(
    "Churn Status", options=list(churn_map.values()), default=list(churn_map.values())
)

gender_filter = st.sidebar.multiselect(
    "Gender", options=data_fe['gender'].dropna().unique(), default=data_fe['gender'].dropna().unique()
)

region_filter = st.sidebar.multiselect(
    "Region Category", options=['City', 'Town', 'Village'], default=['City', 'Town', 'Village']
)

internet_options = [
    col for col in data_fe.columns 
    if col.startswith('internet_option(') and not col.endswith('_log')
]
internet_labels = [col.replace("internet_option(", "").replace(")", "") for col in internet_options]
internet_dict = dict(zip(internet_labels, internet_options))

internet_filter_labels = st.sidebar.multiselect(
    "Internet Option", options=internet_labels, default=internet_labels
)
internet_filter = [internet_dict[label] for label in internet_filter_labels]

# Move Customer Tenure filter below Internet Option filter
tenure_min, tenure_max = int(data_fe['customer_tenure'].min()), int(data_fe['customer_tenure'].max())
tenure_range = st.sidebar.slider("Customer Tenure (Months)", min_value=tenure_min, max_value=tenure_max, value=(tenure_min, tenure_max))

trans_min, trans_max = float(data_fe['avg_transaction_value'].min()), float(data_fe['avg_transaction_value'].max())
transaction_range = st.sidebar.slider("Avg Transaction Value", min_value=trans_min, max_value=trans_max, value=(trans_min, trans_max))

# Apply filters to data
data_fe['Churn_Label'] = data_fe['churn_risk_score'].map(churn_map)

filtered_data = data_fe[(
    data_fe['churn_risk_score'].isin([inv_churn_map[ch] for ch in churn_filter])) &
    (data_fe['gender'].isin(gender_filter)) &
    (data_fe[[f'region_category({r})' for r in region_filter]].sum(axis=1) > 0) &
    (data_fe[internet_filter].sum(axis=1) > 0) &
    (data_fe['customer_tenure'].between(tenure_range[0], tenure_range[1])) &
    (data_fe['avg_transaction_value'].between(transaction_range[0], transaction_range[1]))
]

# Raw Data Display
with st.expander("📄 Show/Hide Raw Data (Filtered)"):
    st.dataframe(filtered_data.head(50), use_container_width=True)
    csv = filtered_data.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered Data", data=csv, file_name="filtered_churn.csv", mime="text/csv")

# 🎯 KPI Cards
st.subheader("📌 Summary Metrics")

total_customers = len(filtered_data)
churned_customers = filtered_data[filtered_data['Churn_Label'] == 'Churned'].shape[0]
not_churned_customers = filtered_data[filtered_data['Churn_Label'] == 'Not Churned'].shape[0]
churn_rate = (churned_customers / total_customers) * 100 if total_customers > 0 else 0

avg_tenure = filtered_data['customer_tenure'].mean()
tenure_percentage = (avg_tenure / tenure_max) * 100 if tenure_max > 0 else 0
avg_transaction = filtered_data['avg_transaction_value'].mean()
transaction_percentage = (avg_transaction / trans_max) * 100 if trans_max > 0 else 0
total_fiber_optic = filtered_data[[col for col in internet_options if 'Fiber_Optic' in col]].sum().sum()
internet_usage_sums = filtered_data[internet_filter].sum()
most_common_internet = internet_usage_sums.idxmax().replace("internet_option(", "").replace(")", "") if not internet_usage_sums.empty else "N/A"

metrics = [
    {"title": "Total Customers", "value": total_customers},
    {"title": "Churned", "value": churned_customers},
    {"title": "Not Churned", "value": not_churned_customers},
    {"title": "Churn Rate", "value": f"{churn_rate:.2f}%"},
    {"title": "Avg Tenure", "value": f"{tenure_percentage:.1f}%"},
    {"title": "Avg Transaction", "value": f"{transaction_percentage:.1f}%"},
    {"title": "Fiber Optic Users", "value": int(total_fiber_optic)},
    {"title": "Top Internet Option", "value": most_common_internet},
]

for i in range(0, len(metrics), 4):
    cols = st.columns(4)
    for col, metric in zip(cols, metrics[i:i+4]):
        col.markdown(f"""
            <div style="background-color:#1e1e2f;padding:20px;border-radius:15px;
                        text-align:center;border:1px solid #333;box-shadow:0 4px 6px rgba(0,0,0,0.3);min-height:110px;">
                <h4 style="color:#cccccc;word-wrap:break-word;line-height:1.3;">{metric['title']}</h4>
                <h2 style="color:#1fa2ff;margin-top:5px;">{metric['value']}</h2>
            </div>
        """, unsafe_allow_html=True)

# 📊 Visual Insights
st.subheader("📈 Visual Insights")

main_colors = ['#1f77b4', '#ff7f0e']

col5, col6 = st.columns(2)
with col5:
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    sns.countplot(data=filtered_data, x='Churn_Label', palette=main_colors, ax=ax1)
    ax1.set_title('Customer Churn Count')
    st.pyplot(fig1)

with col6:
    churn_counts = filtered_data['Churn_Label'].value_counts()
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.pie(churn_counts, labels=churn_counts.index.tolist(), autopct='%1.1f%%', startangle=140,
            colors=main_colors, textprops={'fontsize': 12})
    ax2.set_title('Churn Distribution (%)')
    st.pyplot(fig2)

col7, col8 = st.columns(2)
with col7:
    fig3, ax3 = plt.subplots()
    sns.countplot(x='gender', hue='Churn_Label', data=filtered_data, palette=main_colors, ax=ax3)
    ax3.set_title('Churn by Gender')
    st.pyplot(fig3)

with col8:
    fig4, ax4 = plt.subplots()
    sns.boxplot(x='Churn_Label', y='avg_transaction_value', data=filtered_data, palette=main_colors, ax=ax4)
    ax4.set_title('Avg Transaction Value by Churn Status')
    st.pyplot(fig4)

region_cols = ['region_category(City)', 'region_category(Town)', 'region_category(Village)']
region_churn = filtered_data.groupby('Churn_Label')[region_cols].sum().T

internet_churn = filtered_data.groupby('Churn_Label')[internet_options].sum().T

col9, col10 = st.columns(2)

with col9:
    fig6, ax6 = plt.subplots()
    region_churn.plot(kind='bar', stacked=True, ax=ax6, color=main_colors)
    ax6.set_title('Churn Distribution by Region')
    st.pyplot(fig6.figure)

with col10:
    fig7, ax7 = plt.subplots()
    internet_churn.plot(kind='bar', ax=ax7, color=main_colors)
    ax7.set_title('Churn by Internet Option')
    plt.xticks(rotation=45)
    st.pyplot(fig7.figure)


# 8. Stacked Area Plot Over Time (Joining Date vs Churn)
st.subheader("📈 Customer Joining Trend Over Time by Churn Status")

# Ensure joining_date is in datetime format
data_fe['joining_date'] = pd.to_datetime(data_fe['joining_date'])

# Group by month and churn label
time_churn = data_fe.groupby([pd.Grouper(key='joining_date', freq='M'), 'Churn_Label']).size().unstack(fill_value=0)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
time_churn.plot(kind='line', stacked=True, ax=ax)
ax.set_title('Customer Joining Trend Over Time by Churn Status')
ax.set_xlabel('Joining Date')
ax.set_ylabel('Number of Customers')
ax.legend(title='Churn Status')
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)



# ⭐ Feature Selection
st.subheader("🌟 Feature Selection Using RFE + RandomForest")

data_rfe = data_fe.copy()
data_rfe['joining_date'] = pd.to_datetime(data_rfe['joining_date'], errors='coerce', dayfirst=True)
today = pd.to_datetime(datetime.today().date())
data_rfe['days_since_joining'] = (today - data_rfe['joining_date']).dt.days
data_rfe = data_rfe.drop('joining_date', axis=1)

X_rfe = data_rfe.drop(['churn_risk_score', 'Churn_Label'], axis=1)
y_rfe = data_rfe['churn_risk_score']
X_rfe = X_rfe.select_dtypes(include=['int64', 'float64'])

model = RandomForestClassifier(random_state=42)
rfe = RFE(model, n_features_to_select=15)
rfe.fit(X_rfe, y_rfe)

selected_features = X_rfe.columns[rfe.support_]
model.fit(X_rfe[selected_features], y_rfe)
importances = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

imp_df = importance_df[importance_df['Importance'] > 0.03]

st.write("#### Selected Features with Importance > 0.03")
st.dataframe(imp_df)

fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
ax_imp.barh(imp_df['Feature'], imp_df['Importance'], color='#ff7f0e')
ax_imp.set_xlabel('Importance')
ax_imp.set_ylabel('Features')
ax_imp.set_title('Feature Importance via RFE')
ax_imp.invert_yaxis()
st.pyplot(fig_imp)
