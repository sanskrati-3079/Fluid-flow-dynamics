import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

st.set_page_config(layout="wide")
st.title("💧 Dynamic Fluid Flow Classification (Laminar vs Turbulent)")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Flow.csv file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Dataset Preview")
    st.write(df.head())

    st.subheader("🔍 Data Info")
    st.write(f"Shape: {df.shape}")
    st.write("Missing values:", df.isnull().sum().sum())

    # Visualize target distribution
    st.subheader("📊 Flow Type Distribution")
    fig1, ax1 = plt.subplots()
    df['flow_type'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'], ax=ax1)
    st.pyplot(fig1)

    # Winsorize outliers
    numerical_cols = ['t', 'x', 'y', 'u', 'v', 'p', 'dudx', 'dudy', 'dvdx', 'dvdy', 'dudt', 'dvdt']
    for col in numerical_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)

    # Scale features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Encode target
    df['flow_type_encoded'] = df['flow_type'].map({'laminar': 0, 'turbulent': 1})

    # Feature engineering
    df['u_times_v'] = df['u'] * df['v']
    df['p_div_u'] = df['p'] / (df['u'] + 1e-6)
    df['dudx_times_dvdy'] = df['dudx'] * df['dvdy']

    # Model training (basic logistic regression for demo)
    X = df.drop(['flow_type', 'flow_type_encoded', 'sample_id'], axis=1, errors='ignore')
    y = df['flow_type_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    st.subheader("⚙️ Logistic Regression Model Trained")
    st.write(f"Train Accuracy: {model.score(X_train, y_train):.2f}")
    st.write(f"Test Accuracy: {model.score(X_test, y_test):.2f}")

    # Predict and display results
    st.subheader("🔍 Predict Flow Types")
    df['Predicted Flow Type'] = model.predict(X)
    df['Predicted Flow Type'] = df['Predicted Flow Type'].map({0: 'laminar', 1: 'turbulent'})
    st.write(df[['flow_type', 'Predicted Flow Type']].head(10))

    # Correlation heatmap
    st.subheader("📈 Correlation Heatmap")
    corr = df.select_dtypes(include='number').corr()
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    # Optional download of predictions
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results", csv, "classified_flow.csv", "text/csv")
else:
    st.info("Please upload a `Flow.csv` file to begin.")
