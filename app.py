import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Fluid Dynamics AI Model",
    page_icon="🌊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stNumberInput>div>div>input {
        background-color: white;
    }
    .stSuccess {
        background-color: #dff0d8;
        color: #3c763d;
        padding: 10px;
        border-radius: 5px;
    }
    .stWarning {
        background-color: #fcf8e3;
        color: #8a6d3b;
        padding: 10px;
        border-radius: 5px;
    }
    .header {
        color: #2c3e50;
        text-align: center;
        padding: 20px;
    }
    .subheader {
        color: #34495e;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Header
st.markdown('<h1 class="header">🌊 Fluid Dynamics AI Model</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subheader">Regression & Classification Analysis</h3>', unsafe_allow_html=True)

# Sidebar for file upload
with st.sidebar:
    st.markdown("### 📁 Data Upload")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
    if uploaded_file:
        st.success("Data loaded successfully!")

# Main content
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Display data preview
    st.markdown("### 📊 Data Preview")
    st.dataframe(df.head().style.background_gradient(cmap='Blues'))

    # Features & targets
    X = df[["Inlet_Velocity", "Pressure_Drop", "Geometry_Complexity", "Turbulence_Intensity"]]
    y_reg = df["Avg_Velocity"]
    y_clf = df["Vortex_Formation"]

    # Split data
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    st.session_state.scaler = scaler

    # Training section
    st.markdown("### 🚀 Model Training")
    if st.button("Train Model", key="train_button"):
        with st.spinner("Training in progress..."):
            # Build model
            input_layer = Input(shape=(X.shape[1],))
            x = Dense(64, activation='relu')(input_layer)
            x = Dense(32, activation='relu')(x)
            reg_output = Dense(1, name='avg_velocity')(x)
            clf_output = Dense(1, activation='sigmoid', name='vortex_formation')(x)

            model = Model(inputs=input_layer, outputs=[reg_output, clf_output])
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss={'avg_velocity': 'mse', 'vortex_formation': 'binary_crossentropy'},
                metrics={'avg_velocity': 'mae', 'vortex_formation': 'accuracy'}
            )

            # Train
            history = model.fit(
                X_train_scaled,
                {'avg_velocity': y_reg_train, 'vortex_formation': y_clf_train},
                validation_split=0.1,
                epochs=50,
                batch_size=32,
                verbose=0
            )

            st.session_state.model = model

            # Predict
            y_reg_pred, y_clf_pred = model.predict(X_test_scaled)
            y_clf_pred_binary = (y_clf_pred > 0.5).astype(int)

            # Metrics
            mae = mean_absolute_error(y_reg_test, y_reg_pred)
            acc = accuracy_score(y_clf_test, y_clf_pred_binary)

            # Display metrics in columns
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📈 Model Performance")
                st.metric("Mean Absolute Error (Velocity)", f"{mae:.3f}")
                st.metric("Accuracy (Vortex Formation)", f"{acc:.3f}")

            # Visualization
            with col2:
                st.markdown("### 📊 Training Progress")
                fig, axs = plt.subplots(1, 2, figsize=(12, 4))
                axs[0].plot(history.history['avg_velocity_mae'], label='Train MAE', color='#3498db')
                axs[0].plot(history.history['val_avg_velocity_mae'], label='Val MAE', color='#e74c3c')
                axs[0].set_title("Velocity MAE")
                axs[0].legend()
                axs[0].grid(True, alpha=0.3)

                axs[1].plot(history.history['vortex_formation_accuracy'], label='Train Accuracy', color='#2ecc71')
                axs[1].plot(history.history['val_vortex_formation_accuracy'], label='Val Accuracy', color='#f1c40f')
                axs[1].set_title("Vortex Formation Accuracy")
                axs[1].legend()
                axs[1].grid(True, alpha=0.3)

                st.pyplot(fig)

# Prediction section
if st.session_state.model is not None:
    st.markdown("### 🔮 Make Predictions")
    
    # Create input fields in columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Input Parameters")
        inlet_velocity = st.number_input("Inlet Velocity (m/s)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        pressure_drop = st.number_input("Pressure Drop (Pa)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    with col2:
        st.markdown("#### Flow Characteristics")
        geometry_complexity = st.slider("Geometry Complexity", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        turbulence_intensity = st.slider("Turbulence Intensity", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

    if st.button("Predict", key="predict_button"):
        # Prepare input data
        test_input = np.array([[inlet_velocity, pressure_drop, geometry_complexity, turbulence_intensity]])
        test_input_scaled = st.session_state.scaler.transform(test_input)
        
        # Make prediction
        pred_velocity, pred_vortex = st.session_state.model.predict(test_input_scaled)
        
        # Display results in columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Prediction Results")
            st.metric("Predicted Average Velocity", f"{pred_velocity[0][0]:.2f} m/s")
        with col2:
            vortex_prob = pred_vortex[0][0]
            vortex_pred = "Yes" if vortex_prob > 0.5 else "No"
            st.metric("Vortex Formation", f"{vortex_pred} ({vortex_prob:.1%})")
else:
    st.warning("Please upload your dataset and train the model first.")

