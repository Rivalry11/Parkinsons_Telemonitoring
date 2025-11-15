import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("🔮 Predicción del puntaje motor_UPDRS")
st.markdown("""
Esta sección permite ingresar valores acústicos y clínicos del paciente
para predecir su **motor_UPDRS** usando el modelo final entrenado.
""")

# -----------------------------------------------------
# Cargar MODELO y SCALER
# -----------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "models/modelo_final.pkl"
    scaler_path = "models/scaler.pkl"

    if not os.path.exists(model_path):
        st.error("❌ No se encontró el archivo `modelo_final.pkl` en la carpeta models/.")
        st.stop()

    if not os.path.exists(scaler_path):
        st.error("❌ No se encontró el archivo `scaler.pkl` en la carpeta models/.")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model()

# -----------------------------------------------------
# Definir variables necesarias (features)
# -----------------------------------------------------
VARIABLES = [
    'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
    'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE',
    'DFA', 'PPE'
]

st.subheader("📝 Ingresa los valores del paciente")

user_inputs = {}

# Crear formulario con las variables
for var in VARIABLES:
    user_inputs[var] = st.number_input(
        var, 
        min_value=0.0, 
        max_value=200.0, 
        value=0.0,
        step=0.01
    )

# Convertir a DataFrame
input_df = pd.DataFrame([user_inputs])

# -----------------------------------------------------
# PREDICCIÓN
# -----------------------------------------------------
if st.button("🔮 Predecir motor_UPDRS"):
    # Escalar datos igual que en el entrenamiento
    input_scaled = scaler.transform(input_df.values)

    # Hacer predicción
    pred = model.predict(input_scaled)[0]

    st.success(f"### ✅ Predicción estimada de motor_UPDRS: **{pred:.2f}**")

    st.markdown("""
    **Interpretación:**  
    - Un valor más alto indica una **mayor severidad motora**.  
    - Esta predicción puede servir como ayuda clínica, pero no reemplaza evaluación médica.
    """)

# -----------------------------------------------------
# Mostrar DataFrame de entrada
# -----------------------------------------------------
with st.expander("Ver valores ingresados"):
    st.dataframe(input_df)