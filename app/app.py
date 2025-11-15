import streamlit as st

# ---------------------------------------------
# CONFIGURACIÓN GENERAL DE LA APLICACIÓN
# ---------------------------------------------
st.set_page_config(
    page_title="Parkinson’s Telemonitoring – Dashboard ML",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------
# TÍTULO PRINCIPAL DEL DASHBOARD
# ---------------------------------------------
st.title("🧠 Parkinson’s Telemonitoring – Dashboard de Machine Learning")

st.markdown("""
Bienvenido al dashboard interactivo del proyecto **Parkinson’s Telemonitoring**.

Este sistema presenta:
- 📊 **Análisis Exploratorio (EDA)**  
- 🤖 **Comparación de Modelos de Machine Learning**  
- 🔮 **Predicción en tiempo real de motor_UPDRS**  

Este proyecto fue desarrollado para la entrega académica de Machine Learning.
""")

# ---------------------------------------------
# SIDEBAR (MENÚ LATERAL)
# ---------------------------------------------
st.sidebar.title("📁 Navegación")
st.sidebar.markdown("""
Usa el menú **Pages** de Streamlit (a la izquierda) para cambiar entre:

- **📊 EDA**
- **🤖 Modelos**
- **🔮 Predicción**
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📚 Información del dataset")
st.sidebar.info("""
**Dataset:** Parkinson’s Telemonitoring  
- Fuente: UCI Repository  
- Registros: 5,875  
- Variables: 22 acústicas + motor_UPDRS + total_UPDRS  
""")

st.sidebar.markdown("---")
st.sidebar.subheader("👩‍💻 Realizado por")
st.sidebar.write("**Camila Rubio y Omar Cerezo – 2025**")

# ---------------------------------------------
# CONTENIDO INICIAL (PORTADA)
# ---------------------------------------------
st.subheader("📘 Introducción al Proyecto")

st.markdown("""
Este proyecto utiliza datos de monitoreo telemétrico para predecir la severidad motora
(**motor_UPDRS**) en pacientes con enfermedad de Parkinson.

El objetivo principal es evaluar distintos modelos de regresión y seleccionar el mejor para realizar predicciones reales.

### ✔ ¿Qué encontrarás en este dashboard?

#### 1. **EDA – Exploración de los datos**
Distribuciones, correlaciones, estadísticas y relaciones clave entre variables.

#### 2. **Modelos Predictivos**
Comparación visual e interactiva de ocho modelos:
- Regresión Lineal  
- Ridge / Lasso / ElasticNet  
- Árbol de Decisión  
- Random Forest  
- Gradient Boosting  
- SVR  

Incluye métricas **MSE** y **R²**, y gráficos de dispersión.


### 📌 Nota  
Usa el menú lateral *Pages* para navegar entre módulos.
""")

# ---------------------------------------------
# FOOTER
# ---------------------------------------------
st.markdown("---")
st.caption("© 2025 – Dashboard ML de Parkinson’s Telemonitoring | Develop by Camila Rubio - Omar Cerezo")
