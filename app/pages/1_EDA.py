import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# TÍTULO Y DESCRIPCIÓN
# -----------------------------
st.title("📊 Análisis Exploratorio (EDA) – Parkinson’s Telemonitoring")

st.markdown("""
Este módulo presenta un resumen visual del análisis exploratorio del dataset Parkinson’s Telemonitoring.
Aquí puedes explorar las distribuciones, correlaciones y relaciones entre las variables clave.
""")

# -----------------------------
# CARGA DEL DATASET
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/parkinsons_telemonitoring.csv")
    df = df.rename(columns={'subject#': 'subject_id'})
    return df

df = load_data()

st.subheader("Vista general del dataset")
st.dataframe(df.head())

# -----------------------------
# ESTADÍSTICAS DESCRIPTIVAS
# -----------------------------
st.subheader("📌 Estadísticas descriptivas")
st.dataframe(df.describe())

# -----------------------------
# DISTRIBUCIONES CON HISTOGRAMA + KDE
# -----------------------------
st.subheader("📈 Distribución de variables numéricas")

num_cols = df.select_dtypes(include=['float64', 'int64']).columns

selected_var = st.selectbox("Selecciona una variable", num_cols)

plt.figure(figsize=(6,4))
sns.histplot(df[selected_var], kde=True, color="steelblue")
plt.title(f"Distribución de {selected_var}")
st.pyplot()

# -----------------------------
# BOXLOTS DE VARIABLES PRINCIPALES
# -----------------------------
st.subheader("📦 Boxplots de Variables Principales")

cols_box = ["motor_UPDRS", "total_UPDRS"]

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df[cols_box])
plt.title("Boxplots de motor_UPDRS y total_UPDRS")
st.pyplot(fig)

# -----------------------------
# HEATMAP DE CORRELACIONES
# -----------------------------
st.subheader("🔥 Mapa de correlación")

corr = df.corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=False, cmap="coolwarm")
st.pyplot()

# -----------------------------
# SCATTERPLOT ENTRE TARGETS
# -----------------------------
st.subheader("🔍 Relación entre motor_UPDRS y total_UPDRS")

plt.figure(figsize=(6,4))
sns.scatterplot(x=df["motor_UPDRS"], y=df["total_UPDRS"], hue=df["sex"], palette="Set2")
plt.title("motor_UPDRS vs total_UPDRS por sexo")
st.pyplot()

# -----------------------------
# CONCLUSIONES
# -----------------------------
st.subheader("📝 Conclusiones")

st.markdown("""
- **motor_UPDRS** y **total_UPDRS** están fuertemente correlacionados.
- Varias variables acústicas muestran relaciones con los síntomas motores.
- Se observan distribuciones relativamente consistentes entre pacientes.
- No existen valores nulos significativos después del preprocesamiento.
""")