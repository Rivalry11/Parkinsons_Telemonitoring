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
    df = pd.read_csv("data/parkinsons_updrs.csv")
    df['test_time'] = df['test_time'].astype(int)
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
st.subheader("📈 Distribución de variables")

num_cols = df.select_dtypes(include=['float64', 'int64']).columns

selected_var = st.selectbox("Selecciona una variable", num_cols)

plt.figure(figsize=(6,4))
sns.histplot(df[selected_var], kde=True, color="steelblue")
plt.title(f"Distribución de {selected_var}")
st.pyplot()

st.subheader("📝 Conclusiones")

st.markdown("""
Las variables clínicas (**motor_UPDRS** y **total_UPDRS**) muestran una distribución amplia, indicando distintos niveles de severidad entre pacientes. En contraste, la mayoría de las variables acústicas (**Jitter, Shimmer, NHR**) están fuertemente sesgadas hacia valores bajos, lo cual es típico en medidas de voz. Las variables no lineales (**RPDE, DFA, PPE**) presentan distribuciones más equilibradas. En conjunto, esto muestra que el dataset es diverso y requiere normalización para un buen modelado.
""")

# -----------------------------
# BOXLOTS DE VARIABLES PRINCIPALES
# -----------------------------
st.subheader("📦 Boxplots de Variables Principales")

cols_box = ["motor_UPDRS", "total_UPDRS"]

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df[cols_box])
plt.title("Boxplots de motor_UPDRS y total_UPDRS")
st.pyplot(fig)
st.subheader("📝 Conclusiones")

st.markdown("""
- **total_UPDRS** presenta valores más altos y una mayor variabilidad, lo cual es esperado porque esta medida incluye tanto síntomas motores como no motores.
- **motor_UPDRS** muestra una dispersión ligeramente menor y valores más concentrados alrededor de la mediana.
""")


# -----------------------------
# HEATMAP DE CORRELACIONES
# -----------------------------
st.subheader("🔥 Mapa de correlación")

plt.figure(figsize=(14, 10))

corr_matrix = df.drop(['subject_id', 'sex', 'age'], axis=1).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Mapa de correlación (sin valores negativos en test_time)")
plt.show()
st.pyplot()

st.subheader("📝 Conclusiones")

st.markdown("""
- Este mapa muestra la relación entre todas las variables. Se observan fuertes correlaciones entre las medidas de **Jitter** y **Shimmer**, así como una alta relación entre **motor_UPDRS** y **total_UPDRS**. **HNR** destaca por correlaciones negativas con varias variables acústicas.
""")
