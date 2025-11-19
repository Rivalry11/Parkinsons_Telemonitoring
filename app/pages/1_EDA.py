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
# DICCIONARIO DE DATOS
# -----------------------------

st.subheader("📘 Diccionario de Datos – Parkinson’s Telemonitoring")

data_dict = {
    "Variable": [
        "age", "sex", "test_time",
        "motor_UPDRS", "total_UPDRS",
        "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
        "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
        "NHR", "HNR",
        "RPDE", "DFA", "PPE"
    ],
    "Descripción": [
        "Edad del paciente en años",
        "Sexo (0 = mujer, 1 = hombre)",
        "Días desde la primera medición",
        "Puntaje motor de la escala UPDRS",
        "Puntaje total de la escala UPDRS",
        "Variación porcentual de la frecuencia vocal",
        "Variación absoluta de la frecuencia vocal",
        "Variabilidad en ventana de 3 ciclos",
        "Variabilidad en ventana de 5 ciclos",
        "Medida derivada del RAP",
        "Variación de amplitud vocal",
        "Variación de amplitud en decibelios",
        "Variabilidad en ventana de 3 ciclos",
        "Variabilidad en ventana de 5 ciclos",
        "Variabilidad en ventana de 11 ciclos",
        "Medida derivada de APQ3",
        "Proporción ruido / armónicos",
        "Relación armónicos / ruido",
        "Imprevisibilidad en la señal vocal",
        "Complejidad temporal de la señal",
        "Entropía perceptual del tono vocal"
    ],
    "Tipo": [
        "int", "Binario", "int",
        "float", "float",
        "float", "float", "float", "float", "float",
        "float", "float", "float", "float", "float", "float",
        "float", "float",
        "float", "float", "float"
    ]
}

df_dict = pd.DataFrame(data_dict)

st.dataframe(df_dict, use_container_width=True)

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
Las variables clínicas (**motor_UPDRS** y **total_UPDRS**) muestran una distribución amplia, indicando distintos niveles de severidad entre pacientes. En contraste, la mayoría de las variables acústicas (**Jitter, Shimmer, NHR**) están fuertemente sesgadas hacia valores bajos, lo cual es típico en medidas de voz. Las variables no lineales (**RPDE, DFA, PPE**) presentan distribuciones más equilibradas.
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
