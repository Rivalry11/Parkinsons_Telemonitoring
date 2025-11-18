import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image


# -----------------------------
# TÍTULO
# -----------------------------
st.title("🤖 Comparación de Modelos Predictivos – Parkinson’s Telemonitoring")

st.markdown("""
Esta sección muestra la comparación de varios modelos de Machine Learning aplicados para predecir **motor_UPDRS** 
a partir de las variables acústicas y clínicas del dataset.
""")

# -----------------------------
# CARGA DEL DATASET
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/parkinsons_updrs.csv")
    df = df.rename(columns={'subject#': 'subject_id'})
    df = df.drop(['sex', 'subject_id', 'age'], axis=1, errors='ignore')
    return df

df = load_data()

# -----------------------------
# SEPARACIÓN FEATURES / TARGET
# -----------------------------
X = df.drop(['motor_UPDRS', 'total_UPDRS'], axis=1)
y = df['motor_UPDRS']

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# DEFINICIÓN DE MODELOS
# -----------------------------
models = {
    'Regresión Lineal': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.01),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR (RBF Kernel)': SVR(kernel='rbf')
}

results = {}

# -----------------------------
# ENTRENAMIENTO DE MODELOS
# -----------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'y_pred': y_pred
    }

# -----------------------------
# MÉTRICAS ORDENADAS Y GRÁFICOS
# -----------------------------

# Convertir resultados a DataFrame
df_results = pd.DataFrame(results).T

# Normalizar columna R2 → R²
df_results = df_results.rename(columns={"R2": "R²"})

# Ordenar de mejor a peor rendimiento
df_results_sorted = df_results.sort_values(by="R²", ascending=False)

# Mostrar tabla ordenada
st.subheader("📊 Métricas de rendimiento de cada modelo (ordenadas por R²)")
st.dataframe(df_results_sorted[['MSE', 'R²']])

# -----------------------------
# GRAFICO COMPARATIVO DE MÉTRICAS
# -----------------------------
st.subheader("📈 Comparación gráfica de rendimiento (MSE y R²)")

# Preparar rankings
df_r2 = df_results_sorted.reset_index().rename(columns={"index": "Modelo"})
df_mse = df_results.sort_values(by="MSE", ascending=True).reset_index().rename(columns={"index": "Modelo"})

# Crear la figura
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Gráfico R² (de mejor a peor) ---
sns.barplot(
    x="Modelo", 
    y="R²", 
    data=df_r2,
    palette="crest",
    ax=axes[0]
)
axes[0].set_title("Ranking de Modelos por R² (mejor → peor)")
axes[0].set_ylabel("R²")
axes[0].set_xlabel("Modelo")
axes[0].tick_params(axis='x', rotation=45)

# --- Gráfico MSE (de mejor a peor) ---
sns.barplot(
    x="Modelo", 
    y="MSE", 
    data=df_mse,
    palette="flare",
    ax=axes[1]
)
axes[1].set_title("Ranking de Modelos por MSE (menor error → mayor error)")
axes[1].set_ylabel("MSE")
axes[1].set_xlabel("Modelo")
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle("Comparación ordenada de rendimiento entre modelos", fontsize=15, y=1.05)
plt.tight_layout()
plt.show()
st.pyplot(fig)

st.subheader("📝 Conclusiones")

st.markdown("""
Random Forest fue elegido como modelo final porque obtuvo el mejor R² y el menor MSE, superando al resto de modelos. Esto indica que captura mejor las relaciones no lineales y la complejidad del dataset, mientras que los modelos lineales no lograron adaptarse tan bien.
""")
# -----------------------------
# IMPORTANCIA DE VARIABLES (RANDOM FOREST)
# -----------------------------
st.subheader("🌟 Importancia de variables (Random Forest)")

rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_

feat_importances = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': importances
}).sort_values(by='Importancia', ascending=False)

fig, ax = plt.subplots(figsize=(6, 6))
sns.barplot(data=feat_importances, x='Importancia', y='Variable', palette='crest')
plt.title("Importancia de Variables")
st.pyplot(fig)


st.subheader("🔁 Permutation Importance")

# Mostrar imagen de forma responsiva
try:
    image = Image.open("app/images/permutation_importance.png")

    # Layout responsivo para móviles
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, caption="Permutation Importance – Random Forest", use_column_width=True)

except:
    st.warning("⚠️ No se pudo cargar la imagen. Asegúrate de generar la imagen en el notebook.")

# -----------------------------
# TEXTO AUTOMÁTICO: Interpretación de las 3 variables más importantes
# -----------------------------
st.subheader("📘 Interpretación automática de las 3 variables más importantes")

# Diccionario de descripciones clínicas
descripcion_variables = {
    "test_time": "Momento dentro del seguimiento. Indica progresión temporal de la enfermedad.",
    "Jitter(%)": "Variación rápida de frecuencia. Se relaciona con inestabilidad vocal por alteraciones motoras.",
    "Jitter(Abs)": "Cambios absolutos en frecuencia. Refleja vibración irregular de las cuerdas vocales.",
    "Jitter:RAP": "Promedio de variaciones sucesivas — asociado al temblor fino vocal.",
    "Jitter:PPQ5": "Variación de frecuencia a corto plazo, relacionada con pérdida de control muscular.",
    "Jitter:DDP": "Derivado de RAP — mide inestabilidad de vibración.",
    "Shimmer": "Variación de amplitud — evidencia rigidez y fatiga muscular.",
    "Shimmer(dB)": "Oscilación de amplitud en decibelios — fuerte indicador de deterioro vocal.",
    "Shimmer:APQ3": "Promedio de diferencias de amplitud — estabilidad fonatoria.",
    "Shimmer:APQ5": "Variabilidad de amplitud a corto plazo.",
    "Shimmer:APQ11": "Variación a largo plazo — voz más irregular.",
    "Shimmer:DDA": "Derivado de APQ3 — irregularidad muscular.",
    "NHR": "Relación ruido-armonía. A mayor ruido, peor calidad vocal.",
    "HNR": "Relación armónico-ruido. Valores bajos muestran voz deteriorada.",
    "RPDE": "Medida de complejidad temporal de la señal vocal.",
    "DFA": "Captura la dinámica no lineal del habla.",
    "PPE": "Indicador de irregularidad del tono."
}

# Cargar el dataframe usado para generar las importancias
# (Debe coincidir con el orden de la imagen)
try:
    import pandas as pd
    feat_perm = pd.read_csv("app/images/feat_perm_values.csv")  # OPCIONAL si guardaste los datos

    top3 = feat_perm.head(3)

    st.markdown("### 🥇 Variables más influyentes en el modelo")

    for i, row in top3.iterrows():
        var = row["Variable"]
        imp = row["Importancia"]

        st.markdown(f"""
        **🔹 {var}**  
        Importancia: `{imp:.4f}`  
        **Interpretación:** {descripcion_variables.get(var, "No hay interpretación disponible.")}  
        """)

except:
    st.info("""
    ℹ️ Para generar el texto automático, puedes guardar el dataframe de Permutation Importance
    como `feat_perm_values.csv` desde el notebook.
    """)