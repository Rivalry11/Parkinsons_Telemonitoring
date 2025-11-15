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
# -----------------------------
# PERMUTATION IMPORTANCE — SOLO SI EL USUARIO LO SOLICITA
# -----------------------------
st.subheader("🔁 Permutation Importance (cálculo más lento)")

if st.button("Calcular Permutation Importance"):
    with st.spinner("Calculando... puede tardar unos segundos"):
        from sklearn.inspection import permutation_importance
        result = permutation_importance(
            rf, X_test, y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

        feat_perm = pd.DataFrame({
            "Variable": X.columns,
            "Importancia": result.importances_mean,
            "STD": result.importances_std
        }).sort_values(by="Importancia", ascending=False)

        # Gráfico
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.barplot(data=feat_perm, x="Importancia", y="Variable", palette="viridis")
        plt.title("Permutation Importance – Random Forest")
        st.pyplot(fig2)

        # Tabla explicativa
        st.subheader("📘 Interpretación de las variables más importantes")
        explicacion = {
            "test_time": "Indica progresión temporal del paciente.",
            "Jitter(%)": "Variación de frecuencia — refleja inestabilidad vocal.",
            "Jitter(Abs)": "Cambio absoluto en frecuencia — vibración irregular.",
            "Jitter:RAP": "Variación rápida — temblor fino.",
            "Jitter:PPQ5": "Variación a corto plazo.",
            "Jitter:DDP": "Medida derivada de RAP.",
            "Shimmer": "Variación en amplitud — rigidez muscular.",
            "Shimmer(dB)": "Oscilación dB — severidad vocal.",
            "Shimmer:APQ3": "Amplitud promediada — estabilidad de fonación.",
            "Shimmer:APQ5": "Variabilidad de amplitud.",
            "Shimmer:APQ11": "Variabilidad de amplitud a largo plazo.",
            "Shimmer:DDA": "Variación derivada de APQ3.",
            "NHR": "Ruido presente en la señal vocal.",
            "HNR": "Relación armónico-ruido.",
            "RPDE": "Complejidad temporal de la señal.",
            "DFA": "Dinamismo no lineal de la voz.",
            "PPE": "Estimación de probabilidad de error en tono."
        }

        info_df = pd.DataFrame({
            "Variable": feat_perm["Variable"],
            "Importancia": feat_perm["Importancia"].round(4),
            "Interpretación": feat_perm["Variable"].map(explicacion)
        })

        st.dataframe(info_df)