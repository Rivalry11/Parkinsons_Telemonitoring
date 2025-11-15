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
    df = pd.read_csv("../../data/parkinsons_updrs.csv")
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

# Convertir a DataFrame
df_results = pd.DataFrame(results).T
st.subheader("📊 Métricas de rendimiento de cada modelo")
st.dataframe(df_results[['MSE', 'R2']])

# -----------------------------
# GRAFICO COMPARATIVO DE MÉTRICAS
# -----------------------------
st.subheader("📈 Comparación gráfica de rendimiento (MSE y R²)")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.barplot(x=df_results.index, y=df_results['R2'], palette='crest', ax=axes[0])
axes[0].set_title("R² por modelo")
axes[0].tick_params(axis='x', rotation=45)

sns.barplot(x=df_results.index, y=df_results['MSE'], palette='flare', ax=axes[1])
axes[1].set_title("MSE por modelo")
axes[1].tick_params(axis='x', rotation=45)

st.pyplot(fig)

# -----------------------------
# DISPERSIÓN INDIVIDUAL POR MODELO
# -----------------------------
st.subheader("🔍 Dispersión de predicciones por modelo")

selected_model = st.selectbox("Selecciona un modelo", list(models.keys()))

model = models[selected_model]
y_pred = results[selected_model]['y_pred']

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.figure(figsize=(7, 5))
sns.scatterplot(x=y_test, y=y_pred, color='teal', edgecolor='white')
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
sns.regplot(x=y_test, y=y_pred, scatter=False, color='orange')
plt.title(f"{selected_model}\nR²={results[selected_model]['R2']:.2f} | MSE={results[selected_model]['MSE']:.2f}")
plt.xlabel("Valor real (y_test)")
plt.ylabel("Predicción (y_pred)")

st.pyplot()

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