import streamlit as st

st.title("📌 Conclusiones del Proyecto")
st.write("---")

# 1. Objetivo del proyecto
st.header("🎯 Objetivo del Proyecto")
st.write("""
El propósito de este proyecto fue analizar el dataset *Parkinson’s Telemonitoring* 
con el fin de comprender cómo diversas variables acústicas y clínicas influyen 
en la progresión de la enfermedad, medida mediante la escala **motor_UPDRS**. 
Se desarrollaron análisis exploratorios y varios modelos de regresión para evaluar 
la capacidad predictiva de estas características.
""")

# 2. Hallazgos principales del EDA
st.header("🔍 Hallazgos Principales del EDA")
st.write("""
- Las variables **motor_UPDRS** y **total_UPDRS** presentan una relación lineal fuerte.
- Algunas variables acústicas como **Jitter(%)**, **Shimmer(dB)**, **NHR** y **PPE** 
  mostraron correlaciones importantes con los puntajes UPDRS.
- No se encontraron valores nulos tras la limpieza inicial, aunque sí se identificaron 
  outliers especialmente en las variables acústicas.
- La mayoría de variables presentan distribuciones sesgadas, reflejando variaciones 
  típicas de síntomas motores del Parkinson.
""")

# 3. Resultados de los modelos
st.header("📊 Resultados de los Modelos de Regresión")
st.write("""
Se compararon múltiples modelos: **Regresión Lineal**, **Ridge**, **Lasso**, **Elastic Net**,  
**Árbol de Decisión**, **Random Forest**, **Gradient Boosting** y **SVR**.

Los modelos basados en **Random Forest** y **Gradient Boosting** fueron los que mostraron
mejor rendimiento según el valor de R² y el MSE.

El modelo elegido fue **Random Forest** según el valor de R².
""")

# 4. Interpretación del mejor modelo
st.header("🌟 Interpretación de las Variables Más Importantes")
st.write("""
El análisis de importancia de características mostró que variables como:

- **Jitter(%)**
- **Shimmer(dB)**
- **NHR**
- **PPE**
- **RPDE**

tienen una alta influencia en la predicción de motor_UPDRS.  
Estas variables reflejan alteraciones en la estabilidad vocal y ruido en la señal, 
típicas de pacientes con síntomas motores más avanzados.
""")

# 5. Limitaciones
st.header("⚠️ Limitaciones del Estudio")
st.write("""
- El dataset no incluye información clínica completa (medicación, antecedentes, etc.).
- Solo se utilizaron variables acústicas, lo cual limita el alcance predictivo de la degradacion de la enfermedad en los pacientes.
- El tamaño del dataset es moderado y puede no generalizar a toda la población.
""")


# 6. Conclusión final
st.header("📘 Conclusión Final")
st.write("""
El objetivo del proyecto se cumplió: se logró analizar detalladamente el dataset,
entender las variables clave y evaluar múltiples modelos de regresión.  
Los resultados permiten comprender mejor qué características de la voz pueden reflejar 
el estado motor del paciente, aportando valor para futuros estudios o aplicaciones clínicas.
""")