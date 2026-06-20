# 🏈 NFL Play Predictor

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NFL Big Data Bowl](https://img.shields.io/badge/Data-NFL%20Big%20Data%20Bowl-red.svg)](https://www.kaggle.com/c/nfl-big-data-bowl-2024)

## 📋 Descripción

Sistema de Machine Learning para predecir jugadas de la NFL utilizando datos reales del **NFL Big Data Bowl**. El modelo predice:

- 🎯 **Tipo de jugada**: Pase vs Carrera (Pass/Run)
- 📏 **Distancia de jugada**: Corta vs Larga (Short/Long gain)
- ✅ **Probabilidad de éxito**: Likelihood de conseguir el primer down

## 🎯 Objetivos del Proyecto

1. **Análisis Exploratorio**: Comprender patrones en jugadas de la NFL
2. **Feature Engineering Avanzado**: Crear características relevantes para el contexto del juego
3. **Modelado Predictivo**: Implementar XGBoost y LightGBM para clasificación
4. **Interpretabilidad**: Explicar las predicciones usando SHAP values

## 🛠️ Tecnologías Utilizadas

- **Python 3.9+**
- **Pandas & NumPy**: Manipulación de datos
- **XGBoost & LightGBM**: Modelos de gradient boosting
- **Scikit-learn**: Preprocesamiento y métricas
- **SHAP**: Interpretabilidad del modelo
- **Matplotlib & Seaborn**: Visualización

## 📁 Estructura del Proyecto

```
06_Play_Predictor_NFL/
│
├── 📓 nfl_play_predictor.ipynb    # Notebook principal con análisis completo
├── 🐍 play_predictor.py           # Módulo principal con clases del modelo
├── 🔧 utils.py                    # Funciones auxiliares
├── 📋 requirements.txt            # Dependencias del proyecto
├── 📖 README.md                   # Documentación
├── 🚫 .gitignore                  # Archivos ignorados
│
├── 📂 data/
│   ├── raw/                       # Datos crudos del NFL Big Data Bowl
│   └── processed/                 # Datos procesados
│
└── 📂 models/
    └── *.pkl                      # Modelos entrenados
```

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/nfl-play-predictor.git
cd nfl-play-predictor
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar datos

Los datos se descargan automáticamente desde Kaggle. Necesitas configurar tu API key:

```bash
# Crear directorio de Kaggle
mkdir ~/.kaggle  # Linux/Mac
mkdir %USERPROFILE%\.kaggle  # Windows

# Copiar tu kaggle.json a este directorio
```

## 📊 Uso

### Opción 1: Ejecutar el Notebook

```bash
jupyter notebook nfl_play_predictor.ipynb
```

### Opción 2: Usar el módulo Python

```python
from play_predictor import NFLPlayPredictor

# Inicializar el predictor
predictor = NFLPlayPredictor()

# Cargar y preparar datos
predictor.load_data()
predictor.prepare_features()

# Entrenar modelos
predictor.train_models()

# Predecir
predictions = predictor.predict(new_data)
```

## 📈 Resultados

| Modelo | Accuracy | F1-Score | AUC-ROC |
|--------|----------|----------|---------|
| XGBoost (Pass/Run) | ~75% | ~0.74 | ~0.82 |
| LightGBM (Pass/Run) | ~76% | ~0.75 | ~0.83 |
| XGBoost (Success) | ~68% | ~0.67 | ~0.73 |

## 🔍 Features Principales

### Contexto del Juego
- `down`: Número de down (1-4)
- `yardsToGo`: Yardas necesarias para primer down
- `quarter`: Cuarto del partido
- `gameClock`: Tiempo restante

### Situación del Equipo
- `offenseFormation`: Formación ofensiva
- `defendersInBox`: Defensores en la caja
- `numberOfPassRushers`: Número de pass rushers

### Features Engineered
- `is_passing_situation`: Situación típica de pase
- `score_differential`: Diferencia en el marcador
- `field_position_bucket`: Zona del campo
- `time_pressure`: Presión por tiempo

## 👤 Autor

- 👤 Autor : **César Adrián Delgado Díaz**
- 💼 LinkedIn: [linkedin.com/in/cesar-delgado-diaz](linkedin.com/in/cesar-delgado-diaz)
- 🐙 GitHub: [github.com/tu-usuario](https://github.com/cesar530)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- [NFL Big Data Bowl](https://www.kaggle.com/c/nfl-big-data-bowl-2024) por proporcionar los datos
- La comunidad de Kaggle por las discusiones y notebooks públicos

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!
