"""
NFL Play Predictor - Módulo Principal
=====================================

Sistema de Machine Learning para predecir jugadas de la NFL:
- Tipo de jugada (Pase vs Carrera)
- Distancia de jugada (Corta vs Larga)
- Probabilidad de éxito (First Down)

Autor: César Adrián Delgado Díaz
Proyecto: NFL Play Predictor - Portfolio Personal
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Importaciones opcionales para modelos
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost no instalado. Instalar con: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠ LightGBM no instalado. Instalar con: pip install lightgbm")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# =============================================================================
# CONFIGURACIÓN
# =============================================================================


@dataclass
class ModelConfig:
    """Configuración para los modelos de predicción."""

    # Configuración general
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

    # Configuración XGBoost
    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "use_label_encoder": False,
    })

    # Configuración LightGBM
    lgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1,
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
    })


# =============================================================================
# CLASE PRINCIPAL
# =============================================================================


class NFLPlayPredictor:
    """
    Predictor de jugadas de la NFL usando Machine Learning.

    Esta clase encapsula todo el pipeline de ML:
    - Carga y preparación de datos
    - Feature engineering
    - Entrenamiento de modelos (XGBoost, LightGBM)
    - Evaluación y predicción
    - Interpretabilidad con SHAP

    Attributes
    ----------
    config : ModelConfig
        Configuración del modelo
    data : pd.DataFrame
        Datos procesados
    models : dict
        Modelos entrenados
    feature_cols : list
        Columnas de features

    Examples
    --------
    >>> predictor = NFLPlayPredictor()
    >>> predictor.load_data()
    >>> predictor.prepare_features()
    >>> predictor.train_models(target='isPass')
    >>> predictions = predictor.predict(new_data)
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Inicializa el predictor.

        Parameters
        ----------
        config : ModelConfig, optional
            Configuración personalizada
        """
        self.config = config or ModelConfig()
        self.data: Optional[pd.DataFrame] = None
        self.models: Dict[str, Any] = {}
        self.feature_cols: List[str] = []
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self._trained = False

    def load_data(
        self,
        data_path: str = "data/raw",
        use_sample: bool = False,
        sample_size: int = 10000,
    ) -> "NFLPlayPredictor":
        """
        Carga los datos de jugadas.

        Parameters
        ----------
        data_path : str
            Ruta a los datos
        use_sample : bool
            Si True, usa datos de ejemplo
        sample_size : int
            Tamaño del sample

        Returns
        -------
        self
            Instancia del predictor (para method chaining)
        """
        if use_sample:
            from utils import create_sample_data
            self.data = create_sample_data(sample_size, self.config.random_state)
        else:
            try:
                from utils import load_plays_data
                self.data = load_plays_data(data_path)
            except FileNotFoundError:
                print("⚠ Datos no encontrados. Usando datos de ejemplo...")
                from utils import create_sample_data
                self.data = create_sample_data(sample_size, self.config.random_state)

        print(f"✓ Datos cargados: {self.data.shape}")
        return self

    def prepare_features(self) -> "NFLPlayPredictor":
        """
        Prepara las features para el modelo.

        Returns
        -------
        self
            Instancia del predictor
        """
        if self.data is None:
            raise ValueError("Primero debes cargar los datos con load_data()")

        from utils import engineer_all_features
        self.data = engineer_all_features(self.data)

        # Identificar features numéricas
        self._identify_feature_columns()

        # Encodear variables categóricas
        self._encode_categorical_features()

        print(f"✓ Features preparadas: {len(self.feature_cols)} columnas")
        return self

    def _identify_feature_columns(self) -> None:
        """Identifica las columnas de features."""
        exclude_cols = [
            "gameId", "playId", "isPass", "isSuccess", "playResult",
            "passResult", "gameClock", "personnelO", "personnelD",
            "downAndDistance", "fieldPositionBucket", "distanceCategory",
        ]

        self.feature_cols = [
            col for col in self.data.columns
            if col not in exclude_cols
            and self.data[col].dtype in ["int64", "float64", "int32", "float32", "uint8"]
        ]

    def _encode_categorical_features(self) -> None:
        """Encodea variables categóricas."""
        categorical_cols = [
            "offenseFormation",
            "fieldPositionBucket",
            "distanceCategory",
        ]

        for col in categorical_cols:
            if col in self.data.columns and self.data[col].dtype == "object":
                le = LabelEncoder()
                self.data[f"{col}_encoded"] = le.fit_transform(
                    self.data[col].astype(str)
                )
                self.label_encoders[col] = le
                self.feature_cols.append(f"{col}_encoded")

    def train_models(
        self,
        target: str = "isPass",
        models_to_train: List[str] = None,
    ) -> "NFLPlayPredictor":
        """
        Entrena los modelos de predicción.

        Parameters
        ----------
        target : str
            Variable objetivo ('isPass', 'isSuccess')
        models_to_train : list
            Lista de modelos a entrenar ['xgboost', 'lightgbm']

        Returns
        -------
        self
            Instancia del predictor
        """
        if self.data is None or not self.feature_cols:
            raise ValueError("Primero prepara los datos con load_data() y prepare_features()")

        if models_to_train is None:
            models_to_train = ["xgboost", "lightgbm"]

        # Preparar datos
        X = self.data[self.feature_cols].copy()
        y = self.data[target].copy()

        # Manejar valores faltantes
        X = X.fillna(0)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test

        print(f"\n{'='*60}")
        print(f"ENTRENANDO MODELOS - Target: {target}")
        print(f"{'='*60}")
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Distribución target: {y.value_counts(normalize=True).to_dict()}")

        # Entrenar modelos
        if "xgboost" in models_to_train and HAS_XGBOOST:
            self._train_xgboost(X_train, y_train, X_test, y_test)

        if "lightgbm" in models_to_train and HAS_LIGHTGBM:
            self._train_lightgbm(X_train, y_train, X_test, y_test)

        self._trained = True
        return self

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Entrena modelo XGBoost."""
        print("\n📊 Entrenando XGBoost...")

        model = xgb.XGBClassifier(
            **self.config.xgb_params,
            random_state=self.config.random_state,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        self.models["xgboost"] = model
        self._evaluate_model(model, X_test, y_test, "XGBoost")

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Entrena modelo LightGBM."""
        print("\n📊 Entrenando LightGBM...")

        model = lgb.LGBMClassifier(
            **self.config.lgb_params,
            random_state=self.config.random_state,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
        )

        self.models["lightgbm"] = model
        self._evaluate_model(model, X_test, y_test, "LightGBM")

    def _evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
    ) -> Dict:
        """Evalúa un modelo entrenado."""
        from utils import evaluate_model, print_classification_report

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_proba, model_name)

        print(f"\n{model_name} - Resultados:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics.get('roc_auc', 'N/A'):.4f}" if metrics.get('roc_auc') else "")

        return metrics

    def cross_validate(
        self,
        model_name: str = "xgboost",
        target: str = "isPass",
        cv: int = None,
    ) -> Dict:
        """
        Realiza validación cruzada.

        Parameters
        ----------
        model_name : str
            Nombre del modelo
        target : str
            Variable objetivo
        cv : int
            Número de folds

        Returns
        -------
        dict
            Resultados de CV
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no entrenado")

        cv = cv or self.config.cv_folds
        X = self.data[self.feature_cols].fillna(0)
        y = self.data[target]

        model = self.models[model_name]

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.config.random_state)

        scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")

        print(f"\n📊 Cross-Validation ({model_name}) - {cv} folds:")
        print(f"  ROC-AUC: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "scores": scores,
        }

    def predict(
        self,
        X: pd.DataFrame,
        model_name: str = "xgboost",
        return_proba: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Realiza predicciones.

        Parameters
        ----------
        X : pd.DataFrame
            Datos para predecir
        model_name : str
            Modelo a usar
        return_proba : bool
            Si retornar probabilidades

        Returns
        -------
        array or tuple
            Predicciones y opcionalmente probabilidades
        """
        if not self._trained:
            raise ValueError("Primero entrena los modelos con train_models()")

        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no disponible")

        model = self.models[model_name]

        # Asegurar que las features están en orden
        X_pred = X[self.feature_cols].fillna(0)

        predictions = model.predict(X_pred)

        if return_proba:
            probabilities = model.predict_proba(X_pred)[:, 1]
            return predictions, probabilities

        return predictions

    def get_feature_importance(
        self,
        model_name: str = "xgboost",
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Obtiene la importancia de features.

        Parameters
        ----------
        model_name : str
            Nombre del modelo
        top_n : int
            Top N features

        Returns
        -------
        pd.DataFrame
            DataFrame con importancias
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no entrenado")

        model = self.models[model_name]

        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return importance_df.head(top_n)

    def explain_predictions(
        self,
        X: pd.DataFrame = None,
        model_name: str = "xgboost",
        n_samples: int = 100,
    ):
        """
        Explica predicciones usando SHAP.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Datos para explicar (usa test set si None)
        model_name : str
            Modelo a explicar
        n_samples : int
            Número de muestras para SHAP

        Returns
        -------
        shap.Explanation
            Objeto SHAP con explicaciones
        """
        if not HAS_SHAP:
            raise ImportError("SHAP no instalado. Instalar con: pip install shap")

        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no entrenado")

        if X is None:
            X = self._X_test.head(n_samples)

        model = self.models[model_name]

        print("🔍 Calculando SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[self.feature_cols].fillna(0))

        return shap_values, X

    def save(self, output_dir: str = "models", prefix: str = "nfl_predictor") -> None:
        """
        Guarda el predictor entrenado.

        Parameters
        ----------
        output_dir : str
            Directorio de salida
        prefix : str
            Prefijo para archivos
        """
        os.makedirs(output_dir, exist_ok=True)

        # Guardar cada modelo
        for name, model in self.models.items():
            model_path = os.path.join(output_dir, f"{prefix}_{name}.pkl")
            joblib.dump(model, model_path)
            print(f"✓ Modelo guardado: {model_path}")

        # Guardar configuración y features
        config_path = os.path.join(output_dir, f"{prefix}_config.pkl")
        joblib.dump({
            "feature_cols": self.feature_cols,
            "label_encoders": self.label_encoders,
            "config": self.config,
        }, config_path)
        print(f"✓ Configuración guardada: {config_path}")

    @classmethod
    def load(cls, model_dir: str = "models", prefix: str = "nfl_predictor") -> "NFLPlayPredictor":
        """
        Carga un predictor guardado.

        Parameters
        ----------
        model_dir : str
            Directorio con modelos
        prefix : str
            Prefijo de archivos

        Returns
        -------
        NFLPlayPredictor
            Predictor cargado
        """
        # Cargar configuración
        config_path = os.path.join(model_dir, f"{prefix}_config.pkl")
        config_data = joblib.load(config_path)

        predictor = cls(config_data.get("config"))
        predictor.feature_cols = config_data["feature_cols"]
        predictor.label_encoders = config_data.get("label_encoders", {})

        # Cargar modelos
        for model_file in os.listdir(model_dir):
            if model_file.startswith(prefix) and model_file.endswith(".pkl"):
                if "config" not in model_file:
                    model_name = model_file.replace(f"{prefix}_", "").replace(".pkl", "")
                    model_path = os.path.join(model_dir, model_file)
                    predictor.models[model_name] = joblib.load(model_path)

        predictor._trained = True
        print(f"✓ Predictor cargado con modelos: {list(predictor.models.keys())}")
        return predictor


# =============================================================================
# FUNCIONES DE ALTO NIVEL
# =============================================================================


def train_pass_run_predictor(
    data_path: str = "data/raw",
    use_sample: bool = True,
    sample_size: int = 10000,
) -> NFLPlayPredictor:
    """
    Entrena un predictor de Pase vs Carrera.

    Parameters
    ----------
    data_path : str
        Ruta a datos
    use_sample : bool
        Usar datos de ejemplo
    sample_size : int
        Tamaño de muestra

    Returns
    -------
    NFLPlayPredictor
        Predictor entrenado
    """
    predictor = NFLPlayPredictor()
    predictor.load_data(data_path, use_sample, sample_size)
    predictor.prepare_features()
    predictor.train_models(target="isPass")
    return predictor


def train_success_predictor(
    data_path: str = "data/raw",
    use_sample: bool = True,
    sample_size: int = 10000,
) -> NFLPlayPredictor:
    """
    Entrena un predictor de éxito de jugada.

    Parameters
    ----------
    data_path : str
        Ruta a datos
    use_sample : bool
        Usar datos de ejemplo
    sample_size : int
        Tamaño de muestra

    Returns
    -------
    NFLPlayPredictor
        Predictor entrenado
    """
    predictor = NFLPlayPredictor()
    predictor.load_data(data_path, use_sample, sample_size)
    predictor.prepare_features()
    predictor.train_models(target="isSuccess")
    return predictor


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("NFL PLAY PREDICTOR - Demo")
    print("=" * 60)

    # Entrenar predictor de pase vs carrera
    print("\n1. Entrenando predictor Pase vs Carrera...")
    predictor = train_pass_run_predictor(use_sample=True, sample_size=5000)

    # Mostrar importancia de features
    print("\n2. Top 10 Features más importantes:")
    importance = predictor.get_feature_importance(top_n=10)
    print(importance.to_string(index=False))

    # Cross-validation
    print("\n3. Validación cruzada...")
    if "xgboost" in predictor.models:
        predictor.cross_validate("xgboost")

    print("\n✓ Demo completada!")
