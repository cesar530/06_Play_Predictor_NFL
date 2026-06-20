"""
Utilidades para NFL Play Predictor
==================================

Funciones auxiliares para carga de datos, feature engineering,
visualización y evaluación de modelos.

Autor: César Adrián Delgado Díaz
Proyecto: NFL Play Predictor - Portfolio Personal
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# Configuración de estilo para visualizaciones
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "primary": "#013369",  # NFL Blue
    "secondary": "#D50A0A",  # NFL Red
    "accent": "#FFB612",  # Gold
    "success": "#2E8B57",
    "warning": "#FFA500",
}


# =============================================================================
# FUNCIONES DE CARGA DE DATOS
# =============================================================================


def download_nfl_data(
    competition: str = "nfl-big-data-bowl-2024",
    data_path: str = "data/raw",
    force_download: bool = False,
) -> None:
    """
    Descarga datos del NFL Big Data Bowl desde Kaggle.

    Parameters
    ----------
    competition : str
        Nombre de la competición de Kaggle
    data_path : str
        Ruta donde guardar los datos
    force_download : bool
        Si True, descarga aunque ya existan los archivos
    """
    os.makedirs(data_path, exist_ok=True)

    # Verificar si ya existen datos
    if not force_download and os.listdir(data_path):
        print(f"✓ Datos ya existen en {data_path}")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        print(f"📥 Descargando datos de {competition}...")
        api.competition_download_files(competition, path=data_path, unzip=True)
        print(f"✓ Datos descargados exitosamente en {data_path}")
    except Exception as e:
        print(f"⚠ Error descargando datos: {e}")
        print("Por favor descarga manualmente desde:")
        print(f"https://www.kaggle.com/competitions/{competition}/data")


def load_plays_data(data_path: str = "data/raw") -> pd.DataFrame:
    """
    Carga el dataset de jugadas.

    Parameters
    ----------
    data_path : str
        Ruta a los datos

    Returns
    -------
    pd.DataFrame
        DataFrame con las jugadas
    """
    plays_file = os.path.join(data_path, "plays.csv")

    if not os.path.exists(plays_file):
        raise FileNotFoundError(
            f"No se encontró {plays_file}. "
            "Ejecuta download_nfl_data() primero."
        )

    df = pd.read_csv(plays_file)
    print(f"✓ Cargadas {len(df):,} jugadas")
    return df


def load_games_data(data_path: str = "data/raw") -> pd.DataFrame:
    """
    Carga el dataset de partidos.

    Parameters
    ----------
    data_path : str
        Ruta a los datos

    Returns
    -------
    pd.DataFrame
        DataFrame con los partidos
    """
    games_file = os.path.join(data_path, "games.csv")

    if not os.path.exists(games_file):
        raise FileNotFoundError(f"No se encontró {games_file}")

    df = pd.read_csv(games_file)
    print(f"✓ Cargados {len(df):,} partidos")
    return df


def load_players_data(data_path: str = "data/raw") -> pd.DataFrame:
    """
    Carga el dataset de jugadores.

    Parameters
    ----------
    data_path : str
        Ruta a los datos

    Returns
    -------
    pd.DataFrame
        DataFrame con los jugadores
    """
    players_file = os.path.join(data_path, "players.csv")

    if not os.path.exists(players_file):
        raise FileNotFoundError(f"No se encontró {players_file}")

    df = pd.read_csv(players_file)
    print(f"✓ Cargados {len(df):,} jugadores")
    return df


def create_sample_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Crea datos de ejemplo para demostración cuando no hay datos reales disponibles.

    Parameters
    ----------
    n_samples : int
        Número de muestras a generar
    random_state : int
        Semilla para reproducibilidad

    Returns
    -------
    pd.DataFrame
        DataFrame con datos sintéticos de jugadas NFL
    """
    np.random.seed(random_state)
    
    # Distribuciones realistas basadas en estadísticas de la NFL
    data = {
        "gameId": np.random.randint(1, 100, n_samples),
        "playId": np.arange(1, n_samples + 1),
        "quarter": np.random.choice([1, 2, 3, 4], n_samples, p=[0.27, 0.27, 0.23, 0.23]),
        "down": np.random.choice([1, 2, 3, 4], n_samples, p=[0.35, 0.30, 0.25, 0.10]),
        "yardsToGo": np.clip(np.random.exponential(7, n_samples).astype(int), 1, 30),
        "yardlineNumber": np.random.randint(1, 100, n_samples),
        "absoluteYardlineNumber": np.random.randint(1, 100, n_samples),
        "gameClock": [f"{np.random.randint(0, 15):02d}:{np.random.randint(0, 60):02d}" 
                     for _ in range(n_samples)],
        "preSnapHomeScore": np.random.randint(0, 35, n_samples),
        "preSnapVisitorScore": np.random.randint(0, 35, n_samples),
        "offenseFormation": np.random.choice(
            ["SHOTGUN", "SINGLEBACK", "I_FORM", "EMPTY", "PISTOL", "JUMBO"],
            n_samples,
            p=[0.45, 0.25, 0.10, 0.08, 0.07, 0.05]
        ),
        "defendersInTheBox": np.random.choice(
            [5, 6, 7, 8, 9], n_samples, p=[0.10, 0.30, 0.35, 0.20, 0.05]
        ),
        "numberOfPassRushers": np.random.choice(
            [3, 4, 5, 6, 7], n_samples, p=[0.05, 0.50, 0.30, 0.10, 0.05]
        ),
        "personnelO": np.random.choice(
            ["1 RB, 1 TE, 3 WR", "1 RB, 2 TE, 2 WR", "2 RB, 1 TE, 2 WR", "1 RB, 0 TE, 4 WR"],
            n_samples,
            p=[0.50, 0.25, 0.15, 0.10]
        ),
        "personnelD": np.random.choice(
            ["4 DL, 2 LB, 5 DB", "4 DL, 3 LB, 4 DB", "3 DL, 4 LB, 4 DB", "4 DL, 1 LB, 6 DB"],
            n_samples,
            p=[0.40, 0.30, 0.20, 0.10]
        ),
    }
    
    df = pd.DataFrame(data)
    
    # Crear play type (pase vs carrera) con lógica realista
    # Más probable pase en: 3rd down largo, shotgun, más defensores en caja
    pass_prob = (
        0.55 +  # Base pass rate
        0.15 * (df["down"] >= 3) +  # 3rd/4th down
        0.10 * (df["yardsToGo"] > 7) +  # Long yardage
        0.10 * (df["offenseFormation"] == "SHOTGUN") +
        0.05 * (df["offenseFormation"] == "EMPTY") -
        0.15 * (df["offenseFormation"] == "I_FORM") -
        0.10 * (df["down"] == 1) * (df["yardsToGo"] <= 10)
    )
    pass_prob = np.clip(pass_prob, 0.1, 0.95)
    df["isPass"] = np.random.binomial(1, pass_prob)
    df["passResult"] = np.where(
        df["isPass"] == 1,
        np.random.choice(["C", "I", "S", "IN", "R"], n_samples, p=[0.60, 0.20, 0.08, 0.07, 0.05]),
        np.nan
    )
    
    # Yards gained con distribución realista
    run_yards = np.random.normal(4.2, 4, n_samples)
    pass_yards = np.random.normal(7.5, 10, n_samples)
    df["playResult"] = np.where(df["isPass"] == 1, pass_yards, run_yards).astype(int)
    df["playResult"] = np.clip(df["playResult"], -10, 80)
    
    # Calcular éxito (conseguir el first down o TD)
    df["isSuccess"] = ((df["playResult"] >= df["yardsToGo"]) | 
                       (df["playResult"] >= 10)).astype(int)
    
    print(f"✓ Creados {len(df):,} registros de ejemplo")
    return df


# =============================================================================
# FUNCIONES DE FEATURE ENGINEERING
# =============================================================================


def parse_game_clock(clock_str: str) -> int:
    """
    Convierte el reloj del juego a segundos.

    Parameters
    ----------
    clock_str : str
        Tiempo en formato "MM:SS"

    Returns
    -------
    int
        Segundos totales
    """
    if pd.isna(clock_str):
        return 0
    try:
        parts = str(clock_str).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return 0


def create_game_situation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features relacionadas con la situación del juego.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con jugadas

    Returns
    -------
    pd.DataFrame
        DataFrame con nuevas features
    """
    df = df.copy()

    # Tiempo restante en segundos
    df["gameClockSeconds"] = df["gameClock"].apply(parse_game_clock)

    # Tiempo total restante en el juego (considerando el cuarto)
    df["totalSecondsRemaining"] = (4 - df["quarter"]) * 900 + df["gameClockSeconds"]

    # Diferencia de marcador
    if "preSnapHomeScore" in df.columns and "preSnapVisitorScore" in df.columns:
        df["scoreDifferential"] = df["preSnapHomeScore"] - df["preSnapVisitorScore"]
        df["absScoreDifferential"] = df["scoreDifferential"].abs()
        df["isWinning"] = (df["scoreDifferential"] > 0).astype(int)
        df["isLosing"] = (df["scoreDifferential"] < 0).astype(int)
        df["isTied"] = (df["scoreDifferential"] == 0).astype(int)

    # Situación de passing (típicamente pase)
    df["isPassingSituation"] = (
        ((df["down"] == 3) & (df["yardsToGo"] > 5))
        | ((df["down"] == 4) & (df["yardsToGo"] > 2))
        | (df["yardsToGo"] > 10)
    ).astype(int)

    # Situación de rushing (típicamente carrera)
    df["isRushingSituation"] = (
        ((df["down"] == 1) & (df["yardsToGo"] <= 10))
        | ((df["down"] <= 2) & (df["yardsToGo"] <= 3))
    ).astype(int)

    # Presión por tiempo (2-minute drill)
    df["timePressure"] = (
        (df["gameClockSeconds"] <= 120)
        & (df["quarter"].isin([2, 4]))
    ).astype(int)

    # Red zone
    if "yardlineNumber" in df.columns:
        df["inRedZone"] = (df["yardlineNumber"] <= 20).astype(int)
        df["inGoalLine"] = (df["yardlineNumber"] <= 5).astype(int)

    # Field position buckets
    if "absoluteYardlineNumber" in df.columns:
        df["fieldPositionBucket"] = pd.cut(
            df["absoluteYardlineNumber"],
            bins=[0, 20, 40, 60, 80, 100],
            labels=["own_deep", "own_territory", "midfield", "opp_territory", "red_zone"],
        )

    print("✓ Features de situación de juego creadas")
    return df


def create_down_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features relacionadas con down y distancia.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con jugadas

    Returns
    -------
    pd.DataFrame
        DataFrame con nuevas features
    """
    df = df.copy()

    # Categorías de distancia
    df["distanceCategory"] = pd.cut(
        df["yardsToGo"],
        bins=[0, 3, 7, 12, float("inf")],
        labels=["short", "medium", "long", "very_long"],
    )

    # Down específico combinado con distancia
    df["downAndDistance"] = df["down"].astype(str) + "_" + df["distanceCategory"].astype(str)

    # Es tercera o cuarta y larga?
    df["isThirdOrFourthLong"] = (
        (df["down"].isin([3, 4])) & (df["yardsToGo"] > 7)
    ).astype(int)

    # Probabilidad base de conversión (feature heurística)
    # Basada en estadísticas históricas de la NFL
    conversion_base = {
        (1, "short"): 0.95,
        (1, "medium"): 0.85,
        (1, "long"): 0.70,
        (1, "very_long"): 0.55,
        (2, "short"): 0.90,
        (2, "medium"): 0.75,
        (2, "long"): 0.55,
        (2, "very_long"): 0.40,
        (3, "short"): 0.75,
        (3, "medium"): 0.50,
        (3, "long"): 0.35,
        (3, "very_long"): 0.20,
        (4, "short"): 0.70,
        (4, "medium"): 0.45,
        (4, "long"): 0.30,
        (4, "very_long"): 0.15,
    }
    df["expectedConversionRate"] = df.apply(
        lambda x: conversion_base.get((x["down"], x["distanceCategory"]), 0.5),
        axis=1,
    )

    print("✓ Features de down y distancia creadas")
    return df


def create_formation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features relacionadas con formaciones.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con jugadas

    Returns
    -------
    pd.DataFrame
        DataFrame con nuevas features
    """
    df = df.copy()

    # Encoding de formación ofensiva
    if "offenseFormation" in df.columns:
        # Formaciones típicamente de pase
        pass_formations = ["SHOTGUN", "EMPTY"]
        df["isPassFormation"] = df["offenseFormation"].isin(pass_formations).astype(int)

        # Formaciones típicamente de carrera
        run_formations = ["I_FORM", "SINGLEBACK", "JUMBO"]
        df["isRunFormation"] = df["offenseFormation"].isin(run_formations).astype(int)

    # Defensores en la caja
    if "defendersInTheBox" in df.columns:
        df["lightBox"] = (df["defendersInTheBox"] <= 6).astype(int)
        df["loadedBox"] = (df["defendersInTheBox"] >= 8).astype(int)

    # Parse personnel
    if "personnelO" in df.columns:
        df["numRB"] = df["personnelO"].str.extract(r"(\d+) RB").astype(float).fillna(1)
        df["numTE"] = df["personnelO"].str.extract(r"(\d+) TE").astype(float).fillna(1)
        df["numWR"] = df["personnelO"].str.extract(r"(\d+) WR").astype(float).fillna(2)

    print("✓ Features de formación creadas")
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas las transformaciones de feature engineering.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original

    Returns
    -------
    pd.DataFrame
        DataFrame con todas las features
    """
    df = create_game_situation_features(df)
    df = create_down_distance_features(df)
    df = create_formation_features(df)
    print("✓ Feature engineering completo")
    return df


# =============================================================================
# FUNCIONES DE VISUALIZACIÓN
# =============================================================================


def plot_play_type_distribution(
    df: pd.DataFrame,
    target_col: str = "isPass",
    title: str = "Distribución de Tipo de Jugada",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Grafica la distribución de tipos de jugada.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos
    target_col : str
        Columna objetivo
    title : str
        Título del gráfico
    figsize : tuple
        Tamaño de la figura

    Returns
    -------
    plt.Figure
        Figura de matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Countplot
    counts = df[target_col].value_counts()
    labels = ["Carrera", "Pase"] if target_col == "isPass" else counts.index.tolist()

    axes[0].bar(labels, counts.values, color=[COLORS["secondary"], COLORS["primary"]])
    axes[0].set_title("Conteo por Tipo")
    axes[0].set_ylabel("Cantidad de Jugadas")

    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 100, f"{v:,}", ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=[COLORS["secondary"], COLORS["primary"]],
        explode=(0.02, 0.02),
    )
    axes[1].set_title("Proporción")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Grafica la importancia de features.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame con columnas 'feature' e 'importance'
    top_n : int
        Número de features a mostrar
    title : str
        Título
    figsize : tuple
        Tamaño de figura

    Returns
    -------
    plt.Figure
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    top_features = importance_df.nlargest(top_n, "importance")

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features)))[::-1]

    ax.barh(
        top_features["feature"],
        top_features["importance"],
        color=colors,
    )
    ax.set_xlabel("Importancia")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
    title: str = "Matriz de Confusión",
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Grafica matriz de confusión.

    Parameters
    ----------
    y_true : array-like
        Valores reales
    y_pred : array-like
        Predicciones
    labels : list
        Etiquetas de clases
    title : str
        Título
    figsize : tuple
        Tamaño de figura

    Returns
    -------
    plt.Figure
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = ["Carrera", "Pase"]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_roc_curves(
    results: Dict[str, Dict],
    title: str = "Curvas ROC - Comparación de Modelos",
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Grafica curvas ROC para múltiples modelos.

    Parameters
    ----------
    results : dict
        Diccionario con resultados de modelos
        {model_name: {'y_true': array, 'y_proba': array}}
    title : str
        Título
    figsize : tuple
        Tamaño de figura

    Returns
    -------
    plt.Figure
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["success"]]

    for i, (name, data) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
        auc = roc_auc_score(data["y_true"], data["y_proba"])
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc:.3f})",
            color=colors[i % len(colors)],
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_down_distance_analysis(
    df: pd.DataFrame,
    target_col: str = "isPass",
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Analiza la distribución de jugadas por down y distancia.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos
    target_col : str
        Columna objetivo
    figsize : tuple
        Tamaño de figura

    Returns
    -------
    plt.Figure
        Figura de matplotlib
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Por down
    down_pass_rate = df.groupby("down")[target_col].mean()
    axes[0].bar(down_pass_rate.index, down_pass_rate.values, color=COLORS["primary"])
    axes[0].set_xlabel("Down")
    axes[0].set_ylabel("Tasa de Pases")
    axes[0].set_title("Tasa de Pases por Down")
    axes[0].set_ylim(0, 1)

    # Por distancia
    if "distanceCategory" in df.columns:
        dist_pass_rate = df.groupby("distanceCategory")[target_col].mean()
        axes[1].bar(
            range(len(dist_pass_rate)),
            dist_pass_rate.values,
            color=COLORS["secondary"],
        )
        axes[1].set_xticks(range(len(dist_pass_rate)))
        axes[1].set_xticklabels(dist_pass_rate.index, rotation=45)
        axes[1].set_xlabel("Categoría de Distancia")
        axes[1].set_ylabel("Tasa de Pases")
        axes[1].set_title("Tasa de Pases por Distancia")
        axes[1].set_ylim(0, 1)

    # Heatmap down vs distance
    if "distanceCategory" in df.columns:
        pivot = df.pivot_table(
            values=target_col,
            index="down",
            columns="distanceCategory",
            aggfunc="mean",
        )
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlBu_r", ax=axes[2])
        axes[2].set_title("Tasa de Pases: Down vs Distancia")

    plt.tight_layout()
    return fig


# =============================================================================
# FUNCIONES DE EVALUACIÓN
# =============================================================================


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Modelo",
) -> Dict:
    """
    Evalúa un modelo de clasificación.

    Parameters
    ----------
    y_true : array-like
        Valores reales
    y_pred : array-like
        Predicciones
    y_proba : array-like, optional
        Probabilidades predichas
    model_name : str
        Nombre del modelo

    Returns
    -------
    dict
        Diccionario con métricas
    """
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = None

    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = None,
) -> None:
    """
    Imprime un reporte de clasificación formateado.

    Parameters
    ----------
    y_true : array-like
        Valores reales
    y_pred : array-like
        Predicciones
    target_names : list
        Nombres de las clases
    """
    if target_names is None:
        target_names = ["Carrera", "Pase"]

    print("\n" + "=" * 60)
    print("REPORTE DE CLASIFICACIÓN")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=target_names))


def compare_models(results: List[Dict]) -> pd.DataFrame:
    """
    Compara resultados de múltiples modelos.

    Parameters
    ----------
    results : list
        Lista de diccionarios con métricas

    Returns
    -------
    pd.DataFrame
        DataFrame comparativo
    """
    df = pd.DataFrame(results)
    df = df.set_index("model")

    # Formatear porcentajes
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if x else "N/A")

    return df


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================


def get_feature_columns(
    df: pd.DataFrame,
    exclude_cols: List[str] = None,
) -> List[str]:
    """
    Obtiene las columnas de features para el modelo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos
    exclude_cols : list
        Columnas a excluir

    Returns
    -------
    list
        Lista de columnas de features
    """
    if exclude_cols is None:
        exclude_cols = [
            "gameId",
            "playId",
            "isPass",
            "isSuccess",
            "playResult",
            "passResult",
            "gameClock",
            "personnelO",
            "personnelD",
            "downAndDistance",
            "fieldPositionBucket",
            "distanceCategory",
        ]

    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols
        and df[col].dtype in ["int64", "float64", "int32", "float32"]
    ]

    return feature_cols


def save_model_artifacts(
    model,
    feature_cols: List[str],
    metrics: Dict,
    output_dir: str = "models",
    model_name: str = "model",
) -> None:
    """
    Guarda artefactos del modelo.

    Parameters
    ----------
    model : estimator
        Modelo entrenado
    feature_cols : list
        Lista de features
    metrics : dict
        Métricas del modelo
    output_dir : str
        Directorio de salida
    model_name : str
        Nombre del modelo
    """
    import joblib

    os.makedirs(output_dir, exist_ok=True)

    # Guardar modelo
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)

    # Guardar features
    features_path = os.path.join(output_dir, f"{model_name}_features.txt")
    with open(features_path, "w") as f:
        f.write("\n".join(feature_cols))

    # Guardar métricas
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.txt")
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"✓ Artefactos guardados en {output_dir}/")


if __name__ == "__main__":
    # Test básico
    print("Probando funciones de utils...")
    
    # Crear datos de ejemplo
    df = create_sample_data(1000)
    print(f"\nDataset de ejemplo: {df.shape}")
    
    # Aplicar feature engineering
    df = engineer_all_features(df)
    print(f"Dataset con features: {df.shape}")
    
    # Mostrar features creadas
    print(f"\nColumnas: {df.columns.tolist()}")
