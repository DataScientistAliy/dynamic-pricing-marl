"""
Yordamchi funksiyalar moduli.

Bu modul quyidagilarni o'z ichiga oladi:
    * Precomputed natijalarni generatsiya qilish va yuklash
    * Sample dataset generatsiya qilish
    * Streamlit uchun custom CSS
    * JSON yordamida saqlash va yuklash funksiyalari

BMI loyihasidan olingan ANIQ raqamlar asosida realistic
ma'lumotlar yaratiladi (PPO daromad +37.5%, sodiqliq +42%, churn -54%).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


# ═══════════════════════════════════════════════════════════════
# BMI dan olingan ANIQ statistik raqamlar (Bobi 3 jadvallaridan)
# ═══════════════════════════════════════════════════════════════

BMI_RESULTS_CONFIG: Dict[str, Dict[str, Any]] = {
    "PPO": {
        "revenue_mean": 38672.0,
        "revenue_std": 1756.0,
        "loyalty_mean": 0.713,
        "loyalty_std": 0.048,
        "churn_mean": 0.083,
        "churn_std": 0.021,
        "volatility_mean": 1.92,
        "volatility_std": 0.31,
        "convergence_episode": 145,
        "final_reward": 28.4,
        "training_time_sec": 412.0,
        "memory_mb": 387.0,
    },
    "SAC": {
        "revenue_mean": 36141.0,
        "revenue_std": 2478.0,
        "loyalty_mean": 0.652,
        "loyalty_std": 0.069,
        "churn_mean": 0.103,
        "churn_std": 0.027,
        "volatility_mean": 2.34,
        "volatility_std": 0.39,
        "convergence_episode": 178,
        "final_reward": 26.1,
        "training_time_sec": 528.0,
        "memory_mb": 451.0,
    },
    "DQN": {
        "revenue_mean": 34987.0,
        "revenue_std": 2184.0,
        "loyalty_mean": 0.621,
        "loyalty_std": 0.058,
        "churn_mean": 0.124,
        "churn_std": 0.029,
        "volatility_mean": 2.78,
        "volatility_std": 0.42,
        "convergence_episode": 220,
        "final_reward": 24.7,
        "training_time_sec": 367.0,
        "memory_mb": 312.0,
    },
    "Static": {
        "revenue_mean": 28134.0,
        "revenue_std": 1478.0,
        "loyalty_mean": 0.502,
        "loyalty_std": 0.041,
        "churn_mean": 0.182,
        "churn_std": 0.023,
        "volatility_mean": 0.51,
        "volatility_std": 0.09,
        "convergence_episode": 0,
        "final_reward": 18.9,
        "training_time_sec": 0.0,
        "memory_mb": 45.0,
    },
}


# Pareto Hyper-Volume metrikasi (BMI 3.4-jadvalidan)
HYPERVOLUMES: Dict[str, float] = {
    "PPO": 0.687,
    "SAC": 0.589,
    "DQN": 0.521,
    "Static": 0.214,
}


def _project_root() -> Path:
    """Loyiha asosiy papkasini aniqlash (src/utils.py joyidan)."""
    return Path(__file__).resolve().parent.parent


def _data_dir() -> Path:
    """data/ papkasiga yo'l."""
    return _project_root() / "data"


def generate_precomputed_results(
    n_runs: int = 30,
    n_episodes: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Demo uchun realistic natijalar generatsiya qilish.

    Har algoritm uchun BMI dan olingan o'rtacha va standart og'ish
    asosida normal taqsimotdan tasodifiy qiymatlar olinadi.

    Parameters
    ----------
    n_runs : int
        Har algoritm uchun mustaqil run lar soni (default: 30)
    n_episodes : int
        Learning curve uchun epizodlar soni (default: 300)
    seed : int
        Reproducibility uchun random seed

    Returns
    -------
    dict
        Barcha natijalarni o'z ichiga olgan lug'at
    """
    rng = np.random.default_rng(seed)
    experiments: Dict[str, Any] = {}

    for algo, cfg in BMI_RESULTS_CONFIG.items():
        # 30 ta mustaqil run uchun metrikalar (normal taqsimot)
        revenues = rng.normal(cfg["revenue_mean"], cfg["revenue_std"], n_runs)
        loyalties = np.clip(
            rng.normal(cfg["loyalty_mean"], cfg["loyalty_std"], n_runs),
            0.0,
            1.0,
        )
        churns = np.clip(
            rng.normal(cfg["churn_mean"], cfg["churn_std"], n_runs),
            0.0,
            1.0,
        )
        volatilities = np.clip(
            rng.normal(cfg["volatility_mean"], cfg["volatility_std"], n_runs),
            0.0,
            None,
        )
        avg_prices = np.clip(
            rng.normal(11.5 if algo != "Static" else 10.0, 0.8, n_runs),
            5.0,
            30.0,
        )

        # Learning curve — logarifmik o'sish bilan tebranish
        learning_curve = _generate_learning_curve(
            final_reward=cfg["final_reward"],
            n_episodes=n_episodes,
            convergence_ep=cfg["convergence_episode"],
            noise_level=cfg["volatility_mean"] * 0.6,
            is_static=(algo == "Static"),
            rng=rng,
        )

        experiments[algo] = {
            "revenues": revenues.round(2).tolist(),
            "loyalties": loyalties.round(4).tolist(),
            "churns": churns.round(4).tolist(),
            "price_volatilities": volatilities.round(4).tolist(),
            "avg_prices": avg_prices.round(2).tolist(),
            "learning_curve": [round(float(x), 3) for x in learning_curve],
            "convergence_episode": cfg["convergence_episode"],
            "final_reward": cfg["final_reward"],
            "training_time_sec": cfg["training_time_sec"],
            "memory_mb": cfg["memory_mb"],
            "hypervolume": HYPERVOLUMES[algo],
        }

    return {
        "experiments": experiments,
        "metadata": {
            "dataset": "Online Retail II (UCI)",
            "n_runs": n_runs,
            "n_episodes_learning_curve": n_episodes,
            "seed": seed,
            "env_steps": 30000,
            "timestamp": "2026-01-01",
            "reward_function": "R = 0.7 * revenue_norm + 0.3 * loyalty_delta * 100",
            "algorithms": list(BMI_RESULTS_CONFIG.keys()),
        },
    }


def _generate_learning_curve(
    final_reward: float,
    n_episodes: int,
    convergence_ep: int,
    noise_level: float,
    is_static: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Realistic learning curve generatsiya qilish.

    Forma: logarifmik o'sish + Gauss noise.
    Static agent uchun gorizontal chiziq.
    """
    if is_static:
        # Statik baseline — tebranishsiz
        return np.full(n_episodes, final_reward) + rng.normal(0, 0.3, n_episodes)

    # Boshlang'ich reward (random exploration)
    initial_reward = final_reward * 0.15

    # Logarifmik o'sish funksiyasi
    episodes = np.arange(n_episodes)
    progress = np.clip(episodes / max(convergence_ep, 1), 0.0, 1.0)
    # Smooth log-shaped curve
    growth = 1.0 - np.exp(-3.0 * progress)
    base_curve = initial_reward + (final_reward - initial_reward) * growth

    # Reward noise
    noise = rng.normal(0.0, noise_level, n_episodes)
    return base_curve + noise


def save_results(results_dict: Dict[str, Any], path: str | Path) -> None:
    """
    Natijalar lug'atini JSON formatida saqlash.

    Parameters
    ----------
    results_dict : dict
        Saqlash uchun ma'lumotlar
    path : str or Path
        Saqlanish manzili
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)


def load_precomputed_results(path: str | Path | None = None) -> Dict[str, Any]:
    """
    data/precomputed_results.json dan natijalarni yuklash.

    Agar fayl mavjud bo'lmasa — avtomatik generatsiya qilinadi.

    Parameters
    ----------
    path : str, Path, or None
        JSON fayl manzili. None bo'lsa, standart yo'l ishlatiladi.

    Returns
    -------
    dict
        Barcha natijalarni o'z ichiga olgan lug'at
    """
    if path is None:
        path = _data_dir() / "precomputed_results.json"

    path = Path(path)
    if not path.exists():
        # Fayl mavjud emas — yangi generatsiya
        results = generate_precomputed_results()
        save_results(results, path)
        return results

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_color_palette() -> Dict[str, str]:
    """
    Loyiha bo'yicha bir xil rang sxemasini qaytarish.

    Algoritm rang assignment + UI rang doimiy bo'lishi uchun.
    """
    return {
        # Algoritm ranglari
        "PPO": "#00d4aa",       # teal — yashil-ko'k
        "SAC": "#ffa500",       # to'q sariq
        "DQN": "#6366f1",       # binafsha-ko'k
        "Static": "#94a3b8",    # kulrang
        # UI brand ranglari
        "primary": "#1e3a5f",
        "secondary": "#00d4aa",
        "accent": "#ff6b6b",
        "background": "#0e1117",
        "card_bg": "#1a2035",
        "text": "#e2e8f0",
        "text_muted": "#94a3b8",
        "success": "#22c55e",
        "warning": "#f59e0b",
        "danger": "#ef4444",
    }


def create_custom_css() -> str:
    """
    Streamlit uchun zamonaviy custom CSS qaytarish.

    Glassmorphism, gradient, hover effektlari va animatsiyalar
    bilan professional dizayn.
    """
    return """
    <style>
    /* ═══════════ Asosiy font ═══════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ═══════════ Asosiy fon ═══════════ */
    .stApp {
        background: radial-gradient(ellipse at top, #1a2035 0%, #0e1117 50%, #0a0e15 100%);
    }

    /* ═══════════ Sidebar ═══════════ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2035 0%, #0e1117 100%);
        border-right: 1px solid rgba(0, 212, 170, 0.15);
    }

    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }

    /* ═══════════ Asosiy header — gradient matn ═══════════ */
    .hero-title {
        background: linear-gradient(135deg, #00d4aa 0%, #6366f1 50%, #ff6b6b 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        line-height: 1.1;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.25rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    .section-title {
        background: linear-gradient(90deg, #00d4aa 0%, #6366f1 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2rem;
        margin: 2rem 0 1rem 0;
    }

    /* ═══════════ Metric kartochka ═══════════ */
    .metric-card {
        background: linear-gradient(135deg, rgba(26, 32, 53, 0.9) 0%, rgba(14, 17, 23, 0.9) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00d4aa, #6366f1, #ff6b6b);
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 212, 170, 0.45);
        box-shadow: 0 12px 40px rgba(0, 212, 170, 0.2);
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        color: #e2e8f0;
        font-size: 2.25rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }

    .metric-delta-up {
        color: #22c55e;
        font-size: 0.95rem;
        font-weight: 600;
    }

    .metric-delta-down {
        color: #ef4444;
        font-size: 0.95rem;
        font-weight: 600;
    }

    /* ═══════════ Tugma ═══════════ */
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa 0%, #00a385 100%);
        color: #0e1117;
        border: none;
        border-radius: 10px;
        padding: 0.625rem 1.5rem;
        font-weight: 600;
        transition: all 0.25s ease;
        box-shadow: 0 4px 12px rgba(0, 212, 170, 0.25);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 212, 170, 0.4);
        background: linear-gradient(135deg, #00e8bd 0%, #00b894 100%);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* ═══════════ Tab ═══════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(26, 32, 53, 0.5);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 212, 170, 0.15);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        border-radius: 8px;
        padding: 0.625rem 1.25rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 212, 170, 0.08);
        color: #e2e8f0;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.2), rgba(99, 102, 241, 0.2)) !important;
        color: #00d4aa !important;
        border: 1px solid rgba(0, 212, 170, 0.3);
    }

    /* ═══════════ DataFrame ═══════════ */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(0, 212, 170, 0.15);
    }

    /* ═══════════ Slider ═══════════ */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #00d4aa;
        border: 2px solid #0e1117;
        box-shadow: 0 2px 8px rgba(0, 212, 170, 0.4);
    }

    /* ═══════════ Selectbox ═══════════ */
    .stSelectbox > div > div {
        background: rgba(26, 32, 53, 0.8);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 8px;
        color: #e2e8f0;
    }

    /* ═══════════ Agent card ═══════════ */
    .agent-card {
        background: linear-gradient(135deg, rgba(26, 32, 53, 0.95) 0%, rgba(20, 25, 42, 0.95) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.75rem;
        margin: 0.75rem 0;
        transition: all 0.3s ease;
        height: 100%;
    }

    .agent-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-4px);
        box-shadow: 0 12px 36px rgba(99, 102, 241, 0.15);
    }

    .agent-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
    }

    .agent-title {
        color: #e2e8f0;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .agent-description {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ═══════════ Badge ═══════════ */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-success {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    .badge-info {
        background: rgba(0, 212, 170, 0.15);
        color: #00d4aa;
        border: 1px solid rgba(0, 212, 170, 0.3);
    }

    .badge-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* ═══════════ Footer ═══════════ */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem 0;
        margin-top: 4rem;
        border-top: 1px solid rgba(0, 212, 170, 0.1);
        font-size: 0.875rem;
    }

    /* ═══════════ Streamlit standart elementlarini yashirish ═══════════ */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: rgba(14, 17, 23, 0.7);
        backdrop-filter: blur(10px);
    }

    /* ═══════════ Scrollbar ═══════════ */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #0e1117;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00d4aa, #6366f1);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #00e8bd, #818cf8);
    }

    /* ═══════════ Animatsiyalar ═══════════ */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in-up {
        animation: fadeInUp 0.6s ease-out forwards;
    }

    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 170, 0.3); }
        50% { box-shadow: 0 0 30px rgba(0, 212, 170, 0.6); }
    }

    .pulse-glow {
        animation: pulse-glow 2s ease-in-out infinite;
    }

    /* ═══════════ Info quti ═══════════ */
    .info-box {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.08) 0%, rgba(99, 102, 241, 0.08) 100%);
        border-left: 4px solid #00d4aa;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #e2e8f0;
    }

    .warning-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(255, 107, 107, 0.08) 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #e2e8f0;
    }
    </style>
    """


def format_currency(value: float, currency: str = "£") -> str:
    """
    Pul qiymatini chiroyli formatda qaytarish.

    Misol: format_currency(38672.50) -> "£38,672.50"
    """
    return f"{currency}{value:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Foiz formatida ko'rsatish: 0.375 -> '+37.5%'."""
    sign = "+" if value > 0 else ""
    return f"{sign}{value * 100:.{decimals}f}%"


def get_plotly_template() -> Dict[str, Any]:
    """
    Barcha plotly grafiklari uchun bir xil dark template.

    Returns
    -------
    dict
        Plotly layout konfiguratsiyasi
    """
    colors = get_color_palette()
    return {
        "layout": {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(26, 32, 53, 0.4)",
            "font": {
                "family": "Inter, -apple-system, sans-serif",
                "color": colors["text"],
                "size": 13,
            },
            "xaxis": {
                "gridcolor": "rgba(148, 163, 184, 0.1)",
                "zerolinecolor": "rgba(148, 163, 184, 0.2)",
                "color": colors["text_muted"],
            },
            "yaxis": {
                "gridcolor": "rgba(148, 163, 184, 0.1)",
                "zerolinecolor": "rgba(148, 163, 184, 0.2)",
                "color": colors["text_muted"],
            },
            "legend": {
                "bgcolor": "rgba(14, 17, 23, 0.7)",
                "bordercolor": "rgba(0, 212, 170, 0.2)",
                "borderwidth": 1,
                "font": {"color": colors["text"]},
            },
            "colorway": [
                colors["PPO"], colors["SAC"], colors["DQN"], colors["Static"],
                "#ec4899", "#06b6d4", "#a855f7", "#f97316",
            ],
            "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
            "hoverlabel": {
                "bgcolor": "#1a2035",
                "bordercolor": "#00d4aa",
                "font": {"color": "#e2e8f0", "family": "Inter"},
            },
        }
    }


def generate_sample_csv(n_rows: int = 1000, seed: int = 42) -> "pd.DataFrame":  # type: ignore[name-defined]
    """
    Online Retail II ga o'xshash sample dataset generatsiya qilish.

    Realistic mahsulot nomlari, davlatlar va sanalar bilan.

    Parameters
    ----------
    n_rows : int
        Generatsiya qilinadigan qatorlar soni
    seed : int
        Random seed

    Returns
    -------
    pandas.DataFrame
        Online Retail II formatidagi DataFrame
    """
    import pandas as pd
    from datetime import datetime, timedelta

    rng = np.random.default_rng(seed)

    # Realistic mahsulot kataloglari (UK retail)
    products = [
        ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER"),
        ("71053", "WHITE METAL LANTERN"),
        ("84406B", "CREAM CUPID HEARTS COAT HANGER"),
        ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE"),
        ("84029E", "RED WOOLLY HOTTIE WHITE HEART"),
        ("22752", "SET 7 BABUSHKA NESTING BOXES"),
        ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER"),
        ("22633", "HAND WARMER UNION JACK"),
        ("22632", "HAND WARMER RED POLKA DOT"),
        ("84879", "ASSORTED COLOUR BIRD ORNAMENT"),
        ("22745", "POPPY'S PLAYHOUSE BEDROOM"),
        ("22748", "POPPY'S PLAYHOUSE KITCHEN"),
        ("22310", "IVORY KNITTED MUG COSY"),
        ("84969", "BOX OF 6 ASSORTED COLOUR TEASPOONS"),
        ("22623", "BOX OF VINTAGE JIGSAW BLOCKS"),
        ("22622", "BOX OF VINTAGE ALPHABET BLOCKS"),
        ("21754", "HOME BUILDING BLOCK WORD"),
        ("21755", "LOVE BUILDING BLOCK WORD"),
        ("21777", "RECIPE BOX WITH METAL HEART"),
        ("48187", "DOORMAT NEW ENGLAND"),
        ("22960", "JAM MAKING SET WITH JARS"),
        ("22913", "RED COAT RACK PARIS FASHION"),
        ("22912", "YELLOW COAT RACK PARIS FASHION"),
        ("22914", "BLUE COAT RACK PARIS FASHION"),
        ("21034", "REX CASH+CARRY JUMBO SHOPPER"),
        ("85099B", "JUMBO BAG RED RETROSPOT"),
        ("85099C", "JUMBO BAG BAROQUE BLACK WHITE"),
        ("20725", "LUNCH BAG RED RETROSPOT"),
        ("20727", "LUNCH BAG BLACK SKULL"),
        ("22382", "LUNCH BAG SPACEBOY DESIGN"),
        ("23203", "JUMBO BAG VINTAGE DOILY"),
        ("23204", "JUMBO BAG BABUSHKA"),
        ("47566", "PARTY BUNTING"),
        ("23084", "RABBIT NIGHT LIGHT"),
        ("22086", "PAPER CHAIN KIT 50'S CHRISTMAS"),
        ("23298", "SPOTTY BUNTING"),
        ("22411", "JUMBO SHOPPER VINTAGE RED PAISLEY"),
        ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN"),
        ("22197", "POPCORN HOLDER"),
        ("21212", "PACK OF 72 RETROSPOT CAKE CASES"),
    ]

    # Davlatlar va ehtimolliklar (Online Retail II ga muvofiq)
    countries = [
        "United Kingdom", "Germany", "France", "EIRE",
        "Spain", "Netherlands", "Belgium", "Switzerland",
        "Portugal", "Australia", "Norway", "Italy",
    ]
    country_probs = [0.70, 0.10, 0.08, 0.04, 0.02, 0.02, 0.015, 0.01,
                     0.005, 0.005, 0.0025, 0.0025]

    # Sanalar oralig'i
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range_days = (end_date - start_date).days

    rows: List[Dict[str, Any]] = []
    invoice_counter = 489434

    for i in range(n_rows):
        # Har bir invoice 1-5 ta qatordan iborat bo'lishi mumkin
        if i % rng.integers(1, 6) == 0:
            invoice_counter += 1

        product_idx = rng.integers(0, len(products))
        stock_code, description = products[product_idx]

        # Real narx jenerator: mahsulotga qarab
        price = round(float(rng.uniform(0.5, 50.0)), 2)
        # 90% holatda 1-12 dona, 10% — yirik buyurtma
        if rng.random() < 0.9:
            quantity = int(rng.integers(1, 13))
        else:
            quantity = int(rng.integers(12, 101))

        random_days = rng.integers(0, date_range_days)
        invoice_date = start_date + timedelta(
            days=int(random_days),
            hours=int(rng.integers(8, 20)),
            minutes=int(rng.integers(0, 60)),
        )

        customer_id = int(rng.integers(14000, 18288))
        country = str(rng.choice(countries, p=country_probs))

        rows.append({
            "Invoice": str(invoice_counter),
            "StockCode": stock_code,
            "Description": description,
            "Quantity": quantity,
            "InvoiceDate": invoice_date.strftime("%Y-%m-%d %H:%M:%S"),
            "Price": price,
            "Customer ID": customer_id,
            "Country": country,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("InvoiceDate").reset_index(drop=True)
    return df


def ensure_data_files_exist() -> None:
    """
    data/precomputed_results.json va data/sample_data.csv mavjudligini ta'minlash.

    Agar ular mavjud bo'lmasa — avtomatik yaratiladi.
    """
    data_dir = _data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    json_path = data_dir / "precomputed_results.json"
    csv_path = data_dir / "sample_data.csv"

    if not json_path.exists():
        results = generate_precomputed_results()
        save_results(results, json_path)

    if not csv_path.exists():
        df = generate_sample_csv(n_rows=1000)
        df.to_csv(csv_path, index=False)


# ═══════════════════════════════════════════════════════════════
# Modul to'g'ridan-to'g'ri ishga tushirilsa, data fayllarni yarat
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(">>> precomputed_results.json va sample_data.csv generatsiya qilinmoqda...")
    ensure_data_files_exist()
    print(">>> Tayyor! data/ papkasini tekshiring.")
