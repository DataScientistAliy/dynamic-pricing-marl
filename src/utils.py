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
        # Algoritm ranglari — issiq, yorqin tona
        "PPO": "#ea580c",       # to'q apelsin (eng yaxshi algoritm)
        "SAC": "#7c3aed",       # binafsha
        "DQN": "#0284c7",       # okean ko'ki
        "Static": "#78716c",    # issiq kulrang
        # UI brand ranglari
        "primary": "#f97316",
        "secondary": "#eab308",
        "accent": "#dc2626",
        "background": "#fffaf5",
        "card_bg": "#ffffff",
        "text": "#1c1917",
        "text_muted": "#78716c",
        "success": "#16a34a",
        "warning": "#d97706",
        "danger": "#dc2626",
    }


def create_custom_css() -> str:
    """
    Streamlit uchun yorqin ochiq rangdagi custom CSS.

    Iliq oq/krem fon, apelsin aksent, aniq kartalar.
    """
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    }

    .stApp {
        background: #fffaf5;
        background-image:
            radial-gradient(ellipse 70% 40% at 100% 0%, rgba(249,115,22,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 50% 30% at 0% 100%, rgba(234,179,8,0.05) 0%, transparent 50%);
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: #fff7ed;
        border-right: 2px solid #fed7aa;
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        color: #44403c !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #fed7aa;
        margin: 0.75rem 0;
    }

    .hero-title {
        background: linear-gradient(125deg, #ea580c 0%, #f97316 50%, #d97706 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        line-height: 1.1;
        margin: 0.5rem 0;
        letter-spacing: -0.025em;
    }

    .hero-subtitle {
        color: #78716c;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.7;
    }

    .section-title {
        color: #1c1917;
        font-weight: 700;
        font-size: 1.5rem;
        margin: 2rem 0 0.6rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ffedd5;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .metric-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin: 0.3rem 0;
        border: 1px solid #ffedd5;
        border-left: 4px solid #f97316;
        box-shadow: 0 2px 12px rgba(249,115,22,0.08);
        transition: transform 0.22s ease, box-shadow 0.22s ease;
        position: relative;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 28px rgba(249,115,22,0.15);
    }

    .metric-label {
        color: #a8a29e;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        color: #ea580c;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.3rem;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    .metric-delta-up {
        color: #16a34a;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .metric-delta-down {
        color: #dc2626;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .stButton > button {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.5rem;
        font-weight: 700;
        font-size: 0.92rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(249,115,22,0.28);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(249,115,22,0.4);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #fff7ed;
        padding: 0.4rem;
        border-radius: 12px;
        border: 1px solid #fed7aa;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #a8a29e;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        font-size: 0.86rem;
        transition: all 0.15s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(249,115,22,0.07);
        color: #1c1917;
    }

    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #ea580c !important;
        border: 1px solid #fed7aa;
        box-shadow: 0 2px 8px rgba(249,115,22,0.12);
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #ffedd5;
    }

    .stSelectbox > div > div {
        background: #ffffff;
        border: 1px solid #fed7aa;
        border-radius: 10px;
        color: #1c1917;
    }

    .agent-card {
        background: #ffffff;
        border: 1px solid #ffedd5;
        border-radius: 16px;
        padding: 1.75rem 1.5rem;
        margin: 0.4rem 0;
        transition: all 0.25s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }

    .agent-card:hover {
        border-color: #f97316;
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(249,115,22,0.12);
    }

    .agent-icon {
        font-size: 2.5rem;
        margin-bottom: 0.9rem;
        display: block;
        line-height: 1;
    }

    .agent-title {
        color: #1c1917;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .agent-description {
        color: #78716c;
        font-size: 0.86rem;
        line-height: 1.65;
    }

    .badge {
        display: inline-block;
        padding: 0.18rem 0.6rem;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .badge-success {
        background: #dcfce7;
        color: #16a34a;
        border: 1px solid #bbf7d0;
    }

    .badge-info {
        background: #fff7ed;
        color: #ea580c;
        border: 1px solid #fed7aa;
    }

    .badge-warning {
        background: #fefce8;
        color: #d97706;
        border: 1px solid #fde68a;
    }

    .badge-purple {
        background: #f5f3ff;
        color: #7c3aed;
        border: 1px solid #ddd6fe;
    }

    .footer {
        text-align: center;
        color: #d6d3d1;
        padding: 2.5rem 0 1rem;
        margin-top: 4rem;
        border-top: 1px solid #ffedd5;
        font-size: 0.8rem;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    header[data-testid="stHeader"] {
        background: rgba(255,250,245,0.9);
        border-bottom: 1px solid #ffedd5;
    }

    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: #fff7ed; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #f97316, #fbbf24);
        border-radius: 8px;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in-up { animation: fadeInUp 0.5s ease-out forwards; }

    .info-box {
        background: #fff7ed;
        border-left: 4px solid #f97316;
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.1rem;
        margin: 0.75rem 0;
        color: #44403c;
    }

    .warning-box {
        background: #fefce8;
        border-left: 4px solid #d97706;
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.1rem;
        margin: 0.75rem 0;
        color: #44403c;
    }

    .stat-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(255,255,255,0.25);
        border: 1px solid rgba(255,255,255,0.4);
        border-radius: 50px;
        padding: 0.35rem 0.9rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: #fff;
        margin: 0.2rem;
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
    Barcha plotly grafiklari uchun bir xil ochiq rang template.

    Returns
    -------
    dict
        Plotly layout konfiguratsiyasi
    """
    colors = get_color_palette()
    return {
        "layout": {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(255,247,237,0.6)",
            "font": {
                "family": "Plus Jakarta Sans, -apple-system, sans-serif",
                "color": colors["text"],
                "size": 13,
            },
            "xaxis": {
                "gridcolor": "rgba(214,211,208,0.6)",
                "zerolinecolor": "rgba(214,211,208,0.8)",
                "color": colors["text_muted"],
                "linecolor": "rgba(214,211,208,0.6)",
            },
            "yaxis": {
                "gridcolor": "rgba(214,211,208,0.6)",
                "zerolinecolor": "rgba(214,211,208,0.8)",
                "color": colors["text_muted"],
                "linecolor": "rgba(214,211,208,0.6)",
            },
            "legend": {
                "bgcolor": "rgba(255,247,237,0.9)",
                "bordercolor": "#fed7aa",
                "borderwidth": 1,
                "font": {"color": colors["text"]},
            },
            "colorway": [
                colors["PPO"],    # ea580c orange
                colors["DQN"],    # 0284c7 blue
                colors["SAC"],    # 7c3aed purple
                colors["Static"], # 78716c gray
                "#db2777",        # pink
                "#059669",        # emerald
                "#0891b2",        # cyan
                "#d97706",        # amber
            ],
            "margin": {"l": 55, "r": 25, "t": 55, "b": 45},
            "hoverlabel": {
                "bgcolor": "#ffffff",
                "bordercolor": "#f97316",
                "font": {"color": "#1c1917", "family": "Plus Jakarta Sans"},
            },
        }
    }


def show_progress_sidebar(current_step: int = 0) -> None:
    """
    Sidebar da ulanuvchi step indicator ko'rsatish.

    Parameters
    ----------
    current_step : int
        Joriy sahifa raqami:
        0=Bosh sahifa, 1=EDA, 2=Simulatsiya, 3=Natijalar,
        4=Statistik Tahlil, 5=Tavsiyalar
    """
    import streamlit as st  # lokal import — circular dependency oldini olish

    steps = [
        ("🏠", "Bosh Sahifa",      "Loyiha ko'rinishi"),
        ("📊", "EDA va Dataset",    "Ma'lumotlar tahlili"),
        ("🤖", "RL Simulatsiya",    "Interaktiv demo"),
        ("📈", "Natijalar",         "Taqqoslash"),
        ("🔬", "Statistik Tahlil",  "Testlar & effekt"),
        ("🗺️", "Tavsiyalar",         "Yo'l xaritasi"),
    ]

    html_parts = ['<div style="padding:0.25rem 0;">']

    for i, (icon, name, desc) in enumerate(steps):
        is_active = i == current_step
        is_done   = i < current_step
        is_future = i > current_step

        if is_active:
            dot_bg     = "#f97316"
            dot_border = "#f97316"
            has_inner  = True
            inner_col  = "#ffffff"
            label_col  = "#1c1917"
            desc_col   = "#78716c"
            fw         = "700"
            glow       = "box-shadow:0 0 0 3px rgba(249,115,22,0.2);"
        elif is_done:
            dot_bg     = "#fed7aa"
            dot_border = "#f97316"
            has_inner  = True
            inner_col  = "#f97316"
            label_col  = "#78716c"
            desc_col   = "#a8a29e"
            fw         = "600"
            glow       = ""
        else:
            dot_bg     = "#ffffff"
            dot_border = "#d6d3d1"
            has_inner  = False
            inner_col  = ""
            label_col  = "#a8a29e"
            desc_col   = "#d6d3d1"
            fw         = "400"
            glow       = ""

        inner_html = (
            f'<div style="width:6px;height:6px;border-radius:50%;background:{inner_col};"></div>'
            if has_inner else ""
        )

        is_last = i == len(steps) - 1
        line_color = "#f97316" if is_done else "#e7e5e4"
        line_html = (
            ""
            if is_last
            else f'<div style="width:2px;height:28px;background:{line_color};'
                 f'margin:3px 0;border-radius:2px;flex-shrink:0;"></div>'
        )

        label_bg = "background:rgba(249,115,22,0.07);border-radius:8px;padding:0.2rem 0.5rem;" if is_active else ""

        html_parts.append(f"""
        <div style="display:flex;align-items:flex-start;gap:0.75rem;">
          <div style="display:flex;flex-direction:column;align-items:center;flex-shrink:0;padding-top:2px;">
            <div style="width:16px;height:16px;border-radius:50%;background:{dot_bg};
                        border:2px solid {dot_border};{glow}
                        display:flex;align-items:center;justify-content:center;">
              {inner_html}
            </div>
            {line_html}
          </div>
          <div style="padding-bottom:{'0' if is_last else '0.3rem'};">
            <div style="color:{label_col};font-size:0.82rem;font-weight:{fw};
                        line-height:1.3;{label_bg}">{icon} {name}</div>
            <div style="color:{desc_col};font-size:0.7rem;line-height:1.2;margin-top:1px;">{desc}</div>
          </div>
        </div>
        """)

    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


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
