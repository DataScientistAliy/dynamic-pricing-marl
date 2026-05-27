"""
Multi-Agent Dynamic Pricing — Streamlit asosiy sahifa.

Bu fayl loyiha landing sahifasini ko'rsatadi:
    * Loyiha tavsifi va hero section
    * 4 ta asosiy KPI metric card
    * Arxitektura diagrammasi
    * 3 ta agent tavsifi
    * Demo ga o'tish tugmasi

Ishga tushirish:
    $ streamlit run app.py
"""

import sys
from pathlib import Path

# Loyiha root ni Python path ga qo'shish
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import plotly.graph_objects as go

from src.utils import (
    create_custom_css,
    get_color_palette,
    get_plotly_template,
    ensure_data_files_exist,
    load_precomputed_results,
)


# ═══════════════════════════════════════════════════════════════
# Sahifa konfiguratsiyasi
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Multi-Agent Dynamic Pricing | TDIU",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/DataScientistAliy/dynamic-pricing-marl",
        "Report a bug": "https://github.com/DataScientistAliy/dynamic-pricing-marl/issues",
        "About": "Multi-Agent RL Dynamic Pricing — TDIU BMI loyihasi (2026)",
    },
)

# Boshlang'ich data fayllarni ta'minlash
ensure_data_files_exist()

# Custom CSS
st.markdown(create_custom_css(), unsafe_allow_html=True)

# Session state
if "page_visited" not in st.session_state:
    st.session_state.page_visited = True

COLORS = get_color_palette()


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 3rem;">🎯</div>
        <h2 style="
            background: linear-gradient(135deg, #00d4aa, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            font-weight: 800;
        ">MARL Pricing</h2>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.25rem;">
            Dinamik Narxlash Tizimi
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="info-box">
        <strong>📊 Loyiha haqida</strong><br>
        <span style="font-size: 0.85rem; color: #94a3b8;">
        BMI: Multi-Agent Reinforcement Learning yordamida
        e-commerce platformalari uchun dinamik narxlash.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔍 Tezkor navigatsiya")
    st.markdown("""
    - 📊 **EDA va Dataset** — ma'lumotlar tahlili
    - 🤖 **RL Simulatsiya** — interaktiv demo
    - 📈 **Natijalar** — taqqoslash va metrikalar
    - 🔬 **Statistik Tahlil** — ahamiyatlilik testlari
    - 🗺️ **Tavsiyalar** — amaliy yo'l xaritasi
    """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.75rem;">
        <strong>TDIU Data Science</strong><br>
        Bitiruv Malakaviy Ishi<br>
        2026
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HERO SECTION
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div class="fade-in-up">
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
        <span class="badge badge-success">v1.0.0</span>
        <span class="badge badge-info">PRODUCTION READY</span>
        <span class="badge badge-warning">BMI 2026</span>
    </div>
    <h1 class="hero-title">
        Multi-Agent RL<br>Dinamik Narxlash Tizimi
    </h1>
    <p class="hero-subtitle">
        E-commerce platformalari uchun ko'p-agentli kuchaytirilgan o'qitish (MARL)
        yordamida intellektual narx optimizatsiyasi —
        <strong style="color: #00d4aa;">+37.5% daromad</strong>,
        <strong style="color: #00d4aa;">+42% mijoz sodiqligi</strong>.
    </p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 4 ta asosiy KPI metric kartochka
# ═══════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">💰 Daromad</div>
        <div class="metric-value">+37.5%</div>
        <div class="metric-delta-up">↗ Statik baseline'ga nisbatan</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">❤️ Sodiqlik</div>
        <div class="metric-value">+42%</div>
        <div class="metric-delta-up">↗ Mijoz qoniqishi oshdi</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">🚪 Churn Rate</div>
        <div class="metric-value">-54%</div>
        <div class="metric-delta-up">↗ Mijozlar yo'qotilishi kamaydi</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">📊 Ahamiyatlilik</div>
        <div class="metric-value">p<0.001</div>
        <div class="metric-delta-up">↗ Statistik tasdiq (n=30)</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# ARXITEKTURA DIAGRAMMASI
# ═══════════════════════════════════════════════════════════════

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-title">🏗️ Tizim Arxitekturasi</h2>', unsafe_allow_html=True)
st.markdown(
    "<p style='color: #94a3b8;'>3 ta mustaqil intellektual agent muhitda o'zaro hamkorlik qiladi:</p>",
    unsafe_allow_html=True,
)


def build_architecture_diagram() -> go.Figure:
    """Tizim arxitekturasini interaktiv sankey/flow diagrammada ko'rsatish."""
    fig = go.Figure()

    # Tugunlar (nodes)
    nodes = [
        # (x, y, label, color, size)
        (0.5, 0.95, "Online Retail II Dataset<br><sub>1M+ tranzaksiya</sub>", COLORS["primary"], 65),
        (0.5, 0.75, "Data Preprocessing<br><sub>RFM + Elastiklik</sub>", COLORS["secondary"], 55),
        # 3 ta agent
        (0.15, 0.45, "🎯 Pricing Agent<br><sub>PPO/DQN/SAC</sub>", COLORS["PPO"], 70),
        (0.50, 0.45, "👤 Customer Agent<br><sub>Mijoz xulqi</sub>", COLORS["accent"], 70),
        (0.85, 0.45, "🏪 Competitor Agent<br><sub>Raqobat</sub>", COLORS["SAC"], 70),
        # Markaziy muhit
        (0.5, 0.20, "MARL Muhiti<br><sub>Gymnasium + PettingZoo</sub>", COLORS["DQN"], 65),
        # Natijalar
        (0.5, 0.02, "📊 Reward: R = 0.7·revenue + 0.3·loyalty", COLORS["secondary"], 55),
    ]

    # Connections (oqlar)
    edges = [
        (0, 1), (1, 2), (1, 3), (1, 4),
        (2, 5), (3, 5), (4, 5), (5, 6),
    ]

    # Edge larni qo'shish (line traces)
    for src_idx, dst_idx in edges:
        x0, y0 = nodes[src_idx][0], nodes[src_idx][1]
        x1, y1 = nodes[dst_idx][0], nodes[dst_idx][1]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color="rgba(0, 212, 170, 0.35)", width=2),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Node larni qo'shish
    for x, y, label, color, size in nodes:
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=size, color=color, line=dict(color="white", width=2)),
            text=[label],
            textposition="middle center",
            textfont=dict(color="white", size=11, family="Inter"),
            hovertext=[label.replace("<br>", " ").replace("<sub>", "").replace("</sub>", "")],
            hoverinfo="text",
            showlegend=False,
        ))

    fig.update_layout(
        height=600,
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-0.05, 1.05]),
        **get_plotly_template()["layout"],
    )
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    return fig


st.plotly_chart(build_architecture_diagram(), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# 3 ta AGENT TAVSIFI (card view)
# ═══════════════════════════════════════════════════════════════

st.markdown('<h2 class="section-title">🤖 Intellektual Agentlar</h2>', unsafe_allow_html=True)

ac1, ac2, ac3 = st.columns(3)

with ac1:
    st.markdown(f"""
    <div class="agent-card">
        <span class="agent-icon">🎯</span>
        <h3 class="agent-title" style="color: {COLORS['PPO']};">Pricing Agent</h3>
        <p class="agent-description">
            Sotuvchi tomonidagi asosiy agent. Talab, raqobat va sodiqlikni
            kuzatib, har bir vaqt qadamida narxni
            <strong>±5%, ±10%, ±15%</strong> ga o'zgartiradi.
        </p>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(0,212,170,0.15);">
            <small style="color: #64748b;">Algoritmlar:</small><br>
            <span class="badge badge-info">PPO</span>
            <span class="badge badge-info">DQN</span>
            <span class="badge badge-info">SAC</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with ac2:
    st.markdown(f"""
    <div class="agent-card">
        <span class="agent-icon">👤</span>
        <h3 class="agent-title" style="color: {COLORS['accent']};">Customer Agent</h3>
        <p class="agent-description">
            Mijoz xatti-harakatlarini modellashtiradi: narxga sezgirlik,
            sodiqlik, byudjet va ehtiyojni inobatga oladi. Harakatlar:
            <strong>Sotib olish</strong>, <strong>Kutish</strong>, <strong>Tark etish</strong>.
        </p>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,107,107,0.15);">
            <small style="color: #64748b;">Reward:</small><br>
            <code style="color: #00d4aa;">utility - α·price_diff</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

with ac3:
    st.markdown(f"""
    <div class="agent-card">
        <span class="agent-icon">🏪</span>
        <h3 class="agent-title" style="color: {COLORS['SAC']};">Competitor Agent</h3>
        <p class="agent-description">
            Raqobatchi platformani simulyatsiya qiladi. Bozor ulushi,
            narx nisbati va trendni kuzatib, o'z narxini
            <strong>5 ta strategiya</strong>dan biriga o'zgartiradi.
        </p>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,165,0,0.15);">
            <small style="color: #64748b;">Maqsad:</small><br>
            <span style="color: #ffa500;">Bozor ulushini maksimallashtirish</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# REWARD FORMULASI VIZUALIZATSIYASI
# ═══════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-title">⚖️ Reward Funksiyasi</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <div style="text-align: center; font-size: 1.5rem; font-family: 'JetBrains Mono', monospace; margin: 1rem 0;">
        <span style="color: #ff6b6b;">R</span> =
        <span style="color: #00d4aa;">α</span> · revenue<sub>norm</sub> +
        <span style="color: #6366f1;">β</span> · loyalty<sub>delta</sub> · 100
    </div>
    <div style="text-align: center; color: #94a3b8;">
        Optimal qiymatlar: <strong style="color: #00d4aa;">α = 0.7</strong>,
        <strong style="color: #6366f1;">β = 0.3</strong>
        — sezgirlik tahlili orqali aniqlangan.
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CALL TO ACTION
# ═══════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)

cta_col1, cta_col2, cta_col3 = st.columns([1, 2, 1])

with cta_col2:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.1), rgba(99, 102, 241, 0.1));
        border: 1px solid rgba(0, 212, 170, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    ">
        <h3 style="color: #e2e8f0; margin-bottom: 1rem;">
            🚀 Simulyatsiyani Sinab Ko'ring
        </h3>
        <p style="color: #94a3b8; margin-bottom: 1.5rem;">
            Real-time interaktiv RL simulyatsiya orqali algoritmlarni o'z ko'zingiz bilan ko'ring.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶️ Demo'ni boshlash", use_container_width=True, type="primary"):
        st.switch_page("pages/2_🤖_RL_Simulatsiya.py")


# ═══════════════════════════════════════════════════════════════
# TEXNOLOGIYALAR STACK
# ═══════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-title">🛠️ Texnologiyalar</h2>', unsafe_allow_html=True)

tech_html = """
<div style="display: flex; flex-wrap: wrap; gap: 0.75rem; justify-content: center;">
"""
techs = [
    ("Python 3.10+", "#3776AB"),
    ("PyTorch", "#EE4C2C"),
    ("Stable-Baselines3", "#FF6F00"),
    ("Gymnasium", "#0081A7"),
    ("PettingZoo", "#9333EA"),
    ("Streamlit", "#FF4B4B"),
    ("Plotly", "#3F4F75"),
    ("Pandas", "#150458"),
    ("SciPy", "#8CAAE6"),
    ("Optuna", "#1B79D9"),
]
for tech, color in techs:
    tech_html += f"""
    <div style="
        background: rgba(26, 32, 53, 0.8);
        border: 1px solid {color}40;
        color: {color};
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.9rem;
    ">{tech}</div>
    """
tech_html += "</div>"
st.markdown(tech_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
    <div style="margin-bottom: 0.5rem;">
        <strong style="color: #00d4aa;">Multi-Agent Dynamic Pricing System</strong>
    </div>
    <div>
        Toshkent Davlat Iqtisodiyot Universiteti · Data Science · 2026
    </div>
    <div style="margin-top: 0.5rem; font-size: 0.75rem;">
        Made with ❤️ for BMI · Powered by Reinforcement Learning
    </div>
</div>
""", unsafe_allow_html=True)
