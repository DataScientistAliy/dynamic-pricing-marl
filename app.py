"""
Multi-Agent Dynamic Pricing — Streamlit bosh sahifa.

Ishga tushirish:
    $ streamlit run app.py
"""

import sys
from pathlib import Path

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
    page_title="Dynamic Pricing MARL | TDIU",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/DataScientistAliy/dynamic-pricing-marl",
        "About": "MARL Dynamic Pricing — TDIU BMI 2026",
    },
)

ensure_data_files_exist()
st.markdown(create_custom_css(), unsafe_allow_html=True)

COLORS = get_color_palette()


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding: 1.25rem 0 0.75rem; text-align:center;">
        <div style="
            width: 56px; height: 56px;
            background: linear-gradient(135deg,#f97316,#facc15);
            border-radius: 16px;
            display: inline-flex; align-items:center; justify-content:center;
            font-size: 1.6rem; margin-bottom: 0.75rem;
        ">🔥</div>
        <div style="
            font-size: 1.15rem;
            font-weight: 800;
            color: #f1f5f9;
            letter-spacing: -0.02em;
            line-height: 1.2;
        ">MARL Pricing</div>
        <div style="color: #475569; font-size: 0.78rem; margin-top: 0.2rem;">
            Dinamik Narxlash Tizimi
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background: rgba(249,115,22,0.08);
        border: 1px solid rgba(249,115,22,0.18);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin: 0.5rem 0 1rem;
    ">
        <div style="color:#fb923c; font-size:0.78rem; font-weight:700; margin-bottom:0.4rem; text-transform:uppercase; letter-spacing:0.06em;">
            📌 Loyiha
        </div>
        <div style="color:#94a3b8; font-size:0.82rem; line-height:1.5;">
            BMI: RL orqali e-commerce platformalari
            uchun intellektual narx optimizatsiyasi.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="color:#64748b; font-size:0.75rem; font-weight:700;
                text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;">
        Sahifalar
    </div>
    """, unsafe_allow_html=True)

    nav_items = [
        ("📊", "EDA va Dataset", "Ma'lumotlar tahlili"),
        ("🤖", "RL Simulatsiya",  "Interaktiv demo"),
        ("📈", "Natijalar",      "Algoritmlar taqqoslash"),
        ("🔬", "Statistik Tahlil","Testlar & Cohen's d"),
        ("🗺️", "Tavsiyalar",      "Biznes yo'l xaritasi"),
    ]

    for icon, title, desc in nav_items:
        st.markdown(f"""
        <div style="
            display:flex; align-items:center; gap:0.65rem;
            padding:0.55rem 0.75rem;
            border-radius:10px;
            margin-bottom:0.2rem;
            border: 1px solid transparent;
            transition: all 0.15s;
        " onmouseover="this.style.background='rgba(249,115,22,0.06)'"
          onmouseout="this.style.background='transparent'">
            <span style="font-size:1.1rem;">{icon}</span>
            <div>
                <div style="color:#e2e8f0; font-size:0.85rem; font-weight:600;">{title}</div>
                <div style="color:#475569; font-size:0.73rem;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(249,115,22,0.1);
        text-align:center;
        color:#334155;
        font-size:0.73rem;
    ">
        <strong style="color:#475569;">TDIU · Data Science</strong><br>
        Bitiruv Malakaviy Ishi · 2026
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HERO BANNER — katta, rangli blok
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div style="
    background: linear-gradient(135deg, #1c0a00 0%, #2d1200 30%, #1a0e2e 70%, #0b0e1a 100%);
    border: 1px solid rgba(249,115,22,0.25);
    border-radius: 24px;
    padding: 3.25rem 3rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
">
    <!-- Dekorativ doiralar -->
    <div style="
        position:absolute; top:-60px; right:-60px;
        width:260px; height:260px;
        background: radial-gradient(circle, rgba(249,115,22,0.18) 0%, transparent 70%);
        border-radius:50%;
    "></div>
    <div style="
        position:absolute; bottom:-80px; left:30%;
        width:320px; height:320px;
        background: radial-gradient(circle, rgba(167,139,250,0.10) 0%, transparent 70%);
        border-radius:50%;
    "></div>

    <!-- Badgelar -->
    <div style="margin-bottom:1rem; position:relative;">
        <span class="badge badge-info">v1.0.0</span>&nbsp;
        <span class="badge badge-success">BMI 2026</span>&nbsp;
        <span class="badge badge-purple">TDIU · Data Science</span>
    </div>

    <!-- Sarlavha -->
    <h1 style="
        color: #fff;
        font-size: 3rem;
        font-weight: 900;
        line-height: 1.1;
        margin: 0 0 0.75rem;
        letter-spacing: -0.03em;
        position:relative;
    ">
        Multi-Agent RL<br>
        <span style="
            background: linear-gradient(125deg, #f97316, #facc15, #fb923c);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        ">Dinamik Narxlash</span>
    </h1>

    <p style="
        color: rgba(255,255,255,0.65);
        font-size: 1.1rem;
        margin-bottom: 1.75rem;
        max-width: 600px;
        position:relative;
    ">
        E-commerce platformalari uchun kuchaytirilgan o'qitish (MARL) algoritmlari
        yordamida intellektual narx optimallashtirish tizimi.
    </p>

    <!-- Natijalar chiplari -->
    <div style="display:flex; flex-wrap:wrap; gap:0.5rem; position:relative;">
        <span class="stat-chip">📈 +37.5% Daromad</span>
        <span class="stat-chip">❤️ +42% Sodiqlik</span>
        <span class="stat-chip">📉 −54% Churn</span>
        <span class="stat-chip">✅ p &lt; 0.001</span>
        <span class="stat-chip">🔬 n=30 tajriba</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 4 ta KPI METRIC KARTA
# ═══════════════════════════════════════════════════════════════

col1, col2, col3, col4 = st.columns(4, gap="medium")

kpi_data = [
    ("💰", "Daromad o'sishi", "+37.5%", "↗ Statik baseline'ga nisbatan"),
    ("❤️", "Mijoz sodiqligi", "+42%",   "↗ Loyalty indeksi oshdi"),
    ("🚪", "Churn kamayishi", "−54%",   "↗ Yo'qotishlar minimallashdi"),
    ("📊", "Stat. ahamiyat",  "p<0.001","↗ n=30 · Welch t-test"),
]

for col, (icon, label, value, delta) in zip([col1, col2, col3, col4], kpi_data):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.75rem; margin-bottom:0.5rem; line-height:1;">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta-up">{delta}</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# ARXITEKTURA DIAGRAMMASI
# ═══════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-title">🏗️ Tizim Arxitekturasi</h2>', unsafe_allow_html=True)
st.markdown(
    "<p style='color:#64748b; font-size:0.9rem; margin-bottom:1rem;'>"
    "3 ta mustaqil RL agent bir muhitda o'zaro ta'sir qiladi va optimal narx strategiyasini o'rganadi."
    "</p>",
    unsafe_allow_html=True,
)


def build_architecture_diagram() -> go.Figure:
    """Arxitektura flow diagrammasi."""
    fig = go.Figure()
    tmpl = get_plotly_template()

    nodes = [
        (0.50, 0.92, "📦  Online Retail II Dataset<br><sup>1M+ tranzaksiya, UK Retail</sup>",    "#f97316", 70),
        (0.50, 0.72, "⚙️  Preprocessing Pipeline<br><sup>RFM · Elastiklik · Tozalash</sup>",    "#facc15", 58),
        (0.18, 0.45, "🎯  Pricing Agent<br><sup>PPO / DQN / SAC</sup>",                          "#f97316", 68),
        (0.50, 0.45, "👤  Customer Agent<br><sup>Talab modeli</sup>",                             "#38bdf8", 65),
        (0.82, 0.45, "🏪  Competitor Agent<br><sup>Raqobat simulyatsiya</sup>",                   "#a78bfa", 65),
        (0.50, 0.20, "🌐  MARL Muhiti<br><sup>Gymnasium · PettingZoo AEC</sup>",                 "#34d399", 62),
        (0.50, 0.03, "🏆  Reward:  R = 0.7·rev + 0.3·loyalty·100",                               "#fb923c", 52),
    ]

    edges = [(0,1),(1,2),(1,3),(1,4),(2,5),(3,5),(4,5),(5,6)]

    for s, d in edges:
        x0, y0 = nodes[s][0], nodes[s][1]
        x1, y1 = nodes[d][0], nodes[d][1]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color="rgba(249,115,22,0.3)", width=2.5),
            hoverinfo="skip", showlegend=False,
        ))

    for x, y, label, color, size in nodes:
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=size, color=color, opacity=0.9,
                        line=dict(color="rgba(255,255,255,0.15)", width=2)),
            text=[label],
            textposition="middle center",
            textfont=dict(color="white", size=11, family="Plus Jakarta Sans"),
            hovertext=[label.replace("<br>", " ").replace("<sup>","(").replace("</sup>",")")],
            hoverinfo="text",
            showlegend=False,
        ))

    fig.update_layout(
        height=560,
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-0.05, 1.0]),
        **tmpl["layout"],
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    return fig


st.plotly_chart(build_architecture_diagram(), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# 3 ta AGENT KARTASI
# ═══════════════════════════════════════════════════════════════

st.markdown('<h2 class="section-title">🤖 Intellektual Agentlar</h2>', unsafe_allow_html=True)

ac1, ac2, ac3 = st.columns(3, gap="medium")

agents = [
    (
        "🎯", "Pricing Agent", COLORS["PPO"],
        "Sotuvchi tomonidagi asosiy RL agent. "
        "Narxni episod davomida <strong style='color:#fb923c;'>±5%, ±10%, ±15%</strong> "
        "qadam bilan o'zgartiradi. 7 ta diskret harakat maydoni.",
        ["PPO", "DQN", "SAC"], "badge-info",
    ),
    (
        "👤", "Customer Agent", COLORS["DQN"],
        "Mijoz xulqini modellashtiradi — narxga sezgirlik, "
        "byudjet, qulaylik. Uch xil qaror: "
        "<strong style='color:#38bdf8;'>Sotib olish</strong>, Kutish, Tark etish.",
        ["Utility model", "Price elasticity"], "badge-info",
    ),
    (
        "🏪", "Competitor Agent", COLORS["SAC"],
        "Raqobatchi platformani simulyatsiya qiladi. "
        "Bozor ulushi va narx nisbatini kuzatib "
        "<strong style='color:#a78bfa;'>5 ta strategiya</strong>dan birini tanlaydi.",
        ["Market share", "Price ratio"], "badge-info",
    ),
]

for col, (icon, title, color, desc, tags, badge_cls) in zip([ac1, ac2, ac3], agents):
    tags_html = " ".join(f'<span class="badge {badge_cls}">{t}</span>' for t in tags)
    with col:
        st.markdown(f"""
        <div class="agent-card">
            <span class="agent-icon">{icon}</span>
            <h3 class="agent-title" style="color:{color};">{title}</h3>
            <p class="agent-description">{desc}</p>
            <div style="margin-top:1.1rem; padding-top:0.9rem;
                        border-top:1px solid rgba(255,255,255,0.06);">
                {tags_html}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# REWARD FORMULASI
# ═══════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-title">⚖️ Reward Funksiyasi</h2>', unsafe_allow_html=True)

st.markdown("""
<div style="
    background: #141928;
    border: 1px solid rgba(249,115,22,0.18);
    border-radius: 18px;
    padding: 2rem 2.25rem;
    text-align: center;
">
    <div style="
        font-size: 2rem;
        font-family: 'Space Mono', monospace;
        margin: 0.5rem 0 1rem;
        letter-spacing: 0.02em;
    ">
        <span style="color:#f87171; font-weight:700;">R</span>
        <span style="color:#64748b;"> = </span>
        <span style="color:#f97316; font-weight:700;">α</span>
        <span style="color:#64748b;"> · revenue</span><sub style="color:#64748b;">norm</sub>
        <span style="color:#64748b;">  +  </span>
        <span style="color:#a78bfa; font-weight:700;">β</span>
        <span style="color:#64748b;"> · loyalty</span><sub style="color:#64748b;">Δ</sub>
        <span style="color:#64748b;"> · 100</span>
    </div>
    <div style="
        display:flex; justify-content:center; gap:2rem; flex-wrap:wrap;
        margin-top:0.75rem;
    ">
        <div style="
            background: rgba(249,115,22,0.08); border:1px solid rgba(249,115,22,0.2);
            border-radius:12px; padding:0.75rem 1.5rem;
        ">
            <div style="color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.07em;">
                Daromad koeffitsienti
            </div>
            <div style="color:#f97316; font-size:1.4rem; font-weight:800;
                        font-family:'Space Mono',monospace;">
                α = 0.7
            </div>
        </div>
        <div style="
            background: rgba(167,139,250,0.08); border:1px solid rgba(167,139,250,0.2);
            border-radius:12px; padding:0.75rem 1.5rem;
        ">
            <div style="color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.07em;">
                Sodiqlik koeffitsienti
            </div>
            <div style="color:#a78bfa; font-size:1.4rem; font-weight:800;
                        font-family:'Space Mono',monospace;">
                β = 0.3
            </div>
        </div>
        <div style="
            background: rgba(74,222,128,0.08); border:1px solid rgba(74,222,128,0.2);
            border-radius:12px; padding:0.75rem 1.5rem;
        ">
            <div style="color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.07em;">
                Sezgirlik tahlili
            </div>
            <div style="color:#4ade80; font-size:1.4rem; font-weight:800;
                        font-family:'Space Mono',monospace;">
                Grid search
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# DEMO CTA
# ═══════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)

left_sp, cta_c, right_sp = st.columns([1, 2, 1])
with cta_c:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(249,115,22,0.08), rgba(167,139,250,0.08));
        border: 1px solid rgba(249,115,22,0.25);
        border-radius: 20px;
        padding: 2.25rem;
        text-align: center;
        margin-bottom: 1rem;
    ">
        <div style="font-size:2.5rem; margin-bottom:0.5rem;">🚀</div>
        <h3 style="color:#f1f5f9; margin:0 0 0.75rem; font-size:1.25rem; font-weight:700;">
            Simulyatsiyani Sinab Ko'ring
        </h3>
        <p style="color:#64748b; font-size:0.88rem; margin:0 0 1.25rem;">
            Interaktiv graflar orqali algoritmlar qanday ishlashini real vaqtda kuzating.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶  Demo'ni Boshlash", use_container_width=True, type="primary"):
        st.switch_page("pages/2_🤖_RL_Simulatsiya.py")


# ═══════════════════════════════════════════════════════════════
# ALGORITMLAR NATIJASI — qisqa taqqoslash jadvali
# ═══════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-title">📊 Asosiy Natijalar</h2>', unsafe_allow_html=True)

results_html = """
<div style="
    background:#141928;
    border:1px solid rgba(255,255,255,0.06);
    border-radius:18px;
    overflow:hidden;
">
<table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
    <thead>
        <tr style="background:rgba(249,115,22,0.08); border-bottom:1px solid rgba(249,115,22,0.15);">
            <th style="padding:0.85rem 1.25rem; text-align:left; color:#64748b;
                       font-weight:700; text-transform:uppercase; letter-spacing:0.06em; font-size:0.75rem;">
                Algoritm
            </th>
            <th style="padding:0.85rem 1rem; text-align:right; color:#64748b;
                       font-weight:700; text-transform:uppercase; letter-spacing:0.06em; font-size:0.75rem;">
                Daromad (£)
            </th>
            <th style="padding:0.85rem 1rem; text-align:right; color:#64748b;
                       font-weight:700; text-transform:uppercase; letter-spacing:0.06em; font-size:0.75rem;">
                Sodiqlik
            </th>
            <th style="padding:0.85rem 1rem; text-align:right; color:#64748b;
                       font-weight:700; text-transform:uppercase; letter-spacing:0.06em; font-size:0.75rem;">
                Churn
            </th>
            <th style="padding:0.85rem 1rem; text-align:right; color:#64748b;
                       font-weight:700; text-transform:uppercase; letter-spacing:0.06em; font-size:0.75rem;">
                HV
            </th>
        </tr>
    </thead>
    <tbody>
"""

rows = [
    ("🔥 PPO",    "#f97316", "38,672 ±1,756", "0.713 ±0.048", "0.083 ±0.021", "0.687", True),
    ("💜 SAC",    "#a78bfa", "36,141 ±2,478", "0.652 ±0.069", "0.103 ±0.027", "0.589", False),
    ("🔵 DQN",    "#38bdf8", "34,987 ±2,184", "0.621 ±0.058", "0.124 ±0.029", "0.521", False),
    ("⬜ Static", "#64748b", "28,134 ±1,478", "0.502 ±0.041", "0.182 ±0.023", "0.214", False),
]

for i, (name, color, rev, loy, churn, hv, is_best) in enumerate(rows):
    bg = "rgba(249,115,22,0.06)" if is_best else ("rgba(255,255,255,0.02)" if i % 2 else "transparent")
    star = " ⭐" if is_best else ""
    results_html += f"""
        <tr style="border-bottom:1px solid rgba(255,255,255,0.04); background:{bg};">
            <td style="padding:0.9rem 1.25rem;">
                <span style="color:{color}; font-weight:700; font-size:0.95rem;">{name}{star}</span>
            </td>
            <td style="padding:0.9rem 1rem; text-align:right; color:#f1f5f9; font-family:'Space Mono',monospace; font-size:0.85rem;">
                {rev}
            </td>
            <td style="padding:0.9rem 1rem; text-align:right; color:#f1f5f9; font-family:'Space Mono',monospace; font-size:0.85rem;">
                {loy}
            </td>
            <td style="padding:0.9rem 1rem; text-align:right; color:#f1f5f9; font-family:'Space Mono',monospace; font-size:0.85rem;">
                {churn}
            </td>
            <td style="padding:0.9rem 1rem; text-align:right; font-family:'Space Mono',monospace; font-size:0.85rem; color:{color};">
                {hv}
            </td>
        </tr>
    """

results_html += "</tbody></table></div>"
st.markdown(results_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TEXNOLOGIYALAR
# ═══════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-title">🛠️ Texnologiyalar</h2>', unsafe_allow_html=True)

techs = [
    ("Python 3.10+",      "#3b82f6"),
    ("PyTorch",           "#ef4444"),
    ("Stable-Baselines3", "#f97316"),
    ("Gymnasium",         "#06b6d4"),
    ("PettingZoo",        "#a855f7"),
    ("Streamlit",         "#ff4b4b"),
    ("Plotly",            "#818cf8"),
    ("Pandas",            "#10b981"),
    ("SciPy",             "#f59e0b"),
    ("NumPy",             "#4ade80"),
]

tech_html = '<div style="display:flex; flex-wrap:wrap; gap:0.6rem; margin-top:0.25rem;">'
for tech, color in techs:
    tech_html += f"""
    <div style="
        background: rgba(255,255,255,0.03);
        border: 1px solid {color}30;
        color: {color};
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 0.01em;
    ">{tech}</div>
    """
tech_html += "</div>"
st.markdown(tech_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
    <strong style="color:#475569;">Multi-Agent Dynamic Pricing System</strong><br>
    Toshkent Davlat Iqtisodiyot Universiteti · Data Science · 2026<br>
    <span style="color:#1e293b; font-size:0.72rem;">
        Built with Reinforcement Learning · PPO · DQN · SAC
    </span>
</div>
""", unsafe_allow_html=True)
