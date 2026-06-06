"""
Multi-Agent Dynamic Pricing — bosh sahifa.
Ishga tushirish: streamlit run app.py
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
    show_progress_sidebar,
)

st.set_page_config(
    page_title="Dynamic Pricing MARL | TDIU",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_data_files_exist()
st.markdown(create_custom_css(), unsafe_allow_html=True)
COLORS = get_color_palette()


# ─── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem; text-align:center;">
      <div style="
        width:52px; height:52px;
        background:linear-gradient(135deg,#f97316,#fbbf24);
        border-radius:14px;
        display:inline-flex; align-items:center; justify-content:center;
        font-size:1.5rem; margin-bottom:0.6rem;
      ">🔥</div>
      <div style="font-size:1.05rem;font-weight:800;color:#1c1917;letter-spacing:-0.02em;">
        MARL Pricing
      </div>
      <div style="color:#a8a29e;font-size:0.75rem;margin-top:0.15rem;">
        Dinamik Narxlash · TDIU 2026
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<p style='color:#a8a29e;font-size:0.72rem;font-weight:700;"
        "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.75rem;'>"
        "Jarayon bosqichlari</p>",
        unsafe_allow_html=True,
    )
    show_progress_sidebar(current_step=0)

    st.markdown("---")
    st.markdown("""
    <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;
                padding:0.8rem;margin-top:0.25rem;">
      <div style="color:#ea580c;font-size:0.72rem;font-weight:700;margin-bottom:0.3rem;
                  text-transform:uppercase;letter-spacing:0.06em;">Loyiha</div>
      <div style="color:#78716c;font-size:0.78rem;line-height:1.5;">
        BMI: RL orqali e-commerce uchun intellektual narx optimizatsiyasi.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─── HERO BANNER ────────────────────────────────────────────────
st.markdown("""
<div style="
  background:linear-gradient(135deg, #7c2d12 0%, #9a3412 35%, #431407 70%, #1c0a00 100%);
  border-radius:20px;
  padding:3rem 3rem 2.75rem;
  margin-bottom:1.75rem;
  position:relative;
  overflow:hidden;
">
  <div style="
    position:absolute;top:-40px;right:-40px;
    width:220px;height:220px;
    background:radial-gradient(circle,rgba(251,191,36,0.2) 0%,transparent 70%);
    border-radius:50%;
  "></div>
  <div style="
    position:absolute;bottom:-60px;left:40%;
    width:280px;height:280px;
    background:radial-gradient(circle,rgba(249,115,22,0.15) 0%,transparent 70%);
    border-radius:50%;
  "></div>
  <div style="position:relative;">
    <div style="margin-bottom:0.9rem;">
      <span class="badge badge-info">v1.0.0</span>&nbsp;
      <span class="badge badge-success">BMI 2026</span>&nbsp;
      <span class="badge badge-purple">TDIU · Data Science</span>
    </div>
    <h1 style="
      color:#fff;font-size:2.75rem;font-weight:900;
      line-height:1.1;margin:0 0 0.7rem;letter-spacing:-0.03em;
    ">
      Multi-Agent RL<br>
      <span style="
        background:linear-gradient(125deg,#fbbf24,#f97316,#fb923c);
        background-clip:text;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
      ">Dinamik Narxlash</span>
    </h1>
    <p style="color:rgba(255,255,255,0.65);font-size:1rem;margin-bottom:1.5rem;max-width:580px;">
      E-commerce platformalari uchun kuchaytirilgan o'qitish (MARL) algoritmlari
      yordamida intellektual narx optimallashtirish tizimi.
    </p>
    <div>
      <span class="stat-chip">📈 +37.5% Daromad</span>
      <span class="stat-chip">❤️ +42% Sodiqlik</span>
      <span class="stat-chip">📉 -54% Churn</span>
      <span class="stat-chip">p &lt; 0.001</span>
      <span class="stat-chip">n=30 tajriba</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── KPI KARTALAR ───────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4, gap="medium")

kpi = [
    ("💰", "Daromad o'sishi",  "+37.5%", "↗ Statik baseline'ga nisbatan"),
    ("❤️", "Mijoz sodiqligi",  "+42%",   "↗ Loyalty indeksi oshdi"),
    ("🚪", "Churn kamayishi",  "−54%",   "↗ Yo'qotishlar minimallashdi"),
    ("📊", "Stat. ahamiyat",   "p<0.001","↗ Welch t-test · n=30"),
]
for col, (icon, label, val, delta) in zip([col1, col2, col3, col4], kpi):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div style="font-size:1.6rem;margin-bottom:0.4rem;">{icon}</div>
          <div class="metric-label">{label}</div>
          <div class="metric-value">{val}</div>
          <div class="metric-delta-up">{delta}</div>
        </div>
        """, unsafe_allow_html=True)


# ─── ARXITEKTURA ────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-title">🏗️ Tizim Arxitekturasi</h2>', unsafe_allow_html=True)
st.caption("3 ta mustaqil RL agent bir muhitda o'zaro ta'sir qilib optimal narx strategiyasini o'rganadi.")


def build_arch_fig() -> go.Figure:
    fig = go.Figure()
    tmpl = get_plotly_template()

    nodes = [
        (0.50, 0.92, "📦 Online Retail II\n1M+ tranzaksiya",     "#ea580c", 70),
        (0.50, 0.72, "⚙️ Preprocessing\nRFM · Elastiklik",       "#d97706", 56),
        (0.18, 0.45, "🎯 Pricing Agent\nPPO / DQN / SAC",        "#ea580c", 66),
        (0.50, 0.45, "👤 Customer Agent\nTalab modeli",           "#0284c7", 62),
        (0.82, 0.45, "🏪 Competitor Agent\nRaqobat sim.",         "#7c3aed", 62),
        (0.50, 0.20, "🌐 MARL Muhiti\nGymnasium · PettingZoo",   "#059669", 60),
        (0.50, 0.03, "🏆 R = 0.7·rev + 0.3·loyalty·100",         "#d97706", 50),
    ]
    edges = [(0,1),(1,2),(1,3),(1,4),(2,5),(3,5),(4,5),(5,6)]

    for s, d in edges:
        fig.add_trace(go.Scatter(
            x=[nodes[s][0], nodes[d][0]], y=[nodes[s][1], nodes[d][1]],
            mode="lines",
            line=dict(color="rgba(249,115,22,0.35)", width=2.5),
            hoverinfo="skip", showlegend=False,
        ))
    for x, y, lbl, col, sz in nodes:
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=sz, color=col, opacity=0.85,
                        line=dict(color="rgba(255,255,255,0.5)", width=2)),
            text=[lbl], textposition="middle center",
            textfont=dict(color="white", size=10.5, family="Plus Jakarta Sans"),
            hoverinfo="text", showlegend=False,
        ))

    fig.update_layout(
        height=530, showlegend=False,
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-0.05, 1.0]),
        **tmpl["layout"],
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    return fig


st.plotly_chart(build_arch_fig(), use_container_width=True)


# ─── AGENTLAR ───────────────────────────────────────────────────
st.markdown('<h2 class="section-title">🤖 Intellektual Agentlar</h2>', unsafe_allow_html=True)

ac1, ac2, ac3 = st.columns(3, gap="medium")
agents_data = [
    ("🎯", "Pricing Agent", COLORS["PPO"],
     "Sotuvchi RL agenti. Narxni episod davomida "
     "<strong style='color:#ea580c;'>±5%, ±10%, ±15%</strong> "
     "qadam bilan o'zgartiradi. 7 ta diskret harakat.",
     ["PPO", "DQN", "SAC"]),
    ("👤", "Customer Agent", COLORS["DQN"],
     "Mijoz xulqini modellashtiradi. Narxga sezgirlik, "
     "byudjet va qulaylikni inobatga oladi. "
     "<strong style='color:#0284c7;'>3 qaror:</strong> Sotib olish, Kutish, Tark etish.",
     ["Utility model", "Elasticity"]),
    ("🏪", "Competitor Agent", COLORS["SAC"],
     "Raqobatchi simulyatsiya. Bozor ulushi va narx nisbatini "
     "kuzatib <strong style='color:#7c3aed;'>5 strategiya</strong>dan birini tanlaydi.",
     ["Market share", "Price ratio"]),
]
for col, (icon, title, color, desc, tags) in zip([ac1, ac2, ac3], agents_data):
    tags_html = " ".join(
        f'<span class="badge badge-info" style="margin:0.1rem;">{t}</span>' for t in tags
    )
    with col:
        st.markdown(f"""
        <div class="agent-card">
          <span class="agent-icon">{icon}</span>
          <h3 class="agent-title" style="color:{color};">{title}</h3>
          <p class="agent-description">{desc}</p>
          <div style="margin-top:1rem;padding-top:0.8rem;border-top:1px solid #ffedd5;">
            {tags_html}
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─── REWARD FORMULASI ───────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-title">⚖️ Reward Funksiyasi</h2>', unsafe_allow_html=True)

st.markdown("""
<div style="
  background:#ffffff;border:1px solid #ffedd5;
  border-radius:16px;padding:1.75rem 2rem;
  box-shadow:0 2px 10px rgba(249,115,22,0.07);
  text-align:center;
">
  <div style="
    font-size:1.8rem;font-family:'Space Grotesk',sans-serif;
    margin-bottom:1.25rem;
    color:#1c1917;
    letter-spacing:0.01em;
  ">
    <span style="color:#dc2626;font-weight:800;">R</span>
    <span style="color:#a8a29e;"> = </span>
    <span style="color:#ea580c;font-weight:700;">α</span>
    <span style="color:#78716c;"> · revenue</span>
    <sub style="color:#a8a29e;font-size:0.6em;">norm</sub>
    <span style="color:#78716c;">  +  </span>
    <span style="color:#7c3aed;font-weight:700;">β</span>
    <span style="color:#78716c;"> · loyalty</span>
    <sub style="color:#a8a29e;font-size:0.6em;">delta</sub>
    <span style="color:#78716c;"> · 100</span>
  </div>
  <div style="display:flex;justify-content:center;gap:1.25rem;flex-wrap:wrap;">
    <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:12px;padding:0.7rem 1.4rem;">
      <div style="color:#a8a29e;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.07em;">Daromad</div>
      <div style="color:#ea580c;font-size:1.4rem;font-weight:800;font-family:'Space Grotesk',sans-serif;">α = 0.7</div>
    </div>
    <div style="background:#f5f3ff;border:1px solid #ddd6fe;border-radius:12px;padding:0.7rem 1.4rem;">
      <div style="color:#a8a29e;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.07em;">Sodiqlik</div>
      <div style="color:#7c3aed;font-size:1.4rem;font-weight:800;font-family:'Space Grotesk',sans-serif;">β = 0.3</div>
    </div>
    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;padding:0.7rem 1.4rem;">
      <div style="color:#a8a29e;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.07em;">Sezgirlik</div>
      <div style="color:#16a34a;font-size:1.4rem;font-weight:800;font-family:'Space Grotesk',sans-serif;">Grid ✓</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── NATIJALAR JADVALI ──────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-title">📊 Asosiy Natijalar</h2>', unsafe_allow_html=True)

rows = [
    ("🥇 PPO",    "#ea580c", "38,672 ±1,756", "0.713 ±0.048", "0.083", "0.687", True),
    ("🥈 SAC",    "#7c3aed", "36,141 ±2,478", "0.652 ±0.069", "0.103", "0.589", False),
    ("🥉 DQN",    "#0284c7", "34,987 ±2,184", "0.621 ±0.058", "0.124", "0.521", False),
    ("— Static",  "#78716c", "28,134 ±1,478", "0.502 ±0.041", "0.182", "0.214", False),
]

tbl = """
<div style="background:#ffffff;border:1px solid #ffedd5;border-radius:16px;
            overflow:hidden;box-shadow:0 2px 10px rgba(249,115,22,0.07);">
<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
  <thead>
    <tr style="background:#fff7ed;border-bottom:2px solid #ffedd5;">
      <th style="padding:0.8rem 1.2rem;text-align:left;color:#a8a29e;font-size:0.72rem;
                 text-transform:uppercase;letter-spacing:0.07em;">Algoritm</th>
      <th style="padding:0.8rem 1rem;text-align:right;color:#a8a29e;font-size:0.72rem;
                 text-transform:uppercase;letter-spacing:0.07em;">Daromad (£)</th>
      <th style="padding:0.8rem 1rem;text-align:right;color:#a8a29e;font-size:0.72rem;
                 text-transform:uppercase;letter-spacing:0.07em;">Sodiqlik</th>
      <th style="padding:0.8rem 1rem;text-align:right;color:#a8a29e;font-size:0.72rem;
                 text-transform:uppercase;letter-spacing:0.07em;">Churn</th>
      <th style="padding:0.8rem 1rem;text-align:right;color:#a8a29e;font-size:0.72rem;
                 text-transform:uppercase;letter-spacing:0.07em;">HV</th>
    </tr>
  </thead>
  <tbody>
"""
for name, color, rev, loy, churn, hv, best in rows:
    bg = "background:#fff7ed;" if best else ("background:#fafaf9;" if rows.index((name,color,rev,loy,churn,hv,best))%2 else "")
    star = " ⭐" if best else ""
    tbl += f"""
    <tr style="border-bottom:1px solid #f5f5f4;{bg}">
      <td style="padding:0.85rem 1.2rem;">
        <span style="color:{color};font-weight:700;">{name}{star}</span>
      </td>
      <td style="padding:0.85rem 1rem;text-align:right;color:#1c1917;font-weight:500;">{rev}</td>
      <td style="padding:0.85rem 1rem;text-align:right;color:#1c1917;font-weight:500;">{loy}</td>
      <td style="padding:0.85rem 1rem;text-align:right;color:#1c1917;font-weight:500;">{churn}</td>
      <td style="padding:0.85rem 1rem;text-align:right;color:{color};font-weight:700;">{hv}</td>
    </tr>
    """
tbl += "</tbody></table></div>"
st.markdown(tbl, unsafe_allow_html=True)


# ─── DEMO CTA ───────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, cta, _ = st.columns([1, 2, 1])
with cta:
    st.markdown("""
    <div style="
      background:linear-gradient(135deg,#fff7ed,#fef3c7);
      border:1px solid #fed7aa;border-radius:16px;
      padding:2rem;text-align:center;margin-bottom:1rem;
    ">
      <div style="font-size:2.2rem;margin-bottom:0.4rem;">🚀</div>
      <h3 style="color:#1c1917;margin:0 0 0.6rem;font-size:1.15rem;font-weight:700;">
        Simulyatsiyani Sinab Ko'ring
      </h3>
      <p style="color:#78716c;font-size:0.85rem;margin:0 0 1rem;">
        Real vaqtda algoritmlar qanday ishlashini kuzating.
      </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("▶  Demo'ni Boshlash", use_container_width=True, type="primary"):
        st.switch_page("pages/2_🤖_RL_Simulatsiya.py")


# ─── TEXNOLOGIYALAR ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-title">🛠️ Texnologiyalar</h2>', unsafe_allow_html=True)

techs = [
    ("Python 3.10+", "#2563eb"), ("PyTorch", "#dc2626"),
    ("Stable-Baselines3", "#ea580c"), ("Gymnasium", "#0891b2"),
    ("PettingZoo", "#7c3aed"), ("Streamlit", "#e11d48"),
    ("Plotly", "#6366f1"), ("Pandas", "#059669"),
    ("SciPy", "#d97706"), ("NumPy", "#16a34a"),
]
th = '<div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-top:0.25rem;">'
for name, col in techs:
    th += (f'<div style="background:#ffffff;border:1px solid {col}30;color:{col};'
           f'padding:0.35rem 0.85rem;border-radius:8px;font-weight:600;font-size:0.8rem;">'
           f'{name}</div>')
th += "</div>"
st.markdown(th, unsafe_allow_html=True)


# ─── FOOTER ─────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <strong style="color:#78716c;">Multi-Agent Dynamic Pricing System</strong><br>
  Toshkent Davlat Iqtisodiyot Universiteti · Data Science · 2026<br>
  <span style="color:#d6d3d1;">PPO · DQN · SAC · Gymnasium · PettingZoo</span>
</div>
""", unsafe_allow_html=True)
