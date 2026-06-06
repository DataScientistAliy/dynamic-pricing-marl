"""
Tavsiyalar va yo'l xaritasi sahifasi.

Sections:
    1. O'zbekiston e-commerce sektori
    2. MARL afzalliklari va ROI
    3. 4 bosqichli yo'l xaritasi (Gantt)
    4. Algoritm tavsiya kalkulyatori
    5. Etik mulohazalar
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils import (
    create_custom_css,
    get_color_palette,
    get_plotly_template,
    ensure_data_files_exist,
    show_progress_sidebar,
)


# ═══════════════════════════════════════════════════════════════
# Sahifa konfiguratsiyasi
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Tavsiyalar | MARL Pricing",
    page_icon="🗺️",
    layout="wide",
)

ensure_data_files_exist()
st.markdown(create_custom_css(), unsafe_allow_html=True)

COLORS = get_color_palette()
PLOTLY_TEMPLATE = get_plotly_template()

with st.sidebar:
    st.markdown("""
    <div style="padding:0.75rem 0 0.25rem;text-align:center;">
      <div style="font-size:1rem;font-weight:800;color:#1c1917;">MARL Pricing</div>
      <div style="color:#a8a29e;font-size:0.72rem;">TDIU 2026</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<p style='color:#a8a29e;font-size:0.7rem;font-weight:700;"
        "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;'>"
        "Jarayon bosqichlari</p>",
        unsafe_allow_html=True,
    )
    show_progress_sidebar(current_step=5)


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown('<h1 class="hero-title" style="font-size: 2.5rem;">🗺️ Tavsiyalar va Yo\'l Xaritasi</h1>',
            unsafe_allow_html=True)
st.markdown(
    "<p class='hero-subtitle'>O'zbekiston e-commerce sektori uchun MARL'ni amaliy joriy "
    "etish bo'yicha bosqichma-bosqich strategiya va biznes hisob-kitoblari.</p>",
    unsafe_allow_html=True,
)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════
# SECTION 1: O'zbekiston e-commerce
# ═══════════════════════════════════════════════════════════════

st.markdown('<h2 class="section-title">1️⃣ O\'zbekiston E-commerce Bozori</h2>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">💵 Bozor hajmi 2024</div>
        <div class="metric-value" style="font-size: 1.75rem;">$1.2B</div>
        <div class="metric-delta-up">↗ Doimiy o'sish</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">📈 Prognoz 2026</div>
        <div class="metric-value" style="font-size: 1.75rem;">$2.5B</div>
        <div class="metric-delta-up">↗ 2x o'sish</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">📱 Mobile share</div>
        <div class="metric-value" style="font-size: 1.75rem;">78%</div>
        <div class="metric-delta-up">↗ Mobile-first</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">👥 Uzum mijozlari</div>
        <div class="metric-value" style="font-size: 1.75rem;">4M+</div>
        <div class="metric-delta-up">↗ Yetakchi</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Yetakchi platformalar
st.markdown("**🏪 Yetakchi e-commerce platformalar:**")

platforms_data = pd.DataFrame([
    {"Platforma": "Uzum Market", "Mijozlar (mln)": 4.0, "Bozor ulushi (%)": 35, "Segment": "Marketplace"},
    {"Platforma": "Asaxi.uz", "Mijozlar (mln)": 1.5, "Bozor ulushi (%)": 18, "Segment": "Electronics"},
    {"Platforma": "Olcha.uz", "Mijozlar (mln)": 1.2, "Bozor ulushi (%)": 15, "Segment": "Fashion"},
    {"Platforma": "Texnomart", "Mijozlar (mln)": 0.8, "Bozor ulushi (%)": 10, "Segment": "Electronics"},
    {"Platforma": "Mediapark", "Mijozlar (mln)": 0.6, "Bozor ulushi (%)": 8, "Segment": "Electronics"},
    {"Platforma": "Boshqalar", "Mijozlar (mln)": 1.0, "Bozor ulushi (%)": 14, "Segment": "Various"},
])

pc1, pc2 = st.columns([1, 1])

with pc1:
    st.dataframe(platforms_data, use_container_width=True, hide_index=True)

with pc2:
    fig_share = px.pie(
        platforms_data,
        names="Platforma", values="Bozor ulushi (%)",
        title="🥧 Bozor Ulushi Taqsimoti",
        color_discrete_sequence=["#00d4aa", "#6366f1", "#ff6b6b", "#ffa500", "#22c55e", "#94a3b8"],
        hole=0.45,
    )
    fig_share.update_layout(height=350, **PLOTLY_TEMPLATE["layout"])
    st.plotly_chart(fig_share, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# SECTION 2: MARL afzalliklari va ROI
# ═══════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-title">2️⃣ MARL Joriy Etishning Afzalliklari</h2>',
            unsafe_allow_html=True)

advantages = [
    ("💰 Daromad", 37.5, "PPO algoritmi static baseline'dan +37.5% ko'p daromad keltiradi", COLORS["PPO"]),
    ("❤️ Sodiqlik", 42.0, "Mijoz sodiqlik indeksi 0.502'dan 0.713'ga oshdi", COLORS["secondary"]),
    ("🚪 Churn", 54.0, "Mijozlar yo'qotilishi 18.2%'dan 8.3%'ga tushdi", COLORS["accent"]),
    ("⚡ Reaksiya tezligi", 85.0, "Bozor o'zgarishlariga real-time javob beradi", COLORS["DQN"]),
    ("🎯 Aniqlik", 92.0, "Talabni 92% aniqlik bilan bashorat qiladi", COLORS["SAC"]),
]

for label, value, desc, color in advantages:
    bar_filled = int(value / 10)
    bar = "█" * bar_filled + "░" * (10 - bar_filled)
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(26, 32, 53, 0.9) 0%, rgba(14, 17, 23, 0.9) 100%);
        border: 1px solid {color}33;
        border-left: 4px solid {color};
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 1.5rem;
    ">
        <div style="flex-shrink: 0; width: 180px;">
            <div style="color: #e2e8f0; font-weight: 700; font-size: 1.05rem;">{label}</div>
            <div style="color: {color}; font-size: 1.5rem; font-weight: 800;">+{value}%</div>
        </div>
        <div style="flex-grow: 1;">
            <div style="font-family: 'JetBrains Mono', monospace; color: {color}; font-size: 1.25rem; letter-spacing: 0.1em;">{bar}</div>
            <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.25rem;">{desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ROI hisob-kitobi
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("**💼 ROI Hisob-Kitobi (o'rtacha o'lchamdagi platforma uchun):**")

roi_cols = st.columns(4)
roi_data = [
    ("💵 Boshlang'ich xarajat", "$80-120K", "Infratuzilma + jamoa"),
    ("📊 Yillik daromad oshishi", "$400-500K", "+37.5% revenue lift"),
    ("⏱️ Qaytarish davri", "6-8 oy", "ROI = 350-420%"),
    ("📈 5-yillik NPV", "$1.8M-2.2M", "10% discount rate"),
]
for col, (label, value, desc) in zip(roi_cols, roi_data):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size: 1.4rem;">{value}</div>
            <div style="color: #94a3b8; font-size: 0.8rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SECTION 3: 4 bosqichli Gantt
# ═══════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-title">3️⃣ Bosqichma-bosqich Yo\'l Xaritasi</h2>',
            unsafe_allow_html=True)

# Gantt chart
start = datetime(2026, 6, 1)
phases = [
    {
        "Bosqich": "1. Ma'lumotlar va Infratuzilma",
        "Boshlanish": start,
        "Tugash": start + timedelta(days=60),
        "Rang": "#ef4444",
        "Tavsif": "Data pipeline, ETL, sklab",
        "Byudjet": "$20K",
    },
    {
        "Bosqich": "2. MVP / Pilot Trening",
        "Boshlanish": start + timedelta(days=60),
        "Tugash": start + timedelta(days=150),
        "Rang": "#f59e0b",
        "Tavsif": "PPO trening, env tuning",
        "Byudjet": "$30K",
    },
    {
        "Bosqich": "3. A/B Test",
        "Boshlanish": start + timedelta(days=150),
        "Tugash": start + timedelta(days=210),
        "Rang": "#6366f1",
        "Tavsif": "10% traffic, 2 oy",
        "Byudjet": "$15K",
    },
    {
        "Bosqich": "4. To'liq Joriy Etish",
        "Boshlanish": start + timedelta(days=210),
        "Tugash": start + timedelta(days=330),
        "Rang": "#22c55e",
        "Tavsif": "Production, monitoring",
        "Byudjet": "$25K",
    },
]

gantt_df = pd.DataFrame([
    {"Task": p["Bosqich"], "Start": p["Boshlanish"], "Finish": p["Tugash"],
     "Color": p["Rang"], "Description": p["Tavsif"], "Budget": p["Byudjet"]}
    for p in phases
])

fig_gantt = px.timeline(
    gantt_df,
    x_start="Start", x_end="Finish", y="Task",
    color="Task",
    color_discrete_map={p["Bosqich"]: p["Rang"] for p in phases},
    hover_data=["Description", "Budget"],
    title="📅 4-Bosqichli Joriy Etish Yo'l Xaritasi (11 oy)",
)
fig_gantt.update_yaxes(autorange="reversed")
fig_gantt.update_layout(height=400, showlegend=False, **PLOTLY_TEMPLATE["layout"])
st.plotly_chart(fig_gantt, use_container_width=True)

# Bosqich kartochkalari
st.markdown("**📋 Bosqichlar batafsil:**")
phase_cols = st.columns(4)

phase_details = [
    ("🏗️ Bosqich 1: Ma'lumotlar", "#ef4444", "1-2 oy", "$20K",
     ["Tranzaksiya pipeline", "RFM segmentatsiya", "Data warehouse", "Monitoring stack"],
     "Data Engineer + ML Engineer"),
    ("🤖 Bosqich 2: MVP/Pilot", "#f59e0b", "2-3 oy", "$30K",
     ["PPO trening (n=10K)", "Env hyperparameter tuning", "Reward funksiya kalibratsiyasi",
      "Simulyatsiya validatsiyasi"],
     "ML Engineer + Data Scientist"),
    ("📊 Bosqich 3: A/B Test", "#6366f1", "1-2 oy", "$15K",
     ["Traffic split (90/10)", "Daromad va loyalty kuzatuv", "Statistik tahlil",
      "Risk monitoring"],
     "Data Scientist + Product Manager"),
    ("🚀 Bosqich 4: Production", "#22c55e", "3-4 oy", "$25K",
     ["Real-time inference", "Auto-retrain pipeline", "Dashboard va alerting",
      "GDPR compliance"],
     "ML Engineer + DevOps + Legal"),
]

for col, (title, color, dur, budget, tasks, team) in zip(phase_cols, phase_details):
    with col:
        tasks_html = "".join([f"<li style='margin: 0.25rem 0;'>{t}</li>" for t in tasks])
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(26, 32, 53, 0.95) 0%, rgba(14, 17, 23, 0.95) 100%);
            border: 1px solid {color}40;
            border-top: 4px solid {color};
            border-radius: 12px;
            padding: 1.25rem;
            min-height: 320px;
        ">
            <h4 style="color: {color}; margin-top: 0;">{title}</h4>
            <div style="display: flex; gap: 0.5rem; margin-bottom: 0.75rem;">
                <span class="badge badge-info">{dur}</span>
                <span class="badge badge-warning">{budget}</span>
            </div>
            <strong style="color: #e2e8f0; font-size: 0.85rem;">Vazifalar:</strong>
            <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 1.25rem; margin: 0.5rem 0;">
                {tasks_html}
            </ul>
            <strong style="color: #e2e8f0; font-size: 0.85rem;">Jamoa:</strong>
            <div style="color: #94a3b8; font-size: 0.85rem;">{team}</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SECTION 4: Algoritm Tavsiya Calculator
# ═══════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-title">4️⃣ Algoritm Tavsiya Kalkulyatori</h2>',
            unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    Sizning biznesingiz uchun eng mos algoritmni topish uchun parametrlarni kiriting:
</div>
""", unsafe_allow_html=True)

cc1, cc2, cc3 = st.columns(3)

with cc1:
    sector = st.selectbox(
        "🏪 E-commerce sektori",
        ["Marketplace", "Electronics", "Fashion", "Grocery", "Books/Media"],
    )

with cc2:
    daily_tx = st.slider(
        "📦 Kunlik tranzaksiyalar",
        min_value=100, max_value=100_000, value=10_000, step=500,
    )

with cc3:
    revenue_priority = st.slider(
        "⚖️ Daromad vs Sodiqlik prioriteti",
        min_value=0, max_value=100, value=70,
        help="0 = faqat sodiqlik, 100 = faqat daromad",
    )

# Tavsiya logikasi
def recommend_algorithm(sector: str, tx: int, priority: int) -> dict:
    """Parametrlar asosida algoritm tavsiyasi."""
    if priority >= 60 and tx >= 5000:
        return {
            "algo": "PPO",
            "reason": "Yuqori daromad + barqaror sodiqlik balansi. Kompleks aktsiya makonlari uchun ideal.",
            "expected_revenue": "+37.5%",
            "expected_loyalty": "+42%",
            "training_time": "~7 min",
            "color": COLORS["PPO"],
        }
    elif priority < 40:
        return {
            "algo": "SAC",
            "reason": "Stabillik va explorationga e'tibor. Long-term sodiqlik uchun yaxshi.",
            "expected_revenue": "+28.4%",
            "expected_loyalty": "+30%",
            "training_time": "~9 min",
            "color": COLORS["SAC"],
        }
    elif tx < 1000:
        return {
            "algo": "DQN",
            "reason": "Kichik datasetlar uchun samarali. Diskret action space ga juda mos.",
            "expected_revenue": "+24.4%",
            "expected_loyalty": "+24%",
            "training_time": "~6 min",
            "color": COLORS["DQN"],
        }
    else:
        return {
            "algo": "PPO",
            "reason": "Universal tanlov: stabil va yuqori performance.",
            "expected_revenue": "+37.5%",
            "expected_loyalty": "+42%",
            "training_time": "~7 min",
            "color": COLORS["PPO"],
        }


recommendation = recommend_algorithm(sector, daily_tx, revenue_priority)

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, {recommendation['color']}20, {recommendation['color']}10);
    border: 2px solid {recommendation['color']};
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    text-align: center;
">
    <div style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em;">
        Tavsiya etilgan algoritm
    </div>
    <h1 style="
        background: linear-gradient(135deg, {recommendation['color']}, #00d4aa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4rem;
        margin: 0.5rem 0;
    ">{recommendation['algo']}</h1>
    <p style="color: #e2e8f0; font-size: 1.1rem; margin-bottom: 1.5rem;">
        {recommendation['reason']}
    </p>
    <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
        <div>
            <div style="color: #94a3b8; font-size: 0.8rem;">DAROMAD</div>
            <div style="color: #22c55e; font-size: 1.5rem; font-weight: 700;">{recommendation['expected_revenue']}</div>
        </div>
        <div>
            <div style="color: #94a3b8; font-size: 0.8rem;">SODIQLIK</div>
            <div style="color: #22c55e; font-size: 1.5rem; font-weight: 700;">{recommendation['expected_loyalty']}</div>
        </div>
        <div>
            <div style="color: #94a3b8; font-size: 0.8rem;">TRENING</div>
            <div style="color: #00d4aa; font-size: 1.5rem; font-weight: 700;">{recommendation['training_time']}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SECTION 5: Etik mulohazalar
# ═══════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-title">5️⃣ Etik Mulohazalar</h2>', unsafe_allow_html=True)

et1, et2, et3 = st.tabs(["🎭 Narx kamsitishi", "🔒 GDPR / Maxfiylik", "⚖️ Algoritmik adolatsizlik"])

with et1:
    st.markdown("""
    ### 🎭 Narx Kamsitishi (Price Discrimination)

    **Risklar:**
    - Bir xil mahsulot turli mijozlarga turli narxda taqdim etilishi
    - Mijozning shahri, qurilmasi yoki tarixiga ko'ra narxni o'zgartirish
    - "Personalizatsiyalashgan narx" = ba'zan diskriminatsion narx

    **Tavsiyalar:**
    - ✅ Faqat **vaqtli** narx o'zgarishi (peak vs off-peak), mijozga emas
    - ✅ Shaffof "promo" va "discount" tushuntirishi
    - ✅ Sensitiv ma'lumotlardan foydalanmaslik (yosh, jins, joylashuv)
    - ❌ Individual mijoz profili asosida narx

    **O'zbekiston qonunchiligi:**
    > "Iste'molchilar huquqlarini himoya qilish to'g'risida"gi qonunning 7-moddasi —
    > narx kamsitishi taqiqlanadi.
    """)

with et2:
    st.markdown("""
    ### 🔒 Ma'lumot Maxfiyligi (Privacy)

    **MARL tizimi qanday ma'lumot to'playdi:**
    - Mijoz xarid tarixi (yashirilgan ID bilan)
    - Sessiya davomiyligi va xulq-atvor
    - Geolokatsiya (faqat shahar darajasida)

    **Compliance talablar:**
    - ✅ **O'zbekiston "Shaxsiy ma'lumotlar" qonuni** (2019)
    - ✅ GDPR (Yevropa mijozlari uchun)
    - ✅ Pseudonimizatsiya (hash bilan Customer ID)
    - ✅ "Right to be forgotten" — mijoz so'rovida o'chirish
    - ✅ Opt-in consent for personalization

    **Texnik yechimlar:**
    - Differential privacy (noise inject)
    - Federated learning (model markazda emas, qurilmada o'qitiladi)
    - Encryption at rest + in transit
    """)

with et3:
    st.markdown("""
    ### ⚖️ Algoritmik Adolatsizlik

    **Mumkin bo'lgan muammolar:**
    - Tarixiy ma'lumotlardagi bias modelda davom etadi
    - "Filter bubble" — boy mijozga premium narx, kambag'al mijozga arzon
    - Model "ko'r" — explainability past

    **Yechimlar:**
    - ✅ **Fairness metrics** doimiy monitoring (demographic parity)
    - ✅ **Explainable AI** — SHAP qiymatlari har qarorga
    - ✅ Inson nazorati (Human-in-the-Loop) muhim qarorlar uchun
    - ✅ Mustaqil audit har 6 oyda
    - ✅ Ochiq xulosalar — barcha algoritm transparently dokumentlangan

    **O'zbekiston konteksti:**
    - "Sun'iy intellekt rivoji" strategiyasi (2030 yilgacha)
    - Algoritm shaffofligi davlat darajasida talab qilinmoqda
    """)


# ═══════════════════════════════════════════════════════════════
# YAKUNIY XULOSA
# ═══════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(0, 212, 170, 0.15), rgba(99, 102, 241, 0.15));
    border: 1px solid rgba(0, 212, 170, 0.4);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 2rem;
    text-align: center;
">
    <h2 style="color: #00d4aa; margin: 0 0 1rem 0;">🎯 Asosiy Xulosa</h2>
    <p style="color: #e2e8f0; font-size: 1.1rem; line-height: 1.7;">
        Multi-Agent Reinforcement Learning — O'zbekiston e-commerce sektori uchun
        <strong>real va o'lchanadigan</strong> qiymat yaratuvchi yechim.
        <strong style="color: #22c55e;">11 oylik joriy etish</strong> va
        <strong style="color: #22c55e;">$80-120K investitsiya</strong> bilan
        <strong style="color: #22c55e;">350-420% ROI</strong> ga erishish mumkin.
    </p>
</div>
""", unsafe_allow_html=True)
