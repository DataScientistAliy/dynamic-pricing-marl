"""
RL Simulatsiya sahifasi.

Interaktiv MARL simulyatsiya: foydalanuvchi algoritm, epizodlar, parametrlar tanlaydi.
Demo rejimi precomputed natijalarni darhol ko'rsatadi.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.utils import (
    create_custom_css,
    get_color_palette,
    get_plotly_template,
    load_precomputed_results,
    ensure_data_files_exist,
    format_currency,
    show_progress_sidebar,
)


# ═══════════════════════════════════════════════════════════════
# Sahifa konfiguratsiyasi
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RL Simulatsiya | MARL Pricing",
    page_icon="🤖",
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
    show_progress_sidebar(current_step=2)


@st.cache_data(show_spinner=False)
def _load_results():
    """Precomputed natijalarni cache bilan yuklash."""
    return load_precomputed_results()


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown('<h1 class="hero-title" style="font-size: 2.5rem;">🤖 RL Simulatsiya</h1>',
            unsafe_allow_html=True)
st.markdown(
    "<p class='hero-subtitle'>Real-time interaktiv simulyatsiya: algoritmlarni tanlang, "
    "parametrlarni sozlang va natijalarni jonli kuzating.</p>",
    unsafe_allow_html=True,
)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════
# SIDEBAR — Sozlamalar
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Simulyatsiya Sozlamalari")

    algorithm = st.selectbox(
        "🧠 Algoritm",
        options=["PPO", "DQN", "SAC", "Static"],
        index=0,
        help="Reinforcement Learning algoritmi",
    )

    n_episodes = st.slider(
        "📊 Epizodlar soni",
        min_value=10, max_value=200, value=50, step=10,
    )

    initial_price = st.slider(
        "💰 Boshlang'ich narx (£)",
        min_value=5.0, max_value=30.0, value=10.0, step=0.5,
    )

    st.markdown("---")
    st.markdown("**⚖️ Reward funksiyasi:**")

    alpha = st.slider(
        "α (daromad og'irligi)",
        min_value=0.0, max_value=1.0, value=0.7, step=0.05,
        help="Daromadga e'tibor darajasi",
    )

    beta = 1.0 - alpha
    st.metric("β (sodiqliq og'irligi)", f"{beta:.2f}")

    st.markdown(f"""
    <div class="info-box" style="font-size: 0.85rem;">
        <code>R = {alpha:.2f}·rev + {beta:.2f}·loyalty·100</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    run_real = st.checkbox("⚡ Real trening (sekin)", value=False,
                          help="Real trening Stable-Baselines3 talab qiladi")

    if run_real:
        st.warning("⚠️ Real trening 30+ soniya davom etishi mumkin")

    btn_demo = st.button("▶️ Demo'ni boshlash", use_container_width=True, type="primary")
    btn_reset = st.button("🔄 Tozalash", use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Session state init
# ═══════════════════════════════════════════════════════════════

if btn_reset:
    for key in ["sim_running", "sim_data", "sim_step"]:
        st.session_state.pop(key, None)
    st.rerun()


# ═══════════════════════════════════════════════════════════════
# Simulyatsiya funksiyalari
# ═══════════════════════════════════════════════════════════════

def simulate_episode_realtime(
    algo: str, alpha: float, beta: float, initial_price: float, max_steps: int = 100
) -> dict:
    """
    Bir epizodni real-time simulyatsiya qilish (SingleAgentPricingEnv asosida).

    Demo uchun har algoritm o'z xulq-atvori bo'yicha ishlaydi.
    """
    rng = np.random.default_rng(hash(algo) % (2**32))
    price = initial_price
    loyalty = 0.5
    base_demand = 100.0
    elasticity = -1.5

    prices, demands, loyalties, rewards, revenues = [], [], [], [], []

    for step in range(max_steps):
        # Algoritmga ko'ra harakat tanlash
        if algo == "Static":
            action_delta = 0.0
        elif algo == "PPO":
            # PPO — smart, loyalty va revenue ni balanslaydi
            if loyalty > 0.7:
                action_delta = float(rng.choice([0.05, 0.10], p=[0.7, 0.3]))
            elif loyalty < 0.3:
                action_delta = float(rng.choice([-0.10, -0.05], p=[0.6, 0.4]))
            else:
                action_delta = float(rng.choice([-0.05, 0.0, 0.05], p=[0.3, 0.4, 0.3]))
        elif algo == "DQN":
            # DQN — value-based, koproq tebranadi
            action_delta = float(rng.choice([-0.15, -0.05, 0.0, 0.05, 0.15],
                                            p=[0.15, 0.2, 0.3, 0.2, 0.15]))
        else:  # SAC
            # SAC — actor-critic, o'rtacha xatti-harakat
            action_delta = float(rng.choice([-0.10, -0.05, 0.0, 0.05, 0.10],
                                            p=[0.15, 0.25, 0.2, 0.25, 0.15]))

        # Narx yangilanishi
        price = float(np.clip(price * (1 + action_delta), initial_price * 0.3, initial_price * 3))

        # Talab (elastiklik)
        demand = base_demand * (price / initial_price) ** elasticity * (0.5 + loyalty)
        demand = max(demand, 0.0)

        # Sodiqlik o'zgarishi
        prev_loyalty = loyalty
        loyalty_change = -action_delta * 0.5 + rng.normal(0, 0.01)
        loyalty = float(np.clip(loyalty + loyalty_change, 0.0, 1.0))

        # Reward
        revenue = price * demand
        revenue_norm = revenue / (initial_price * base_demand * 1.5)
        reward = alpha * revenue_norm + beta * (loyalty - prev_loyalty) * 100

        prices.append(price)
        demands.append(demand)
        loyalties.append(loyalty)
        rewards.append(reward)
        revenues.append(revenue)

    return {
        "prices": prices,
        "demands": demands,
        "loyalties": loyalties,
        "rewards": rewards,
        "revenues": revenues,
        "total_reward": float(np.sum(rewards)),
        "total_revenue": float(np.sum(revenues)),
        "mean_loyalty": float(np.mean(loyalties)),
        "churn_rate": float(np.mean(np.array(loyalties) < 0.1)),
        "price_volatility": float(np.std(prices)),
    }


# ═══════════════════════════════════════════════════════════════
# DEMO REJIMI
# ═══════════════════════════════════════════════════════════════

if btn_demo or "sim_data" in st.session_state:
    if btn_demo:
        with st.spinner(f"🚀 {algorithm} algoritmi ishga tushdi..."):
            # Real-time animatsiya simulyatsiyasi
            sim_data = simulate_episode_realtime(
                algorithm, alpha, beta, initial_price, max_steps=n_episodes
            )
            st.session_state["sim_data"] = sim_data
            st.session_state["sim_algo"] = algorithm
            st.session_state["sim_params"] = {
                "alpha": alpha, "beta": beta, "initial_price": initial_price,
                "n_episodes": n_episodes,
            }

    sim_data = st.session_state["sim_data"]
    algo_used = st.session_state["sim_algo"]

    # ═══════ Jonli metrikalar ═══════
    st.markdown('<h2 class="section-title">📊 Simulyatsiya Natijalari</h2>',
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💰 Jami Daromad</div>
            <div class="metric-value" style="font-size: 1.5rem;">£{sim_data['total_revenue']:,.0f}</div>
            <div class="metric-delta-up">{algo_used}</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">❤️ O'rt. Sodiqlik</div>
            <div class="metric-value">{sim_data['mean_loyalty']:.3f}</div>
            <div class="metric-delta-up">Indeks (0-1)</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🚪 Churn Rate</div>
            <div class="metric-value">{sim_data['churn_rate']*100:.1f}%</div>
            <div class="metric-delta-up">Past loyalty</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📈 Volatillik</div>
            <div class="metric-value">{sim_data['price_volatility']:.2f}</div>
            <div class="metric-delta-up">Narx std</div>
        </div>
        """, unsafe_allow_html=True)

    # ═══════ Grafiklar ═══════
    st.markdown("<br>", unsafe_allow_html=True)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "💰 Narx Dinamikasi", "❤️ Sodiqlik Indeksi",
            "🎯 Reward Trend", "📊 Talab (Demand)",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    x_steps = list(range(1, len(sim_data["prices"]) + 1))

    fig.add_trace(
        go.Scatter(x=x_steps, y=sim_data["prices"],
                   mode="lines+markers", line=dict(color=COLORS["PPO"], width=2),
                   marker=dict(size=4), name="Narx"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_steps, y=sim_data["loyalties"],
                   mode="lines+markers", line=dict(color=COLORS["accent"], width=2),
                   marker=dict(size=4), name="Sodiqlik"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=x_steps, y=sim_data["rewards"],
                   mode="lines", line=dict(color=COLORS["secondary"], width=2),
                   fill="tozeroy", fillcolor="rgba(0, 212, 170, 0.1)", name="Reward"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x_steps, y=sim_data["demands"],
                   mode="lines", line=dict(color=COLORS["SAC"], width=2),
                   fill="tozeroy", fillcolor="rgba(255, 165, 0, 0.1)", name="Talab"),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="Qadam", row=2, col=1)
    fig.update_xaxes(title_text="Qadam", row=2, col=2)
    fig.update_yaxes(title_text="Narx (£)", row=1, col=1)
    fig.update_yaxes(title_text="Sodiqlik", row=1, col=2)
    fig.update_yaxes(title_text="Reward", row=2, col=1)
    fig.update_yaxes(title_text="Talab", row=2, col=2)

    fig.update_layout(
        height=700,
        showlegend=False,
        **PLOTLY_TEMPLATE["layout"],
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Hisoblash effektivligi va konvergentsiya
# ═══════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-title">⚙️ Hisoblash Effektivligi</h2>', unsafe_allow_html=True)

results = _load_results()
exp = results["experiments"]

efficiency_data = []
for algo in ["PPO", "DQN", "SAC", "Static"]:
    d = exp[algo]
    efficiency_data.append({
        "Algoritm": algo,
        "Trening vaqti (s)": d["training_time_sec"],
        "RAM (MB)": d["memory_mb"],
        "Konvergentsiya epizodi": d["convergence_episode"] if d["convergence_episode"] > 0 else "N/A",
        "Yakuniy reward": round(d["final_reward"], 2),
        "Hyper-Volume": d["hypervolume"],
    })
eff_df = pd.DataFrame(efficiency_data)

ec1, ec2 = st.columns([1, 1])

with ec1:
    st.markdown("**📊 Algoritmlar bo'yicha effektivlik:**")
    st.dataframe(eff_df, use_container_width=True, hide_index=True)

with ec2:
    # Trening vaqt vs reward scatter
    fig_eff = go.Figure()
    for algo in ["PPO", "DQN", "SAC", "Static"]:
        d = exp[algo]
        fig_eff.add_trace(go.Scatter(
            x=[d["training_time_sec"]], y=[d["final_reward"]],
            mode="markers+text", text=[algo], textposition="top center",
            marker=dict(size=20 + d["memory_mb"] / 20, color=COLORS[algo],
                       line=dict(color="white", width=2)),
            name=algo,
        ))
    fig_eff.update_layout(
        title="⚡ Trening vaqt vs Yakuniy Reward (bubble = RAM)",
        xaxis_title="Trening vaqti (s)",
        yaxis_title="Yakuniy reward",
        height=400,
        showlegend=False,
        **PLOTLY_TEMPLATE["layout"],
    )
    st.plotly_chart(fig_eff, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Konvergentsiya tahlili — formula 3.1
# ═══════════════════════════════════════════════════════════════

st.markdown('<h2 class="section-title">📈 Konvergentsiya Tahlili</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>Konvergentsiya kriteriyasi (BMI Formula 3.1):</strong>
    <code>CV(20) = σ(R[t-19:t]) / |μ(R[t-19:t])| < 0.05</code>
    <br>
    Ya'ni so'nggi 20 epizodda reward o'zgarish koeffitsienti 5% dan past tushganda model konvergensiyaga keldi deb hisoblanadi.
</div>
""", unsafe_allow_html=True)

fig_conv = go.Figure()
for algo in ["PPO", "DQN", "SAC", "Static"]:
    curve = exp[algo]["learning_curve"]
    conv_ep = exp[algo]["convergence_episode"]

    fig_conv.add_trace(go.Scatter(
        x=list(range(len(curve))), y=curve,
        mode="lines", line=dict(color=COLORS[algo], width=2),
        name=algo, opacity=0.85,
    ))

    # Konvergentsiya nuqtasi belgisi
    if conv_ep > 0 and conv_ep < len(curve):
        fig_conv.add_trace(go.Scatter(
            x=[conv_ep], y=[curve[conv_ep]],
            mode="markers", marker=dict(size=15, color=COLORS[algo], symbol="star",
                                        line=dict(color="white", width=2)),
            name=f"{algo} konvergent: ep {conv_ep}",
            showlegend=False,
        ))

fig_conv.update_layout(
    title="📈 Learning Curve va Konvergentsiya Nuqtalari",
    xaxis_title="Epizod",
    yaxis_title="Reward (smoothed)",
    height=500,
    hovermode="x unified",
    **PLOTLY_TEMPLATE["layout"],
)
st.plotly_chart(fig_conv, use_container_width=True)
