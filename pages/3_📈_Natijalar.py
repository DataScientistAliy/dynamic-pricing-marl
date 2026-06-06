"""
Natijalar sahifasi — algoritmlar taqqoslash.

Tabs:
    1. Asosiy metrikalar (BMI 3.3-jadval)
    2. Boxplot tahlili
    3. Radar chart
    4. Pareto tahlili va Hyper-Volume
    5. Learning curves
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.evaluation import pareto_analysis
from src.utils import (
    create_custom_css,
    get_color_palette,
    get_plotly_template,
    load_precomputed_results,
    ensure_data_files_exist,
    show_progress_sidebar,
)


# ═══════════════════════════════════════════════════════════════
# Sahifa konfiguratsiyasi
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Natijalar | MARL Pricing",
    page_icon="📈",
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
    show_progress_sidebar(current_step=3)


@st.cache_data(show_spinner=False)
def _load_results():
    return load_precomputed_results()


results = _load_results()
exp = results["experiments"]
ALGOS = ["PPO", "DQN", "SAC", "Static"]


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown('<h1 class="hero-title" style="font-size: 2.5rem;">📈 Natijalar va Taqqoslash</h1>',
            unsafe_allow_html=True)
st.markdown(
    "<p class='hero-subtitle'>4 ta algoritm (PPO, DQN, SAC, Static) bo'yicha 30 ta mustaqil "
    "experimentdan to'plangan natijalar.</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# Sahifa boshida hero KPI
hc1, hc2, hc3, hc4 = st.columns(4)
ppo_rev = float(np.mean(exp["PPO"]["revenues"]))
static_rev = float(np.mean(exp["Static"]["revenues"]))
ppo_loyalty = float(np.mean(exp["PPO"]["loyalties"]))
static_loyalty = float(np.mean(exp["Static"]["loyalties"]))
ppo_churn = float(np.mean(exp["PPO"]["churns"]))
static_churn = float(np.mean(exp["Static"]["churns"]))

with hc1:
    st.markdown(f"""
    <div class="metric-card" style="border-left: 3px solid {COLORS['PPO']};">
        <div class="metric-label">🏆 G'olib Algoritm</div>
        <div class="metric-value" style="color: {COLORS['PPO']};">PPO</div>
        <div class="metric-delta-up">£{ppo_rev:,.0f} o'rtacha</div>
    </div>
    """, unsafe_allow_html=True)

with hc2:
    rev_improv = (ppo_rev - static_rev) / static_rev * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">💰 Daromad ↑</div>
        <div class="metric-value">+{rev_improv:.1f}%</div>
        <div class="metric-delta-up">PPO vs Static</div>
    </div>
    """, unsafe_allow_html=True)

with hc3:
    loy_improv = (ppo_loyalty - static_loyalty) / static_loyalty * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">❤️ Sodiqlik ↑</div>
        <div class="metric-value">+{loy_improv:.1f}%</div>
        <div class="metric-delta-up">PPO vs Static</div>
    </div>
    """, unsafe_allow_html=True)

with hc4:
    churn_improv = (static_churn - ppo_churn) / static_churn * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🚪 Churn ↓</div>
        <div class="metric-value">-{churn_improv:.1f}%</div>
        <div class="metric-delta-up">PPO vs Static</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Asosiy Metrikalar",
    "📦 Boxplot Tahlili",
    "🕸️ Radar Chart",
    "🎯 Pareto Tahlili",
    "📈 Learning Curves",
])


# ═══════════════════════════════════════════════════════════════
# Tab 1: Asosiy metrikalar
# ═══════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<h3 style="color: #00d4aa;">📋 BMI 3.3-jadval: Algoritm Taqqoslash (n=30)</h3>',
                unsafe_allow_html=True)

    # Hero jadval
    table_data = []
    for algo in ALGOS:
        d = exp[algo]
        table_data.append({
            "Algoritm": algo,
            "💰 Daromad (£)": f"{np.mean(d['revenues']):,.0f} ± {np.std(d['revenues']):,.0f}",
            "❤️ Sodiqlik": f"{np.mean(d['loyalties']):.3f} ± {np.std(d['loyalties']):.3f}",
            "🚪 Churn": f"{np.mean(d['churns']):.3f} ± {np.std(d['churns']):.3f}",
            "📈 Volatillik": f"{np.mean(d['price_volatilities']):.2f} ± {np.std(d['price_volatilities']):.2f}",
            "🎯 Hyper-Volume": f"{d['hypervolume']:.3f}",
        })
    metrics_df = pd.DataFrame(table_data)

    # Styled table
    def highlight_winner(s):
        is_ppo = s.values == "PPO"
        return ["background: linear-gradient(90deg, rgba(0,212,170,0.2), transparent); font-weight: 700;"
                if v else "" for v in is_ppo]

    styled = metrics_df.style.apply(highlight_winner, subset=["Algoritm"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Bar charts (2x2)
    st.markdown("<br>", unsafe_allow_html=True)
    fig_bars = make_subplots(
        rows=2, cols=2,
        subplot_titles=("💰 Daromad", "❤️ Sodiqlik", "🚪 Churn (kichik yaxshi)", "📈 Volatillik"),
        vertical_spacing=0.18, horizontal_spacing=0.12,
    )

    for col_idx, (metric_key, title, row, col) in enumerate([
        ("revenues", "Revenue", 1, 1),
        ("loyalties", "Loyalty", 1, 2),
        ("churns", "Churn", 2, 1),
        ("price_volatilities", "Volatility", 2, 2),
    ]):
        means = [np.mean(exp[a][metric_key]) for a in ALGOS]
        stds = [np.std(exp[a][metric_key]) for a in ALGOS]
        colors_list = [COLORS[a] for a in ALGOS]

        fig_bars.add_trace(
            go.Bar(
                x=ALGOS, y=means,
                error_y=dict(type="data", array=stds, color="white", thickness=1.5),
                marker=dict(color=colors_list, line=dict(color="white", width=1)),
                text=[f"{v:.2f}" if v < 1000 else f"{v:,.0f}" for v in means],
                textposition="outside",
                name=title,
                showlegend=False,
            ),
            row=row, col=col,
        )

    fig_bars.update_layout(height=700, **PLOTLY_TEMPLATE["layout"])
    st.plotly_chart(fig_bars, use_container_width=True)

    st.markdown("""
    <div class="info-box">
        <strong>📝 Xulosa:</strong> PPO algoritmi 4 metrikadan 3 tasida (daromad, sodiqliq, churn)
        eng yaxshi natijani ko'rsatadi. Volatillik bo'yicha Static eng past, lekin uning daromadi
        eng yomon. PPO — biznes uchun optimal tanlov.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Tab 2: Boxplot
# ═══════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<h3 style="color: #00d4aa;">📦 Boxplot: Taqsimotlar Tahlili</h3>',
                unsafe_allow_html=True)

    fig_box = make_subplots(
        rows=2, cols=2,
        subplot_titles=("💰 Daromad", "❤️ Sodiqlik", "🚪 Churn Rate", "📈 Volatillik"),
        vertical_spacing=0.15, horizontal_spacing=0.12,
    )

    for col_idx, (metric_key, row, col) in enumerate([
        ("revenues", 1, 1), ("loyalties", 1, 2),
        ("churns", 2, 1), ("price_volatilities", 2, 2),
    ]):
        for algo in ALGOS:
            fig_box.add_trace(
                go.Box(
                    y=exp[algo][metric_key],
                    name=algo,
                    marker=dict(color=COLORS[algo]),
                    boxmean="sd",
                    showlegend=(col_idx == 0),
                ),
                row=row, col=col,
            )

    fig_box.update_layout(height=750, **PLOTLY_TEMPLATE["layout"])
    st.plotly_chart(fig_box, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Tab 3: Radar chart
# ═══════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<h3 style="color: #00d4aa;">🕸️ Radar Chart: Ko\'p o\'lchovli Taqqoslash</h3>',
                unsafe_allow_html=True)

    # 5 o'lchov uchun normalize qilingan qiymatlar
    categories = ["Daromad", "Sodiqlik", "Past Churn", "Past Volatillik", "Konvergensiya"]

    # Har metrika uchun maks. qiymat
    max_rev = max(np.mean(exp[a]["revenues"]) for a in ALGOS)
    max_loy = max(np.mean(exp[a]["loyalties"]) for a in ALGOS)
    max_inv_churn = max(1 - np.mean(exp[a]["churns"]) for a in ALGOS)
    max_inv_vol = max(1 / max(np.mean(exp[a]["price_volatilities"]), 0.1) for a in ALGOS)

    fig_radar = go.Figure()
    for algo in ALGOS:
        d = exp[algo]
        values = [
            np.mean(d["revenues"]) / max_rev,
            np.mean(d["loyalties"]) / max_loy,
            (1 - np.mean(d["churns"])) / max_inv_churn,
            (1 / max(np.mean(d["price_volatilities"]), 0.1)) / max_inv_vol,
            d["hypervolume"] / max(exp[a]["hypervolume"] for a in ALGOS),
        ]
        values_closed = values + [values[0]]
        cats_closed = categories + [categories[0]]

        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=cats_closed,
            fill="toself",
            fillcolor=f"rgba({int(COLORS[algo][1:3], 16)}, "
                      f"{int(COLORS[algo][3:5], 16)}, "
                      f"{int(COLORS[algo][5:7], 16)}, 0.2)",
            line=dict(color=COLORS[algo], width=2),
            marker=dict(size=8, color=COLORS[algo]),
            name=algo,
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.1], showticklabels=False,
                           gridcolor="rgba(148, 163, 184, 0.2)"),
            angularaxis=dict(gridcolor="rgba(148, 163, 184, 0.2)",
                            tickfont=dict(size=12, color="#e2e8f0")),
            bgcolor="rgba(26, 32, 53, 0.4)",
        ),
        height=600,
        title="🕸️ 5 o'lchovli Performance Profile (normallashtirilgan)",
        showlegend=True,
        **{k: v for k, v in PLOTLY_TEMPLATE["layout"].items() if k != "xaxis" and k != "yaxis"},
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("""
    <div class="info-box">
        Radar chart har algoritmning <strong>kuchli</strong> va <strong>zaif</strong> tomonlarini
        vizual ko'rsatadi. PPO eng kengaytirilgan profilga ega — barcha o'lchovlar bo'yicha
        muvozanatli ishlaydi.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Tab 4: Pareto tahlili
# ═══════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<h3 style="color: #00d4aa;">🎯 Pareto Tahlili va Hyper-Volume</h3>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>Pareto frontasi</strong> — boshqa algoritm tomonidan ustun bo'lmagan optimal nuqtalar.
        <strong>Hyper-Volume (HV)</strong> — frontaning sifati o'lchovi (0 dan 1 gacha, yuqori — yaxshi).
    </div>
    """, unsafe_allow_html=True)

    revenue_data = {a: exp[a]["revenues"] for a in ALGOS}
    loyalty_data = {a: exp[a]["loyalties"] for a in ALGOS}
    pareto_res = pareto_analysis(revenue_data, loyalty_data)

    pc1, pc2 = st.columns([2, 1])

    with pc1:
        fig_pareto = go.Figure()
        # Barcha runlar nuqtalari
        for algo in ALGOS:
            fig_pareto.add_trace(go.Scatter(
                x=exp[algo]["revenues"],
                y=exp[algo]["loyalties"],
                mode="markers",
                marker=dict(size=8, color=COLORS[algo], opacity=0.5,
                           line=dict(color="white", width=1)),
                name=f"{algo} (runs)",
                hovertemplate=f"<b>{algo}</b><br>Revenue: %{{x:,.0f}}<br>Loyalty: %{{y:.3f}}",
            ))

        # O'rtacha nuqtalar — katta belgilar
        for point in pareto_res["points"]:
            fig_pareto.add_trace(go.Scatter(
                x=[point["revenue"]],
                y=[point["loyalty"]],
                mode="markers+text",
                marker=dict(size=22, color=COLORS[point["algorithm"]], symbol="diamond",
                           line=dict(color="white", width=2)),
                text=[point["algorithm"]],
                textposition="top center",
                textfont=dict(size=14, color="white"),
                name=f"{point['algorithm']} (mean)",
                showlegend=False,
                hovertemplate=f"<b>{point['algorithm']}</b><br>Mean Revenue: {point['revenue']:,.0f}"
                              f"<br>Mean Loyalty: {point['loyalty']:.3f}",
            ))

        # Pareto front chizig'i
        if len(pareto_res["pareto_front"]) > 1:
            sorted_front = sorted(pareto_res["pareto_front"], key=lambda x: x["revenue"])
            fig_pareto.add_trace(go.Scatter(
                x=[p["revenue"] for p in sorted_front],
                y=[p["loyalty"] for p in sorted_front],
                mode="lines",
                line=dict(color="rgba(0, 212, 170, 0.7)", width=3, dash="dash"),
                name="Pareto Frontasi",
            ))

        fig_pareto.update_layout(
            title="🎯 Revenue vs Loyalty — Pareto Frontasi",
            xaxis_title="O'rtacha Daromad (£)",
            yaxis_title="O'rtacha Sodiqliq",
            height=550,
            **PLOTLY_TEMPLATE["layout"],
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

    with pc2:
        st.markdown("**📊 BMI 3.4-jadval: HV qiymatlari**")
        hv_data = []
        for algo in ALGOS:
            is_pareto = any(p["algorithm"] == algo for p in pareto_res["pareto_front"])
            hv_data.append({
                "Algoritm": algo,
                "HV": f"{exp[algo]['hypervolume']:.3f}",
                "Pareto": "✅" if is_pareto else "❌",
            })
        st.dataframe(pd.DataFrame(hv_data), use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="info-box">
            <strong>Yetakchi:</strong> PPO (HV = 0.687)<br>
            Pareto frontasiga eng yaqin.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Tab 5: Learning curves
# ═══════════════════════════════════════════════════════════════

with tab5:
    st.markdown('<h3 style="color: #00d4aa;">📈 Learning Curves — Konvergentsiya Dinamikasi</h3>',
                unsafe_allow_html=True)

    smooth_window = st.slider("🎚️ Smoothing oyna kengligi", 1, 30, 10)

    fig_lc = make_subplots(
        rows=1, cols=2,
        subplot_titles=("📊 Raw Learning Curves", "📈 Smoothed (oyna)"),
        horizontal_spacing=0.1,
    )

    for algo in ALGOS:
        curve = np.array(exp[algo]["learning_curve"])

        # Raw
        fig_lc.add_trace(
            go.Scatter(
                x=list(range(len(curve))), y=curve,
                mode="lines", line=dict(color=COLORS[algo], width=1),
                opacity=0.5, name=algo, showlegend=False,
            ),
            row=1, col=1,
        )

        # Smoothed (moving average)
        if smooth_window > 1 and len(curve) > smooth_window:
            smoothed = pd.Series(curve).rolling(smooth_window, min_periods=1).mean()
        else:
            smoothed = curve
        fig_lc.add_trace(
            go.Scatter(
                x=list(range(len(smoothed))), y=smoothed,
                mode="lines", line=dict(color=COLORS[algo], width=2.5),
                name=algo,
            ),
            row=1, col=2,
        )

        # Konvergentsiya nuqtasi
        conv_ep = exp[algo]["convergence_episode"]
        if conv_ep > 0 and conv_ep < len(curve):
            fig_lc.add_trace(
                go.Scatter(
                    x=[conv_ep], y=[smoothed.iloc[conv_ep] if hasattr(smoothed, 'iloc') else smoothed[conv_ep]],
                    mode="markers",
                    marker=dict(size=14, color=COLORS[algo], symbol="star",
                              line=dict(color="white", width=2)),
                    showlegend=False,
                ),
                row=1, col=2,
            )

    fig_lc.update_xaxes(title_text="Epizod", row=1, col=1)
    fig_lc.update_xaxes(title_text="Epizod", row=1, col=2)
    fig_lc.update_yaxes(title_text="Reward", row=1, col=1)
    fig_lc.update_yaxes(title_text="Reward (smoothed)", row=1, col=2)
    fig_lc.update_layout(height=500, **PLOTLY_TEMPLATE["layout"])
    st.plotly_chart(fig_lc, use_container_width=True)

    # BMI 3.1-jadval — konvergensiya
    st.markdown("**📋 BMI 3.1-jadval: Konvergentsiya va Stabillik**")
    conv_data = []
    for algo in ALGOS:
        d = exp[algo]
        curve = np.array(d["learning_curve"])
        stab = float(np.std(curve[-50:])) if len(curve) > 50 else 0.0
        conv_data.append({
            "Algoritm": algo,
            "Konvergensiya epizodi": d["convergence_episode"] if d["convergence_episode"] > 0 else "—",
            "Yakuniy reward": f"{d['final_reward']:.2f}",
            "Stabillik (so'nggi 50 ep. std)": f"{stab:.2f}",
            "Trening vaqti (s)": d["training_time_sec"],
        })
    st.dataframe(pd.DataFrame(conv_data), use_container_width=True, hide_index=True)
