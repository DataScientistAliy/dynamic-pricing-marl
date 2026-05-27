"""
EDA va Dataset sahifasi.

Tarkib:
    1. Dataset yuklash (file uploader yoki sample data)
    2. Ma'lumotlarni tozalash (5 bosqich, statistika bilan)
    3. RFM tahlili va segmentatsiya
    4. EDA vizualizatsiyalar (narxlar, trend, top mahsulotlar, korrelyatsiya)
    5. Narx elastikligini OLS regression yordamida hisoblash
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import DataLoader
from src.utils import (
    create_custom_css,
    get_color_palette,
    get_plotly_template,
    ensure_data_files_exist,
    format_currency,
)


# ═══════════════════════════════════════════════════════════════
# Sahifa konfiguratsiyasi
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EDA va Dataset | MARL Pricing",
    page_icon="📊",
    layout="wide",
)

ensure_data_files_exist()
st.markdown(create_custom_css(), unsafe_allow_html=True)

COLORS = get_color_palette()
PLOTLY_TEMPLATE = get_plotly_template()


@st.cache_data(show_spinner=False)
def _load_default_data() -> pd.DataFrame:
    """data/sample_data.csv ni yuklash (cache bilan)."""
    return pd.read_csv(PROJECT_ROOT / "data" / "sample_data.csv")


@st.cache_data(show_spinner=False)
def _clean_data_cached(df_bytes: bytes) -> tuple:
    """Tozalangan DataFrame va statistika qaytarish (bytes orqali cache)."""
    df = pd.read_csv(io.BytesIO(df_bytes))
    loader = DataLoader()
    df_clean = loader.clean_data(df)
    return df_clean, loader.cleaning_stats


@st.cache_data(show_spinner=False)
def _calc_rfm_cached(df_csv: str) -> pd.DataFrame:
    """RFM jadval (cache)."""
    df = pd.read_csv(io.StringIO(df_csv), parse_dates=["InvoiceDate"])
    loader = DataLoader()
    return loader.calculate_rfm(df)


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown('<h1 class="hero-title" style="font-size: 2.5rem;">📊 EDA va Dataset Tahlili</h1>',
            unsafe_allow_html=True)
st.markdown(
    "<p class='hero-subtitle'>Online Retail II datasetini chuqur tahlil qilish: tozalash, "
    "RFM segmentatsiyasi, narx elastikligi va vizual tushuncha.</p>",
    unsafe_allow_html=True,
)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════
# SECTION 1: Dataset yuklash
# ═══════════════════════════════════════════════════════════════

st.markdown('<h2 class="section-title">1️⃣ Dataset Yuklash</h2>', unsafe_allow_html=True)

c1, c2 = st.columns([2, 1])

with c1:
    uploaded_file = st.file_uploader(
        "📁 Faylni tanlang (.xlsx yoki .csv)",
        type=["xlsx", "csv"],
        help="Online Retail II formatidagi fayl",
    )

with c2:
    st.markdown("""
    <div class="info-box" style="margin-top: 0;">
        <strong>💡 Maslahat</strong><br>
        <small>Fayl yuklamasangiz, avtomatik <code>sample_data.csv</code> (1000 qator) ishlatiladi.</small>
    </div>
    """, unsafe_allow_html=True)

# Datasetni yuklash
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file, engine="openpyxl")
        st.success(f"✅ {uploaded_file.name} muvaffaqiyatli yuklandi: {len(df_raw):,} qator")
    except Exception as e:
        st.error(f"❌ Yuklash xatosi: {e}. Sample data ishlatiladi.")
        df_raw = _load_default_data()
else:
    df_raw = _load_default_data()
    st.info("ℹ️ Sample dataset yuklandi (1000 qator). O'zingizning faylingizni yuklang.")

# Ustun nomlarini standartlashtirish
loader = DataLoader()
df_raw = loader._standardize_columns(df_raw)

st.session_state["df_raw"] = df_raw

# Dataset haqida ekspander
with st.expander("📖 Dataset haqida batafsil"):
    st.markdown("""
    **Online Retail II Dataset (UCI Machine Learning Repository)**

    - **Tavsif:** Buyuk Britaniyadagi onlayn rozetka tranzaksiyalari
    - **Davri:** 2009-yil 12 — 2011-yil 12 (2 yil)
    - **Hajmi:** ~1,067,371 tranzaksiya, ~5,942 unique mijoz
    - **Ustunlar:**
        - `Invoice` — buyurtma raqami
        - `StockCode` — mahsulot kodi
        - `Description` — mahsulot nomi
        - `Quantity` — miqdor (qaytarish manfiy)
        - `InvoiceDate` — sana va vaqt
        - `Price` — birlik narxi (£)
        - `Customer ID` — mijoz identifikatori
        - `Country` — davlat
    - **Manba:** https://archive.ics.uci.edu/dataset/502/online+retail+ii
    """)

# Asosiy statistika kartochkalari
sc1, sc2, sc3, sc4 = st.columns(4)

with sc1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📦 Tranzaksiyalar</div>
        <div class="metric-value" style="font-size: 1.75rem;">{len(df_raw):,}</div>
        <div class="metric-delta-up">Yuklangan</div>
    </div>
    """, unsafe_allow_html=True)

with sc2:
    unique_customers = df_raw["Customer ID"].nunique() if "Customer ID" in df_raw.columns else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">👥 Mijozlar</div>
        <div class="metric-value" style="font-size: 1.75rem;">{unique_customers:,}</div>
        <div class="metric-delta-up">Unique</div>
    </div>
    """, unsafe_allow_html=True)

with sc3:
    unique_products = df_raw["StockCode"].nunique() if "StockCode" in df_raw.columns else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🏷️ Mahsulotlar</div>
        <div class="metric-value" style="font-size: 1.75rem;">{unique_products:,}</div>
        <div class="metric-delta-up">Turlari</div>
    </div>
    """, unsafe_allow_html=True)

with sc4:
    unique_countries = df_raw["Country"].nunique() if "Country" in df_raw.columns else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🌍 Davlatlar</div>
        <div class="metric-value" style="font-size: 1.75rem;">{unique_countries}</div>
        <div class="metric-delta-up">Bozor</div>
    </div>
    """, unsafe_allow_html=True)

# Birinchi 10 qatorni ko'rsatish
with st.expander("👀 Dastlabki 10 qator (raw data)"):
    st.dataframe(df_raw.head(10), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# SECTION 2: Ma'lumotlarni tozalash
# ═══════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h2 class="section-title">2️⃣ Ma\'lumotlarni Tozalash</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    Tozalash quyidagi 5 bosqichdan iborat: missing values → manfiy miqdor →
    nol/manfiy narx → dublikatlar → IQR outlier filter.
</div>
""", unsafe_allow_html=True)

if st.button("🧹 Tozalashni boshlash", type="primary"):
    progress = st.progress(0, text="Tozalash boshlandi...")
    csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
    df_clean, stats = _clean_data_cached(csv_bytes)
    progress.progress(100, text="✅ Tozalash yakunlandi!")
    st.session_state["df_clean"] = df_clean
    st.session_state["cleaning_stats"] = stats

# Avval tozalangan bo'lsa — ko'rsatish
if "df_clean" in st.session_state:
    df_clean = st.session_state["df_clean"]
    stats = st.session_state["cleaning_stats"]

    # Bosqichlar jadvali
    steps_data = [
        ("Boshlang'ich", stats.get("initial", 0), 0),
        ("Customer ID bo'sh", stats.get("after_dropna_customer", 0),
         stats.get("removed_no_customer", 0)),
        ("Manfiy miqdor", stats.get("after_positive_quantity", 0),
         stats.get("removed_negative_qty", 0)),
        ("Manfiy narx", stats.get("after_positive_price", 0),
         stats.get("removed_non_positive_price", 0)),
        ("Dublikatlar", stats.get("after_dedup", 0),
         stats.get("removed_duplicates", 0)),
        ("Outlier (IQR)", stats.get("after_iqr_filter", 0),
         stats.get("removed_outliers", 0)),
    ]
    steps_df = pd.DataFrame(steps_data, columns=["Bosqich", "Qolgan qator", "O'chirilgan"])

    cc1, cc2 = st.columns([1, 1])

    with cc1:
        st.markdown("**📋 Tozalash bosqichlari:**")
        st.dataframe(steps_df, use_container_width=True, hide_index=True)

    with cc2:
        # Funnel chart
        fig_funnel = go.Figure(go.Funnel(
            y=[s[0] for s in steps_data],
            x=[s[1] for s in steps_data],
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(
                color=[COLORS["primary"], COLORS["DQN"], COLORS["accent"],
                       COLORS["SAC"], COLORS["secondary"], COLORS["PPO"]],
            ),
            connector={"line": {"color": "rgba(0, 212, 170, 0.4)", "width": 1}},
        ))
        fig_funnel.update_layout(
            title="🔻 Tozalash Funnel",
            height=400,
            **PLOTLY_TEMPLATE["layout"],
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

    # Yakuniy KPI
    retention = stats.get("retention_pct", 0)
    if retention >= 80:
        st.success(f"✅ Saqlanish darajasi: **{retention}%** — yuqori sifat")
    elif retention >= 60:
        st.warning(f"⚠️ Saqlanish darajasi: **{retention}%** — o'rtacha")
    else:
        st.error(f"❌ Saqlanish darajasi: **{retention}%** — past sifat")


# ═══════════════════════════════════════════════════════════════
# SECTION 3: RFM segmentatsiyasi
# ═══════════════════════════════════════════════════════════════

if "df_clean" in st.session_state:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">3️⃣ RFM Segmentatsiyasi</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>RFM</strong> — mijozlar segmentatsiyasi uchun klassik metod:
        <strong>Recency</strong> (oxirgi xarid),
        <strong>Frequency</strong> (xaridlar soni),
        <strong>Monetary</strong> (jami summa).
        Har o'lchov 1-5 ball oladi, yig'indi 3-15 oralig'ida.
    </div>
    """, unsafe_allow_html=True)

    if st.button("🎯 RFM Hisoblash", type="primary"):
        with st.spinner("RFM matritsasi hisoblanmoqda..."):
            df_clean = st.session_state["df_clean"]
            rfm = _calc_rfm_cached(df_clean.to_csv(index=False))
            st.session_state["rfm"] = rfm

    if "rfm" in st.session_state:
        rfm = st.session_state["rfm"]

        # Asosiy statistika
        seg_counts = rfm["Segment"].value_counts()
        s1, s2, s3, s4, s5 = st.columns(5)
        segments_info = [
            ("🏆 Champion", "Champion", "#22c55e"),
            ("❤️ Loyal", "Loyal", "#00d4aa"),
            ("⚡ Potential", "Potential", "#6366f1"),
            ("⚠️ At_Risk", "At_Risk", "#f59e0b"),
            ("💔 Lost", "Lost", "#ef4444"),
        ]
        for col, (label, segment, color) in zip([s1, s2, s3, s4, s5], segments_info):
            count = int(seg_counts.get(segment, 0))
            pct = 100 * count / max(len(rfm), 1)
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 3px solid {color};">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="font-size: 1.5rem;">{count}</div>
                    <div style="color: {color}; font-size: 0.85rem; font-weight: 600;">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        # Vizualizatsiyalar
        rt1, rt2 = st.tabs(["📊 Segment Taqsimoti", "🌐 3D RFM Scatter"])

        with rt1:
            cc1, cc2 = st.columns(2)

            with cc1:
                fig_pie = px.pie(
                    rfm,
                    names="Segment",
                    title="🥧 Mijoz Segmentlari Taqsimoti",
                    color_discrete_sequence=[
                        "#22c55e", "#00d4aa", "#6366f1", "#f59e0b", "#ef4444",
                    ],
                    hole=0.5,
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                fig_pie.update_layout(height=400, **PLOTLY_TEMPLATE["layout"])
                st.plotly_chart(fig_pie, use_container_width=True)

            with cc2:
                # Segment bo'yicha o'rtacha monetary
                seg_money = rfm.groupby("Segment")["Monetary"].mean().reset_index()
                seg_money = seg_money.sort_values("Monetary", ascending=False)
                fig_bar = px.bar(
                    seg_money,
                    x="Segment", y="Monetary",
                    title="💰 Segment bo'yicha O'rtacha Monetary",
                    color="Segment",
                    color_discrete_map={
                        "Champion": "#22c55e", "Loyal": "#00d4aa",
                        "Potential": "#6366f1", "At_Risk": "#f59e0b", "Lost": "#ef4444",
                    },
                )
                fig_bar.update_layout(height=400, showlegend=False, **PLOTLY_TEMPLATE["layout"])
                st.plotly_chart(fig_bar, use_container_width=True)

        with rt2:
            fig_3d = px.scatter_3d(
                rfm,
                x="Recency", y="Frequency", z="Monetary",
                color="Segment",
                hover_data=["Customer ID", "RFM_Score"],
                title="🌐 3D RFM Scatter",
                color_discrete_map={
                    "Champion": "#22c55e", "Loyal": "#00d4aa",
                    "Potential": "#6366f1", "At_Risk": "#f59e0b", "Lost": "#ef4444",
                },
                opacity=0.7,
            )
            fig_3d.update_layout(height=600, **PLOTLY_TEMPLATE["layout"])
            st.plotly_chart(fig_3d, use_container_width=True)

        # RFM jadvali
        with st.expander("📋 To'liq RFM jadvali ko'rish"):
            st.dataframe(
                rfm.sort_values("RFM_Score", ascending=False),
                use_container_width=True,
                hide_index=True,
            )


# ═══════════════════════════════════════════════════════════════
# SECTION 4: EDA vizualizatsiyalar
# ═══════════════════════════════════════════════════════════════

if "df_clean" in st.session_state:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">4️⃣ Exploratory Data Analysis</h2>',
                unsafe_allow_html=True)

    df_clean = st.session_state["df_clean"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "💸 Narxlar", "📅 Vaqt Tahlili", "🏆 Top Mahsulotlar", "🔗 Korrelyatsiya",
    ])

    with tab1:
        cc1, cc2 = st.columns(2)
        with cc1:
            fig_hist = px.histogram(
                df_clean,
                x="Price",
                nbins=50,
                title="📊 Narxlar Histogrami",
                color_discrete_sequence=[COLORS["PPO"]],
            )
            fig_hist.update_layout(height=400, **PLOTLY_TEMPLATE["layout"])
            st.plotly_chart(fig_hist, use_container_width=True)

        with cc2:
            fig_box = px.box(
                df_clean,
                y="Price",
                title="📦 Narxlar Box Plot (outlier'lar)",
                color_discrete_sequence=[COLORS["secondary"]],
            )
            fig_box.update_layout(height=400, **PLOTLY_TEMPLATE["layout"])
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("**📈 Narx statistikasi:**")
        price_stats = df_clean["Price"].describe()
        st.dataframe(
            pd.DataFrame(price_stats).T.style.format("{:.2f}"),
            use_container_width=True,
        )

    with tab2:
        if "InvoiceDate" in df_clean.columns:
            df_temp = df_clean.copy()
            df_temp["InvoiceDate"] = pd.to_datetime(df_temp["InvoiceDate"], errors="coerce")
            df_temp = df_temp.dropna(subset=["InvoiceDate"])
            df_temp["Month"] = df_temp["InvoiceDate"].dt.to_period("M").astype(str)

            monthly = df_temp.groupby("Month").agg(
                tranzaksiya_soni=("Invoice", "count"),
                jami_daromad=("TotalPrice", "sum") if "TotalPrice" in df_temp.columns else ("Price", "sum"),
            ).reset_index()

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=monthly["Month"], y=monthly["tranzaksiya_soni"],
                mode="lines+markers", name="Tranzaksiyalar soni",
                line=dict(color=COLORS["PPO"], width=3),
                marker=dict(size=8),
            ))
            fig_trend.add_trace(go.Scatter(
                x=monthly["Month"], y=monthly["jami_daromad"],
                mode="lines+markers", name="Jami daromad (£)",
                line=dict(color=COLORS["accent"], width=3, dash="dash"),
                marker=dict(size=8),
                yaxis="y2",
            ))
            fig_trend.update_layout(
                title="📅 Oylik Tranzaksiya va Daromad Trend",
                height=450,
                yaxis=dict(title="Tranzaksiyalar soni"),
                yaxis2=dict(title="Daromad (£)", overlaying="y", side="right"),
                hovermode="x unified",
                **PLOTLY_TEMPLATE["layout"],
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("InvoiceDate ustuni mavjud emas")

    with tab3:
        if "Description" in df_clean.columns:
            top_products = df_clean.groupby("Description").agg(
                soni=("Quantity", "sum"),
                daromad=("Price", lambda x: (x * df_clean.loc[x.index, "Quantity"]).sum()),
            ).reset_index().sort_values("daromad", ascending=False).head(15)

            fig_top = px.bar(
                top_products,
                x="daromad", y="Description",
                orientation="h",
                title="🏆 Top 15 Mahsulot (Daromad bo'yicha)",
                color="daromad",
                color_continuous_scale="Teal",
            )
            fig_top.update_layout(
                height=500,
                yaxis=dict(categoryorder="total ascending"),
                **PLOTLY_TEMPLATE["layout"],
            )
            st.plotly_chart(fig_top, use_container_width=True)

    with tab4:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr = df_clean[numeric_cols].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="🔗 Korrelyatsiya Matritsasi",
            )
            fig_corr.update_layout(height=500, **PLOTLY_TEMPLATE["layout"])
            st.plotly_chart(fig_corr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# SECTION 5: Narx elastikligi
# ═══════════════════════════════════════════════════════════════

if "df_clean" in st.session_state:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">5️⃣ Narx Elastikligi</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>Narx elastikligi (ε)</strong> — narx 1% o'zgarganda talabning necha % o'zgarishi.
        Formula: <code>ln(Q) = α + ε · ln(P)</code> (OLS regression).
        <br><br>
        BMI da hisoblangan: <strong style="color: #00d4aa;">ε ≈ -1.5</strong> (elastik).
    </div>
    """, unsafe_allow_html=True)

    if st.button("📈 Elastiklikni Hisoblash", type="primary"):
        with st.spinner("OLS regression hisoblanmoqda..."):
            df_clean = st.session_state["df_clean"]
            elasticity_result = loader.calculate_elasticity(df_clean)
            st.session_state["elasticity"] = elasticity_result

    if "elasticity" in st.session_state:
        elast = st.session_state["elasticity"]
        ec1, ec2, ec3 = st.columns(3)

        with ec1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📉 Elastiklik (ε)</div>
                <div class="metric-value" style="color: #ff6b6b;">{elast['elasticity']:.3f}</div>
                <div class="metric-delta-up">{"Elastik" if abs(elast['elasticity']) > 1 else "Noelastik"}</div>
            </div>
            """, unsafe_allow_html=True)

        with ec2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📊 R²</div>
                <div class="metric-value">{elast['r_squared']:.4f}</div>
                <div class="metric-delta-up">Model sifati</div>
            </div>
            """, unsafe_allow_html=True)

        with ec3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🔢 Kuzatuvlar</div>
                <div class="metric-value">{elast['n_obs']:,}</div>
                <div class="metric-delta-up">OLS uchun</div>
            </div>
            """, unsafe_allow_html=True)

        # Scatter + regression line
        df_clean = st.session_state["df_clean"]
        sample = df_clean[(df_clean["Price"] > 0) & (df_clean["Quantity"] > 0)].sample(
            min(2000, len(df_clean)), random_state=42
        )
        log_price = np.log(sample["Price"])
        log_qty = np.log(sample["Quantity"])

        fig_elast = go.Figure()
        fig_elast.add_trace(go.Scatter(
            x=log_price, y=log_qty,
            mode="markers",
            marker=dict(size=5, color=COLORS["PPO"], opacity=0.4),
            name="Kuzatuvlar",
        ))
        # Regression chizigi
        x_line = np.linspace(log_price.min(), log_price.max(), 100)
        y_line = elast["intercept"] + elast["elasticity"] * x_line
        fig_elast.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            line=dict(color=COLORS["accent"], width=3),
            name=f"OLS: ε = {elast['elasticity']:.3f}",
        ))
        fig_elast.update_layout(
            title="📈 Log-Log Narx-Talab Regression",
            xaxis_title="ln(Price)",
            yaxis_title="ln(Quantity)",
            height=450,
            **PLOTLY_TEMPLATE["layout"],
        )
        st.plotly_chart(fig_elast, use_container_width=True)
