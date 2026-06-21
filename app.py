import pandas as pd
import streamlit as st
import plotly.express as px

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Dashboard APBD Indonesia",
    layout="wide"
)

# Custom CSS for Premium look and feel
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&display=swap" rel="stylesheet">
    <style>
    /* Styling modern premium dark */
    .stApp {
        background-color: #0b0f19;
        color: #e5e7eb;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, .stSubheader {
        font-family: 'Outfit', sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    h1 {
        background: linear-gradient(135deg, #60a5fa 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem !important;
    }
    div[data-testid="metric-container"] {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.3);
    }
    div[data-testid="metric-container"] label {
        color: #9ca3af;
        font-weight: 500;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .stDataFrame, div[data-testid="stTable"] {
        background: rgba(17, 24, 39, 0.5);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background-color: #111827;
    }
    </style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data

def load_data():

    return pd.read_parquet(
        "processed/final_mining.parquet"
    )

@st.cache_data

def load_forecast():

    return pd.read_parquet(
        "processed/forecast.parquet"
    )

# load

df = load_data()

forecast_df = load_forecast()

# =========================================================
# TITLE
# =========================================================

st.title(
    "Dashboard Fiscal Mining APBD Indonesia"
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Filter")

selected_year = st.sidebar.selectbox(
    "Tahun",
    sorted(df["tahun"].unique())
)

selected_cluster = st.sidebar.multiselect(
    "Cluster",
    df["cluster_label"].unique(),
    default=df["cluster_label"].unique()
)

# =========================================================
# FILTER DATA
# =========================================================

filtered = df[
    (df["tahun"] == selected_year)
    &
    (df["cluster_label"].isin(selected_cluster))
]

# =========================================================
# METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Jumlah Daerah",
    filtered["pemda"].nunique()
)

col2.metric(
    "Rata Fiscal Score",
    round(
        filtered[
            "fiscal_score"
        ].mean(),
        3
    )
)

col3.metric(
    "Rata Rasio PAD",
    round(
        filtered[
            "rasio_pad"
        ].mean(),
        3
    )
)

col4.metric(
    "Daerah Anomali",
    len(
        filtered[
            filtered["anomaly"] == -1
        ]
    )
)

# =========================================================
# PCA CLUSTER
# =========================================================

st.subheader("Cluster Fiskal")

fig = px.scatter(

    filtered,

    x="pca1",

    y="pca2",

    color="cluster_label",

    hover_data=[
        "pemda",
        "fiscal_score"
    ]
)

st.plotly_chart(
    fig,
    width='stretch'
)

# =========================================================
# TOP FISCAL SCORE
# =========================================================

st.subheader(
    "Top Fiscal Score"
)

ranking = filtered.sort_values(
    by="fiscal_score",
    ascending=False
)

st.dataframe(
    ranking[[
        "pemda",
        "provinsi",
        "fiscal_score",
        "cluster_label"
    ]].head(20)
)

# =========================================================
# ANOMALY
# =========================================================

st.subheader("Daerah Anomali")

anomali = filtered[
    filtered["anomaly"] == -1
]

st.dataframe(
    anomali[[
        "pemda",
        "provinsi",
        "fiscal_score",
        "cluster_label"
    ]]
)

# =========================================================
# FORECAST
# =========================================================

st.subheader("🔮 Simulasi & Proyeksi Fiskal Tahun 2026")

all_pemdas = sorted(df["pemda"].unique())
selected_pemda = st.selectbox("Pilih Daerah untuk Analisis Tren & Proyeksi 2026", all_pemdas)

if selected_pemda:
    # Get historical data for the selected region
    pemda_hist = df[df["pemda"] == selected_pemda].sort_values("tahun")
    # Get forecasted data
    pemda_fore = forecast_df[forecast_df["pemda"] == selected_pemda]
    
    if not pemda_fore.empty and not pemda_hist.empty:
        fore_pad = pemda_fore["forecast_pad_2026"].values[0]
        fore_belanja = pemda_fore["forecast_belanja_2026"].values[0]
        
        last_year = pemda_hist["tahun"].max()
        last_pad = pemda_hist[pemda_hist["tahun"] == last_year]["pad"].values[0]
        last_belanja = pemda_hist[pemda_hist["tahun"] == last_year]["total_belanja"].values[0]
        
        # Calculate growth rates
        growth_pad_proj = ((fore_pad - last_pad) / last_pad) * 100 if last_pad > 0 else 0.0
        growth_belanja_proj = ((fore_belanja - last_belanja) / last_belanja) * 100 if last_belanja > 0 else 0.0
        
        # Display projections metrics in cards
        f_col1, f_col2 = st.columns(2)
        
        f_col1.metric(
            label="Proyeksi PAD 2026",
            value=f"Rp {fore_pad:,.2f}",
            delta=f"{growth_pad_proj:+.2f}% dibanding {last_year}"
        )
        f_col2.metric(
            label="Proyeksi Belanja 2026",
            value=f"Rp {fore_belanja:,.2f}",
            delta=f"{growth_belanja_proj:+.2f}% dibanding {last_year}"
        )
        
        # Build interactive Plotly chart showing history + projection
        import plotly.graph_objects as go
        
        fig_line = go.Figure()
        
        # PAD
        fig_line.add_trace(go.Scatter(
            x=pemda_hist["tahun"], 
            y=pemda_hist["pad"],
            mode='lines+markers',
            name='PAD Historis',
            line=dict(color='#10b981', width=3),
            marker=dict(size=8)
        ))
        fig_line.add_trace(go.Scatter(
            x=[last_year, 2026],
            y=[last_pad, fore_pad],
            mode='lines+markers',
            name='Proyeksi PAD 2026',
            line=dict(color='#f59e0b', width=3, dash='dash'),
            marker=dict(size=12, symbol='star', color='#ef4444')
        ))
        
        # Belanja
        fig_line.add_trace(go.Scatter(
            x=pemda_hist["tahun"],
            y=pemda_hist["total_belanja"],
            mode='lines+markers',
            name='Belanja Historis',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=8)
        ))
        fig_line.add_trace(go.Scatter(
            x=[last_year, 2026],
            y=[last_belanja, fore_belanja],
            mode='lines+markers',
            name='Proyeksi Belanja 2026',
            line=dict(color='#8b5cf6', width=3, dash='dash'),
            marker=dict(size=12, symbol='star', color='#ef4444')
        ))
        
        fig_line.update_layout(
            title=f"Tren Anggaran Historis vs Proyeksi 2026 - {selected_pemda}",
            xaxis_title="Tahun",
            yaxis_title="Nilai (Rupiah)",
            xaxis=dict(tickmode='linear', tick0=2021, dtick=1),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Data historis atau proyeksi tidak lengkap untuk daerah ini.")

st.markdown("---")
st.subheader("📋 Ringkasan Proyeksi 2026 (Seluruh Daerah)")
formatted_forecast = forecast_df.copy()
formatted_forecast.columns = ["Pemda / Daerah", "Proyeksi PAD 2026 (Rp)", "Proyeksi Belanja 2026 (Rp)"]
st.dataframe(formatted_forecast, height=400)