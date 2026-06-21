import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score

# Config
DATA_PATH = "processed/fiscal_features.parquet"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Set style untuk grafik
sns.set_theme(style="whitegrid")

def load_and_preprocess(path):
    print("\n--- 1. LOAD & PREPROCESS DATA ---")
    df = pd.read_parquet(path)
    print(f"Dimensi Data: {df.shape[0]} Rows, {df.shape[1]} Cols")
    
    feature_columns = [
        "rasio_pad", "rasio_modal", "rasio_pegawai", "serapan"
    ]
    
    # Handle Missing & Infinite
    X = df[feature_columns].copy().fillna(0)
    X.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    
    return df, X_scaled

def run_analytics(df, X_scaled):
    print("\n--- 2. RUN CLUSTERING & ANOMALY DETECTION ---")
    
    # K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    joblib.dump(kmeans, f"{MODEL_DIR}/kmeans.pkl")
    
    score = silhouette_score(X_scaled, df["cluster"])
    print(f"Silhouette Score: {score:.4f}")
    
    # PCA for Visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df["pca1"] = pca_result[:, 0]
    df["pca2"] = pca_result[:, 1]
    
    # Anomaly Detection
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = iso.fit_predict(X_scaled)
    joblib.dump(iso, f"{MODEL_DIR}/anomaly.pkl")
    
    return df

def run_forecasting(df):
    print("\n--- 3. RUN FORECASTING (2026) ---")
    forecast_rows = []
    future_year = pd.DataFrame({"tahun": [2026]})
    
    for pemda in df["pemda"].unique():
        temp = df[df["pemda"] == pemda].sort_values("tahun")
        if len(temp) < 2:
            continue
            
        X_year = temp[["tahun"]]
        
        # Ensure positive values before log transform
        pad_vals = temp["pad"].clip(lower=1.0)
        belanja_vals = temp["total_belanja"].clip(lower=1.0)
        
        try:
            # Log-linear regression for PAD
            model_pad = LinearRegression().fit(X_year, np.log(pad_vals))
            pred_pad = np.exp(model_pad.predict(future_year)[0])
            
            # Log-linear regression for Belanja
            model_belanja = LinearRegression().fit(X_year, np.log(belanja_vals))
            pred_belanja = np.exp(model_belanja.predict(future_year)[0])
        except Exception:
            # Fallback to standard linear regression with clipping if log fails
            model_pad = LinearRegression().fit(X_year, temp["pad"])
            pred_pad = max(0.0, model_pad.predict(future_year)[0])
            
            model_belanja = LinearRegression().fit(X_year, temp["total_belanja"])
            pred_belanja = max(0.0, model_belanja.predict(future_year)[0])
            
        # Apply guardrails to prevent extreme forecasting spikes/drops (max 2x or min 0.5x of last observed year)
        last_pad = temp["pad"].iloc[-1]
        last_belanja = temp["total_belanja"].iloc[-1]
        
        pred_pad = np.clip(pred_pad, last_pad * 0.5, last_pad * 2.0)
        pred_belanja = np.clip(pred_belanja, last_belanja * 0.5, last_belanja * 2.0)
        
        forecast_rows.append({
            "pemda": pemda,
            "forecast_pad_2026": pred_pad,
            "forecast_belanja_2026": pred_belanja
        })
        
    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df.to_parquet("processed/forecast.parquet", index=False)
    print("Forecasting selesai & disimpan.")
    return forecast_df

def post_processing(df):
    print("\n--- 4. POST PROCESSING & LABELING ---")
    
    # 1. Hitung rata-rata fitur per cluster
    features_to_sum = ["rasio_pad", "rasio_transfer", "rasio_pegawai", "rasio_modal", "fiscal_score"]
    cluster_summary = df.groupby("cluster")[features_to_sum].mean()
    
    # 2. Urutkan indeks cluster berdasarkan rasio_pad dari TERTINGGI ke TERENDAH
    # Cluster dengan PAD tertinggi akan berada di urutan pertama, dst.
    sorted_clusters = cluster_summary.sort_values(by="rasio_pad", ascending=False).index.tolist()
    
    cluster_label = {
        sorted_clusters[0]: "Mandiri Fiskal",
        sorted_clusters[1]: "Fiskal Menengah",
        sorted_clusters[2]: "Tergantung Pusat"
    }
            
    df["cluster_label"] = df["cluster"].map(cluster_label)
    
    # Update cluster_summary index to use readable labels and sort it for heatmap plotting
    cluster_summary.index = cluster_summary.index.map(cluster_label)
    cluster_summary = cluster_summary.reindex([
        "Mandiri Fiskal",
        "Fiskal Menengah",
        "Tergantung Pusat"
    ])
    
    print("\nTOP 5 FISCAL SCORE:")
    print(df.sort_values(by="fiscal_score", ascending=False)[["pemda", "tahun", "fiscal_score"]].head(5))
    
    return df, cluster_summary

def generate_all_reports(df, cluster_summary, forecast_df=None):
    """
    Satu fungsi terintegrasi untuk menampilkan seluruh laporan visual eksekutif,
    grafik analitik advanced, dan tabel statistik tanpa pengulangan komputasi.
    """
    print("\n" + "="*60)
    print("      MEMULAI GENERASI LAPORAN")
    print("="*60)
    
    # 1. PRE-COMPUTATION (Cukup dilakukan sekali untuk semua grafik/tabel)
    tahun_terakhir = df['tahun'].max()
    df_terakhir = df[df['tahun'] == tahun_terakhir]
    df['Status Data'] = df['anomaly'].map({1: 'Normal', -1: 'Anomali'})
    
    # -----------------------------------------------------------------
    # BAGIAN A: MENAMPILKAN TABEL-TABEL UTAMA (Console Report)
    # -----------------------------------------------------------------
    print(f"\n[TABEL 1] RINGKASAN STATISTIK FITUR UTAMA (TAHUN {tahun_terakhir})")
    print("-" * 75)
    features_to_show = ["rasio_pad", "rasio_transfer", "rasio_pegawai", "rasio_modal", "serapan", "fiscal_score"]
    summary_table = df_terakhir[features_to_show].describe().T[['mean', '50%', 'min', 'max']]
    summary_table.columns = ['Rata-rata', 'Median (Q2)', 'Minimum', 'Maksimum']
    
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    print(summary_table)
    
    print(f"\n[TABEL 2] PROFIL DAERAH: TOP 3 & BOTTOM 3 FISCAL SCORE (TAHUN {tahun_terakhir})")
    print("-" * 75)
    df_sorted = df_terakhir.sort_values(by='fiscal_score', ascending=False)
    extreme_table = pd.concat([df_sorted.head(3), df_sorted.tail(3)])[[
        'pemda', 'rasio_pad', 'rasio_transfer', 'rasio_pegawai', 'fiscal_score'
    ]]
    extreme_table['Kategori Kinerja'] = ['Top 1', 'Top 2', 'Top 3', 'Bottom 3', 'Bottom 2', 'Bottom 1']
    print(extreme_table.set_index('Kategori Kinerja'))
    pd.reset_option('display.float_format')

    # -----------------------------------------------------------------
    # BAGIAN B: GENERASI GRAFIK (Menggunakan Grid 3x2)
    # -----------------------------------------------------------------
    print("\n--- GENERATING INTEGRATED GRAPHICS ---")
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Kita buat canvas besar berisi 6 grafik sekaligus (3 baris, 2 kolom)
    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    
    # G1: K-Means & Anomali Overlay (PCA)
    sns.scatterplot(
        ax=axes[0, 0], x="pca1", y="pca2", hue="cluster_label", 
        style="Status Data", markers={"Normal": "o", "Anomali": "X"},
        data=df, palette="Set1", alpha=0.8, s=100
    )
    axes[0, 0].set_title("K-Means Clustering & Sebaran Anomali (PCA Space)", fontsize=12, fontweight='bold')
    
    # G2: Heatmap Karakteristik
    sns.heatmap(ax=axes[0, 1], data=cluster_summary, annot=True, cmap="YlGnBu", fmt=".2f")
    axes[0, 1].set_title("Profil Karakteristik Rata-rata per Cluster", fontsize=12, fontweight='bold')
    
    # G3: Kemandirian vs Alokasi Modal (Advanced Scatter)
    sns.scatterplot(
        ax=axes[1, 0], data=df_terakhir, x='rasio_pad', y='rasio_modal', 
        hue='anomaly', palette={1: '#2ca02c', -1: '#d62728'},
        size='fiscal_score', sizes=(30, 300), alpha=0.75
    )
    axes[1, 0].set_title(f'Produktivitas Fiskal: PAD vs Belanja Modal ({tahun_terakhir})', fontsize=12, fontweight='bold')
    
    # G4: Violin Plot Distribusi Nasional
    df_melt = pd.melt(df, value_vars=['rasio_pegawai', 'rasio_modal', 'rasio_bansos'], var_name='Jenis Belanja', value_name='Rasio')
    df_melt['Jenis Belanja'] = df_melt['Jenis Belanja'].map({'rasio_pegawai': 'Belanja Pegawai', 'rasio_modal': 'Belanja Modal', 'rasio_bansos': 'Belanja Bansos'})
    sns.violinplot(ax=axes[1, 1], data=df_melt, x='Jenis Belanja', y='Rasio', palette='Pastel1', inner='quartile', hue='Jenis Belanja')
    axes[1, 1].set_title('Densitas & Alokasi Anggaran Nasional', fontsize=12, fontweight='bold')
    
    # G5: Forecasting Line Plot (Menggunakan data langsung dari RAM, bukan baca file)
    try:
        # Pilihlah Pemda dengan Fiscal Score tinggi sebagai perwakilan visualisasi yang bagus
        contoh_pemda = df_terakhir.sort_values(by='fiscal_score', ascending=False)['pemda'].iloc[0]
    except Exception:
        contoh_pemda = df['pemda'].unique()[0]
    
    df_pemda = df[df['pemda'] == contoh_pemda].sort_values('tahun')
    axes[2, 0].plot(df_pemda['tahun'], df_pemda['pad'], marker='o', color='#1f77b4', linewidth=2.5, label='PAD Historis')
    
    if forecast_df is not None:
        try:
            pred_2026 = forecast_df[forecast_df['pemda'] == contoh_pemda]['forecast_pad_2026'].values[0]
            axes[2, 0].plot([df_pemda['tahun'].max(), 2026], [df_pemda['pad'].iloc[-1], pred_2026], linestyle='--', color='#ff7f0e')
            axes[2, 0].scatter(2026, pred_2026, color='#d62728', zorder=5, s=120, label='Proyeksi 2026')
            axes[2, 0].set_xticks(list(df_pemda['tahun'].unique()) + [2026])
        except IndexError:
            pass
    axes[2, 0].set_title(f'Simulasi Proyeksi PAD (Forecasting): {contoh_pemda}', fontsize=12, fontweight='bold')
    axes[2, 0].legend(loc='upper left')
    
    # G6: Komposisi Cluster Saat Ini
    cluster_counts = df_terakhir['cluster_label'].value_counts().reset_index()
    cluster_counts.columns = ['Kategori Cluster', 'Jumlah Pemda']
    sns.barplot(ax=axes[2, 1], data=cluster_counts, x='Jumlah Pemda', y='Kategori Cluster', palette='magma', hue='Jumlah Pemda')
    axes[2, 1].set_title(f'Komposisi Klasifikasi Tipologi Pemda ({tahun_terakhir})', fontsize=12, fontweight='bold')
    
    # Penyelesaian
    plt.tight_layout()
    plt.savefig("processed/report_visual.png", dpi=300)
    plt.show()

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    df, X_scaled = load_and_preprocess(DATA_PATH)
    df = run_analytics(df, X_scaled)
    forecast_df = run_forecasting(df)
    df, cluster_summary = post_processing(df)
    
    # Save Final Data
    df.to_parquet("processed/final_mining.parquet", index=False)
    df.to_csv("processed/final_mining.csv", index=False)
    print("\nData final berhasil disimpan.")
    
    # Tampilkan laporan date dan grafik visualisasi
    generate_all_reports(df, cluster_summary, forecast_df)
    
    print("\n--- PROSES SELESAI ---")