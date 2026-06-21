import logging
from pathlib import Path
import numpy as np
import pandas as pd

# =========================================================
# CONFIGURATION & CONSTANTS
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RAW_DIR = Path("raw_apbd")
OUTPUT_DIR = Path("processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2021, 2022, 2023, 2024, 2025]

AKUN_MAP = {
    "Pendapatan Daerah": "total_pendapatan",
    "PAD": "pad",
    "Pajak Daerah": "pajak_daerah",
    "Retribusi Daerah": "retribusi",
    "Pendapatan Transfer Pemerintah Pusat": "transfer",
    "Belanja Daerah": "total_belanja",
    "Belanja Pegawai": "belanja_pegawai",
    "Belanja Modal": "belanja_modal",
    "Belanja Bantuan Sosial": "bansos"
}


# =========================================================
# PIPELINE FUNCTIONS
# =========================================================

def load_and_merge_raw_files(raw_dir: Path, years: list) -> pd.DataFrame:
    """Loads individual yearly Parquet files and stacks them into one DataFrame."""
    all_dfs = []
    for year in years:
        file_path = raw_dir / f"apbd_{year}.parquet"
        if not file_path.exists():
            logging.warning(f"File not found, skipping: {file_path}")
            continue
        
        logging.info(f"Loading raw file: {file_path}")
        all_dfs.append(pd.read_parquet(file_path))
        
    if not all_dfs:
        raise FileNotFoundError("No raw Parquet data files could be located.")
        
    return pd.concat(all_dfs, ignore_index=True).drop_duplicates()


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes string values, data types, and clips anomalous numeric records."""
    logging.info("Executing basic dataset data cleaning...")
    df.columns = df.columns.str.lower()
    
    # Strip string text columns effectively
    text_cols = ["provinsi", "pemda", "akun"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    # Coerce to numeric values safely and clear invalid negative bounds
    for num_col in ["anggaran", "realisasi"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
            df.loc[df[num_col] < 0, num_col] = np.nan
            
    # Map raw Indonesian financial categories into analytics keys
    df["akun_key"] = df["akun"].map(AKUN_MAP)
    return df[df["akun_key"].notna()].copy()


def pivot_and_impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """Reshapes long-form account logs into a wide time-series and fills analytical holes."""
    logging.info("Pivoting long-form layout into wide analytical time-series columns...")
    
    index_cols = ["kode_provinsi", "provinsi", "kode_pemda", "pemda", "tahun"]
    pivot_df = df.pivot_table(
        index=index_cols,
        columns="akun_key",
        values="realisasi",
        aggfunc="sum"
    ).reset_index()
    pivot_df.columns.name = None
    
    # Identify financial metrics to isolate from structural geography indexes
    numeric_cols = [c for c in pivot_df.columns if c not in index_cols]
    pivot_df = pivot_df.sort_values(by=["pemda", "tahun"])
    
    # Time-series bounding interpolation per Pemda entity
    logging.info("Imputing time-series missing values per regional entity...")
    for col in numeric_cols:
        pivot_df[col] = (
            pivot_df.groupby("pemda")[col]
            .transform(lambda x: x.interpolate(method="linear", limit_direction="both"))
        )
    pivot_df[numeric_cols] = pivot_df[numeric_cols].fillna(0)
    
    # Drop volatile regions containing less than 2 distinct tracking years
    region_counts = pivot_df.groupby("pemda")["tahun"].transform("count")
    return pivot_df[region_counts >= 2].copy()


def engineer_fiscal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates structural ratios, year-over-year momentum shifts, and fiscal indices."""
    logging.info("Calculating structural budget ratios and growth metrics...")
    
    # Safely compute allocation weights to avoid NaN on zero division
    df["rasio_pad"] = df["pad"] / df["total_pendapatan"]
    df["rasio_transfer"] = df["transfer"] / df["total_pendapatan"]
    df["rasio_pegawai"] = df["belanja_pegawai"] / df["total_belanja"]
    df["rasio_modal"] = df["belanja_modal"] / df["total_belanja"]
    df["rasio_bansos"] = df["bansos"] / df["total_belanja"]
    df["serapan"] = df["total_belanja"] / df["total_pendapatan"]
    
    # Scrub out infinite anomalies caused by computing divisions against a 0 baseline
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    # Time-series sorting optimization prior to window shifting operations
    df = df.sort_values(by=["pemda", "tahun"])
    
    # Year-Over-Year growth trends tracking
    df["growth_pad"] = df.groupby("pemda")["pad"].pct_change()
    df["growth_belanja"] = df.groupby("pemda")["total_belanja"].pct_change()
    df["growth_pendapatan"] = df.groupby("pemda")["total_pendapatan"].pct_change()
    df.fillna(0, inplace=True)
    
    # Algorithmic calculation of the Fiscal Score index
    logging.info("Compiling composite weighted Fiscal Scores...")
    df["fiscal_score"] = (
        (df["rasio_pad"] * 0.35) + 
        ((1 - df["rasio_transfer"]) * 0.25) + 
        (df["rasio_modal"] * 0.20) + 
        ((1 - df["rasio_pegawai"]) * 0.20)
    )
    return df.drop_duplicates()


# =========================================================
# CORE RUNNER
# =========================================================

def main():
    # 1. Pipeline Consolidation & Loading
    raw_df = load_and_merge_raw_files(RAW_DIR, YEARS)
    raw_df.to_parquet(OUTPUT_DIR / "apbd_merged.parquet", index=False)
    logging.info(f"Consolidated base metrics compiled successfully. Initial rows: {len(raw_df)}")
    
    # 2. Structural Scrubbing & Reshaping
    cleaned_df = clean_raw_data(raw_df)
    pivoted_df = pivot_and_impute_data(cleaned_df)
    pivoted_df.to_parquet(OUTPUT_DIR / "apbd_clean.parquet", index=False)
    
    # 3. Calculations & Mathematical Feature Transforms
    final_features_df = engineer_fiscal_features(pivoted_df)
    
    # 4. Exports & Summaries Generation
    final_features_df.to_parquet(OUTPUT_DIR / "fiscal_features.parquet", index=False)
    final_features_df.to_csv(OUTPUT_DIR / "fiscal_features.csv", index=False)
    logging.info("Feature compilation pipeline complete. Artifacts stored to disc.")
    
    # Log Concise Execution Profile Summaries
    print(f"\n{'='*40}\nEXECUTION PIPELINE PROCESS SUMMARY\n{'='*40}")
    print(f"Total Logged Processing Rows  : {len(final_features_df)}")
    print(f"Unique Local Entities (Pemda) : {final_features_df['pemda'].nunique()}")
    print(f"Unique Provinces (Provinsi)   : {final_features_df['provinsi'].nunique()}")
    print(f"Operational Year Constraints   : {final_features_df['tahun'].min()} - {final_features_df['tahun'].max()}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()