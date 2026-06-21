import logging
import os
import random
import time
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

# Configure logging style
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================================================
# CONFIG
# =========================================================
YEARS = [2021, 2022, 2023, 2024, 2025]
PERIODE = 12
BASE_PORTAL_URL = "https://djpk.kemenkeu.go.id/portal/data/apbd"
BASE_DOWNLOAD_URL = "https://djpk.kemenkeu.go.id/portal/csv_apbd"
OUTPUT_DIR = "raw_apbd"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
}

TARGET_AKUN = {
    "Pendapatan Daerah",
    "PAD",
    "Pajak Daerah",
    "Retribusi Daerah",
    "Pendapatan Transfer Pemerintah Pusat",
    "Belanja Daerah",
    "Belanja Pegawai",
    "Belanja Modal",
    "Belanja Bantuan Sosial",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
session = requests.Session()
session.headers.update(HEADERS)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def safe_request(url, params=None, retries=3, timeout=(10, 60)):
    """Handles network requests safely with exponential backoff."""
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except RequestException as e:
            wait_time = 2**attempt
            logging.warning(
                f"Request failed: {e}. Retry {attempt + 1}/{retries} in {wait_time}s..."
            )
            time.sleep(wait_time)
    logging.error(f"Failed to fetch data from URL: {url} after max retries.")
    return None


def get_provinsi_list():
    """Scrapes the master list of Provinces."""
    logging.info("Fetching province list...")
    r = safe_request(BASE_PORTAL_URL)
    if not r:
        return {}

    soup = BeautifulSoup(r.text, "html.parser")
    select = soup.find("select", {"id": "sel_provinsi"})
    if not select:
        logging.error("Province dropdown menu element selection not found.")
        return {}

    provinsi_map = {}
    for option in select.find_all("option"):
        kode = option.get("value", "").strip()
        nama = option.text.strip()
        if kode in ["", "--"]:
            continue
        provinsi_map[kode] = nama.replace("Provinsi ", "").strip()
    return provinsi_map


def get_pemda_list(provinsi_kode, tahun):
    """Scrapes the list of local government items (Pemda) for a specific province."""
    params = {
        "periode": PERIODE,
        "tahun": tahun,
        "provinsi": provinsi_kode,
        "pemda": "00",
    }
    r = safe_request(BASE_PORTAL_URL, params=params)
    if not r:
        return {}

    soup = BeautifulSoup(r.text, "html.parser")
    select = soup.find("select", {"id": "sel_pemda"})
    if not select:
        return {}

    pemda_map = {
        opt.get("value", "").strip(): opt.text.strip()
        for opt in select.find_all("option")
        if opt.get("value", "").strip() not in ["", "--"]
    }

    # Extract all values excluding parent '00' fallback
    non_00 = {k: v for k, v in pemda_map.items() if k != "00"}
    return non_00 if non_00 else pemda_map


def parse_spreadsheetml(xml_content):
    """Parses Excel XML SpreadsheetML layouts directly via Pandas."""
    try:
        # SpreadsheetML namespaces
        namespaces = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
        # Read XML rows directly into a dataframe structural form
        df_xml = pd.read_xml(
            xml_content, xpath=".//ss:Row", namespaces=namespaces
        )
        if len(df_xml) < 2:
            return None

        # Resolve cell lists inside standard column definitions
        # Note: If structural anomalies persist, fallback to explicit parser logic.
        return df_xml
    except Exception:
        # Fallback to structural extraction if plain read_xml engine encounters custom Excel variants
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_content)
        rows = []
        for row in root.findall(".//ss:Row", namespaces):
            rows.append(
                [
                    c.find("ss:Data", namespaces).text
                    if c.find("ss:Data", namespaces) is not None
                    else None
                    for c in row.findall("ss:Cell", namespaces)
                ]
            )
        if len(rows) < 2:
            return None
        return pd.DataFrame(rows[1:], columns=rows[0])


def normalize_and_clean_df(df):
    """Normalizes target financial keys and eliminates noise rows."""
    rename_map = {}
    for col in df.columns:
        cl = str(col).strip().lower()
        if "akun" in cl:
            rename_map[col] = "akun"
        elif "anggaran" in cl:
            rename_map[col] = "anggaran"
        elif "realisasi" in cl:
            rename_map[col] = "realisasi"

    df = df.rename(columns=rename_map)
    required = {"akun", "anggaran", "realisasi"}
    if not required.issubset(df.columns):
        return None

    # Vectorized text clean-up
    df = df.dropna(subset=["akun"])
    df["akun"] = df["akun"].astype(str).str.strip()
    df = df[df["akun"].isin(TARGET_AKUN)].copy()

    def clean_num(val):
        if pd.isna(val):
            return None
        # Handle numeric types directly
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)
        val_str = str(val).strip().lower()
        if val_str in ["", "none", "nan", "null"]:
            return None
        # Try raw float conversion first (for clean floats like '1233181611609.4')
        try:
            return float(val_str)
        except ValueError:
            pass
        # Fallback to cleaning Indonesian format with separators
        try:
            cleaned = val_str.replace(".", "").replace(",", ".")
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    df["anggaran"] = df["anggaran"].apply(clean_num)
    df["realisasi"] = df["realisasi"].apply(clean_num)
    return df[["akun", "anggaran", "realisasi"]]


def merge_year_files(tahun):
    """Combines isolated provincial data frames into single consolidated year groups."""
    temp_dir = f"{OUTPUT_DIR}/{tahun}"
    if not os.path.exists(temp_dir):
        return

    files = [
        os.path.join(temp_dir, f)
        for f in os.listdir(temp_dir)
        if f.endswith(".parquet")
    ]
    if not files:
        return

    logging.info(f"Merging partition chunks for year: {tahun}...")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            logging.error(f"Error reading chunk file {f}: {e}")

    if dfs:
        final_df = pd.concat(dfs, ignore_index=True).drop_duplicates()
        output_file = f"{OUTPUT_DIR}/apbd_{tahun}.parquet"
        final_df.to_parquet(output_file, index=False)
        logging.info(f"Saved: {output_file} | Total rows: {len(final_df)}")


# =========================================================
# MAIN EXECUTIVE RUNNER
# =========================================================
def run_scraper():
    provinsi_map = get_provinsi_list()
    if not provinsi_map:
        logging.error("No provinces extracted. Exiting pipeline runtime.")
        return

    for tahun in YEARS:
        logging.info(f"=== STARTING SCRAPING PIPELINE FOR YEAR: {tahun} ===")

        for prov_kode, prov_nama in provinsi_map.items():
            logging.info(f"Processing Province: [{prov_kode}] {prov_nama}")
            prov_rows = []

            try:
                pemda_map = get_pemda_list(prov_kode, tahun)
                if not pemda_map:
                    continue

                from concurrent.futures import ThreadPoolExecutor, as_completed

                def download_and_parse(pemda_kode, pemda_nama):
                    download_url = f"{BASE_DOWNLOAD_URL}?type=apbd&periode={PERIODE}&tahun={tahun}&provinsi={prov_kode}&pemda={pemda_kode}"
                    r = safe_request(download_url)
                    if not r or len(r.content) < 100:
                        return None
                    raw_df = parse_spreadsheetml(r.content)
                    if raw_df is None:
                        return None
                    clean_df = normalize_and_clean_df(raw_df)
                    if clean_df is None or clean_df.empty:
                        return None
                    clean_df["kode_provinsi"] = prov_kode
                    clean_df["provinsi"] = prov_nama
                    clean_df["kode_pemda"] = pemda_kode
                    clean_df["pemda"] = pemda_nama
                    clean_df["tahun"] = tahun
                    return clean_df

                # Download pemda data in parallel for the current province
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = {
                        executor.submit(download_and_parse, p_kode, p_nama): p_nama
                        for p_kode, p_nama in pemda_map.items()
                    }
                    for future in as_completed(futures):
                        try:
                            clean_df = future.result()
                            if clean_df is not None:
                                prov_rows.append(clean_df)
                        except Exception as e:
                            logging.error(f"Error downloading pemda {futures[future]}: {e}")

                # Save intermediate checkpoint files
                if prov_rows:
                    prov_df = pd.concat(prov_rows, ignore_index=True)
                    year_dir = f"{OUTPUT_DIR}/{tahun}"
                    os.makedirs(year_dir, exist_ok=True)
                    prov_df.to_parquet(
                        f"{year_dir}/{prov_kode}.parquet", index=False
                    )

            except Exception as e:
                logging.error(f"Critical interruption tracking {prov_nama}: {e}")
                continue

        merge_year_files(tahun)


if __name__ == "__main__":
    run_scraper()