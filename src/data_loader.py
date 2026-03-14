from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover
    class _Stub:
        @staticmethod
        def cache_data(**kwargs):
            def deco(fn):
                return fn
            return deco
    st = _Stub()

CANONICAL_STATES = {
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Delhi", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Ladakh", "Lakshadweep",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha",
    "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal", "Chandigarh", "Jammu And Kashmir", "Andaman And Nicobar Islands",
    "Puducherry", "Dadra And Nagar Haveli And Daman And Diu",
}

STATE_VARIANT_MAP = {
    "WEST BENGAL": "West Bengal",
    "WESTBENGAL": "West Bengal",
    "Westbengal": "West Bengal",
    "West bengal": "West Bengal",
    "West  Bengal": "West Bengal",
    "West Bangal": "West Bengal",
    "West Bengli": "West Bengal",
    "Orissa": "Odisha",
    "ODISHA": "Odisha",
    "odisha": "Odisha",
    "Tamilnadu": "Tamil Nadu",
    "Tamilnadu ": "Tamil Nadu",
    "Jammu & Kashmir": "Jammu And Kashmir",
    "Jammu and Kashmir": "Jammu And Kashmir",
    "Pondicherry": "Puducherry",
    "Pondi": "Puducherry",
    "Telengana": "Telangana",
    "Chattisgarh": "Chhattisgarh",
    "Rajastan": "Rajasthan",
}


def title_case_location(text: str | None) -> str:
    if text is None:
        return ""
    s = str(text).strip().replace("*", "")
    return " ".join(part.capitalize() for part in s.split()) if s else ""


def clean_for_validation(raw: str | None) -> str | None:
    if pd.isna(raw):
        return None
    s = str(raw).lower().strip().replace("&", "and").replace("*", "")
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s or None


def standardize_state_column(df: pd.DataFrame, col: str = "state") -> pd.DataFrame:
    out = df.copy()
    out[col] = out[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    out[col] = out[col].replace(STATE_VARIANT_MAP)
    out[col] = out[col].str.title()
    out = out[out[col].isin(CANONICAL_STATES)].copy()
    return out


def assign_area_type(df: pd.DataFrame, pincode_area_map: dict[int, str]) -> pd.DataFrame:
    out = df.copy()
    out["pincode"] = pd.to_numeric(out["pincode"], errors="coerce")
    out["area_type"] = out["pincode"].apply(lambda x: pincode_area_map.get(int(x), "Unknown") if pd.notna(x) else "Unknown")
    return out


def _concat_csvs(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)


def _looks_like_data_root(path: Path) -> bool:
    return path.exists() and (path / "api_data_aadhar_enrolment").exists() and (path / "api_data_aadhar_demographic").exists() and (path / "api_data_aadhar_biometric").exists()


def discover_data_root(base_dir: Path) -> Path:
    candidates = [
        base_dir / "data",
        base_dir.parent / "data",
        Path.cwd() / "data",
        Path.cwd(),
        Path("/mnt/data"),
    ]
    for candidate in candidates:
        if _looks_like_data_root(candidate):
            return candidate
        if candidate.exists():
            for sub in candidate.glob("*"):
                if sub.is_dir() and _looks_like_data_root(sub):
                    return sub
    raise FileNotFoundError("Could not find the dataset. Put the extracted drive-download contents inside a folder named 'data/' next to app.py.")


@st.cache_data(show_spinner="Loading and preparing Aadhaar datasets...")
def load_data(data_root_str: str) -> dict[str, pd.DataFrame | dict[int, str]]:
    data_root = Path(data_root_str)
    enrolment = _concat_csvs(data_root / "api_data_aadhar_enrolment")
    demographic = _concat_csvs(data_root / "api_data_aadhar_demographic")
    biometric = _concat_csvs(data_root / "api_data_aadhar_biometric")

    for df in (enrolment, demographic, biometric):
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
        df["month"] = df["date"].dt.to_period("M").astype(str)

    enrolment = standardize_state_column(enrolment)
    demographic = standardize_state_column(demographic)
    biometric = standardize_state_column(biometric)

    for df in (enrolment, demographic, biometric):
        df["district_clean"] = df["district"].map(clean_for_validation)
        df.dropna(subset=["district_clean"], inplace=True)

    urban = pd.read_csv(data_root / "lgd urban.csv")
    urban_pincodes = set(pd.to_numeric(urban["pincode"], errors="coerce").dropna().astype(int).tolist())

    rural_pincodes: set[int] = set()
    for file in (data_root / "rural-urban").glob("*.csv"):
        rdf = pd.read_csv(file)
        if "pincode" in rdf.columns:
            rural_pincodes.update(pd.to_numeric(rdf["pincode"], errors="coerce").dropna().astype(int).tolist())

    pincode_area_map: dict[int, str] = {p: "Urban" for p in urban_pincodes}
    for p in rural_pincodes:
        pincode_area_map.setdefault(p, "Rural")

    enrolment = assign_area_type(enrolment, pincode_area_map)
    demographic = assign_area_type(demographic, pincode_area_map)
    biometric = assign_area_type(biometric, pincode_area_map)

    zones = pd.read_csv(data_root / "Zones.csv")
    zones.rename(columns={"District": "district_clean", "Count": "centre_count"}, inplace=True)
    zones["district_clean"] = zones["district_clean"].map(clean_for_validation)
    zones["centre_count"] = pd.to_numeric(zones["centre_count"], errors="coerce").fillna(0)

    return {
        "enrolment": enrolment,
        "demographic": demographic,
        "biometric": biometric,
        "zones": zones,
    }
