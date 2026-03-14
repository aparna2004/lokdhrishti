from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.cluster import BisectingKMeans
except Exception:  # pragma: no cover
    BisectingKMeans = None
    from sklearn.cluster import KMeans

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


STATE_NAME_FIXES = {
    "Andaman And Nicobar Islands": "Andaman and Nicobar",
    "Jammu And Kashmir": "Jammu and Kashmir",
    "Dadra And Nagar Haveli And Daman And Diu": "Dadra and Nagar Haveli and Daman and Diu",
}


def compute_kpis(enrolment: pd.DataFrame, demographic: pd.DataFrame, biometric: pd.DataFrame) -> dict:
    total_enrolments = int(enrolment[["age_0_5", "age_5_17", "age_18_greater"]].sum().sum())
    total_demo = int(demographic[["demo_age_5_17", "demo_age_17_"]].sum().sum())
    total_bio = int(biometric[["bio_age_5_17", "bio_age_17_"]].sum().sum())
    monthly = monthly_service_balance(enrolment, biometric)
    return {
        "districts": int(enrolment["district_clean"].nunique()),
        "states": int(enrolment["state"].nunique()),
        "months": int(enrolment["month"].nunique()),
        "date_min": enrolment["date"].min(),
        "date_max": enrolment["date"].max(),
        "total_enrolments": total_enrolments,
        "total_demographic_updates": total_demo,
        "total_biometric_updates": total_bio,
        "biometric_dominant_months": int((monthly["total_biometric_updates"] > monthly["total_enrolments"]).sum()),
    }


def monthly_service_balance(enrolment: pd.DataFrame, biometric: pd.DataFrame) -> pd.DataFrame:
    enrol = enrolment.groupby("month")[["age_0_5", "age_5_17", "age_18_greater"]].sum().sum(axis=1).rename("total_enrolments")
    bio = biometric.groupby("month")[["bio_age_5_17", "bio_age_17_"]].sum().sum(axis=1).rename("total_biometric_updates")
    out = pd.concat([enrol, bio], axis=1).fillna(0).reset_index().sort_values("month")
    out["service_balance"] = (out["total_enrolments"] - out["total_biometric_updates"]) / (out["total_enrolments"] + out["total_biometric_updates"])
    out["update_share"] = out["total_biometric_updates"] / (out["total_enrolments"] + out["total_biometric_updates"])
    return out


def district_age_enrolment(enrolment: pd.DataFrame) -> pd.DataFrame:
    grouped = enrolment.groupby(["state", "district_clean"])[["age_0_5", "age_5_17", "age_18_greater"]].sum().reset_index()
    grouped["total_enrolments"] = grouped[["age_0_5", "age_5_17", "age_18_greater"]].sum(axis=1)
    grouped["share_0_5"] = grouped["age_0_5"] / grouped["total_enrolments"]
    grouped["share_5_17"] = grouped["age_5_17"] / grouped["total_enrolments"]
    grouped["share_18_greater"] = grouped["age_18_greater"] / grouped["total_enrolments"]
    return grouped.replace([np.inf, -np.inf], 0).fillna(0)


def district_age_heatmap(district_age: pd.DataFrame, top_n: int = 40) -> pd.DataFrame:
    cols = ["district_clean", "share_0_5", "share_5_17", "share_18_greater"]
    return district_age.sort_values("share_0_5").head(top_n)[cols].set_index("district_clean")


def low_childhood_trend(enrolment: pd.DataFrame, district_age: pd.DataFrame, low_n: int = 12) -> pd.DataFrame:
    low_districts = set(district_age.sort_values("share_0_5").head(low_n)["district_clean"])
    trend = enrolment[enrolment["district_clean"].isin(low_districts)].groupby("month")["age_0_5"].sum().reset_index(name="age_0_5")
    return trend.sort_values("month")


def state_late_childhood(enrolment: pd.DataFrame) -> pd.DataFrame:
    grouped = enrolment.groupby("state")[["age_0_5", "age_5_17", "age_18_greater"]].sum().reset_index()
    grouped["total_enrolments"] = grouped[["age_0_5", "age_5_17", "age_18_greater"]].sum(axis=1)
    grouped["early_childhood_share"] = grouped["age_0_5"] / grouped["total_enrolments"]
    grouped["late_childhood_share"] = grouped["age_5_17"] / grouped["total_enrolments"]
    grouped["state_geo"] = grouped["state"].replace(STATE_NAME_FIXES)
    return grouped.replace([np.inf, -np.inf], 0).fillna(0).sort_values("late_childhood_share", ascending=False)


@st.cache_data(show_spinner=False)
def load_india_geojson(base_dir_str: str) -> dict | None:
    base_dir = Path(base_dir_str)
    local = base_dir / "assets" / "india_state.geojson"
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))
    try:
        import urllib.request
        with urllib.request.urlopen("https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson", timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        local.write_text(json.dumps(data), encoding="utf-8")
        return data
    except Exception:
        return None


def district_update_ratios(enrolment: pd.DataFrame, demographic: pd.DataFrame, biometric: pd.DataFrame) -> pd.DataFrame:
    e = enrolment.groupby(["state", "district_clean"])[["age_0_5", "age_5_17", "age_18_greater"]].sum().reset_index()
    e["total_enrolments"] = e[["age_0_5", "age_5_17", "age_18_greater"]].sum(axis=1)
    d = demographic.groupby(["state", "district_clean"])[["demo_age_5_17", "demo_age_17_"]].sum().reset_index()
    d["total_demographic_updates"] = d[["demo_age_5_17", "demo_age_17_"]].sum(axis=1)
    b = biometric.groupby(["state", "district_clean"])[["bio_age_5_17", "bio_age_17_"]].sum().reset_index()
    b["total_biometric_updates"] = b[["bio_age_5_17", "bio_age_17_"]].sum(axis=1)
    out = e.merge(d[["state", "district_clean", "total_demographic_updates"]], on=["state", "district_clean"], how="outer")
    out = out.merge(b[["state", "district_clean", "total_biometric_updates"]], on=["state", "district_clean"], how="outer").fillna(0)
    out["total_updates"] = out["total_demographic_updates"] + out["total_biometric_updates"]
    out["demographic_update_ratio"] = out["total_demographic_updates"] / out["total_enrolments"]
    out["biometric_update_ratio"] = out["total_biometric_updates"] / out["total_enrolments"]
    out["total_update_ratio"] = out["total_updates"] / out["total_enrolments"]
    out = out.replace([np.inf, -np.inf], 0).fillna(0)
    out["update_mix"] = np.where(out["biometric_update_ratio"] > out["demographic_update_ratio"], "Biometric-led", "Demographic-led")
    return out


def high_update_demand(district_ratios: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    filtered = district_ratios[district_ratios["total_enrolments"] >= 100].copy()
    threshold = float(filtered["total_update_ratio"].mean() + filtered["total_update_ratio"].std())
    return filtered[filtered["total_update_ratio"] > threshold].sort_values("total_update_ratio", ascending=False), threshold


def forecast_biometric_demand(enrolment: pd.DataFrame, biometric: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    e = enrolment.groupby(["month", "state", "district_clean"])[["age_0_5", "age_5_17", "age_18_greater"]].sum().reset_index()
    b = biometric.groupby(["month", "state", "district_clean"])[["bio_age_5_17", "bio_age_17_"]].sum().reset_index()
    model_df = e.merge(b, on=["month", "state", "district_clean"], how="left").fillna(0).sort_values(["state", "district_clean", "month"])
    base_cols = ["age_0_5", "age_5_17", "age_18_greater", "bio_age_5_17", "bio_age_17_"]
    for lag in [1, 3, 6]:
        for col in base_cols:
            model_df[f"{col}_lag_{lag}"] = model_df.groupby(["state", "district_clean"])[col].shift(lag)

    feature_cols = [
        "age_0_5", "age_5_17", "age_18_greater",
        "age_0_5_lag_1", "age_5_17_lag_1", "age_18_greater_lag_1",
        "age_0_5_lag_3", "age_5_17_lag_3", "age_18_greater_lag_3",
        "bio_age_5_17_lag_1", "bio_age_17__lag_1",
        "bio_age_5_17_lag_3", "bio_age_17__lag_3",
        "bio_age_5_17_lag_6", "bio_age_17__lag_6",
    ]

    model_df = model_df.dropna(subset=feature_cols + ["bio_age_5_17"]).copy()
    X, y = model_df[feature_cols], model_df["bio_age_5_17"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    latest_month = model_df["month"].max()
    latest = model_df[model_df["month"] == latest_month].copy()
    latest["predicted_bio_age_5_17"] = model.predict(latest[feature_cols])
    latest["forecast_gap"] = latest["predicted_bio_age_5_17"] - latest["bio_age_5_17"]
    latest = latest.sort_values("predicted_bio_age_5_17", ascending=False)
    fi = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    metrics = {"mae": float(mean_absolute_error(y_test, y_pred)), "r2": float(r2_score(y_test, y_pred)), "latest_month": latest_month}
    return latest, metrics, fi


def rural_urban_child_share(enrolment: pd.DataFrame) -> pd.DataFrame:
    grouped = enrolment.groupby(["state", "district_clean", "area_type"])[["age_0_5", "age_5_17", "age_18_greater"]].sum().reset_index()
    grouped["total_enrolments"] = grouped[["age_0_5", "age_5_17", "age_18_greater"]].sum(axis=1)
    grouped = grouped[grouped["area_type"].isin(["Urban", "Rural"])].copy()
    grouped["child_enrolment_share"] = grouped["age_0_5"] / grouped["total_enrolments"]
    grouped = grouped.replace([np.inf, -np.inf], 0).fillna(0)
    grouped["hidden_exclusion_flag"] = np.where(
        (grouped["area_type"] == "Urban") & (grouped["total_enrolments"] >= grouped["total_enrolments"].median()) & (grouped["child_enrolment_share"] < grouped["child_enrolment_share"].quantile(0.2)),
        "Watchlist",
        "Normal",
    )
    return grouped


def tn_capacity_analysis(district_ratios: pd.DataFrame, zones: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    tn = district_ratios[district_ratios["state"] == "Tamil Nadu"].copy()
    out = tn.merge(zones[["district_clean", "centre_count"]], on="district_clean", how="left")
    out["centre_count"] = out["centre_count"].fillna(0)
    out["total_demand"] = out["total_enrolments"] + out["total_demographic_updates"] + out["total_biometric_updates"]
    out["service_load_per_centre"] = (out["total_demand"] / out["centre_count"]).replace([np.inf, -np.inf], 0).fillna(0)
    threshold = float(out["service_load_per_centre"].mean() + out["service_load_per_centre"].std())
    return out.sort_values("service_load_per_centre", ascending=False), threshold


def district_service_metrics(enrolment: pd.DataFrame, demographic: pd.DataFrame, biometric: pd.DataFrame) -> pd.DataFrame:
    e = enrolment.groupby(["state", "district_clean"])[["age_0_5", "age_5_17", "age_18_greater"]].sum()
    e["total_enrolments"] = e.sum(axis=1)
    d = demographic.groupby(["state", "district_clean"])[["demo_age_5_17", "demo_age_17_"]].sum()
    d["total_demographic_updates"] = d.sum(axis=1)
    b = biometric.groupby(["state", "district_clean"])[["bio_age_5_17", "bio_age_17_"]].sum()
    b["total_biometric_updates"] = b.sum(axis=1)
    return e.merge(d, left_index=True, right_index=True, how="outer").merge(b, left_index=True, right_index=True, how="outer").fillna(0).reset_index()


def cluster_districts(service_metrics: pd.DataFrame) -> pd.DataFrame:
    features = ["total_enrolments", "total_demographic_updates", "total_biometric_updates"]
    scaled = StandardScaler().fit_transform(service_metrics[features])
    out = service_metrics.copy()
    out["cluster_agnes"] = AgglomerativeClustering(n_clusters=5, linkage="ward").fit_predict(scaled)
    if BisectingKMeans is not None:
        diana = BisectingKMeans(n_clusters=5, random_state=42)
    else:  # pragma: no cover
        diana = KMeans(n_clusters=5, random_state=42, n_init=10)
    out["cluster_diana"] = diana.fit_predict(scaled)
    out["cluster_dbscan"] = DBSCAN(eps=0.5, min_samples=5).fit_predict(scaled)
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
    out["pca_1"], out["pca_2"] = coords[:, 0], coords[:, 1]
    return out
