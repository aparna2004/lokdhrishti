from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import (
    cluster_districts,
    compute_kpis,
    district_age_enrolment,
    district_age_heatmap,
    district_service_metrics,
    district_update_ratios,
    forecast_biometric_demand,
    high_update_demand,
    load_india_geojson,
    low_childhood_trend,
    monthly_service_balance,
    rural_urban_child_share,
    state_late_childhood,
    tn_capacity_analysis,
)
from src.data_loader import discover_data_root, load_data, title_case_location

st.set_page_config(page_title="LokDrishti", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
    .block-container {max-width: 1500px; padding-top: 1rem; padding-bottom: 2rem;}
    .title-wrap {padding: .2rem 0 1rem 0;}
    .title-wrap h1 {font-size: 2rem; margin: 0; font-weight: 700;}
    .title-wrap p {margin: .2rem 0 0 0; color: #aeb8cc; font-size: 1rem;}
    .section-card {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        background: rgba(255,255,255,0.025);
        margin-top: .4rem;
    }
    .section-card h4 {margin: 0 0 .6rem 0; font-size: 1rem;}
    .section-card p {margin: 0 0 .45rem 0; color: #dbe3f4; line-height: 1.55;}
    .label-muted {color: #9aa7c0; font-size: .9rem;}
    div[data-testid='stMetric'] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 12px 14px;
        border-radius: 16px;
    }
    div[data-testid='stSidebar'] .stRadio label {padding-bottom: .25rem;}
</style>
""",
    unsafe_allow_html=True,
)

base_dir = Path(__file__).resolve().parent
DATA_ROOT = discover_data_root(base_dir)
data = load_data(str(DATA_ROOT))

enrolment = data["enrolment"]
demographic = data["demographic"]
biometric = data["biometric"]
zones = data["zones"]

kpis = compute_kpis(enrolment, demographic, biometric)
monthly_balance = monthly_service_balance(enrolment, biometric)
district_age = district_age_enrolment(enrolment)
state_late = state_late_childhood(enrolment)
district_ratios = district_update_ratios(enrolment, demographic, biometric)
high_update, update_threshold = high_update_demand(district_ratios)
forecast_df, forecast_metrics, feature_importance = forecast_biometric_demand(enrolment, biometric)
area_share = rural_urban_child_share(enrolment)
tn_all, tn_threshold = tn_capacity_analysis(district_ratios, zones)
service_metrics = district_service_metrics(enrolment, demographic, biometric)
clustered_df = cluster_districts(service_metrics)
india_geojson = load_india_geojson(str(base_dir))

state_options = ["All States"] + sorted(enrolment["state"].dropna().unique().tolist())
section_options = [
    "Executive overview",
    "Coverage maturity",
    "Update demand",
    "Forecasting",
    "Rural vs urban",
    "Tamil Nadu capacity",
    "Clustering",
    "Data explorer",
]


def titleize_series(df: pd.DataFrame, col: str, new_col: str = "label") -> pd.DataFrame:
    out = df.copy()
    out[new_col] = out[col].map(title_case_location)
    return out


def render_text_block(title: str, paragraphs: list[str]) -> None:
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    st.markdown(f"<div class='section-card'><h4>{title}</h4>{body}</div>", unsafe_allow_html=True)


st.sidebar.title("LokDrishti")
section = st.sidebar.radio("Navigate", section_options, label_visibility="collapsed")
show_tables = st.sidebar.toggle("Show tables", True)
st.sidebar.caption(f"Coverage window: {kpis['date_min'].date()} to {kpis['date_max'].date()}")

st.markdown(
    """
<div class='title-wrap'>
    <h1>LokDrishti</h1>
    <p>National and district-level analysis of Aadhaar enrolment, update demand, service load, and operating patterns.</p>
</div>
""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Districts", f"{kpis['districts']:,}")
m2.metric("States / UTs", f"{kpis['states']:,}")
m3.metric("Enrolments", f"{kpis['total_enrolments']:,}")
m4.metric("Demographic updates", f"{kpis['total_demographic_updates']:,}")
m5.metric("Biometric updates", f"{kpis['total_biometric_updates']:,}")


if section == "Executive overview":
    top_states = state_late.sort_values("late_childhood_share", ascending=False).head(5)[["state", "late_childhood_share"]]
    low_child_district = district_age.sort_values("share_0_5").head(1).iloc[0]
    hotspot = high_update.head(1).iloc[0]
    tn_peak = tn_all.head(1).iloc[0]

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig = px.line(
            monthly_balance,
            x="month",
            y=["total_enrolments", "total_biometric_updates"],
            markers=True,
            title="Monthly service activity",
            labels={"value": "Volume", "month": "Month", "variable": "Service"},
        )
        fig.update_layout(height=430, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(
            top_states,
            x="late_childhood_share",
            y="state",
            orientation="h",
            title="States with highest late-childhood enrolment share",
            labels={"late_childhood_share": "5–17 share", "state": "State"},
        )
        fig2.update_layout(height=430, yaxis={"categoryorder": "total ascending"}, xaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    render_text_block(
        "Executive summary",
        [
            f"The overall service profile is update-heavy: biometric updates exceeded new enrolments in {kpis['biometric_dominant_months']} of {kpis['months']} observed months, indicating a strong operational tilt toward record maintenance.",
            f"Coverage maturity varies sharply across districts. The lowest observed early-childhood enrolment share appears in {title_case_location(low_child_district['district_clean'])}, {low_child_district['state']}, where the 0–5 share is {low_child_district['share_0_5']:.1%}.",
            f"District update pressure is most pronounced in {title_case_location(hotspot['district_clean'])}, {hotspot['state']}, where the total update-to-enrolment ratio reaches {hotspot['total_update_ratio']:.2f}x.",
            f"Within Tamil Nadu, {title_case_location(tn_peak['district_clean'])} shows the highest estimated service load per centre at {tn_peak['service_load_per_centre']:,.0f}, which makes it a priority candidate for capacity review.",
        ],
    )

elif section == "Coverage maturity":
    selected_state = st.sidebar.selectbox("State", state_options, index=0)
    min_enrolments = st.sidebar.slider("Minimum district enrolments", 0, 10000, 100, 100)
    top_n = st.sidebar.slider("District count", 5, 40, 12, 1)

    district_age_view = district_age.copy() if selected_state == "All States" else district_age[district_age["state"] == selected_state].copy()
    state_late_view = state_late.copy() if selected_state == "All States" else state_late[state_late["state"] == selected_state].copy()
    district_age_view = district_age_view[district_age_view["total_enrolments"] >= min_enrolments].copy()

    c1, c2 = st.columns([1.2, 1])
    with c1:
        if selected_state == "All States" and india_geojson is not None:
            fig = px.choropleth(
                state_late_view,
                geojson=india_geojson,
                featureidkey="properties.NAME_1",
                locations="state_geo",
                color="late_childhood_share",
                color_continuous_scale="Reds",
                hover_name="state",
                hover_data={"late_childhood_share": ":.2%", "early_childhood_share": ":.2%", "total_enrolments": ":,"},
                title="State-wise late-childhood enrolment share",
            )
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(height=520, margin=dict(l=0, r=0, t=60, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter(
                state_late if selected_state == "All States" else state_late_view,
                x="early_childhood_share",
                y="late_childhood_share",
                size="total_enrolments",
                hover_name="state",
                title="Coverage maturity by state",
                labels={"early_childhood_share": "0–5 share", "late_childhood_share": "5–17 share"},
            )
            fig.update_layout(height=520, xaxis_tickformat=".0%", yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        heatmap_df = district_age_heatmap(district_age_view, min(40, max(10, top_n * 2)))
        fig2 = px.imshow(
            heatmap_df,
            aspect="auto",
            color_continuous_scale="Viridis",
            labels=dict(x="Age segment", y="District", color="Share"),
            title="District age-share heatmap",
        )
        fig2.update_layout(height=520)
        st.plotly_chart(fig2, use_container_width=True)

    b1, b2 = st.columns([1.1, 1])
    low_child = titleize_series(district_age_view.sort_values("share_0_5").head(top_n), "district_clean", "district_label")
    with b1:
        fig3 = px.bar(
            low_child,
            x="share_0_5",
            y="district_label",
            color="state",
            orientation="h",
            title="Lowest early-childhood coverage districts",
            labels={"share_0_5": "0–5 share", "district_label": "District"},
        )
        fig3.update_layout(height=470, yaxis={"categoryorder": "total ascending"}, xaxis_tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)
    with b2:
        trend = low_childhood_trend(enrolment if selected_state == "All States" else enrolment[enrolment["state"] == selected_state], district_age_view, min(12, top_n))
        fig4 = px.line(trend, x="month", y="age_0_5", markers=True, title="Monthly trend for low-coverage districts")
        fig4.update_layout(height=470, xaxis_title="Month", yaxis_title="Age 0–5 enrolments")
        st.plotly_chart(fig4, use_container_width=True)

    avg_early = district_age_view["share_0_5"].mean() if len(district_age_view) else 0
    weakest = low_child.iloc[0] if len(low_child) else None
    strongest_state = state_late.sort_values("late_childhood_share", ascending=False).iloc[0]
    render_text_block(
        "Key insights",
        [
            f"Early-childhood enrolment remains uneven across the country. The average district-level 0–5 enrolment share in the current filtered view is {avg_early:.1%}, but a visible tail of districts sits materially below this level.",
            f"The weakest district in the current view is {weakest['district_label']}, {weakest['state']}, with only {weakest['share_0_5']:.1%} of enrolments coming from the 0–5 cohort." if weakest is not None else "District coverage maturity varies widely across the filtered view.",
            f"At the state level, {strongest_state['state']} shows the highest late-childhood enrolment share at {strongest_state['late_childhood_share']:.1%}, which suggests delayed entry into the Aadhaar lifecycle relative to better-matured states.",
            "Taken together, the map, district heatmap, and trend views point to localized enrolment-access gaps rather than a uniform national pattern, making district-focused intervention design more appropriate than blanket action.",
        ],
    )
    if show_tables:
        st.dataframe(low_child[["district_label", "state", "share_0_5", "share_5_17", "share_18_greater", "total_enrolments"]], use_container_width=True)

elif section == "Update demand":
    selected_state = st.sidebar.selectbox("State", state_options, index=0)
    min_enrolments = st.sidebar.slider("Minimum district enrolments", 0, 10000, 100, 100)
    top_n = st.sidebar.slider("District count", 5, 40, 15, 1)

    ratios_view = district_ratios.copy() if selected_state == "All States" else district_ratios[district_ratios["state"] == selected_state].copy()
    ratios_view = ratios_view[ratios_view["total_enrolments"] >= min_enrolments].copy()
    high_view = ratios_view[ratios_view["total_update_ratio"] > update_threshold].sort_values("total_update_ratio", ascending=False).head(top_n)
    high_view = titleize_series(high_view, "district_clean", "district_label")

    c1, c2 = st.columns([1.15, 1])
    with c1:
        fig = px.scatter(
            ratios_view,
            x="total_enrolments",
            y="total_updates",
            size="total_update_ratio",
            color="update_mix",
            hover_name="district_clean",
            title="District enrolments vs total updates",
            labels={"total_updates": "Total updates", "total_enrolments": "Total enrolments"},
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(
            high_view,
            x="total_update_ratio",
            y="district_label",
            color="update_mix",
            orientation="h",
            title="Highest update-pressure districts",
            labels={"total_update_ratio": "Updates per enrolment", "district_label": "District"},
        )
        fig2.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

    monthly = monthly_balance.copy()
    fig3 = px.area(
        monthly,
        x="month",
        y="update_share",
        title="Monthly biometric update share of combined activity",
        labels={"update_share": "Biometric update share", "month": "Month"},
    )
    fig3.update_layout(height=360, yaxis_tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)

    hotspot = high_view.iloc[0] if len(high_view) else None
    update_dominant_months = int((monthly_balance["update_share"] > 0.5).sum())
    render_text_block(
        "Key insights",
        [
            f"High update demand is defined here as districts whose update-to-enrolment ratio exceeds {update_threshold:.2f}x. This flags places where service pressure is driven more by record maintenance than by net enrolment expansion.",
            f"The current filtered view is led by {hotspot['district_label']}, {hotspot['state']}, with a total update ratio of {hotspot['total_update_ratio']:.2f}x and a {hotspot['update_mix'].lower()} demand profile." if hotspot is not None else "The filtered view does not currently surface a district above the threshold.",
            f"Biometric updates dominate the monthly mix in {update_dominant_months} out of {kpis['months']} observed months, reinforcing that this is a structural service pattern rather than an isolated spike.",
            "Operationally, districts in the upper-right zone of the scatter plot deserve the closest attention because they combine a meaningful enrolment base with a large volume of updates, making them harder to absorb through routine capacity alone.",
        ],
    )
    if show_tables:
        st.dataframe(high_view[["district_label", "state", "total_enrolments", "total_demographic_updates", "total_biometric_updates", "total_update_ratio", "update_mix"]], use_container_width=True)

elif section == "Forecasting":
    selected_state = st.sidebar.selectbox("State", state_options, index=0)
    top_n = st.sidebar.slider("Forecast district count", 5, 30, 12, 1)
    forecast_view = forecast_df.copy() if selected_state == "All States" else forecast_df[forecast_df["state"] == selected_state].copy()
    forecast_view = titleize_series(forecast_view.head(top_n), "district_clean", "district_label")
    fi_view = feature_importance.head(10).copy()
    fi_view["feature_label"] = fi_view["feature"].str.replace("_", " ").str.title()

    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = px.bar(
            forecast_view,
            x="predicted_bio_age_5_17",
            y="district_label",
            color="state",
            orientation="h",
            title=f"Forecasted biometric demand for {forecast_metrics['latest_month']}",
            labels={"predicted_bio_age_5_17": "Predicted biometric updates", "district_label": "District"},
        )
        fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(
            fi_view,
            x="importance",
            y="feature_label",
            orientation="h",
            title="Top model drivers",
            labels={"importance": "Importance", "feature_label": "Feature"},
        )
        fig2.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        forecast_view,
        x="bio_age_5_17",
        y="predicted_bio_age_5_17",
        size="age_5_17",
        color="state",
        hover_name="district_label",
        title="Observed vs forecasted biometric demand",
        labels={"bio_age_5_17": "Observed current biometric updates", "predicted_bio_age_5_17": "Predicted biometric updates"},
    )
    fig3.update_layout(height=360)
    st.plotly_chart(fig3, use_container_width=True)

    lead = forecast_view.iloc[0] if len(forecast_view) else None
    render_text_block(
        "Key insights",
        [
            f"The forecasting model reports an MAE of {forecast_metrics['mae']:.2f} and an R² of {forecast_metrics['r2']:.2f} on the held-out split, indicating a strong fit to the available district-month patterns.",
            f"The highest projected demand in the current filtered view is in {lead['district_label']}, {lead['state']}, with an estimated {lead['predicted_bio_age_5_17']:,.0f} biometric updates for the 5–17 cohort." if lead is not None else "No forecast rows are available for the current filtered view.",
            "Feature-importance results show that current enrolment composition and recent biometric history together explain most of the variation in next-step biometric demand, which is consistent with a cohort-driven service cycle.",
            "These forecasts are best interpreted as demand signals for planning and prioritisation, especially when combined with the hotspot and capacity tabs rather than used in isolation.",
        ],
    )
    if show_tables:
        st.dataframe(forecast_view[["district_label", "state", "age_5_17", "bio_age_5_17", "predicted_bio_age_5_17", "forecast_gap"]], use_container_width=True)

elif section == "Rural vs urban":
    selected_state = st.sidebar.selectbox("State", state_options, index=0)
    top_n = st.sidebar.slider("District count", 5, 30, 12, 1)
    min_total = st.sidebar.slider("Minimum total enrolments", 0, 10000, 100, 100)

    area_view = area_share.copy() if selected_state == "All States" else area_share[area_share["state"] == selected_state].copy()
    area_view = area_view[area_view["total_enrolments"] >= min_total].copy()
    urban_watch = titleize_series(
        area_view[(area_view["area_type"] == "Urban") & (area_view["hidden_exclusion_flag"] == "Watchlist")].sort_values("child_enrolment_share").head(top_n),
        "district_clean",
        "district_label",
    )

    c1, c2 = st.columns([1, 1.15])
    with c1:
        fig = px.box(
            area_view,
            x="area_type",
            y="child_enrolment_share",
            color="area_type",
            title="Child enrolment share by area type",
            labels={"child_enrolment_share": "0–5 share", "area_type": "Area type"},
        )
        fig.update_layout(height=470, yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.scatter(
            area_view[area_view["area_type"] == "Urban"],
            x="total_enrolments",
            y="child_enrolment_share",
            color="hidden_exclusion_flag",
            size="total_enrolments",
            hover_name="district_clean",
            title="Urban districts and hidden-exclusion watchlist",
            labels={"child_enrolment_share": "0–5 share", "total_enrolments": "Total enrolments"},
        )
        fig2.update_layout(height=470, yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(
        urban_watch,
        x="child_enrolment_share",
        y="district_label",
        orientation="h",
        color="state",
        title="Urban watchlist districts with lowest child enrolment share",
        labels={"child_enrolment_share": "0–5 share", "district_label": "District"},
    )
    fig3.update_layout(height=360, yaxis={"categoryorder": "total ascending"}, xaxis_tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)

    urban_med = area_view[area_view["area_type"] == "Urban"]["child_enrolment_share"].median()
    rural_med = area_view[area_view["area_type"] == "Rural"]["child_enrolment_share"].median()
    lead = urban_watch.iloc[0] if len(urban_watch) else None
    render_text_block(
        "Key insights",
        [
            f"The median child-enrolment share in the current filtered view is {urban_med:.1%} for urban areas and {rural_med:.1%} for rural areas, showing that urban access does not automatically translate into better early-childhood coverage.",
            f"The watchlist is headed by {lead['district_label']}, {lead['state']}, where the urban child-enrolment share is only {lead['child_enrolment_share']:.1%} despite a meaningful service base." if lead is not None else "No urban watchlist districts appear in the current filtered view.",
            "Urban districts with moderate or high total enrolments but weak 0–5 participation are important because they can hide exclusion pockets behind otherwise strong aggregate service activity.",
            "This pattern is operationally relevant for targeted outreach through local health, child-welfare, and community channels rather than assuming that city environments are uniformly better covered.",
        ],
    )
    if show_tables:
        st.dataframe(urban_watch[["district_label", "state", "total_enrolments", "child_enrolment_share", "hidden_exclusion_flag"]], use_container_width=True)

elif section == "Tamil Nadu capacity":
    top_n = st.sidebar.slider("District count", 5, 30, 12, 1)
    min_centres = st.sidebar.slider("Minimum centres", 0, int(max(1, tn_all["centre_count"].max())), 0, 1)
    tn_view = titleize_series(tn_all[tn_all["centre_count"] >= min_centres].copy(), "district_clean", "district_label")
    top_view = tn_view.head(top_n)

    c1, c2 = st.columns([1.05, 1])
    with c1:
        fig = px.bar(
            top_view,
            x="service_load_per_centre",
            y="district_label",
            orientation="h",
            color="centre_count",
            title="Service load per centre",
            labels={"service_load_per_centre": "Load per centre", "district_label": "District", "centre_count": "Centres"},
        )
        fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.scatter(
            tn_view,
            x="centre_count",
            y="service_load_per_centre",
            size="total_demand",
            color="total_demand",
            hover_name="district_label",
            title="Demand and centre footprint",
            labels={"centre_count": "Number of centres", "service_load_per_centre": "Load per centre"},
        )
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)

    peak = top_view.iloc[0] if len(top_view) else None
    flagged = int((tn_view["service_load_per_centre"] > tn_threshold).sum())
    render_text_block(
        "Key insights",
        [
            f"Tamil Nadu districts above {tn_threshold:,.0f} service events per centre are treated as higher-pressure locations in this view. There are currently {flagged} districts above that level.",
            f"The highest service load per centre is observed in {peak['district_label']} at {peak['service_load_per_centre']:,.0f}, with {int(peak['centre_count'])} centres supporting an estimated total demand of {peak['total_demand']:,.0f}." if peak is not None else "No Tamil Nadu district is visible in the current filtered view.",
            "The scatter plot helps separate two different cases: districts with genuinely high demand volumes and districts where moderate demand becomes problematic because the centre base is thin.",
            "This view is most useful for capacity planning, centre expansion review, and field prioritisation within Tamil Nadu rather than for national comparison.",
        ],
    )
    if show_tables:
        st.dataframe(top_view[["district_label", "centre_count", "total_demand", "service_load_per_centre", "total_update_ratio"]], use_container_width=True)

elif section == "Clustering":
    selected_state = st.sidebar.selectbox("State", state_options, index=0)
    cluster_method = st.sidebar.selectbox(
        "Cluster method",
        ["cluster_dbscan", "cluster_agnes", "cluster_diana"],
        format_func=lambda x: x.replace("cluster_", "").upper(),
    )
    cluster_view = clustered_df.copy() if selected_state == "All States" else clustered_df[clustered_df["state"] == selected_state].copy()
    cluster_view = titleize_series(cluster_view, "district_clean", "district_label")
    summary = cluster_view.groupby(cluster_method).size().reset_index(name="district_count").sort_values("district_count", ascending=False)

    c1, c2 = st.columns([1.2, 0.9])
    with c1:
        fig = px.scatter(
            cluster_view,
            x="pca_1",
            y="pca_2",
            color=cluster_method,
            hover_name="district_label",
            title=f"Cluster layout ({cluster_method.replace('cluster_', '').upper()})",
            labels={"pca_1": "PCA 1", "pca_2": "PCA 2", cluster_method: "Cluster"},
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(
            summary,
            x=cluster_method,
            y="district_count",
            title="Cluster sizes",
            labels={cluster_method: "Cluster", "district_count": "District count"},
        )
        fig2.update_layout(height=520)
        st.plotly_chart(fig2, use_container_width=True)

    if cluster_method == "cluster_dbscan":
        outliers = cluster_view[cluster_view[cluster_method] == -1]
        insight_line = f"DBSCAN identifies {len(outliers)} districts as outliers in the current filtered view, which is useful for surfacing uncommon service-demand profiles."
    else:
        largest = summary.iloc[0]
        insight_line = f"The largest cluster in the current view is cluster {largest[cluster_method]} with {int(largest['district_count'])} districts, which indicates the dominant service-profile archetype under this method."

    render_text_block(
        "Key insights",
        [
            "Clustering groups districts by their combined enrolment, demographic-update, and biometric-update profiles, allowing service patterns to be compared as operational archetypes rather than as isolated points.",
            insight_line,
            "Agglomerative and bisecting approaches are better suited for broad segmentation, while DBSCAN is more useful for highlighting outliers and unusual density patterns that may merit separate field review.",
            "The PCA plot should be read as a relative structure map: districts that sit far from dense central groupings are not necessarily problematic, but they do represent materially different operating contexts.",
        ],
    )
    if show_tables:
        st.dataframe(cluster_view[["district_label", "state", "total_enrolments", "total_demographic_updates", "total_biometric_updates", cluster_method]], use_container_width=True)

elif section == "Data explorer":
    dataset_name = st.sidebar.selectbox("Dataset", ["Enrolment", "Demographic", "Biometric", "District ratios", "Forecast", "Tamil Nadu capacity", "Clusters"])
    mapping = {
        "Enrolment": enrolment,
        "Demographic": demographic,
        "Biometric": biometric,
        "District ratios": district_ratios,
        "Forecast": forecast_df,
        "Tamil Nadu capacity": tn_all,
        "Clusters": clustered_df,
    }
    df = mapping[dataset_name].copy()
    st.subheader(dataset_name)
    st.caption(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    st.dataframe(df, use_container_width=True, height=560)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download current table as CSV", csv, file_name=f"{dataset_name.lower().replace(' ', '_')}.csv", mime="text/csv")
