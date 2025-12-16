import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from scipy import stats

# =========================
# Config
# =========================
st.set_page_config(page_title="PHQ-4 Weekly Feature Explorer", layout="wide")

TARGET = "avg_phq4_score"

A_FEATURES = [
    "avg_sleep_duration",
    "avg_unlock_num_ep0_full", "avg_unlock_num_ep1_night", "avg_unlock_num_ep2_day", "avg_unlock_num_ep3_evening",
    "avg_loc_visits_day",
    "avg_loc_food_dur", "avg_loc_health_dur", "avg_loc_home_dur",
    "avg_loc_leisure_dur", "avg_loc_other_dorm_dur", "avg_loc_self_dorm_dur",
    "avg_loc_social_dur", "avg_loc_study_dur", "avg_loc_workout_dur",
    "avg_loc_worship_dur",
    "avg_steps_ep0_full", "avg_steps_ep1_night", "avg_steps_ep2_day", "avg_steps_ep3_evening",
    "avg_unlock_duration_ep0_full_hrs", "avg_unlock_duration_ep1_night_hrs",
    "avg_unlock_duration_ep2_day_hrs", "avg_unlock_duration_ep3_evening_hrs",
    "sleep_start_num", "sleep_end_num",
]
S_FEATURES = ["avg_stress", "avg_covid_concern", "avg_social_level", "avg_social_distancing"]
R_FEATURES = ["R1_sleep_midpoint_std", "R2_daily_steps_std", "R3_unlock_duration_std", "R4_sleep_consistency"]
B_FEATURES = ["B1_recovery_time_balance", "B2_night_cognitive_load", "B3_daytime_fragmentation"]

CATEGORY_MAP = {
    "A (Absolute Load)": A_FEATURES,
    "S (stress/survey)": S_FEATURES,
    "R (Rhythm/Instability)": R_FEATURES,
    "B (Balance/Fragmentation)": B_FEATURES,
}

ID_COLS_CANDIDATES = ["uid", "year", "week", "gender", "race"]

# =========================
# Helpers
# =========================
def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    # Cohen's d with pooled std; for unequal n; robust enough for reporting
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    if sp == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / sp

def eta_squared_oneway(groups: list[np.ndarray]) -> float:
    # η² = SS_between / SS_total
    # groups: list of arrays
    clean_groups = [g[~np.isnan(g)] for g in groups if len(g[~np.isnan(g)]) > 0]
    if len(clean_groups) < 2:
        return np.nan
    all_vals = np.concatenate(clean_groups)
    if len(all_vals) < 2:
        return np.nan
    grand_mean = np.mean(all_vals)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    ss_between = np.sum([len(g) * (np.mean(g) - grand_mean) ** 2 for g in clean_groups])
    if ss_total == 0:
        return np.nan
    return ss_between / ss_total

# =========================
# Data loading
# =========================
@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    for c in ["year", "week"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year", "week"]).copy()
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].astype(int)

    df = df[(df["week"] >= 1) & (df["week"] <= 53)].copy()

    df["week_start_date"] = pd.to_datetime(
        df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )
    df["month"] = df["week_start_date"].dt.month
    df["month_name"] = df["week_start_date"].dt.strftime("%b")
    return df

st.title("Weekly Features → PHQ-4 Explorer")

with st.sidebar:
    st.header("Data")
    csv_path = st.text_input("CSV path", value="final_weekly.csv")
    st.caption("Expected: year, week, gender, race, avg_phq4_score + feature columns")

try:
    df = load_data(csv_path)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

if TARGET not in df.columns:
    st.error(f"Target column '{TARGET}' not found in CSV.")
    st.stop()

# =========================
# Sidebar filters
# =========================
with st.sidebar:
    st.header("Filters")

    years = sorted(df["year"].dropna().unique().tolist())
    sel_years = st.multiselect("Year", years, default=years)

    months = sorted(df["month"].dropna().unique().tolist())
    month_labels = {m: pd.Timestamp(year=2000, month=m, day=1).strftime("%b") for m in months}
    sel_months = st.multiselect(
        "Month",
        options=months,
        default=months,
        format_func=lambda m: f"{m:02d} ({month_labels.get(m,'')})",
    )

    if "gender" in df.columns:
        genders = sorted(df["gender"].dropna().astype(str).unique().tolist())
        sel_genders = st.multiselect("Gender", genders, default=genders)
    else:
        sel_genders = None

    if "race" in df.columns:
        races = sorted(df["race"].dropna().astype(str).unique().tolist())
        sel_races = st.multiselect("Race", races, default=races)
    else:
        sel_races = None

# Apply filters
f = df.copy()
f = f[f["year"].isin(sel_years)]
f = f[f["month"].isin(sel_months)]
if sel_genders is not None and "gender" in f.columns:
    f = f[f["gender"].astype(str).isin(sel_genders)]
if sel_races is not None and "race" in f.columns:
    f = f[f["race"].astype(str).isin(sel_races)]

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (filtered)", f"{len(f):,}")
c2.metric("Unique users", f"{f['uid'].nunique():,}" if "uid" in f.columns else "—")
c3.metric("Year range", f"{f['year'].min()}–{f['year'].max()}" if len(f) else "—")
c4.metric("Weeks covered", f"{f['week'].min()}–{f['week'].max()}" if len(f) else "—")

if len(f) == 0:
    st.warning("No data after filtering. Adjust filters.")
    st.stop()

# =========================
# Feature selection UI
# =========================
st.subheader("1) Pick feature(s) by category")

left, right = st.columns([1, 1])

with left:
    sel_categories = st.multiselect(
        "Categories",
        options=list(CATEGORY_MAP.keys()),
        default=["A (Absolute Load)"],
    )

    selected_features = []
    for cat in sel_categories:
        for feat in CATEGORY_MAP[cat]:
            if feat in f.columns:
                selected_features.append(feat)
    selected_features = sorted(list(dict.fromkeys(selected_features)))

with right:
    if selected_features:
        st.write("Available features (existing in CSV):")
        st.code(", ".join(selected_features), language="text")
    else:
        st.warning("No selected features found in the CSV.")
        st.stop()

# Common options for color
COLOR_OPTIONS = ["(none)"] + [c for c in ["gender", "race", "year", "month_name", "covid_period"] if c in f.columns]

# =========================
# 2) Single feature vs target
# =========================
st.subheader("2) Feature → Target plot (avg_phq4_score)")

plot_col1, plot_col2 = st.columns([1, 1])
with plot_col1:
    x_feat = st.selectbox("X feature", options=selected_features, index=0)
    color_by = st.selectbox("Color by", options=COLOR_OPTIONS, index=0)
    trendline = st.checkbox("Add trendline (OLS)", value=True)

with plot_col2:
    opacity = st.slider("Point opacity", 0.1, 1.0, 0.5, 0.05)
    show_marginals = st.checkbox("Show marginal distributions", value=False)


scatter_cols = [x_feat, TARGET, "year", "week", "week_start_date"]
if color_by != "(none)":
    scatter_cols.append(color_by)

# ✅ de-duplicate while preserving order (fix narwhals DuplicateError)
scatter_cols = list(dict.fromkeys(scatter_cols))

scatter_df = f.loc[:, scatter_cols].copy()

scatter_df[x_feat] = safe_numeric(scatter_df[x_feat])
scatter_df[TARGET] = safe_numeric(scatter_df[TARGET])
scatter_df = scatter_df.dropna(subset=[x_feat, TARGET])

if len(scatter_df) == 0:
    st.warning("No non-missing rows for selected feature & target.")
else:
    fig_scatter = px.scatter(
        scatter_df,
        x=x_feat,
        y=TARGET,
        color=None if color_by == "(none)" else color_by,
        trendline="ols" if trendline else None,
        opacity=opacity,
        hover_data=["year", "week", "week_start_date"],
        marginal_x="histogram" if show_marginals else None,
        marginal_y="histogram" if show_marginals else None,
    )
    fig_scatter.update_layout(height=520)
    st.plotly_chart(fig_scatter, use_container_width=True)

# =========================
# 3) Distributions
# =========================
st.subheader("3) Distributions")

dist_left, dist_right = st.columns([1, 1])
with dist_left:
    dist_feats = st.multiselect("Pick features", options=selected_features, default=[x_feat])
    dist_kind = st.radio("View", options=["Histogram", "Boxplot"], horizontal=True)
with dist_right:
    bin_count = st.slider("Histogram bins", 10, 120, 40, 5)
    facet_by = st.selectbox("Facet by", options=COLOR_OPTIONS, index=0)

for feat in dist_feats:
    ddf_cols = [feat]
    if facet_by != "(none)":
        ddf_cols.append(facet_by)
    ddf = f[ddf_cols].copy()
    ddf[feat] = safe_numeric(ddf[feat])
    ddf = ddf.dropna(subset=[feat])

    if len(ddf) == 0:
        st.info(f"{feat}: no data after dropping NA.")
        continue

    if dist_kind == "Histogram":
        fig = px.histogram(
            ddf,
            x=feat,
            nbins=bin_count,
            color=None if facet_by == "(none)" else facet_by,
        )
    else:
        fig = px.box(
            ddf,
            y=feat,
            color=None if facet_by == "(none)" else facet_by,
            points="outliers",
        )
    fig.update_layout(title=f"Distribution: {feat}", height=380)
    st.plotly_chart(fig, use_container_width=True)

# Target distribution
st.markdown("**Target distribution: avg_phq4_score**")
tcols = [TARGET]
if facet_by != "(none)":
    tcols.append(facet_by)
tddf = f[tcols].copy()
tddf[TARGET] = safe_numeric(tddf[TARGET])
tddf = tddf.dropna(subset=[TARGET])
fig_t = px.histogram(tddf, x=TARGET, nbins=bin_count, color=None if facet_by == "(none)" else facet_by)
fig_t.update_layout(height=320)
st.plotly_chart(fig_t, use_container_width=True)

# =========================
# 4) Correlation to Target
# =========================
st.subheader("4) Correlation (Pearson) to avg_phq4_score within selected categories")

corr_df = f[selected_features + [TARGET]].copy()
for col in selected_features + [TARGET]:
    corr_df[col] = safe_numeric(corr_df[col])
corr_df = corr_df.dropna(subset=[TARGET])

cors = []
for feat in selected_features:
    tmp = corr_df[[feat, TARGET]].dropna()
    if len(tmp) < 30:
        cors.append((feat, np.nan, len(tmp)))
        continue
    r = tmp[feat].corr(tmp[TARGET])
    cors.append((feat, r, len(tmp)))

cors_out = pd.DataFrame(cors, columns=["feature", "pearson_r", "n_used"])
cors_out["abs_r"] = cors_out["pearson_r"].abs()
cors_rank = cors_out.sort_values(by="abs_r", ascending=False)

topk = st.slider("Show top-K (by |r|)", 5, min(50, len(cors_rank)), min(20, len(cors_rank)))
show_df = cors_rank.head(topk)

cA, cB = st.columns([1, 1])
with cA:
    st.dataframe(show_df[["feature", "pearson_r", "n_used"]], use_container_width=True, height=420)
with cB:
    fig_corr = px.bar(show_df, x="abs_r", y="feature", orientation="h", hover_data=["pearson_r", "n_used"])
    fig_corr.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_corr, use_container_width=True)

# =========================
# 5) Weekly trend (mean)
# =========================
st.subheader("5) Weekly trend (mean)")

trend_feat = st.selectbox("Trend feature", options=[TARGET] + selected_features, index=0)
group_by = st.selectbox("Group by", options=["(none)"] + [c for c in ["gender", "race", "year", "covid_period"] if c in f.columns], index=0)

trend_cols = ["week_start_date", trend_feat]
if group_by != "(none)":
    trend_cols.append(group_by)

td = f[trend_cols].copy()
td[trend_feat] = safe_numeric(td[trend_feat])
td = td.dropna(subset=["week_start_date", trend_feat])

if len(td) == 0:
    st.info("No data for trend plot.")
else:
    if group_by == "(none)":
        agg = td.groupby("week_start_date", as_index=False)[trend_feat].mean()
        fig_tr = px.line(agg, x="week_start_date", y=trend_feat)
    else:
        agg = td.groupby(["week_start_date", group_by], as_index=False)[trend_feat].mean()
        fig_tr = px.line(agg, x="week_start_date", y=trend_feat, color=group_by)
    fig_tr.update_layout(height=420)
    st.plotly_chart(fig_tr, use_container_width=True)

# ============================================================
# NEW FEATURE #1: Multi-feature facet scatter
# ============================================================
st.subheader("6) Multi-feature facet scatter (many X → avg_phq4_score)")

m1, m2 = st.columns([1, 1])

with m1:
    default_multi = [x_feat] + [feat for feat in selected_features if feat != x_feat][:5]
    multi_feats = st.multiselect(
        "Pick many features (X's)",
        options=selected_features,
        default=default_multi,
    )
    multi_color = st.selectbox("Color by (multi)", options=COLOR_OPTIONS, index=0)
    multi_trend = st.checkbox("Add trendline (multi)", value=False)

with m2:
    col_wrap = st.slider("Panels per row (col_wrap)", 2, 5, 3, 1)
    sample_max = st.slider("Max points (downsample if huge)", 1000, 50000, 12000, 1000)
    st.caption("Facet plot can get heavy; downsampling keeps it responsive.")

if len(multi_feats) == 0:
    st.info("Select at least 1 feature for facet scatter.")
else:
    long_cols = ["year", "week", "week_start_date", TARGET] + (["uid"] if "uid" in f.columns else [])
    if multi_color != "(none)":
        long_cols.append(multi_color)

    base = f[long_cols + multi_feats].copy()
    for c in multi_feats + [TARGET]:
        base[c] = safe_numeric(base[c])

    # Melt to long format: feature_name, feature_value
    long_df = base.melt(
        id_vars=long_cols,
        value_vars=multi_feats,
        var_name="feature",
        value_name="x_value",
    )
    long_df = long_df.dropna(subset=["x_value", TARGET])

    # Downsample if needed
    if len(long_df) > sample_max:
        long_df = long_df.sample(sample_max, random_state=42)

    if len(long_df) == 0:
        st.warning("No data available for multi-feature plot after dropping NA.")
    else:
        fig_multi = px.scatter(
            long_df,
            x="x_value",
            y=TARGET,
            color=None if multi_color == "(none)" else multi_color,
            facet_col="feature",
            facet_col_wrap=col_wrap,
            trendline="ols" if multi_trend else None,
            opacity=0.55,
            hover_data=["year", "week", "week_start_date"] + (["uid"] if "uid" in long_df.columns else []),
        )
        fig_multi.update_layout(height=350 + 260 * int(np.ceil(len(multi_feats) / col_wrap)))
        fig_multi.update_xaxes(matches=None)  # allow each panel its own x scale
        st.plotly_chart(fig_multi, use_container_width=True)

# ============================================================
# NEW FEATURE #2: Group difference test + effect size
# ============================================================
st.subheader("7) Group difference test (gender / race): mean difference + p-value + effect size")

g1, g2 = st.columns([1, 1])
with g1:
    test_feature = st.selectbox("Test feature", options=[TARGET] + selected_features, index=0)
with g2:
    group_var = st.selectbox(
        "Group variable",
        options=[c for c in ["gender", "race"] if c in f.columns] or ["(not available)"],
        index=0,
    )

if group_var == "(not available)":
    st.info("gender/race column not found in data.")
else:
    min_n = st.slider("Min group size (filter small groups)", 5, 200, 20, 5)
    drop_missing_group = st.checkbox("Drop rows with missing group value", value=True)

    tdf = f[[test_feature, group_var]].copy()
    tdf[test_feature] = safe_numeric(tdf[test_feature])
    if drop_missing_group:
        tdf[group_var] = tdf[group_var].astype(str)
        tdf = tdf.replace({group_var: {"nan": np.nan}})
        tdf = tdf.dropna(subset=[group_var])
    tdf = tdf.dropna(subset=[test_feature])

    # Group summary
    summary = (
        tdf.groupby(group_var)[test_feature]
        .agg(n="count", mean="mean", std="std", median="median")
        .reset_index()
        .sort_values("n", ascending=False)
    )
    summary = summary[summary["n"] >= min_n].copy()

    if len(summary) < 2:
        st.warning("Not enough groups with sufficient sample size after filtering.")
    else:
        # restrict data to kept groups
        keep_groups = summary[group_var].astype(str).tolist()
        tdf2 = tdf[tdf[group_var].astype(str).isin(keep_groups)].copy()

        st.markdown("**Group summary (after min_n filter):**")
        st.dataframe(summary, use_container_width=True)

        # Decide test type
        ngroups = summary.shape[0]
        groups_vals = [tdf2[tdf2[group_var].astype(str) == g][test_feature].values for g in keep_groups]

        result_rows = []
        if ngroups == 2:
            gA, gB = keep_groups[0], keep_groups[1]
            x = groups_vals[0]
            y = groups_vals[1]

            # Welch t-test
            t_stat, p_val = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
            d = cohens_d(x, y)

            result_rows.append({
                "test": "Welch t-test (2 groups)",
                "groups": f"{gA} vs {gB}",
                "stat": t_stat,
                "p_value": p_val,
                "effect_size": d,
                "effect_label": "Cohen's d (A-B)",
            })

        else:
            # One-way ANOVA
            f_stat, p_val = stats.f_oneway(*groups_vals)
            eta2 = eta_squared_oneway(groups_vals)

            result_rows.append({
                "test": "One-way ANOVA (>=3 groups)",
                "groups": f"{ngroups} groups",
                "stat": f_stat,
                "p_value": p_val,
                "effect_size": eta2,
                "effect_label": "η² (eta-squared)",
            })

        res = pd.DataFrame(result_rows)

        r1, r2 = st.columns([1, 1])
        with r1:
            st.markdown("**Test result:**")
            st.dataframe(res, use_container_width=True)

        with r2:
            # Visual: boxplot by group
            fig_box = px.box(
                tdf2,
                x=group_var,
                y=test_feature,
                points="outliers",
            )
            fig_box.update_layout(height=420, xaxis_title=group_var, yaxis_title=test_feature)
            st.plotly_chart(fig_box, use_container_width=True)

        st.caption(
            "Notes: Two groups uses Welch t-test (unequal variances). 3+ groups uses one-way ANOVA. "
            "Effect sizes: Cohen’s d (2 groups) and η² for ANOVA."
        )

# =========================
# Preview data
# =========================
with st.expander("Preview filtered data"):
    preview_cols = [c for c in ID_COLS_CANDIDATES if c in f.columns] + [TARGET] + selected_features[:10]
    st.dataframe(f[preview_cols].head(200), use_container_width=True)
