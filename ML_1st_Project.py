from datetime import date, timedelta
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import streamlit as st
import plotly.graph_objects as go

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WeatherCast AI",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #0b0f1a !important;
    color: #e2e8f0 !important;
    font-family: 'Syne', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 10%, #0e1f3d 0%, #0b0f1a 60%) !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"]    { display: none; }
[data-testid="stDecoration"] { display: none; }

.block-container { padding: 2rem 3rem 3rem 3rem !important; max-width: 1200px; }

/* Hero */
.hero-wrap  { text-align: center; padding: 2rem 1rem 1.2rem; }
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white; font-family: 'Space Mono', monospace;
    font-size: 0.62rem; letter-spacing: 0.18em; text-transform: uppercase;
    padding: 0.28rem 1rem; border-radius: 50px; margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: clamp(2.2rem, 5vw, 3.4rem); line-height: 1.1;
    background: linear-gradient(135deg, #e2e8f0 0%, #38bdf8 50%, #818cf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.5rem;
}
.hero-sub { color: #64748b; font-size: 0.95rem; }

/* Metric cards */
.metric-row  { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 150px;
    background: #111827; border: 1px solid #1e2d45;
    border-radius: 14px; padding: 1.1rem 1.3rem;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.metric-card.blue::before   { background: linear-gradient(90deg, #38bdf8, #6366f1); }
.metric-card.green::before  { background: linear-gradient(90deg, #34d399, #0ea5e9); }
.metric-card.orange::before { background: linear-gradient(90deg, #fb923c, #fbbf24); }
.metric-card.purple::before { background: linear-gradient(90deg, #818cf8, #c084fc); }
.metric-label { font-family: 'Space Mono', monospace; font-size: 0.62rem;
    text-transform: uppercase; letter-spacing: 0.15em;
    color: #64748b; margin-bottom: 0.4rem; }
.metric-value { font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 1.8rem; color: #e2e8f0; line-height: 1; }
.metric-unit  { font-size: 0.95rem; color: #64748b; font-weight: 400; }
.metric-sub   { font-size: 0.72rem; color: #64748b; margin-top: 0.3rem; }

/* Info banner */
.info-banner {
    background: linear-gradient(135deg, rgba(56,189,248,.08), rgba(99,102,241,.08));
    border: 1px solid rgba(56,189,248,.2); border-radius: 12px;
    padding: 0.9rem 1.3rem; margin-bottom: 1.4rem;
    display: flex; align-items: center; gap: 0.8rem;
}
.info-banner .icon { font-size: 1.3rem; }
.info-banner .text { font-size: 0.88rem; color: #e2e8f0; }
.info-banner .loc  { font-weight: 700; color: #38bdf8; }

/* Section header */
.section-header {
    font-family: 'Space Mono', monospace; font-size: 0.68rem;
    text-transform: uppercase; letter-spacing: 0.2em;
    color: #38bdf8; margin-bottom: 0.7rem; margin-top: 0.4rem;
}

/* Diff colors */
.diff-good  { color: #34d399; font-weight: 700; }
.diff-warn  { color: #fbbf24; font-weight: 700; }
.diff-alert { color: #fb923c; font-weight: 700; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #1a2540 !important; border: 1px solid #1e2d45 !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-family: 'Syne', sans-serif !important;
}

/* Button */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important;
    letter-spacing: 0.05em !important; padding: 0.65rem 2.5rem !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.3) !important;
    width: 100% !important; transition: all 0.3s ease !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(14,165,233,0.45) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #161d2e !important; border: 1px solid #1e2d45 !important;
    border-radius: 12px !important;
}

/* Footer */
.footer {
    text-align: center; padding: 1.8rem 0 0.8rem;
    color: #334155; font-size: 0.75rem; font-family: 'Space Mono', monospace;
    border-top: 1px solid #1e2d45; margin-top: 2rem;
}

hr { border-color: #1e2d45 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Backend Logic (unchanged) ────────────────────────────────────────────────

GEOCODE_URL  = "https://geocoding-api.open-meteo.com/v1/search"
HIST_URL     = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def geocode_city(name):
    params = {"name": name, "count": 1, "language": "en", "format": "json"}
    r = requests.get(GEOCODE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("results"):
        raise ValueError(f"Could not geocode city '{name}'. Try a different spelling.")
    res = data["results"][0]
    return {
        "name":      res.get("name"),
        "latitude":  res["latitude"],
        "longitude": res["longitude"],
        "timezone":  res.get("timezone", "auto"),
        "country":   res.get("country"),
        "admin1":    res.get("admin1"),
    }


def fetch_history(lat, lon, start_date, end_date, timezone="auto"):
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date.isoformat(),
        "end_date":   end_date.isoformat(),
        "daily":      ["temperature_2m_max", "temperature_2m_min"],
        "timezone":   timezone,
    }
    r = requests.get(HIST_URL, params=params, timeout=60)
    r.raise_for_status()
    d = r.json()
    daily = d.get("daily", {})
    if not daily or "time" not in daily:
        raise RuntimeError("Historical data not available for the requested range.")
    df = pd.DataFrame(daily)
    for col in ["temperature_2m_max", "temperature_2m_min"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["temp_mean"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2.0
    df["date"] = pd.to_datetime(df["time"])
    before = len(df)
    df = df.dropna(subset=["temp_mean"]).reset_index(drop=True)
    after = len(df)
    if after < before:
        st.toast(f"ℹ️ Dropped {before - after} rows with missing data")
    return df[["date", "temperature_2m_min", "temperature_2m_max", "temp_mean"]]


def fetch_forecast(lat, lon, timezone="auto"):
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "daily":         ["temperature_2m_max", "temperature_2m_min"],
        "forecast_days": 7,
        "timezone":      timezone,
    }
    r = requests.get(FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()
    d = r.json()
    daily = d.get("daily", {})
    if not daily or "time" not in daily:
        return pd.DataFrame(columns=["date", "temperature_2m_min", "temperature_2m_max", "temp_mean"])
    df = pd.DataFrame(daily)
    df["date"]               = pd.to_datetime(df["time"])
    df["temperature_2m_max"] = pd.to_numeric(df["temperature_2m_max"], errors="coerce")
    df["temperature_2m_min"] = pd.to_numeric(df["temperature_2m_min"], errors="coerce")
    df["temp_mean"]          = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2.0
    return df[["date", "temperature_2m_min", "temperature_2m_max", "temp_mean"]]


def build_xy(df):
    df    = df.sort_values("date").reset_index(drop=True)
    start = df["date"].min()
    df["x"] = (df["date"] - start).dt.days.astype(int)
    X = df[["x"]].values
    y = df["temp_mean"].values.astype(float)
    return df, X, y, start


def fit_poly_regression(X, y, degree=3):
    model = Pipeline([
        ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
        ("linreg", LinearRegression())
    ])
    model.fit(X, y)
    return model


# ─── City Data (52 cities, ascending order) ───────────────────────────────────

CITY_META = {
    "Agartala":           {"emoji": "🌿", "desc": "Gateway to Northeast"},
    "Agra":               {"emoji": "🕌", "desc": "City of the Taj Mahal"},
    "Ahmedabad":          {"emoji": "🏙️", "desc": "Manchester of India"},
    "Aizawl":             {"emoji": "⛰️", "desc": "City in the Clouds"},
    "Amritsar":           {"emoji": "✨", "desc": "City of the Golden Temple"},
    "Bengaluru":          {"emoji": "💻", "desc": "Silicon Valley of India"},
    "Bhopal":             {"emoji": "🌊", "desc": "City of Lakes"},
    "Bhubaneswar":        {"emoji": "🛕", "desc": "Temple City of India"},
    "Chandigarh":         {"emoji": "🌳", "desc": "The City Beautiful"},
    "Chennai":            {"emoji": "🌴", "desc": "Gateway to South India"},
    "Coimbatore":         {"emoji": "🏭", "desc": "Manchester of South India"},
    "Dehradun":           {"emoji": "🏔️", "desc": "Gateway to the Himalayas"},
    "Delhi":              {"emoji": "🏛️", "desc": "Capital Territory"},
    "Dharamsala":         {"emoji": "🙏", "desc": "Home of the Dalai Lama"},
    "Dispur":             {"emoji": "🦏", "desc": "Capital of Assam"},
    "Gangtok":            {"emoji": "🏔️", "desc": "Jewel of the Himalayas"},
    "Guwahati":           {"emoji": "🌿", "desc": "Gateway to Northeast India"},
    "Gwalior":            {"emoji": "🏰", "desc": "City of Forts"},
    "Hyderabad":          {"emoji": "🕌", "desc": "City of Pearls"},
    "Imphal":             {"emoji": "🌸", "desc": "Jewel of India"},
    "Indore":             {"emoji": "🍽️", "desc": "Food Capital of India"},
    "Itanagar":           {"emoji": "🌲", "desc": "City of Dawn-Lit Mountains"},
    "Jaipur":             {"emoji": "🏰", "desc": "The Pink City"},
    "Jammu":              {"emoji": "🛕", "desc": "City of Temples"},
    "Jodhpur":            {"emoji": "💙", "desc": "The Blue City"},
    "Kochi":              {"emoji": "⛵", "desc": "Queen of the Arabian Sea"},
    "Kohima":             {"emoji": "⚔️", "desc": "Land of the Nagas"},
    "Kolkata":            {"emoji": "🌊", "desc": "City of Joy"},
    "Lucknow":            {"emoji": "🍢", "desc": "City of Nawabs"},
    "Ludhiana":           {"emoji": "🌾", "desc": "Wheat Bowl of India"},
    "Madurai":            {"emoji": "🛕", "desc": "Athens of the East"},
    "Mangaluru":          {"emoji": "🌊", "desc": "Cradle of Indian Banking"},
    "Mumbai":             {"emoji": "🌆", "desc": "Financial Capital of India"},
    "Mysuru":             {"emoji": "👑", "desc": "City of Palaces"},
    "Nagpur":             {"emoji": "🍊", "desc": "Orange City of India"},
    "Panaji":             {"emoji": "🏖️", "desc": "Pearl of the Orient"},
    "Patna":              {"emoji": "🌸", "desc": "Ancient City of Pataliputra"},
    "Puducherry":         {"emoji": "🇫🇷", "desc": "French Riviera of the East"},
    "Pune":               {"emoji": "🎓", "desc": "Oxford of the East"},
    "Raipur":             {"emoji": "🌿", "desc": "Rice Bowl of India"},
    "Rajkot":             {"emoji": "🏏", "desc": "City of Champions"},
    "Ranchi":             {"emoji": "💧", "desc": "City of Waterfalls"},
    "Shillong":           {"emoji": "🎵", "desc": "Scotland of the East"},
    "Shimla":             {"emoji": "❄️", "desc": "Queen of the Hills"},
    "Siliguri":           {"emoji": "🍃", "desc": "Gateway to Northeast"},
    "Srinagar":           {"emoji": "🛶", "desc": "Paradise on Earth"},
    "Surat":              {"emoji": "💎", "desc": "Diamond City of India"},
    "Thiruvananthapuram": {"emoji": "🌴", "desc": "Evergreen City of India"},
    "Udaipur":            {"emoji": "🏯", "desc": "City of Lakes"},
    "Varanasi":           {"emoji": "🪔", "desc": "Spiritual Capital of India"},
    "Vijayawada":         {"emoji": "🌊", "desc": "City of Victory"},
    "Visakhapatnam":      {"emoji": "⚓", "desc": "Jewel of the East Coast"},
}

# Always sorted alphabetically
cities = sorted(CITY_META.keys())


# ─── Helpers ──────────────────────────────────────────────────────────────────

def temp_color(t):
    if t < 15:   return "#38bdf8"
    elif t < 25: return "#34d399"
    elif t < 32: return "#fbbf24"
    else:        return "#fb923c"

def diff_class(d):
    if abs(d) < 1:   return "diff-good"
    elif abs(d) < 3: return "diff-warn"
    else:            return "diff-alert"


# ─── UI Layout ────────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🛰 AI-Powered Forecast Engine</div>
    <div class="hero-title">WeatherCast AI</div>
    <div class="hero-sub">Polynomial regression meets real-time meteorological data</div>
</div>
""", unsafe_allow_html=True)

# City selector + Run button
city_options = [f"{CITY_META[c]['emoji']}  {c} — {CITY_META[c]['desc']}" for c in cities]

col_sel, col_btn = st.columns([3, 1], gap="medium")

with col_sel:
    st.markdown('<div class="section-header">📍 Select City</div>', unsafe_allow_html=True)
    selected_display = st.selectbox(
        label="Select City",
        options=city_options,
        label_visibility="collapsed",
    )
    selected_city = cities[city_options.index(selected_display)]

with col_btn:
    st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)
    run = st.button("⚡  Run Forecast")

st.markdown("---")


# ─── Main Forecast ────────────────────────────────────────────────────────────

if run:
    with st.spinner(f"Fetching data for {selected_city}…"):
        try:
            # Geocode city
            place        = geocode_city(selected_city)
            lat, lon, tz = place["latitude"], place["longitude"], place["timezone"]

            # Fetch historical data
            today    = date.today()
            start    = today - timedelta(days=480)
            hist_end = today - timedelta(days=3)
            hist_df  = fetch_history(lat, lon, start, hist_end, tz)

            if hist_df.empty or len(hist_df) < 5:
                st.error(f"Not enough historical samples for {selected_city}.")
                st.stop()

            # Build X / y
            hist_df, X, y, base_date = build_xy(hist_df)
            mask    = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X, y    = X[mask], y[mask]
            hist_df = hist_df.loc[mask].reset_index(drop=True)

            if len(y) < 5:
                st.error(f"Too few clean samples for {selected_city}.")
                st.stop()

            # Train model
            model = fit_poly_regression(X, y, degree=5)

            # Predict tomorrow
            tomorrow   = today + timedelta(days=1)
            x_tomorrow = np.array([[(pd.Timestamp(tomorrow) - base_date).days]])
            y_pred     = float(model.predict(x_tomorrow)[0])

            # Official 7-day forecast
            fc_df  = fetch_forecast(lat, lon, tz)
            fc_val = None
            if not fc_df.empty:
                td = fc_df.loc[fc_df["date"].dt.date == tomorrow, "temp_mean"]
                if not td.empty:
                    fc_val = float(td.iloc[0])

            # ── Location banner ───────────────────────────────────────────
            parts   = [p for p in [place.get("name"), place.get("admin1"), place.get("country")] if p]
            loc_str = ", ".join(parts)
            meta    = CITY_META.get(selected_city, {"emoji": "📍", "desc": ""})

            st.markdown(f"""
            <div class="info-banner">
                <div class="icon">{meta['emoji']}</div>
                <div class="text">
                    Showing forecast for <span class="loc">{loc_str}</span>
                    &nbsp;·&nbsp; Tomorrow &nbsp;
                    <strong>{tomorrow.strftime('%A, %d %B %Y')}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Metric cards ──────────────────────────────────────────────
            diff     = (y_pred - fc_val) if fc_val is not None else None
            dc       = diff_class(diff) if diff is not None else "diff-good"
            diff_str = f'<span class="{dc}">{diff:+.2f} °C</span>' if diff is not None else "N/A"
            fc_str   = f"{fc_val:.2f}" if fc_val is not None else "—"

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card blue">
                    <div class="metric-label">Model Prediction</div>
                    <div class="metric-value" style="color:{temp_color(y_pred)}">
                        {y_pred:.1f}<span class="metric-unit"> °C</span>
                    </div>
                    <div class="metric-sub">Polynomial regression (deg 3)</div>
                </div>
                <div class="metric-card green">
                    <div class="metric-label">Open-Meteo Forecast</div>
                    <div class="metric-value">{fc_str}<span class="metric-unit"> °C</span></div>
                    <div class="metric-sub">Official weather API</div>
                </div>
                <div class="metric-card orange">
                    <div class="metric-label">Difference (Model − Forecast)</div>
                    <div class="metric-value" style="font-size:1.4rem">{diff_str}</div>
                    <div class="metric-sub">How close the model is</div>
                </div>
                <div class="metric-card purple">
                    <div class="metric-label">Training Days</div>
                    <div class="metric-value">{len(y)}<span class="metric-unit"> days</span></div>
                    <div class="metric-sub">Historical samples used</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Chart 1 — History + Regression Curve ─────────────────────
            st.markdown('<div class="section-header">📈 Temperature History & Model Fit</div>',
                        unsafe_allow_html=True)

            x_range    = np.linspace(X.min(), X.max() + 5, 300).reshape(-1, 1)
            y_range    = model.predict(x_range)
            date_range = [base_date + pd.Timedelta(days=int(xi)) for xi in x_range.flatten()]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.concat([hist_df["date"], hist_df["date"][::-1]]),
                y=pd.concat([hist_df["temperature_2m_max"], hist_df["temperature_2m_min"][::-1]]),
                fill="toself", fillcolor="rgba(56,189,248,0.07)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Min–Max Range", hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=hist_df["date"], y=hist_df["temp_mean"],
                mode="lines+markers", name="Historical Mean",
                line=dict(color="#38bdf8", width=1.5),
                marker=dict(size=3, color="#38bdf8"),
            ))
            fig.add_trace(go.Scatter(
                x=date_range, y=y_range,
                mode="lines", name="Polynomial Fit (deg 3)",
                line=dict(color="#818cf8", width=2.5, dash="dot"),
            ))
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(tomorrow)], y=[y_pred],
                mode="markers", name=f"Model Prediction ({y_pred:.1f}°C)",
                marker=dict(size=14, color="#fb923c", symbol="star",
                            line=dict(color="white", width=1.5)),
            ))
            if fc_val is not None:
                fig.add_trace(go.Scatter(
                    x=[pd.Timestamp(tomorrow)], y=[fc_val],
                    mode="markers", name=f"Open-Meteo ({fc_val:.1f}°C)",
                    marker=dict(size=12, color="#34d399", symbol="diamond",
                                line=dict(color="white", width=1.5)),
                ))
            fig.update_layout(
                paper_bgcolor="#111827", plot_bgcolor="#111827",
                font=dict(family="Syne", color="#94a3b8"),
                legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d45",
                            borderwidth=1, font=dict(size=11)),
                xaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45",
                           tickfont=dict(size=11), title=""),
                yaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45",
                           tickfont=dict(size=11), title="Temperature (°C)"),
                margin=dict(l=10, r=10, t=20, b=20),
                height=370, hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Chart 2 — 7-Day Forecast ──────────────────────────────────
            if not fc_df.empty:
                st.markdown('<div class="section-header">📅 7-Day Forecast Overview</div>',
                            unsafe_allow_html=True)
                fc_df["day_label"] = fc_df["date"].dt.strftime("%a %d")
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=fc_df["day_label"], y=fc_df["temperature_2m_max"],
                    name="Max Temp", marker_color="rgba(251,146,60,0.75)",
                    text=[f"{v:.1f}°" for v in fc_df["temperature_2m_max"]],
                    textposition="outside", textfont=dict(size=11, color="#fb923c"),
                ))
                fig2.add_trace(go.Bar(
                    x=fc_df["day_label"], y=fc_df["temperature_2m_min"],
                    name="Min Temp", marker_color="rgba(56,189,248,0.75)",
                    text=[f"{v:.1f}°" for v in fc_df["temperature_2m_min"]],
                    textposition="outside", textfont=dict(size=11, color="#38bdf8"),
                ))
                fig2.add_trace(go.Scatter(
                    x=fc_df["day_label"], y=fc_df["temp_mean"],
                    mode="lines+markers", name="Mean Temp",
                    line=dict(color="#34d399", width=2),
                    marker=dict(size=8, color="#34d399"),
                ))
                fig2.update_layout(
                    paper_bgcolor="#111827", plot_bgcolor="#111827",
                    font=dict(family="Syne", color="#94a3b8"),
                    barmode="group",
                    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d45", borderwidth=1),
                    xaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45"),
                    yaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45",
                               title="Temperature (°C)"),
                    margin=dict(l=10, r=10, t=20, b=20),
                    height=330,
                )
                st.plotly_chart(fig2, use_container_width=True)

            # ── Raw data table ────────────────────────────────────────────
            with st.expander("🗂  View Raw Historical Data"):
                display_df = hist_df[["date", "temperature_2m_min",
                                      "temperature_2m_max", "temp_mean"]].copy()
                display_df.columns = ["Date", "Min Temp (°C)", "Max Temp (°C)", "Mean Temp (°C)"]
                display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
                st.dataframe(display_df, use_container_width=True, height=280)

        except requests.exceptions.ConnectionError:
            st.error("🌐 Network error — please check your internet connection.")
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Please try again.")
        except ValueError as e:
            st.error(f"⚠️ {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

else:
    # Landing state
    c1, c2, c3 = st.columns(3, gap="large")
    cards = [
        ("🧠", "ML Prediction",   "Degree-3 polynomial regression trained on 120 days of history"),
        ("📡", "Live API Data",   "Real-time historical & forecast data from Open-Meteo"),
        ("📊", "Visual Insights", "Interactive charts with temperature trends & 7-day outlook"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1e2d45;border-radius:14px;
                        padding:1.5rem 1.3rem;text-align:center;">
                <div style="font-size:1.9rem;margin-bottom:0.7rem">{icon}</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.95rem;
                            color:#e2e8f0;margin-bottom:0.45rem">{title}</div>
                <div style="font-size:0.82rem;color:#64748b;line-height:1.5">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
    st.info("👆 Select a city above and click **⚡ Run Forecast** to get started.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    WeatherCast AI &nbsp;·&nbsp; Powered by Open-Meteo API &nbsp;·&nbsp;
    Built with Streamlit &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)