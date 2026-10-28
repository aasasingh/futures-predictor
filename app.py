import streamlit as st
import plotly.graph_objects as go
import os
import random
import time
from datetime import datetime

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recession-O-Meter",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark Theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global background */
    .stApp { background-color: #0e1117; color: #e0e0e0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* Headers */
    h1 { color: #ffffff !important; letter-spacing: -0.5px; }
    h2, h3 { color: #c9d1d9 !important; }

    /* Metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #8b949e;
        margin-top: 2px;
    }

    /* Score band pills */
    .pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0 4px;
    }

    /* Commentary box */
    .commentary-box {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #30363d;
        border-left: 4px solid #f0883e;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 8px;
        font-size: 0.95rem;
        color: #c9d1d9;
        font-style: italic;
    }

    /* Divider */
    hr { border-color: #30363d; }

    /* Slider labels */
    .sidebar-indicator-label {
        font-size: 0.82rem;
        color: #8b949e;
        margin-bottom: -10px;
    }

    /* Weight badges */
    .weight-badge {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 8px 14px;
        font-size: 0.8rem;
        color: #8b949e;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ─── FRED Data ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_fred_indicators(api_key: str | None = None) -> tuple[dict, bool]:
    """
    Returns (indicators_dict, is_live).
    Pulls the latest FRED values when a key is given, otherwise (or if the
    request fails) falls back to placeholder values.
    """
    if api_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=api_key)
            start = (datetime.now().replace(year=datetime.now().year - 1)).strftime("%Y-%m-%d")

            def latest(series_id: str) -> float:
                return float(fred.get_series(series_id, observation_start=start).dropna().iloc[-1])

            return {
                "yield_10yr": latest("DGS10"),
                "yield_2yr":  latest("DGS2"),
                "spread":     latest("T10Y2Y"),   # negative = inverted = bad
                "unemp":      latest("UNRATE"),
                "sentiment":  latest("UMCSENT"),
            }, True
        except Exception:
            pass

    # ── PLACEHOLDER VALUES (used without a key) ───────────────────────────────
    mock = {
        "yield_10yr":  4.28,   # %
        "yield_2yr":   4.61,   # %
        "spread":     -0.33,   # 10yr minus 2yr — negative = inverted curve
        "unemp":       4.1,    # %
        "sentiment":  67.8,    # University of Michigan index (100 = neutral)
    }
    return mock, False


def score_fred_indicators(data: dict) -> tuple[float, dict]:
    """
    Converts raw FRED values into a 0-100 risk sub-score.
    Returns (score, details_dict).
    """
    details = {}

    # ── Yield Curve Spread ────────────────────────────────────────────────────
    spread = data["spread"]
    # Spread < -0.5 → very bad (100), spread > 0.5 → good (0)
    curve_score = max(0.0, min(100.0, (-spread + 0.5) / 1.5 * 100))
    details["Yield Curve (10y-2y)"] = {
        "raw": f"{spread:+.2f}%",
        "score": curve_score,
        "status": "⚠️ Inverted" if spread < 0 else "✅ Normal",
    }

    # ── Unemployment Rate ─────────────────────────────────────────────────────
    unemp = data["unemp"]
    # 3.5% = great (0), 7%+ = very bad (100)
    unemp_score = max(0.0, min(100.0, (unemp - 3.5) / 3.5 * 100))
    details["Unemployment Rate"] = {
        "raw": f"{unemp:.1f}%",
        "score": unemp_score,
        "status": "✅ Healthy" if unemp < 4.5 else ("⚠️ Elevated" if unemp < 6 else "🚨 High"),
    }

    # ── Consumer Sentiment ────────────────────────────────────────────────────
    sent = data["sentiment"]
    # 100+ = great (0), 50 and below = very bad (100)
    sent_score = max(0.0, min(100.0, (100 - sent) / 50 * 100))
    details["Consumer Sentiment"] = {
        "raw": f"{sent:.1f}",
        "score": sent_score,
        "status": "✅ Confident" if sent > 80 else ("⚠️ Cautious" if sent > 60 else "🚨 Pessimistic"),
    }

    # Weighted average: curve 40%, unemp 35%, sentiment 25%
    fred_score = curve_score * 0.40 + unemp_score * 0.35 + sent_score * 0.25
    return fred_score, details


@st.cache_data(ttl=3600, show_spinner=False)
def get_trends_indicators():
    """
    Fetches Google Trends interest-over-time for recession vibe indicators.
    Returns (indicators_dict, meta_dict).
    Falls back to neutral (50) with live=False if a term fails.
    """
    from pytrends.request import TrendReq

    INDICATORS = {
        "Strip Club Visits": "strip club near me",
        "Ramen Sales":       "instant ramen",
        "LinkedIn Banners":  "layoffs",
        "Generic Pet Food":  "cheap dog food",
    }
    BEER_TERM = "cheap beer"
    WINE_TERM = "wine"

    results = {}
    live_count = 0

    try:
        pytrends = TrendReq(
            hl="en-US",
            tz=300,
            timeout=(10, 30),
            retries=2,
            backoff_factor=0.5,
        )
    except Exception as e:
        # If TrendReq itself fails, return all-fallback
        fallback = {k: {"score": 50.0, "term": v, "live": False}
                    for k, v in {**INDICATORS, "Beer vs Wine Ratio": f"{BEER_TERM} vs {WINE_TERM}"}.items()}
        return fallback, {"fetched_at": datetime.now().strftime("%I:%M %p"), "live_count": 0}

    # ── Single-term indicators ─────────────────────────────────────────────────
    for name, term in INDICATORS.items():
        try:
            pytrends.build_payload([term], timeframe="today 12-m", geo="US")
            df = pytrends.interest_over_time()
            # Drop partial weeks (last row often flagged isPartial=True)
            if "isPartial" in df.columns:
                df = df[~df["isPartial"]]
            if not df.empty and term in df.columns:
                # Average of last 4 complete weeks = stable "current" signal
                score = float(df[term].iloc[-4:].mean())
                results[name] = {"score": round(score, 1), "term": term, "live": True}
                live_count += 1
            else:
                results[name] = {"score": 50.0, "term": term, "live": False}
        except Exception:
            results[name] = {"score": 50.0, "term": term, "live": False}
        time.sleep(0.8)  # stay polite to the unofficial API

    # ── Beer vs Wine comparison ────────────────────────────────────────────────
    try:
        pytrends.build_payload([BEER_TERM, WINE_TERM], timeframe="today 12-m", geo="US")
        df = pytrends.interest_over_time()
        if "isPartial" in df.columns:
            df = df[~df["isPartial"]]
        if not df.empty and BEER_TERM in df.columns and WINE_TERM in df.columns:
            beer_recent = float(df[BEER_TERM].iloc[-4:].mean())
            wine_recent = float(df[WINE_TERM].iloc[-4:].mean())
            total = beer_recent + wine_recent
            if total > 0:
                # beer_share: 30% → score 0, 70% → score 100
                # Typical ratio ~25-35% beer, so 30% is neutral baseline
                beer_share = beer_recent / total
                doom = (beer_share - 0.30) / 0.40 * 100
                score = round(max(0.0, min(100.0, doom)), 1)
            else:
                score = 50.0
            results["Beer vs Wine Ratio"] = {
                "score": score,
                "term": f'"{BEER_TERM}" vs "{WINE_TERM}"',
                "live": True,
            }
            live_count += 1
        else:
            results["Beer vs Wine Ratio"] = {
                "score": 50.0,
                "term": f'"{BEER_TERM}" vs "{WINE_TERM}"',
                "live": False,
            }
    except Exception:
        results["Beer vs Wine Ratio"] = {
            "score": 50.0,
            "term": f'"{BEER_TERM}" vs "{WINE_TERM}"',
            "live": False,
        }

    meta = {
        "fetched_at": datetime.now().strftime("%I:%M %p"),
        "live_count": live_count,
    }
    return results, meta


def score_trends_indicators(trends: dict) -> float:
    """Average 0-100 scores from trends dict → single 0-100 vibe sub-score."""
    scores = [v["score"] for v in trends.values()]
    return sum(scores) / len(scores) if scores else 50.0


def get_risk_label(score: float) -> tuple[str, str]:
    """Returns (label, hex_color) for a given 0-100 score."""
    if score < 20:
        return "No Sweat", "#2ea043"
    elif score < 40:
        return "Mild Jitters", "#7cce5b"
    elif score < 55:
        return "Brewing Storm", "#f0883e"
    elif score < 70:
        return "Code Orange", "#e3643c"
    elif score < 85:
        return "Brace Yourself", "#da3633"
    else:
        return "Full Send Into Doom", "#b91c1c"


def get_commentary(score: float, trends: dict, fred_data: dict) -> str:
    """Generates snarky commentary based on the overall score."""
    spread = fred_data["spread"]
    unemp  = fred_data["unemp"]

    # trends values are 0-100; ≥65 = elevated signal (≈ old slider ≥7/10)
    strip   = trends.get("Strip Club Visits",  {}).get("score", 50)
    ramen   = trends.get("Ramen Sales",        {}).get("score", 50)
    linked  = trends.get("LinkedIn Banners",   {}).get("score", 50)
    beer    = trends.get("Beer vs Wine Ratio", {}).get("score", 50)
    petfood = trends.get("Generic Pet Food",   {}).get("score", 50)

    # Build context snippets
    curve_quip = (
        "the yield curve is doing its best impression of a water slide"
        if spread < -0.3 else
        "the yield curve is almost flat enough to eat dinner on"
        if spread < 0.2 else
        "the yield curve is actually looking semi-normal, which is suspicious"
    )

    unemp_quip = (
        f"unemployment is ticking up ({unemp}%) and LinkedIn is basically a job fair"
        if unemp > 4.5 else
        f"people still have jobs ({unemp}%), for now"
    )

    ramen_quip = (
        "Google says people are searching 'instant ramen' at record levels"
        if ramen >= 65 else
        "some people are still buying fancy noodles, bless them"
    )

    strip_quip = (
        "searches for 'strip club near me' are through the roof"
        if strip >= 65 else
        "surprisingly, not everyone is drowning their sorrows yet"
    )

    beer_quip = (
        "people have quietly switched from wine to whatever's cheapest — Google confirms it"
        if beer >= 55 else
        "the wine aisle is holding steady, somehow"
    )

    linked_quip = (
        "Google Trends says 'layoffs' is spiking like it's 2008"
        if linked >= 65 else
        "a suspiciously normal number of people are Googling 'layoffs'"
    )

    pet_quip = (
        "'cheap dog food' searches are going parabolic — even Fido feels the pinch"
        if petfood >= 65 else
        "pets are still getting name-brand food, at least someone's eating well"
    )

    low_comments = [
        f"Economy's chillin' — {curve_quip}. Go touch grass.",
        f"Relax, {unemp_quip}. The vibes are fine.",
        f"Things look okay. {ramen_quip.capitalize()}. Suspicious, but okay.",
    ]

    mid_comments = [
        f"Well... {curve_quip}. {ramen_quip.capitalize()}. Could be worse. Probably.",
        f"Hmm. {unemp_quip}. Also, {beer_quip}. Light a candle.",
        f"The signs are mixed: {curve_quip}, and {linked_quip}. Maybe update your resumé. Just in case.",
    ]

    high_comments = [
        f"Buckle up. {curve_quip}, {ramen_quip}, and {strip_quip}. Godspeed.",
        f"Yikes. {unemp_quip}. {linked_quip}. {pet_quip}. We had a good run.",
        f"The yield curve looks like a ski slope and {ramen_quip}. {beer_quip}. Pour one out.",
    ]

    doom_comments = [
        f"We are so cooked. {curve_quip}. {linked_quip}. {pet_quip}. {beer_quip}. "
        f"The four horsemen are carpooling to save on gas.",
        f"Full doom mode activated. {curve_quip}. {ramen_quip}. {strip_quip}. "
        f"Start a sourdough starter and move to the woods.",
        f"The economy has entered the chat. Then immediately logged off. {curve_quip}. "
        f"{unemp_quip}. {pet_quip}. May god have mercy on your 401(k).",
    ]

    if score < 30:
        return random.choice(low_comments)
    elif score < 55:
        return random.choice(mid_comments)
    elif score < 75:
        return random.choice(high_comments)
    else:
        return random.choice(doom_comments)


def make_gauge(score: float, color: str) -> go.Figure:
    """Creates a Plotly gauge chart for the overall score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 52, "color": "#ffffff"}, "suffix": ""},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#30363d",
                "tickfont": {"color": "#8b949e", "size": 11},
                "nticks": 6,
            },
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#161b22",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20],  "color": "#0d1f12"},
                {"range": [20, 40], "color": "#142010"},
                {"range": [40, 55], "color": "#2d1e0a"},
                {"range": [55, 70], "color": "#2d1208"},
                {"range": [70, 85], "color": "#2a0d0d"},
                {"range": [85, 100],"color": "#1f0808"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.85,
                "value": score,
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(t=20, b=10, l=30, r=30),
        height=280,
        font={"color": "#e0e0e0"},
    )
    return fig


def make_mini_bar(score: float, color: str, label: str) -> go.Figure:
    """Small horizontal bar for individual indicator scores."""
    fig = go.Figure(go.Bar(
        x=[score],
        y=[label],
        orientation="h",
        marker_color=color,
        text=[f"{score:.0f}"],
        textposition="outside",
        textfont={"color": "#e0e0e0", "size": 11},
        hovertemplate=f"{label}: {score:.1f}<extra></extra>",
    ))
    fig.add_shape(
        type="rect",
        x0=0, x1=100, y0=-0.5, y1=0.5,
        fillcolor="#21262d", line_width=0,
        layer="below",
    )
    fig.update_layout(
        xaxis=dict(range=[0, 115], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=True, tickfont={"color": "#8b949e", "size": 10}),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(t=2, b=2, l=4, r=40),
        height=40,
        showlegend=False,
    )
    return fig


def score_to_color(score: float) -> str:
    """Gradient from green → yellow → red."""
    if score < 20:   return "#2ea043"
    elif score < 40: return "#7cce5b"
    elif score < 55: return "#f0883e"
    elif score < 70: return "#e3643c"
    elif score < 85: return "#da3633"
    else:            return "#b91c1c"


# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌐 Google Trends")
    st.markdown("*Live search interest — US, last 12 months*")

    # Fetch data here so we can show status in sidebar
    with st.spinner("Fetching trends data…"):
        trends_data, trends_meta = get_trends_indicators()

    live_count = trends_meta["live_count"]
    total_indicators = len(trends_data)
    fetched_at = trends_meta["fetched_at"]

    if live_count == total_indicators:
        st.success(f"✅ All live · fetched {fetched_at}", icon=None)
    elif live_count > 0:
        st.warning(f"⚠️ {live_count}/{total_indicators} live · fetched {fetched_at}")
    else:
        st.error("📵 Google Trends unavailable — using estimates")

    if st.button("🔄 Refresh trends", width="stretch"):
        get_trends_indicators.clear()
        st.rerun()

    st.markdown("---")

    EMOJI_MAP = {
        "Strip Club Visits":  ("💃", "Stress-relief spending"),
        "Ramen Sales":        ("🍜", "Trading down from real food"),
        "LinkedIn Banners":   ("📦", "Job anxiety search spike"),
        "Beer vs Wine Ratio": ("🍺", "Cheap beer vs wine interest"),
        "Generic Pet Food":   ("🐾", "Downgrading Fluffy's kibble"),
    }

    for name, info in trends_data.items():
        emoji, hint = EMOJI_MAP.get(name, ("📊", ""))
        score = info["score"]
        live  = info["live"]
        term  = info["term"]
        color = score_to_color(score)
        badge = "✅" if live else "〜"

        st.markdown(
            f'<div style="margin-bottom:12px;">'
            f'<div style="font-size:0.85rem; font-weight:600; color:#c9d1d9;">'
            f'{emoji} {name} '
            f'<span style="font-size:0.7rem; color:#8b949e;">{badge}</span>'
            f'</div>'
            f'<div style="font-size:0.7rem; color:#8b949e; margin:1px 0 4px 0;">{term}</div>'
            f'<div style="background:#21262d; border-radius:4px; height:6px; overflow:hidden;">'
            f'<div style="width:{score}%; height:100%; background:{color}; border-radius:4px;"></div>'
            f'</div>'
            f'<div style="font-size:0.72rem; color:{color}; text-align:right;">{score:.0f}/100</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="weight-badge">⚖️ Real indicators: <b>70%</b><br>Vibe indicators: <b>30%</b></div>', unsafe_allow_html=True)
    st.markdown("")

    fred_key = st.text_input(
        "🔑 FRED API Key (optional)",
        value="",
        type="password",
        placeholder="Leave blank to use FRED_API_KEY or mock data",
        help="Get a free key at fred.stlouisfed.org",
    )


# ─── Compute Scores ──────────────────────────────────────────────────────────────
fred_data, fred_live = get_fred_indicators(fred_key or os.environ.get("FRED_API_KEY"))
fred_score, fred_details = score_fred_indicators(fred_data)
vibe_score  = score_trends_indicators(trends_data)

# 70% real, 30% vibe
total_score = fred_score * 0.70 + vibe_score * 0.30

risk_label, risk_color = get_risk_label(total_score)

# Seed random with score bucket so commentary is stable per score range
# but changes when score changes meaningfully
random.seed(int(total_score // 5))
commentary  = get_commentary(total_score, trends_data, fred_data)


# ─── Main Layout ─────────────────────────────────────────────────────────────────
st.markdown("# 📉 Recession-O-Meter")
st.markdown("*Your real-time guide to economic existential dread — powered by FRED + Google Trends*")
st.markdown("---")

col_gauge, col_right = st.columns([1, 1], gap="large")

with col_gauge:
    st.markdown(f"### Overall Risk Score")
    st.plotly_chart(make_gauge(total_score, risk_color), width="stretch", config={"displayModeBar": False})

    # Risk label pill
    st.markdown(
        f'<div style="text-align:center; margin-top:-10px;">'
        f'<span style="background:{risk_color}22; color:{risk_color}; border:1px solid {risk_color}55; '
        f'padding:6px 20px; border-radius:20px; font-weight:700; font-size:1.1rem;">'
        f'{risk_label}'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    # Weight breakdown
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">📊 Real Indicators</div>'
            f'<div class="metric-value" style="color:{score_to_color(fred_score)}">{fred_score:.0f}</div>'
            f'<div class="metric-sub">70% weight</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">🎭 Vibe Check</div>'
            f'<div class="metric-value" style="color:{score_to_color(vibe_score)}">{vibe_score:.0f}</div>'
            f'<div class="metric-sub">30% weight</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

with col_right:
    st.markdown("### 📊 Real Economic Indicators")
    if fred_live:
        st.caption("Source: FRED (live)")
    else:
        st.caption("Source: FRED (mock data — add API key to use live data)")

    for name, detail in fred_details.items():
        col_a, col_b = st.columns([3, 1])
        with col_a:
            bar_color = score_to_color(detail["score"])
            st.plotly_chart(
                make_mini_bar(detail["score"], bar_color, name),
                width="stretch",
                config={"displayModeBar": False},
            )
        with col_b:
            st.markdown(
                f'<div style="padding-top:6px; font-size:0.82rem; color:#8b949e;">'
                f'{detail["raw"]}<br><span style="font-size:0.75rem">{detail["status"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown("### 🎭 Vibe Indicator Breakdown")
    st.caption("Source: Google Trends · US · last 12 months")
    EMOJI_DISPLAY = {
        "Strip Club Visits":  "💃",
        "Ramen Sales":        "🍜",
        "LinkedIn Banners":   "📦",
        "Beer vs Wine Ratio": "🍺",
        "Generic Pet Food":   "🐾",
    }
    for name, info in trends_data.items():
        score     = info["score"]
        live      = info["live"]
        bar_color = score_to_color(score)
        live_tag  = "" if live else " ·~est."
        label     = f"{EMOJI_DISPLAY.get(name, '')} {name}{live_tag}"
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.plotly_chart(
                make_mini_bar(score, bar_color, label),
                width="stretch",
                config={"displayModeBar": False},
            )
        with col_b:
            status = "✅ live" if live else "〜 est."
            st.markdown(
                f'<div style="padding-top:6px; font-size:0.78rem; color:#8b949e;">'
                f'{score:.0f}/100<br>'
                f'<span style="font-size:0.7rem;">{status}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─── Commentary ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🤖 The Oracle Speaks")
st.markdown(f'<div class="commentary-box">"{commentary}"</div>', unsafe_allow_html=True)

st.markdown("")
st.caption(
    "⚠️ Not financial advice. Seriously. Please don't make any life decisions based on this. "
    "Strip club visit data is not a recognized NBER recession indicator. Yet."
)
