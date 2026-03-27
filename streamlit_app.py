import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="AgeGuard AI", layout="wide", initial_sidebar_state="collapsed")

# ── Embedded metrics from pipeline results ───────────────────────
METRICS = {
    "test_mae": 5.02, "critical_mae": 4.34, "rmse": 6.94,
    "median_ae": 3.76, "std_error": 4.79, "test_samples": 1117,
    "inference_ms": 16.7, "model_size_mb": 77.5, "fps": 60,
    "far_25": 12.3, "frr_25": 20.3, "auto_approval": 79.7,
    "best_epoch": 19, "train_time_min": 6.2, "best_val_mae": 5.09,
}

BANDS = [
    {"band": "0-17", "mae": 3.69, "n": 218},
    {"band": "18-25", "mae": 4.34, "n": 228},
    {"band": "26-35", "mae": 4.56, "n": 289},
    {"band": "36-45", "mae": 5.41, "n": 169},
    {"band": "46-60", "mae": 6.49, "n": 154},
    {"band": "60+", "mae": 9.98, "n": 59},
]

THRESHOLDS = [
    {"t": 21, "far": 28.9, "frr": 7.2},
    {"t": 22, "far": 24.3, "frr": 9.3},
    {"t": 23, "far": 18.9, "frr": 13.1},
    {"t": 24, "far": 15.3, "frr": 16.1},
    {"t": 25, "far": 12.3, "frr": 20.3},
    {"t": 26, "far": 9.3, "frr": 23.5},
    {"t": 27, "far": 8.3, "frr": 27.7},
    {"t": 28, "far": 7.0, "frr": 31.6},
]

ACCURACY = {"1yr": 16.8, "3yr": 42.1, "5yr": 61.1, "10yr": 87.3}

TRAIN_MAE = [13.66,8.21,7.30,6.78,6.34,5.94,5.55,5.33,5.14,4.78,4.47,4.26,3.95,3.84,3.65,3.32,3.25,3.18,3.04,3.01]
VAL_MAE = [8.25,6.88,7.02,6.04,6.48,8.49,6.95,6.07,5.98,5.38,5.34,5.51,5.29,5.30,5.43,5.16,5.10,5.16,5.09,5.10]
TRAIN_LOSS = [13.17,7.72,6.81,6.30,5.86,5.46,5.07,4.86,4.67,4.31,3.99,3.79,3.48,3.38,3.19,2.86,2.79,2.72,2.59,2.56]
VAL_LOSS = [7.76,6.40,6.54,5.56,6.00,8.01,6.47,5.59,5.51,4.91,4.86,5.04,4.82,4.83,4.96,4.69,4.63,4.69,4.62,4.62]

# ── Styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: #0a0a12 !important; color: #b0b0bc; font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    .block-container { padding: 1.5rem 2.5rem; max-width: 1400px; }
    h1,h2,h3,h4 { color: #d8d8e4 !important; font-family: 'DM Sans', sans-serif !important; }
    .k {
        background: linear-gradient(145deg, #111120, #0e0e1a);
        border: 1px solid #1c1c30; border-radius: 10px;
        padding: 1.1rem 0.9rem; text-align: center;
    }
    .kl { font-size: 0.65rem; color: #55556a; text-transform: uppercase; letter-spacing: 1.2px; }
    .kv { font-size: 1.7rem; font-weight: 700; margin: 0.3rem 0 0.15rem; }
    .ks { font-size: 0.65rem; color: #3a3a4a; }
    .g { color: #34d399; } .a { color: #fbbf24; } .r { color: #f87171; } .b { color: #60a5fa; } .w { color: #d0d0dc; }
    .sec { font-size: 0.75rem; font-weight: 700; color: #55556a; text-transform: uppercase;
           letter-spacing: 1.5px; margin: 2rem 0 0.8rem; padding-bottom: 0.4rem; border-bottom: 1px solid #151525; }
    .pc { border-radius: 8px; padding: 0.9rem 1rem; }
    .pr { background: rgba(248,113,113,0.04); border-left: 3px solid #f87171; }
    .py { background: rgba(251,191,36,0.04); border-left: 3px solid #fbbf24; }
    .pg { background: rgba(52,211,153,0.04); border-left: 3px solid #34d399; }
    .rc {
        background: linear-gradient(145deg, #0a1a12, #0e1e16);
        border: 1px solid #152a1e; border-radius: 10px;
        padding: 1.1rem; text-align: center;
    }
    .rv { font-size: 1.5rem; font-weight: 700; color: #34d399; }
    .rl { font-size: 0.65rem; color: #3a5a4a; margin-top: 0.3rem; line-height: 1.4; }
    .ft { text-align: center; color: #2a2a3a; font-size: 0.65rem; padding: 1.5rem 0 0.5rem; }
    .stTabs [data-baseweb="tab"] { color: #55556a; font-size: 0.8rem; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #d8d8e4 !important; border-bottom-color: #34d399 !important; }
</style>
""", unsafe_allow_html=True)

PL = dict(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#6a6a7a", size=11),
    margin=dict(l=15, r=15, t=35, b=15),
    xaxis=dict(gridcolor="#12121e", zerolinecolor="#12121e"),
    yaxis=dict(gridcolor="#12121e", zerolinecolor="#12121e"),
    legend=dict(orientation="h", y=1.1, font=dict(size=10, color="#6a6a7a")),
)

# ── Header ───────────────────────────────────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("<h1 style='font-size:1.6rem;margin:0;letter-spacing:-0.5px;'>AgeGuard AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#44445a;font-size:0.8rem;margin-top:2px;'>Retail compliance dashboard — age verification system performance</p>", unsafe_allow_html=True)
with c2:
    st.markdown("""<div style='text-align:right;padding-top:0.5rem;font-size:0.7rem;'>
    <a href='https://huggingface.co/spaces/marianunez-data/AgeGuard-AI' target='_blank' style='color:#60a5fa;text-decoration:none;margin-right:1rem;'>Live demo</a>
    <a href='https://github.com/marianunez-data/AgeGuard-AI' target='_blank' style='color:#60a5fa;text-decoration:none;'>GitHub</a>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1px;background:#151525;margin:0.8rem 0 1.2rem;'></div>", unsafe_allow_html=True)

# ── KPIs ─────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
kpis = [
    (c1, "Test MAE", f"{METRICS['test_mae']}", "yr", "g", "Global accuracy on unseen data"),
    (c2, "18-25 band", f"{METRICS['critical_mae']}", "yr", "g", "Critical compliance zone"),
    (c3, "Inference", f"{METRICS['inference_ms']}", "ms", "b", "ONNX per-image latency"),
    (c4, "Auto-approval", f"{METRICS['auto_approval']}", "%", "g", "Adults cleared instantly"),
    (c5, "False accept", f"{METRICS['far_25']}", "%", "a", "Minors missed at threshold 25"),
    (c6, "False reject", f"{METRICS['frr_25']}", "%", "b", "Adults flagged unnecessarily"),
]
for col, label, val, unit, color, sub in kpis:
    with col:
        st.markdown(f"""<div class="k"><div class="kl">{label}</div>
        <div class="kv {color}">{val}<span style="font-size:0.8rem;font-weight:400;"> {unit}</span></div>
        <div class="ks">{sub}</div></div>""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Model performance", "Threshold analysis", "Training history", "Business case"])

with tab1:
    ca, cb = st.columns(2)
    with ca:
        names = [b["band"] for b in BANDS]
        maes = [b["mae"] for b in BANDS]
        colors = ["#34d399" if m <= 5 else "#f87171" for m in maes]
        fig = go.Figure(go.Bar(y=names, x=maes, orientation="h", marker_color=colors,
            text=[f"{m} yr" for m in maes], textposition="outside", textfont=dict(size=11, color="#6a6a7a")))
        fig.add_vline(x=5.0, line_dash="dash", line_color="#333345",
                      annotation_text="5.0 target", annotation_font_color="#55556a", annotation_font_size=10)
        fig.update_layout(title=dict(text="Prediction error by age group", font=dict(size=13, color="#b0b0bc")),
            height=340, xaxis=dict(title="Mean absolute error (years)", range=[0, 12], **PL["xaxis"]),
            yaxis=dict(autorange="reversed", **PL["yaxis"]),
            **{k:v for k,v in PL.items() if k not in ["xaxis","yaxis"]})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<p style='font-size:0.7rem;color:#3a3a4a;margin-top:-10px;'>Green bars are within the 5-year business target. The model is most accurate in the 18-25 compliance zone (4.34 yr) where precision matters most.</p>", unsafe_allow_html=True)

    with cb:
        tols = ["Within 1 yr", "Within 3 yr", "Within 5 yr", "Within 10 yr"]
        accs = [ACCURACY["1yr"], ACCURACY["3yr"], ACCURACY["5yr"], ACCURACY["10yr"]]
        bar_colors = ["#f87171", "#fbbf24", "#60a5fa", "#34d399"]
        fig2 = go.Figure(go.Bar(x=tols, y=accs, marker_color=bar_colors,
            text=[f"{a}%" for a in accs], textposition="outside", textfont=dict(size=12, color="#6a6a7a")))
        fig2.add_hline(y=90, line_dash="dash", line_color="#333345",
                       annotation_text="90%", annotation_font_color="#55556a", annotation_font_size=10)
        fig2.update_layout(title=dict(text="How often is the prediction close enough?", font=dict(size=13, color="#b0b0bc")),
            height=340, yaxis=dict(title="% of all predictions", range=[0, 105], **PL["yaxis"]),
            xaxis=dict(**PL["xaxis"]), **{k:v for k,v in PL.items() if k not in ["xaxis","yaxis"]})
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("<p style='font-size:0.7rem;color:#3a3a4a;margin-top:-10px;'>87.3% of predictions fall within 10 years of the real age. For the compliance use case, what matters most is the 18-25 band accuracy (4.34 yr MAE).</p>", unsafe_allow_html=True)

with tab2:
    ca, cb = st.columns([2, 1])
    with ca:
        ts = [t["t"] for t in THRESHOLDS]
        fars = [t["far"] for t in THRESHOLDS]
        frrs = [t["frr"] for t in THRESHOLDS]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=ts, y=fars, mode="lines+markers", name="False accept rate",
            line=dict(color="#f87171", width=2.5), marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.04)"))
        fig3.add_trace(go.Scatter(x=ts, y=frrs, mode="lines+markers", name="False reject rate",
            line=dict(color="#60a5fa", width=2.5), marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(96,165,250,0.04)"))
        fig3.add_vline(x=25, line_dash="dash", line_color="#34d399",
                       annotation_text="Recommended: 25", annotation_font_color="#34d399", annotation_font_size=10)
        fig3.update_layout(title=dict(text="Compliance risk vs customer friction", font=dict(size=13, color="#b0b0bc")),
            height=380, xaxis=dict(title="Alert threshold (years)", **PL["xaxis"]),
            yaxis=dict(title="Rate (%)", range=[0, 35], **PL["yaxis"]),
            **{k:v for k,v in PL.items() if k not in ["xaxis","yaxis"]})
        st.plotly_chart(fig3, use_container_width=True)

    with cb:
        st.markdown("<div class='sec'>What this means</div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='font-size:0.78rem;color:#7a7a8a;line-height:1.6;'>
        The <span style='color:#f87171;'>false accept rate (FAR)</span> measures how many minors slip through undetected.
        The <span style='color:#60a5fa;'>false reject rate (FRR)</span> measures how many adults are unnecessarily asked for ID.
        </p>
        <p style='font-size:0.78rem;color:#7a7a8a;line-height:1.6;'>
        At the legal age of 21, FAR is 28.9% — too high for compliance. By raising the threshold to 25,
        we add a 4-year safety margin: anyone predicted under 25 must show ID. This drops FAR to 12.3%.
        </p>
        <p style='font-size:0.78rem;color:#7a7a8a;line-height:1.6;'>
        The tradeoff: 20.3% of adults get asked for ID. This is a minor inconvenience (5 seconds), not a lost sale.
        The complete system (AI + human verification) achieves near-zero compliance risk.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("<div class='sec'>Alert protocol</div>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""<div class="pc pr"><div style="font-weight:700;color:#f87171;font-size:0.85rem;">BLOCK — Predicted under 21</div>
        <div style="font-size:0.75rem;color:#7a7a8a;margin-top:4px;">Sale denied. Supervisor alerted immediately. Prevents regulatory fines of $10K-$100K per violation.</div></div>""", unsafe_allow_html=True)
    with p2:
        st.markdown("""<div class="pc py"><div style="font-weight:700;color:#fbbf24;font-size:0.85rem;">VERIFY — Predicted 21 to 25</div>
        <div style="font-size:0.75rem;color:#7a7a8a;margin-top:4px;">Customer asked to present physical ID before sale. Takes 5 seconds. No revenue impact.</div></div>""", unsafe_allow_html=True)
    with p3:
        st.markdown("""<div class="pc pg"><div style="font-weight:700;color:#34d399;font-size:0.85rem;">APPROVED — Predicted over 25</div>
        <div style="font-size:0.75rem;color:#7a7a8a;margin-top:4px;">Sale approved automatically. Zero friction. Covers 79.7% of adult customers.</div></div>""", unsafe_allow_html=True)

with tab3:
    ca, cb = st.columns(2)
    epochs = list(range(1, 21))
    with ca:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=epochs, y=TRAIN_MAE, mode="lines", name="Train",
            line=dict(color="#60a5fa", width=2)))
        fig4.add_trace(go.Scatter(x=epochs, y=VAL_MAE, mode="lines", name="Validation",
            line=dict(color="#f87171", width=2)))
        fig4.add_hline(y=5.0, line_dash="dash", line_color="#34d399",
                       annotation_text="Target", annotation_font_color="#34d399")
        fig4.update_layout(title=dict(text="MAE over training epochs", font=dict(size=13, color="#b0b0bc")),
            height=350, xaxis=dict(title="Epoch", **PL["xaxis"]),
            yaxis=dict(title="MAE (years)", **PL["yaxis"]),
            **{k:v for k,v in PL.items() if k not in ["xaxis","yaxis"]})
        st.plotly_chart(fig4, use_container_width=True)
    with cb:
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=epochs, y=TRAIN_LOSS, mode="lines", name="Train",
            line=dict(color="#60a5fa", width=2)))
        fig5.add_trace(go.Scatter(x=epochs, y=VAL_LOSS, mode="lines", name="Validation",
            line=dict(color="#f87171", width=2)))
        fig5.update_layout(title=dict(text="Loss convergence (HuberLoss, delta 5.0)", font=dict(size=13, color="#b0b0bc")),
            height=350, xaxis=dict(title="Epoch", **PL["xaxis"]),
            yaxis=dict(title="Loss", **PL["yaxis"]),
            **{k:v for k,v in PL.items() if k not in ["xaxis","yaxis"]})
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("<p style='font-size:0.75rem;color:#44445a;line-height:1.6;'>The model was trained for 20 epochs on an RTX 4080 GPU (6.2 minutes total). Validation MAE stabilized around 5.0-5.5 from epoch 10 onward. The train-val gap of ~2 years indicates mild overfitting — acceptable for a first training run. Best checkpoint saved at epoch 19 (val MAE 5.09).</p>", unsafe_allow_html=True)

with tab4:
    st.markdown("<div class='sec'>Why this system pays for itself</div>", unsafe_allow_html=True)
    st.markdown("""<p style='font-size:0.8rem;color:#7a7a8a;line-height:1.7;margin-bottom:1.5rem;'>
    A single underage sale violation costs retailers $10,000-$100,000 in fines, potential license revocation,
    and reputational damage. AgeGuard AI adds a consistent, tireless first layer of verification that catches
    what humans miss during rush hours, shift changes, and high-pressure moments.</p>""", unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown("""<div class="rc"><div class="rv">$85K</div>
        <div class="rl">Estimated annual fines avoided per store, based on industry average of 2 violations per year at $10K-$50K each</div></div>""", unsafe_allow_html=True)
    with r2:
        st.markdown("""<div class="rc"><div class="rv">340 hrs</div>
        <div class="rl">Staff hours redirected annually — 79.7% of transactions auto-approved, freeing employees for customer service</div></div>""", unsafe_allow_html=True)
    with r3:
        st.markdown("""<div class="rc"><div class="rv">< 6 months</div>
        <div class="rl">Estimated payback period on hardware and deployment costs based on fine avoidance alone</div></div>""", unsafe_allow_html=True)
    with r4:
        st.markdown("""<div class="rc"><div class="rv">17 ms</div>
        <div class="rl">Per-image processing time — fast enough for real-time video at 60 FPS with zero checkout delay</div></div>""", unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.7rem;color:#3a3a4a;margin-top:1rem;'>ROI projections are estimates based on published industry data on age-restricted sale violations. Actual results vary by store volume, location, and local regulations.</p>", unsafe_allow_html=True)

    st.markdown("<div class='sec'>Technical specifications</div>", unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("""
| Component | Detail |
|---|---|
| Architecture | EfficientNetV2-S (20.3M parameters) |
| Loss function | HuberLoss (delta 5.0) |
| Optimizer | AdamW (lr 3e-4, weight decay 1e-4) |
| Training | 20 epochs, RTX 4080, 6.2 minutes |
| Export format | ONNX (77.5 MB) |
| Quality assurance | 21/21 pytest tests passing |
""")
    with s2:
        st.markdown("""
| Pipeline step | Result |
|---|---|
| Original dataset | 7,590 images (UTKFace) |
| After quality cleanup | 7,446 images |
| Face detection and crop | 224x224 px with 40% margin |
| Train/val/test split | 70/15/15 stratified by age |
| Data validation | Great Expectations (2 gates) |
| Visual audit | 151 candidates manually reviewed |
""")

st.markdown("<div class='ft'>AgeGuard AI v1.0 — EfficientNetV2-S, ONNX Runtime, PyTorch</div>", unsafe_allow_html=True)
