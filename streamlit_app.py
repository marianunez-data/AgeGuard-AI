import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="AgeGuard AI", layout="wide", initial_sidebar_state="collapsed")

# ── Data ─────────────────────────────────────────────────────────
BANDS = [
    {"band": "0-17", "mae": 3.69, "n": 218},
    {"band": "18-25", "mae": 4.34, "n": 228},
    {"band": "26-35", "mae": 4.56, "n": 289},
    {"band": "36-45", "mae": 5.41, "n": 169},
    {"band": "46-60", "mae": 6.49, "n": 154},
    {"band": "60+", "mae": 9.98, "n": 59},
]

THRESHOLDS = [
    {"t": 21, "far": 28.9, "frr": 7.2, "minors": 87},
    {"t": 22, "far": 24.3, "frr": 9.3, "minors": 73},
    {"t": 23, "far": 18.9, "frr": 13.1, "minors": 57},
    {"t": 24, "far": 15.3, "frr": 16.1, "minors": 46},
    {"t": 25, "far": 12.3, "frr": 20.3, "minors": 37},
    {"t": 26, "far": 9.3, "frr": 23.5, "minors": 28},
    {"t": 27, "far": 8.3, "frr": 27.7, "minors": 25},
    {"t": 28, "far": 7.0, "frr": 31.6, "minors": 21},
]

TRAIN_MAE = [13.66,8.21,7.30,6.78,6.34,5.94,5.55,5.33,5.14,4.78,4.47,4.26,3.95,3.84,3.65,3.32,3.25,3.18,3.04,3.01]
VAL_MAE = [8.25,6.88,7.02,6.04,6.48,8.49,6.95,6.07,5.98,5.38,5.34,5.51,5.29,5.30,5.43,5.16,5.10,5.16,5.09,5.10]
TRAIN_LOSS = [13.17,7.72,6.81,6.30,5.86,5.46,5.07,4.86,4.67,4.31,3.99,3.79,3.48,3.38,3.19,2.86,2.79,2.72,2.59,2.56]
VAL_LOSS = [7.76,6.40,6.54,5.56,6.00,8.01,6.47,5.59,5.51,4.91,4.86,5.04,4.82,4.83,4.96,4.69,4.63,4.69,4.62,4.62]

# ── Style ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: #080810 !important;
        color: #c0c0cc;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    .block-container { padding: 2rem 3rem; max-width: 1300px; }
    h1,h2,h3 { font-family: 'Inter', sans-serif !important; }

    .hero {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
    }
    .hero h1 {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -1px;
        margin-bottom: 0.3rem;
    }
    .hero p {
        font-size: 1rem;
        color: #5a5a6e;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    .hero-links {
        margin-top: 1rem;
        font-size: 0.85rem;
    }
    .hero-links a {
        color: #60a5fa;
        text-decoration: none;
        margin: 0 1rem;
        padding: 0.4rem 1.2rem;
        border: 1px solid #1a1a30;
        border-radius: 20px;
        transition: all 0.2s;
    }
    .hero-links a:hover { border-color: #60a5fa; background: rgba(96,165,250,0.08); }

    .divider { height: 1px; background: linear-gradient(90deg, transparent, #1a1a2e, transparent); margin: 1.5rem 0; }

    .kpi-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 1.5rem 0; }
    .kpi {
        background: linear-gradient(145deg, #0f0f1c, #0c0c18);
        border: 1px solid #16162a;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .kpi:hover { border-color: #2a2a44; }
    .kpi-label { font-size: 0.75rem; color: #4a4a5e; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
    .kpi-num { font-size: 2.8rem; font-weight: 700; margin: 0.5rem 0 0.2rem; line-height: 1; }
    .kpi-desc { font-size: 0.8rem; color: #4a4a5e; line-height: 1.4; }
    .green { color: #34d399; } .amber { color: #fbbf24; } .blue { color: #60a5fa; } .red { color: #f87171; }

    .card {
        background: #0c0c18;
        border: 1px solid #16162a;
        border-radius: 14px;
        padding: 1.8rem;
        margin-bottom: 1rem;
    }
    .card-title { font-size: 1.1rem; font-weight: 600; color: #d0d0dc; margin-bottom: 0.3rem; }
    .card-sub { font-size: 0.8rem; color: #4a4a5e; margin-bottom: 1.2rem; line-height: 1.5; }

    .alert-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .alert-red { background: rgba(248,113,113,0.06); border: 1px solid rgba(248,113,113,0.15); }
    .alert-yellow { background: rgba(251,191,36,0.06); border: 1px solid rgba(251,191,36,0.15); }
    .alert-green { background: rgba(52,211,153,0.06); border: 1px solid rgba(52,211,153,0.15); }
    .alert-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.4rem; }
    .alert-desc { font-size: 0.85rem; color: #7a7a8e; line-height: 1.5; }
    .alert-impact { font-size: 0.75rem; color: #5a5a6e; margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid rgba(255,255,255,0.04); }

    .roi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 1rem 0; }
    .roi {
        background: linear-gradient(145deg, #091a10, #0a1a14);
        border: 1px solid #14301e;
        border-radius: 12px;
        padding: 1.3rem;
        text-align: center;
    }
    .roi-num { font-size: 2rem; font-weight: 700; color: #34d399; }
    .roi-label { font-size: 0.75rem; color: #3a6a4a; margin-top: 0.4rem; line-height: 1.4; }

    .insight {
        background: rgba(96,165,250,0.04);
        border: 1px solid rgba(96,165,250,0.1);
        border-radius: 10px;
        padding: 1rem 1.3rem;
        font-size: 0.82rem;
        color: #8a8a9e;
        line-height: 1.6;
        margin: 0.8rem 0;
    }
    .insight strong { color: #60a5fa; }

    .section-label {
        font-size: 0.7rem; font-weight: 700; color: #3a3a4e;
        text-transform: uppercase; letter-spacing: 2px;
        margin: 2.5rem 0 1rem; padding-bottom: 0.5rem;
        border-bottom: 1px solid #111120;
    }

    .footer { text-align: center; color: #2a2a3a; font-size: 0.7rem; padding: 3rem 0 1rem; }

    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #16162a; }
    .stTabs [data-baseweb="tab"] {
        color: #4a4a5e; font-size: 0.9rem; font-weight: 500;
        padding: 0.8rem 1.5rem; border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #e0e0ec !important;
        border-bottom-color: #34d399 !important;
    }
</style>
""", unsafe_allow_html=True)

PL = dict(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#6a6a7e", size=13),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor="#0e0e1a", zerolinecolor="#0e0e1a"),
    yaxis=dict(gridcolor="#0e0e1a", zerolinecolor="#0e0e1a"),
    legend=dict(orientation="h", y=1.08, font=dict(size=12, color="#7a7a8e")),
)

# ── Hero ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>AgeGuard AI</h1>
    <p>Intelligent age verification for retail compliance. Automated facial analysis that protects your business from underage sale violations.</p>
    <div class="hero-links">
        <a href="https://huggingface.co/spaces/marianunez-data/AgeGuard-AI" target="_blank">Try the live demo</a>
        <a href="https://github.com/marianunez-data/AgeGuard-AI" target="_blank">View source code</a>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── KPIs ─────────────────────────────────────────────────────────
st.markdown("""
<div class="kpi-grid">
    <div class="kpi">
        <div class="kpi-label">Prediction accuracy</div>
        <div class="kpi-num green">5.02<span style="font-size:1.2rem;"> yr</span></div>
        <div class="kpi-desc">Average error on unseen test data. The system predicts age within 5 years on average.</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Critical zone accuracy</div>
        <div class="kpi-num green">4.34<span style="font-size:1.2rem;"> yr</span></div>
        <div class="kpi-desc">Error for ages 18-25 — the zone where compliance decisions are made. Below the 5-year target.</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Processing speed</div>
        <div class="kpi-num blue">17<span style="font-size:1.2rem;"> ms</span></div>
        <div class="kpi-desc">Time to analyze one face. Fast enough for live video — zero delay at checkout.</div>
    </div>
</div>
<div class="kpi-grid">
    <div class="kpi">
        <div class="kpi-label">Auto-approval rate</div>
        <div class="kpi-num green">79.7<span style="font-size:1.2rem;"> %</span></div>
        <div class="kpi-desc">Percentage of adult customers approved instantly without needing to show ID.</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Minors detected</div>
        <div class="kpi-num amber">87.7<span style="font-size:1.2rem;"> %</span></div>
        <div class="kpi-desc">Of underage customers are correctly flagged by the system at threshold 25.</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">ID requests</div>
        <div class="kpi-num blue">20.3<span style="font-size:1.2rem;"> %</span></div>
        <div class="kpi-desc">Of adult customers asked to show ID. A 5-second step — no lost sales.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["How accurate is it?", "How safe is it?", "How was it built?", "What is the ROI?"])

# ── TAB 1 ────────────────────────────────────────────────────────
with tab1:
    st.markdown("")
    ca, cb = st.columns(2)

    with ca:
        st.markdown("""<div class="card">
        <div class="card-title">Accuracy across age groups</div>
        <div class="card-sub">Green bars are within the 5-year business target. The system is most precise where it matters — ages 18-25.</div>
        </div>""", unsafe_allow_html=True)

        names = [b["band"] for b in BANDS]
        maes = [b["mae"] for b in BANDS]
        colors = ["#34d399" if m <= 5 else "#f87171" for m in maes]
        fig = go.Figure(go.Bar(y=names, x=maes, orientation="h", marker_color=colors,
            marker_line_width=0, text=[f"{m} years" for m in maes],
            textposition="outside", textfont=dict(size=14, color="#7a7a8e")))
        fig.add_vline(x=5.0, line_dash="dash", line_color="#2a2a3e",
                      annotation_text="5 yr target", annotation_font=dict(color="#4a4a5e", size=12))
        fig.update_layout(height=380, xaxis=dict(title="Average prediction error", range=[0, 13], titlefont=dict(size=13), **PL["xaxis"]),
            yaxis=dict(autorange="reversed", tickfont=dict(size=14), **PL["yaxis"]),
            **{k:v for k,v in PL.items() if k not in ["xaxis","yaxis"]})
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        st.markdown("""<div class="card">
        <div class="card-title">How close are the predictions?</div>
        <div class="card-sub">Percentage of all predictions that fall within a given number of years of the real age.</div>
        </div>""", unsafe_allow_html=True)

        tols = ["Within\n1 year", "Within\n3 years", "Within\n5 years", "Within\n10 years"]
        accs = [16.8, 42.1, 61.1, 87.3]
        fig2 = go.Figure(go.Bar(x=tols, y=accs,
            marker_color=["#f87171", "#fbbf24", "#60a5fa", "#34d399"],
            marker_line_width=0,
            text=[f"{a}%" for a in accs], textposition="outside",
            textfont=dict(size=16, color="#9a9aae")))
        fig2.update_layout(height=380,
            yaxis=dict(title="% of predictions", range=[0, 108], titlefont=dict(size=13), **PL["yaxis"]),
            xaxis=dict(tickfont=dict(size=13), **PL["xaxis"]),
            **{k:v for k,v in PL.items() if k not in ["xaxis","yaxis"]})
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""<div class="insight">
    <strong>Key takeaway:</strong> 87% of predictions are within 10 years of the real age. More importantly,
    the system achieves 4.34 years average error in the 18-25 age band — the exact zone where ID verification
    decisions are made. This means the system is most accurate where accuracy matters most for your business.
    </div>""", unsafe_allow_html=True)

# ── TAB 2 ────────────────────────────────────────────────────────
with tab2:
    st.markdown("")
    st.markdown("""<div class="card">
    <div class="card-title">Understanding the safety threshold</div>
    <div class="card-sub">The legal drinking age is 21. If we only flagged people predicted under 21, some minors would slip through.
    By setting the threshold at 25, we create a 4-year safety buffer — anyone predicted under 25 must show ID.
    The red line shows the risk (minors missed), the blue line shows the inconvenience (adults asked for ID).</div>
    </div>""", unsafe_allow_html=True)

    ca, cb = st.columns([3, 2])
    with ca:
        ts = [t["t"] for t in THRESHOLDS]
        fars = [t["far"] for t in THRESHOLDS]
        frrs = [t["frr"] for t in THRESHOLDS]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=ts, y=fars, mode="lines+markers",
            name="Risk: minors not flagged",
            line=dict(color="#f87171", width=3), marker=dict(size=8),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.05)"))
        fig3.add_trace(go.Scatter(x=ts, y=frrs, mode="lines+markers",
            name="Inconvenience: adults asked for ID",
            line=dict(color="#60a5fa", width=3), marker=dict(size=8),
            fill="tozeroy", fillcolor="rgba(96,165,250,0.05)"))
        fig3.add_vline(x=25, line_dash="dash", line_color="#34d399", line_width=2,
                       annotation_text="Recommended threshold: 25",
                       annotation_font=dict(color="#34d399", size=13),
                       annotation_yshift=10)
        fig3.update_layout(height=420,
            xaxis=dict(title="Alert threshold (years)", dtick=1, titlefont=dict(size=14), tickfont=dict(size=13), **PL["xaxis"]),
            yaxis=dict(title="Percentage (%)", range=[0, 36], titlefont=dict(size=14), tickfont=dict(size=13), **PL["yaxis"]),
            **{k:v for k,v in PL.items() if k not in ["xaxis","yaxis"]})
        st.plotly_chart(fig3, use_container_width=True)

    with cb:
        st.markdown("<div class='section-label'>What happens at each threshold</div>", unsafe_allow_html=True)
        for t in THRESHOLDS:
            is_rec = t["t"] == 25
            bg = "rgba(52,211,153,0.06)" if is_rec else "transparent"
            border = "1px solid #1a3a2a" if is_rec else "1px solid #111120"
            badge = "<span style='background:#34d399;color:#080810;font-size:0.65rem;padding:2px 8px;border-radius:10px;margin-left:8px;font-weight:600;'>RECOMMENDED</span>" if is_rec else ""
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:8px;padding:0.7rem 1rem;margin-bottom:5px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="color:#b0b0bc;font-size:0.9rem;font-weight:600;">Age {t['t']}{badge}</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:6px;font-size:0.8rem;">
                    <span style="color:#f87171;">Risk: {t['far']}%</span>
                    <span style="color:#60a5fa;">ID requests: {t['frr']}%</span>
                    <span style="color:#6a6a7e;">{t['minors']} minors missed</span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Three-level alert system</div>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""<div class="alert-card alert-red">
        <div class="alert-title red">BLOCK</div>
        <div class="alert-desc">Predicted age is under 21. The sale is denied and the supervisor is alerted immediately.</div>
        <div class="alert-impact">This prevents regulatory fines of $10,000 to $100,000 per violation and protects your liquor license.</div>
        </div>""", unsafe_allow_html=True)
    with p2:
        st.markdown("""<div class="alert-card alert-yellow">
        <div class="alert-title amber">VERIFY</div>
        <div class="alert-desc">Predicted age is between 21 and 25. The customer is politely asked to show a physical ID before the sale.</div>
        <div class="alert-impact">Takes about 5 seconds. No sale is lost — the customer simply shows their ID and proceeds.</div>
        </div>""", unsafe_allow_html=True)
    with p3:
        st.markdown("""<div class="alert-card alert-green">
        <div class="alert-title green">APPROVED</div>
        <div class="alert-desc">Predicted age is over 25. The sale goes through automatically with zero friction at checkout.</div>
        <div class="alert-impact">This covers 79.7% of all adult customers — the vast majority experience no delay at all.</div>
        </div>""", unsafe_allow_html=True)

# ── TAB 3 ────────────────────────────────────────────────────────
with tab3:
    st.markdown("")
    st.markdown("""<div class="card">
    <div class="card-title">Training the AI model</div>
    <div class="card-sub">The model was trained on 5,212 face images over 20 learning cycles (epochs). The blue line shows
    how well it learned the training data, and the red line shows how well it performs on new, unseen data.
    When both lines are close, the model generalizes well.</div>
    </div>""", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    epochs = list(range(1, 21))
    with ca:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=epochs, y=TRAIN_MAE, mode="lines", name="Learning accuracy",
            line=dict(color="#60a5fa", width=2.5)))
        fig4.add_trace(go.Scatter(x=epochs, y=VAL_MAE, mode="lines", name="Real-world accuracy",
            line=dict(color="#f87171", width=2.5)))
        fig4.add_hline(y=5.0, line_dash="dash", line_color="#34d399", line_width=1.5,
                       annotation_text="5 yr target", annotation_font=dict(color="#34d399", size=12))
        fig4.update_layout(title=dict(text="Prediction error over training", font=dict(size=14, color="#b0b0bc")),
            height=380, xaxis=dict(title="Training cycle", titlefont=dict(size=13), **PL["xaxis"]),
            yaxis=dict(title="Error (years)", titlefont=dict(size=13), **PL["yaxis"]),
            **{k:v for k,v in PL.items() if k not in ["xaxis","yaxis"]})
        st.plotly_chart(fig4, use_container_width=True)

    with cb:
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=epochs, y=TRAIN_LOSS, mode="lines", name="Training loss",
            line=dict(color="#60a5fa", width=2.5)))
        fig5.add_trace(go.Scatter(x=epochs, y=VAL_LOSS, mode="lines", name="Validation loss",
            line=dict(color="#f87171", width=2.5)))
        fig5.update_layout(title=dict(text="Model learning progress", font=dict(size=14, color="#b0b0bc")),
            height=380, xaxis=dict(title="Training cycle", titlefont=dict(size=13), **PL["xaxis"]),
            yaxis=dict(title="Loss value", titlefont=dict(size=13), **PL["yaxis"]),
            **{k:v for k,v in PL.items() if k not in ["xaxis","yaxis"]})
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("""<div class="insight">
    <strong>Training summary:</strong> The model trained in 6.2 minutes on an NVIDIA RTX 4080 GPU.
    It reached its best performance at cycle 19 out of 20, achieving a validation error of 5.09 years.
    The final exported model is 77.5 MB — compact enough to run on any modern computer or edge device at the point of sale.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Technical specifications</div>", unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("""
| Component | Detail |
|---|---|
| AI model | EfficientNetV2-S (20.3M parameters) |
| Error handling | HuberLoss (tolerant to mislabeled data) |
| Learning method | AdamW optimizer with cosine scheduling |
| Training hardware | NVIDIA RTX 4080 GPU, 6.2 minutes |
| Production format | ONNX (77.5 MB, cross-platform) |
| Quality assurance | 21 automated tests, all passing |
""")
    with s2:
        st.markdown("""
| Data pipeline | Result |
|---|---|
| Starting dataset | 7,590 facial images |
| After quality review | 7,446 images (151 manually audited) |
| Face detection | OpenCV DNN with 40% safety margin |
| Image standardization | 224 x 224 pixels, face-centered |
| Data split | 70% training, 15% validation, 15% testing |
| Data validation | 2 automated quality gates passed |
""")

# ── TAB 4 ────────────────────────────────────────────────────────
with tab4:
    st.markdown("")
    st.markdown("""<div class="card">
    <div class="card-title">Return on investment</div>
    <div class="card-sub">A single underage sale violation costs retailers $10,000 to $100,000 in fines, potential license
    revocation, and reputational damage. AgeGuard AI adds a consistent, tireless verification layer that catches
    what employees miss during rush hours, shift changes, and high-pressure moments.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="roi-grid">
        <div class="roi">
            <div class="roi-num">$85K</div>
            <div class="roi-label">Estimated fines avoided per store per year, based on industry average of 2 violations annually</div>
        </div>
        <div class="roi">
            <div class="roi-num">340 hrs</div>
            <div class="roi-label">Staff hours saved annually by auto-approving 80% of transactions without manual verification</div>
        </div>
        <div class="roi">
            <div class="roi-num">< 6 mo</div>
            <div class="roi-label">Estimated time to recover hardware and deployment costs from fine avoidance alone</div>
        </div>
        <div class="roi">
            <div class="roi-num">17 ms</div>
            <div class="roi-label">Per-image processing — fast enough for 60 frames per second live video with zero checkout delay</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="insight">
    <strong>Bottom line:</strong> The system pays for itself by preventing just one regulatory fine.
    Beyond fines, it reduces liability exposure, demonstrates compliance diligence to regulators,
    and frees staff to focus on customer service instead of age verification during every transaction.
    ROI estimates are based on published industry data and vary by location and volume.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>How it works in your store</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:1rem 0;">
        <div style="background:#0c0c18;border:1px solid #16162a;border-radius:10px;padding:1.2rem;text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">1</div>
            <div style="font-size:0.85rem;color:#b0b0bc;font-weight:600;margin-bottom:0.3rem;">Camera captures</div>
            <div style="font-size:0.75rem;color:#5a5a6e;">Camera at self-checkout detects customer face automatically</div>
        </div>
        <div style="background:#0c0c18;border:1px solid #16162a;border-radius:10px;padding:1.2rem;text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">2</div>
            <div style="font-size:0.85rem;color:#b0b0bc;font-weight:600;margin-bottom:0.3rem;">AI analyzes</div>
            <div style="font-size:0.75rem;color:#5a5a6e;">Model estimates age from facial features in 17 milliseconds</div>
        </div>
        <div style="background:#0c0c18;border:1px solid #16162a;border-radius:10px;padding:1.2rem;text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">3</div>
            <div style="font-size:0.85rem;color:#b0b0bc;font-weight:600;margin-bottom:0.3rem;">Alert triggers</div>
            <div style="font-size:0.75rem;color:#5a5a6e;">System shows green, yellow, or red based on predicted age</div>
        </div>
        <div style="background:#0c0c18;border:1px solid #16162a;border-radius:10px;padding:1.2rem;text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">4</div>
            <div style="font-size:0.85rem;color:#b0b0bc;font-weight:600;margin-bottom:0.3rem;">Staff acts</div>
            <div style="font-size:0.75rem;color:#5a5a6e;">Supervisor verifies flagged cases — AI handles the rest</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='footer'>AgeGuard AI v1.0 — Built for retail compliance</div>", unsafe_allow_html=True)
