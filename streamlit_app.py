import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="AgeGuard AI — Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ── Load data from JSON artifacts ────────────────────────────────
@st.cache_data
def load_data():
    base = Path(__file__).parent
    data = {}
    files = {
        "training": base / "data" / "training_summary.json",
        "evaluation": base / "data" / "test_evaluation.json",
        "threshold": base / "data" / "threshold_analysis.json",
        "split": base / "data" / "split_stats.json",
        "onnx": base / "data" / "onnx_export_report.json",
    }
    for key, path in files.items():
        if path.exists():
            with open(path) as f:
                data[key] = json.load(f)
    return data

data = load_data()
training = data.get("training", {})
evaluation = data.get("evaluation", {})
threshold = data.get("threshold", {})
split = data.get("split", {})
onnx = data.get("onnx", {})

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #2a2a4a;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0 0.25rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #666;
        margin-top: 0.25rem;
    }
    .green { color: #00C853; }
    .amber { color: #FFB300; }
    .red { color: #FF4444; }
    .blue { color: #448AFF; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ccc;
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2a2a4a;
    }
    .alert-box {
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .alert-red { background: rgba(255,68,68,0.1); border-left: 4px solid #FF4444; }
    .alert-yellow { background: rgba(255,179,0,0.1); border-left: 4px solid #FFB300; }
    .alert-green { background: rgba(0,200,83,0.1); border-left: 4px solid #00C853; }
    .roi-card {
        background: linear-gradient(135deg, #0a2e1a 0%, #1a3a2e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #1a4a2e;
    }
    .roi-value { font-size: 1.8rem; font-weight: 700; color: #00C853; }
    .roi-label { font-size: 0.8rem; color: #888; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown("# AgeGuard AI")
    st.markdown("**Retail compliance dashboard** — Real-time age verification performance metrics")
with col_badge:
    st.markdown("")
    st.markdown("[![Demo](https://img.shields.io/badge/🤗_Live_Demo-HuggingFace-yellow)](https://huggingface.co/spaces/marianunez-data/AgeGuard-AI)")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/marianunez-data/AgeGuard-AI)")

st.divider()

# ── KPI Metrics Row ──────────────────────────────────────────────
glob = evaluation.get("global_metrics", {})
biz = evaluation.get("business_metrics", {})
bench = onnx.get("benchmark", {})

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Test MAE</div>
        <div class="metric-value green">{glob.get('mae', 'N/A')} yr</div>
        <div class="metric-sub">Target: ≤ 5.0 years</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Critical band MAE</div>
        <div class="metric-value green">{biz.get('alert_zone_mae', 'N/A')} yr</div>
        <div class="metric-sub">Ages 18-25 (decision zone)</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Inference speed</div>
        <div class="metric-value blue">{bench.get('avg_latency_ms', 'N/A')} ms</div>
        <div class="metric-sub">ONNX Runtime (target: &lt;50ms)</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Auto-approval</div>
        <div class="metric-value green">79.7%</div>
        <div class="metric-sub">Adults cleared without ID</div>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">False accept rate</div>
        <div class="metric-value amber">{biz.get('false_accept_rate_pct', 'N/A')}%</div>
        <div class="metric-sub">At threshold = 25</div>
    </div>""", unsafe_allow_html=True)

# ── Charts Row 1 ─────────────────────────────────────────────────
st.markdown('<div class="section-header">Model performance</div>', unsafe_allow_html=True)

col_band, col_train = st.columns(2)

with col_band:
    bands = evaluation.get("per_band_mae", [])
    if bands:
        names = [b["band"] for b in bands]
        maes = [b["mae"] for b in bands]
        colors = ["#00C853" if m <= 5 else "#FF4444" for m in maes]

        fig_band = go.Figure(go.Bar(
            y=names, x=maes, orientation="h",
            marker_color=colors,
            text=[f"{m} yr" for m in maes],
            textposition="outside",
            textfont=dict(size=13),
        ))
        fig_band.add_vline(x=5.0, line_dash="dash", line_color="#888",
                           annotation_text="Target 5.0", annotation_position="top")
        fig_band.update_layout(
            title="MAE by age band",
            height=350, margin=dict(l=20, r=40, t=50, b=20),
            xaxis=dict(title="MAE (years)", range=[0, 12]),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
        )
        st.plotly_chart(fig_band, use_container_width=True)

with col_train:
    history = training.get("history", {})
    if history:
        epochs = list(range(1, len(history["train_mae"]) + 1))
        fig_curves = go.Figure()
        fig_curves.add_trace(go.Scatter(
            x=epochs, y=history["train_mae"], mode="lines+markers",
            name="Train MAE", line=dict(color="#448AFF", width=2),
            marker=dict(size=5),
        ))
        fig_curves.add_trace(go.Scatter(
            x=epochs, y=history["val_mae"], mode="lines+markers",
            name="Val MAE", line=dict(color="#FF4444", width=2),
            marker=dict(size=5),
        ))
        fig_curves.add_hline(y=5.0, line_dash="dash", line_color="#00C853",
                             annotation_text="Target MAE")
        fig_curves.update_layout(
            title="Training curves (MAE)",
            height=350, margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(title="Epoch"),
            yaxis=dict(title="MAE (years)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_curves, use_container_width=True)

# ── Charts Row 2 ─────────────────────────────────────────────────
st.markdown('<div class="section-header">Business threshold optimization</div>', unsafe_allow_html=True)

col_far, col_acc = st.columns(2)

with col_far:
    analysis = threshold.get("analysis", [])
    if analysis:
        thresholds = [a["threshold"] for a in analysis]
        fars = [a["far_pct"] for a in analysis]
        frrs = [a["frr_pct"] for a in analysis]

        fig_far = go.Figure()
        fig_far.add_trace(go.Scatter(
            x=thresholds, y=fars, mode="lines+markers",
            name="FAR (compliance risk)",
            line=dict(color="#FF4444", width=2.5),
            marker=dict(size=7), fill="tozeroy",
            fillcolor="rgba(255,68,68,0.08)",
        ))
        fig_far.add_trace(go.Scatter(
            x=thresholds, y=frrs, mode="lines+markers",
            name="FRR (customer friction)",
            line=dict(color="#448AFF", width=2.5),
            marker=dict(size=7), fill="tozeroy",
            fillcolor="rgba(68,138,255,0.08)",
        ))
        fig_far.add_vline(x=25, line_dash="dash", line_color="#00C853",
                          annotation_text="Recommended θ=25")
        fig_far.update_layout(
            title="FAR vs FRR tradeoff",
            height=350, margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(title="Decision threshold (years)"),
            yaxis=dict(title="Rate (%)", range=[0, 35]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_far, use_container_width=True)

with col_acc:
    acc = evaluation.get("accuracy_within", {})
    if acc:
        tolerances = [1, 3, 5, 10]
        accuracies = [acc.get("1_year", 0), acc.get("3_years", 0),
                      acc.get("5_years", 0), acc.get("10_years", 0)]

        fig_acc = go.Figure(go.Bar(
            x=[f"±{t} yr" for t in tolerances],
            y=accuracies,
            marker_color=["#FF4444", "#FFB300", "#448AFF", "#00C853"],
            text=[f"{a}%" for a in accuracies],
            textposition="outside",
            textfont=dict(size=14, color="#ccc"),
        ))
        fig_acc.add_hline(y=90, line_dash="dash", line_color="#888",
                          annotation_text="90% target")
        fig_acc.update_layout(
            title="Accuracy by error tolerance",
            height=350, margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(title="% of predictions", range=[0, 105]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

# ── Alert Protocol ───────────────────────────────────────────────
st.markdown('<div class="section-header">Alert protocol</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="alert-box alert-red">
        <div>
            <div style="font-size:1.5rem;">🔴</div>
        </div>
        <div>
            <div style="font-weight:600;color:#FF4444;">BLOCK</div>
            <div style="font-size:0.85rem;color:#aaa;">Predicted &lt; 21 — Deny sale, alert supervisor</div>
            <div style="font-size:0.75rem;color:#666;margin-top:4px;">Prevents fines of $10K-$100K per incident</div>
        </div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="alert-box alert-yellow">
        <div>
            <div style="font-size:1.5rem;">🟡</div>
        </div>
        <div>
            <div style="font-weight:600;color:#FFB300;">VERIFY</div>
            <div style="font-size:0.85rem;color:#aaa;">Predicted 21-25 — Request physical ID</div>
            <div style="font-size:0.75rem;color:#666;margin-top:4px;">Minor friction, zero lost sales</div>
        </div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="alert-box alert-green">
        <div>
            <div style="font-size:1.5rem;">🟢</div>
        </div>
        <div>
            <div style="font-weight:600;color:#00C853;">APPROVED</div>
            <div style="font-size:0.85rem;color:#aaa;">Predicted &gt; 25 — Automatic approval</div>
            <div style="font-size:0.75rem;color:#666;margin-top:4px;">Zero friction, maximum throughput</div>
        </div>
    </div>""", unsafe_allow_html=True)

# ── ROI Section ──────────────────────────────────────────────────
st.markdown('<div class="section-header">Projected ROI (per store, annual)</div>', unsafe_allow_html=True)

r1, r2, r3, r4 = st.columns(4)
with r1:
    st.markdown("""
    <div class="roi-card">
        <div class="roi-value">$85K</div>
        <div class="roi-label">Fines avoided<br>(est. 2 incidents/yr)</div>
    </div>""", unsafe_allow_html=True)

with r2:
    st.markdown("""
    <div class="roi-card">
        <div class="roi-value">340 hrs</div>
        <div class="roi-label">Staff hours saved<br>(80% auto-approval)</div>
    </div>""", unsafe_allow_html=True)

with r3:
    st.markdown("""
    <div class="roi-card">
        <div class="roi-value">&lt;6 mo</div>
        <div class="roi-label">Payback period<br>on deployment</div>
    </div>""", unsafe_allow_html=True)

with r4:
    st.markdown("""
    <div class="roi-card">
        <div class="roi-value">60 FPS</div>
        <div class="roi-label">Processing capacity<br>(real-time video)</div>
    </div>""", unsafe_allow_html=True)

# ── Technical Summary ────────────────────────────────────────────
st.markdown('<div class="section-header">Technical summary</div>', unsafe_allow_html=True)

t1, t2 = st.columns(2)

with t1:
    st.markdown("**Model pipeline**")
    st.markdown("""
    | Component | Detail |
    |---|---|
    | Architecture | EfficientNetV2-S (20.3M params) |
    | Loss function | HuberLoss (δ=5.0) |
    | Optimizer | AdamW (lr=3e-4) |
    | Training | 20 epochs, RTX 4080, 6.2 min |
    | Export | ONNX (77.5 MB) |
    | Tests | 21/21 pytest passing |
    """)

with t2:
    st.markdown("**Data pipeline**")
    st.markdown("""
    | Step | Result |
    |---|---|
    | Original dataset | 7,590 images (UTKFace) |
    | After cleaning | 7,446 images |
    | Face cropping | 224×224 px, 40% margin |
    | Split | 70/15/15 stratified |
    | Validation | Great Expectations (2 gates) |
    | Visual review | 151 candidates audited |
    """)

# ── Footer ───────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#666;font-size:0.8rem;'>"
    "AgeGuard AI v1.0 · EfficientNetV2-S · ONNX Runtime · "
    "Built for retail compliance"
    "</div>",
    unsafe_allow_html=True,
)
