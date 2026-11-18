from pathlib import Path
import io
import zipfile
import json
import tempfile

import numpy as np
import cv2
import streamlit as st
from PIL import Image

from src.image_tiling import divide_into_tiles
from src.network_simulator import simulate_packet_loss
from src.rl_environment import ReconstructionTransmissionEnv
from src.dqn_agent import DQNAgent
from src.evaluation import evaluate_image, roi_psnr

st.set_page_config(
    page_title="RECON | Medical Imaging Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A deliberately simple, high-contrast LIGHT theme.
# Avoid broad selectors such as ".stApp div { color: ... }": they override
# Streamlit's internal widget states and can make labels/options disappear.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --page: #f4f7fb;
  --surface: #ffffff;
  --surface-soft: #edf3f8;
  --border: #d8e2ec;
  --ink: #172b3f;
  --muted: #52677d;
  --teal: #087f8c;
  --teal-dark: #06616b;
  --blue: #245e9b;
  --shadow: 0 8px 24px rgba(25, 48, 73, .07);
}

/* App canvas and typography */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
}
.stApp {
  background: var(--page);
  color: var(--ink);
}
[data-testid="stHeader"] {
  background: rgba(244,247,251,.92);
}
.block-container {
  max-width: 1500px;
  padding-top: 1.6rem;
  padding-bottom: 3rem;
}
h1, h2, h3, h4 {
  color: #142b40 !important;
  font-family: 'Space Grotesk', sans-serif !important;
  letter-spacing: -.02em;
}
p, li, .stMarkdown, .stText, .stCaption, small {
  color: #263d53;
}
[data-testid="stCaptionContainer"] p,
.stCaption {
  color: #52677d !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #eaf1f7;
  border-right: 1px solid #d4e0eb;
}
section[data-testid="stSidebar"] > div {
  background: #eaf1f7;
}
section[data-testid="stSidebar"] .block-container {
  padding-top: 1.25rem;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
  color: #19344d !important;
}

/* Header / hero */
.hero {
  padding: 1.65rem 1.8rem;
  border: 1px solid #d4e2ec;
  border-radius: 20px;
  background: linear-gradient(115deg, #ffffff 0%, #e8f4f7 100%);
  box-shadow: var(--shadow);
  margin-bottom: 1.15rem;
}
.eyebrow {
  color: #087f8c !important;
  text-transform: uppercase;
  letter-spacing: .15em;
  font-size: .72rem;
  font-weight: 700;
}
.hero h1 {
  color: #142b40 !important;
  font-size: clamp(1.8rem, 3vw, 2.55rem);
  margin: .3rem 0 .4rem 0;
}
.hero p {
  color: #405a70 !important;
  margin: 0;
  font-size: 1rem;
  line-height: 1.6;
}
.pill {
  display: inline-block;
  border: 1px solid #a8d4d7;
  background: #e2f4f3;
  color: #075c64 !important;
  border-radius: 99px;
  padding: .32rem .72rem;
  font-size: .76rem;
  font-weight: 700;
  margin: .35rem .3rem 0 0;
}
.section-kicker {
  color: #087f8c !important;
  font-size: .74rem;
  letter-spacing: .13em;
  text-transform: uppercase;
  font-weight: 700;
  margin: .9rem 0 .45rem;
}

/* Content panels and custom metric cards */
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1rem 1.1rem;
  box-shadow: var(--shadow);
}
.metric-card {
  background: #ffffff;
  border: 1px solid var(--border);
  border-top: 3px solid #0b8992;
  border-radius: 14px;
  padding: .95rem 1rem;
  min-height: 104px;
  box-shadow: var(--shadow);
}
.metric-label {
  color: #52677d !important;
  font-size: .78rem;
  font-weight: 700;
}
.metric-value {
  color: #142b40 !important;
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.55rem;
  font-weight: 700;
  margin-top: .28rem;
}
.metric-help {
  color: #61758a !important;
  font-size: .72rem;
  margin-top: .15rem;
}

/* Native Streamlit metric widgets */
div[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: .8rem .9rem;
  box-shadow: 0 3px 12px rgba(25,48,73,.04);
}
div[data-testid="stMetricLabel"] p {
  color: #52677d !important;
  font-weight: 600;
}
div[data-testid="stMetricValue"] {
  color: #142b40 !important;
}
div[data-testid="stMetricDelta"] {
  color: #087f8c !important;
}

/* Widget labels and help text */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stRadio > label,
.stCheckbox > label,
.stSlider > label,
.stSelectSlider > label,
.stNumberInput > label,
.stFileUploader > label {
  color: #19344d !important;
  font-weight: 600 !important;
}
[data-testid="stHelp"] {
  color: #52677d !important;
}

/* Text inputs, number inputs, select boxes */
input, textarea {
  color: #172b3f !important;
  background: #ffffff !important;
  -webkit-text-fill-color: #172b3f !important;
  border-color: #bdcbd8 !important;
}
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div,
[data-baseweb="select"] > div {
  background: #ffffff !important;
  border-color: #bdcbd8 !important;
}
[data-baseweb="select"] * {
  color: #172b3f !important;
}
[role="listbox"], [role="option"], [data-baseweb="popover"] {
  background: #ffffff !important;
  color: #172b3f !important;
}
[role="option"]:hover, [aria-selected="true"] {
  background: #e2f2f3 !important;
  color: #123b48 !important;
}

/* Radio buttons, checkboxes and sliders */
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
  color: #19344d !important;
}
[data-testid="stRadio"] p {
  color: #19344d !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"] {
  color: #52677d !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
  background: #ffffff;
  border: 1px dashed #8da7bb;
  border-radius: 13px;
  padding: .55rem;
}
[data-testid="stFileUploader"] section {
  background: #f4f8fb !important;
  border-radius: 9px;
}
[data-testid="stFileUploader"] section *,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
  color: #29445c !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
  color: #ffffff !important;
  background: linear-gradient(100deg, #087f8c, #245e9b) !important;
  border: 1px solid #087f8c !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  min-height: 2.65rem;
}
.stButton > button *,
.stDownloadButton > button * {
  color: #ffffff !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
  background: #06616b !important;
  border-color: #06616b !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: .4rem;
  border-bottom: 1px solid #d4e0eb;
}
.stTabs [data-baseweb="tab"] {
  background: #e8eff5;
  border-radius: 9px 9px 0 0;
  padding: .65rem 1rem;
}
.stTabs [data-baseweb="tab"] * {
  color: #29445c !important;
  font-weight: 600;
}
.stTabs [aria-selected="true"] {
  background: #ffffff !important;
  border-bottom: 3px solid #087f8c !important;
}
.stTabs [aria-selected="true"] * {
  color: #075f68 !important;
}

/* Alerts, tables, charts, separators */
[data-testid="stAlert"] {
  border-radius: 12px;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
  color: #263d53 !important;
}
[data-testid="stDataFrame"],
[data-testid="stTable"] {
  color: #172b3f !important;
}
hr {
  border-color: #d4e0eb !important;
}
footer {
  visibility: hidden;
}

/* Responsive spacing */
@media (max-width: 800px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .hero { padding: 1.2rem; }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------- HELPERS ---------------------------------
@st.cache_resource
def load_agent():
    path = Path("models/recon_dqn.pt")
    if not path.exists():
        return None, "No trained 3-action checkpoint found. The app will use a clearly labelled rule-based fallback."
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("state_dim") != 13 or ck.get("action_dim") != 3:
            return None, "Incompatible checkpoint: expected 13 states and 3 actions. Retrain the model."
        agent = DQNAgent(state_dim=13, action_dim=3, device="cpu")
        agent.load(str(path))
        return agent, "Loaded trained 3-action DQN checkpoint."
    except Exception as exc:
        return None, f"Checkpoint loading failed: {exc}"


def png_bytes(arr):
    out = io.BytesIO()
    Image.fromarray(arr).save(out, format="PNG")
    return out.getvalue()


def run_pipeline(original, roi_mask, tile_size, initial_loss, bandwidth, latency,
                 jitter, packet_loss, seed):
    tiles = divide_into_tiles(original, tile_size=tile_size)
    _, lost = simulate_packet_loss(
        tiles, loss_probability=float(initial_loss), seed=int(seed)
    )
    if not lost:
        tid = max(tiles)
        lost = {tid: tiles[tid]}

    env = ReconstructionTransmissionEnv(
        tiles=tiles, lost_tiles=lost, roi_mask=roi_mask,
        roi_priority_map=roi_mask, evaluation_roi_mask=roi_mask,
        seed=int(seed), payload_bytes=1200, max_retries=4
    )
    env.reset()
    env.network.update_conditions(
        bandwidth_mbps=float(bandwidth), latency_ms=float(latency),
        jitter_ms=float(jitter), loss_probability=float(packet_loss)
    )
    # Recompute the observation after applying the UI-selected network values.
    state = env._state()
    agent, status = load_agent()
    receiver_before = env.canvas.copy()
    decisions, total_reward = [], 0.0

    for _ in range(len(lost)):
        if agent is not None:
            action = agent.act(state, training=False)
            policy = "Trained 3-action DQN"
        else:
            tid = env.ids[env.i]
            tile = env.lost_tiles[tid]
            roi_fraction = 0.0
            if roi_mask is not None:
                y, x = tile["position"]
                h, w = tile["data"].shape[:2]
                roi_fraction = float((roi_mask[y:y+h, x:x+w] > 0).mean())
            config = env.network.config
            action = (
                2 if roi_fraction >= 0.03 and config.loss_probability < 0.30
                and config.bandwidth_mbps >= 0.5
                else (1 if config.loss_probability < 0.12
                      and config.bandwidth_mbps >= 1 and config.latency_ms < 100
                      else 0)
            )
            policy = "Rule-based fallback — not trained RL"

        state, reward, done, info = env.step(action)
        total_reward += reward
        decisions.append(info)
        if done:
            break

    reconstructed = np.clip(env.canvas, 0, 1)
    metrics = evaluate_image(
        original, reconstructed,
        missing_mask=env.initial_missing_mask,
        data_range=1.0
    )
    metrics["Recovery_Bytes"] = int(env.total_bytes)
    metrics["Priority_Metadata_Bytes"] = int(env.priority_metadata_bytes)
    metrics["Recovery_Delay_ms"] = float(
        sum(float(item.get("delay_ms", 0.0)) for item in decisions)
    )
    if roi_mask is not None:
        metrics["ROI_PSNR_dB"] = roi_psnr(
            original, reconstructed, roi_mask, data_range=1.0
        )
    return reconstructed, receiver_before, metrics, decisions, total_reward, policy, status, env


def show_metric(label, value, help_text=""):
    if value is None:
        display = "N/A"
    elif isinstance(value, (float, np.floating)) and np.isinf(value):
        display = "∞"
    elif isinstance(value, (float, np.floating)):
        display = f"{value:.3f}"
    else:
        display = str(value)
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{display}</div>'
        f'<div class="metric-help">{help_text}</div></div>',
        unsafe_allow_html=True
    )


# ----------------------------- SIDEBAR ---------------------------------
with st.sidebar:
    st.markdown("##  RECON")
    st.caption("Adaptive medical-image transmission")
    st.markdown("---")
    st.markdown("### 01 · Image source")
    mode = st.radio(
        "Input type", ["BraTS NIfTI (FLAIR + SEG)", "2D image (PNG/JPG)"],
        label_visibility="collapsed"
    )
    original, roi_mask, slice_idx = None, None, None

    if mode.startswith("BraTS"):
        flair_file = st.file_uploader(
            "FLAIR volume (.nii / .nii.gz)", type=["nii", "gz"], key="flair"
        )
        seg_file = st.file_uploader(
            "Matching segmentation (.nii / .nii.gz)", type=["nii", "gz"], key="seg"
        )
        slice_choice = st.radio(
            "Axial slice", ["Largest tumour area", "Choose slice index"]
        )
        chosen_idx = st.number_input(
            "Slice index", min_value=0, value=80, step=1
        ) if slice_choice == "Choose slice index" else None
    else:
        image_file = st.file_uploader(
            "Upload image", type=["png", "jpg", "jpeg", "tif", "tiff"]
        )
        mask_file = st.file_uploader(
            "Optional ROI mask", type=["png", "jpg", "jpeg", "tif", "tiff"]
        )

    st.markdown("---")
    st.markdown("### 02 · Network conditions")
    tile_size = st.number_input(
        "Tile size (pixels)",
        min_value=4,
        max_value=256,
        value=32,
        step=1,
        help="Enter any integer tile size from 4 to 256, such as 8, 10, 16, 24, 32, 48 or 64."
    )
    tile_size = int(tile_size)
    initial_loss = st.slider("Initial tile loss", 0.0, 0.8, 0.25, 0.05)
    bandwidth = st.select_slider(
        "Bandwidth (Mbps)", options=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0], value=5.0
    )
    latency = st.slider("Latency (ms)", 5, 250, 40, 5)
    jitter = st.slider("Jitter (ms)", 0, 50, 5, 1)
    packet_loss = st.slider("Packet loss probability", 0.0, 0.6, 0.15, 0.01)
    seed = st.number_input("Random seed", 0, 999999, 42)
    st.markdown("---")
    run_clicked = st.button("▶  RUN RECON", type="primary", use_container_width=True)
    agent, agent_status = load_agent()
    st.caption("● " + ("DQN checkpoint available" if agent else "Rule-based fallback active"))
    st.caption("Actions: 0 Local recovery · 1 ARQ · 2 QoS-prioritized ARQ")

# ----------------------------- LOAD INPUT ---------------------------------
if mode.startswith("BraTS"):
    if flair_file is None or seg_file is None:
        original = None
    else:
        try:
            import nibabel as nib
            with tempfile.TemporaryDirectory() as td:
                flair_path = Path(td) / flair_file.name
                seg_path = Path(td) / seg_file.name
                flair_path.write_bytes(flair_file.getvalue())
                seg_path.write_bytes(seg_file.getvalue())
                fimg = nib.load(str(flair_path))
                simg = nib.load(str(seg_path))
                fv = np.asarray(fimg.get_fdata(), dtype=np.float32)
                sv = np.asarray(simg.get_fdata(), dtype=np.int16)
            if fv.shape != sv.shape or fv.ndim != 3:
                raise ValueError(
                    f"Expected matching 3-D volumes; got FLAIR {fv.shape}, SEG {sv.shape}"
                )
            if flair_file.name.replace("_flair", "") != seg_file.name.replace("_seg", ""):
                st.warning("The uploaded filenames do not appear to be a matching patient pair. Verify before proceeding.")
            if chosen_idx is None:
                slice_idx = int(np.argmax(np.count_nonzero(sv, axis=(0, 1))))
            else:
                slice_idx = int(chosen_idx)
            if not 0 <= slice_idx < fv.shape[2]:
                raise IndexError(f"Slice index must be between 0 and {fv.shape[2]-1}")
            raw = fv[:, :, slice_idx]
            seg_slice = sv[:, :, slice_idx]
            nonzero = raw > 0
            if np.any(nonzero):
                low, high = np.percentile(raw[nonzero], [1, 99])
                raw = np.clip((raw - low) / max(float(high-low), 1e-8), 0, 1)
            else:
                raw = np.zeros_like(raw, dtype=np.float32)
            original = cv2.resize(raw.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA)
            roi_mask = cv2.resize(
                (seg_slice > 0).astype(np.uint8), (256, 256),
                interpolation=cv2.INTER_NEAREST
            )
        except Exception as exc:
            st.error(f"Could not load the uploaded NIfTI volumes: {exc}")
            st.stop()
else:
    if image_file is not None:
        original = np.asarray(
            Image.open(image_file).convert("L").resize(
                (256, 256), Image.Resampling.LANCZOS
            ), dtype=np.float32
        ) / 255.0
        if mask_file is not None:
            roi_mask = (
                np.asarray(
                    Image.open(mask_file).convert("L").resize(
                        (256, 256), Image.Resampling.NEAREST
                    )
                ) > 0
            ).astype(np.uint8)

# ----------------------------- MAIN HEADER ---------------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Research platform · DIP × Computer Networks × Optimization × RL</div>
  <h1>RECON <span style="color:#087f8c">/</span> Intelligent Image Recovery</h1>
  <p>Optimization-driven reconstruction and adaptive retransmission for medical images over unreliable networks.</p>
  <div style="margin-top:.7rem">
    <span class="pill">BraTS-ready</span><span class="pill">3-action policy</span>
    <span class="pill">Packet-loss simulation</span><span class="pill">ROI-aware recovery</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Research prototype only — not for diagnosis or clinical decisions. "
    "BraTS segmentation is treated as sender-available ROI metadata and for offline evaluation; "
    "it is not supplied to the agent as missing image content."
)

if original is None:
    st.markdown("### Welcome to the RECON workspace")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Start an imaging session")
        st.write("Upload a matching BraTS FLAIR volume and SEG mask in the sidebar, or select a 2D image.")
        st.write("Configure simulated network conditions, then run the recovery pipeline.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Decision policy")
        st.write("**0 · Local reconstruction** — recover missing tiles at the receiver.")
        st.write("**1 · Normal ARQ** — request missing packets again.")
        st.write("**2 · QoS-prioritized ARQ** — prioritize medically important ROI packets.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ----------------------------- INPUT PREVIEW ---------------------------------
st.markdown('<div class="section-kicker">01 / Imaging workspace</div>', unsafe_allow_html=True)
head1, head2, head3 = st.columns(3)
head1.metric("Input mode", "BraTS NIfTI" if mode.startswith("BraTS") else "2D image")
head2.metric("Image resolution", f"{original.shape[1]} × {original.shape[0]}")
head3.metric("Axial slice", str(slice_idx) if slice_idx is not None else "—")
preview_cols = st.columns([1.35, 1])
with preview_cols[0]:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Source image · FLAIR")
    st.image(original, clamp=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with preview_cols[1]:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Tumour ROI metadata")
    if roi_mask is not None:
        st.image(roi_mask * 255, clamp=True, use_container_width=True)
        st.caption(f"ROI coverage: {100*np.mean(roi_mask > 0):.2f}% of selected slice")
    else:
        st.info("No ROI mask supplied. ROI-specific metrics and prioritization are unavailable.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-kicker">02 / Adaptive transmission</div>', unsafe_allow_html=True)
network_cols = st.columns(5)
network_cols[0].metric("Tile size", f"{tile_size} px")
network_cols[1].metric("Initial tile loss", f"{initial_loss:.0%}")
network_cols[2].metric("Bandwidth", f"{bandwidth:g} Mbps")
network_cols[3].metric("Latency", f"{latency} ms")
network_cols[4].metric("Packet loss", f"{packet_loss:.0%}")

if run_clicked:
    with st.spinner("Simulating packet loss, adaptive decisions and image recovery..."):
        recon, before, metrics, decisions, reward, policy, status, env = run_pipeline(
            original, roi_mask, tile_size, initial_loss, bandwidth,
            latency, jitter, packet_loss, seed
        )
    st.success("RECON simulation completed.")
    st.markdown('<div class="section-kicker">03 / Run results</div>', unsafe_allow_html=True)

    action_counts = {
        "Local reconstruction": sum(d["action"] == 0 for d in decisions),
        "Normal ARQ": sum(d["action"] == 1 for d in decisions),
        "QoS-prioritized ARQ": sum(d["action"] == 2 for d in decisions),
    }
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Affected tiles processed", len(decisions))
    k2.metric("Local reconstruction", action_counts["Local reconstruction"])
    k3.metric("Normal ARQ", action_counts["Normal ARQ"])
    k4.metric("QoS-prioritized ARQ", action_counts["QoS-prioritized ARQ"])
    st.caption(f"Policy: **{policy}**  ·  Total reward: **{reward:.4f}**  ·  {status}")

    tab_images, tab_metrics, tab_policy, tab_report = st.tabs(
        ["🖼 Image comparison", "📊 Quality metrics", "🧠 Decision trace", "⬇ Export report"]
    )
    with tab_images:
        im1, im2, im3 = st.columns(3)
        with im1:
            st.markdown("**Original reference**")
            st.image(original, clamp=True, use_container_width=True)
        with im2:
            st.markdown("**Receiver before recovery**")
            st.image(np.clip(before, 0, 1), clamp=True, use_container_width=True)
        with im3:
            st.markdown("**Reconstructed output**")
            st.image(recon, clamp=True, use_container_width=True)
    with tab_metrics:
        metric_items = list(metrics.items())
        metric_cols = st.columns(min(4, max(1, len(metric_items))))
        for idx, (key, value) in enumerate(metric_items):
            with metric_cols[idx % len(metric_cols)]:
                show_metric(key.replace("_", " "), value, "Measured on this run")
        st.caption("Metrics compare the reconstructed image with the original reference slice.")
    with tab_policy:
        st.markdown("#### Action distribution")
        st.bar_chart(
            {"Number of decisions": action_counts},
            x_label="Policy action", y_label="Tiles"
        )
        st.markdown("#### Per-tile decision trace")
        rows = [{
            "Tile": d.get("tile_id"),
            "Action": d.get("action_name"),
            "Action ID": d.get("action"),
            "Packet priority": d.get("packet_priority"),
            "Quality reward": round(float(d.get("quality", 0)), 4),
            "ROI quality": d.get("roi_quality"),
            "Bytes sent": d.get("bytes_sent"),
            "Delay (ms)": round(float(d.get("delay_ms", 0)), 2),
            "Fallback used": d.get("fallback_used"),
        } for d in decisions]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    with tab_report:
        report = {
            "policy": policy, "total_reward": reward, "metrics": metrics,
            "action_counts": action_counts, "decisions": decisions,
            "simulation": {
                "tile_size": tile_size, "initial_tile_loss": initial_loss,
                "bandwidth_mbps": bandwidth, "latency_ms": latency,
                "jitter_ms": jitter, "packet_loss_probability": packet_loss,
                "seed": int(seed), "slice_index": slice_idx
            }
        }
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "original_slice.png",
                png_bytes(np.uint8(np.clip(original, 0, 1) * 255))
            )
            archive.writestr(
                "receiver_before_recovery.png",
                png_bytes(np.uint8(np.clip(before, 0, 1) * 255))
            )
            archive.writestr(
                "reconstructed_slice.png",
                png_bytes(np.uint8(np.clip(recon, 0, 1) * 255))
            )
            if roi_mask is not None:
                archive.writestr("tumour_roi_mask.png", png_bytes(roi_mask * 255))
            archive.writestr("run_report.json", json.dumps(report, indent=2, default=str))
        st.download_button(
            "⬇ Download complete run report (.zip)",
            zip_buffer.getvalue(), "RECON_run_report.zip",
            "application/zip", use_container_width=True
        )
        st.download_button(
            "⬇ Download reconstructed slice (.png)",
            png_bytes(np.uint8(np.clip(recon, 0, 1) * 255)),
            "reconstructed_slice.png", "image/png", use_container_width=True
        )
else:
    st.info("Adjust the network settings in the sidebar and select **RUN RECON** to execute a simulation.")

st.markdown("---")
st.caption(
    "RECON · Research prototype · Results depend on the selected image, network settings, "
    "random seed and availability of a compatible trained checkpoint."
)
