from pathlib import Path
import io
import zipfile
import json
import tempfile

import numpy as np
import cv2
import streamlit as st
from PIL import Image
import torch

from src.unet_inpainting import load_inpainting_model
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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --page: #07111f;
  --page-glow: #0b1c2e;
  --surface: #0e1d2e;
  --surface-raised: #12243a;
  --surface-soft: #162b42;
  --border: rgba(157, 184, 207, .16);
  --border-strong: rgba(157, 184, 207, .26);
  --ink: #edf5fb;
  --muted: #9aafc3;
  --teal: #45d6c5;
  --teal-dark: #18a99d;
  --blue: #6ba8ff;
  --shadow: 0 16px 42px rgba(0, 0, 0, .22);
}

/* Global canvas and typography */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
}
.stApp {
  background:
    radial-gradient(ellipse at 78% 0%, rgba(32, 111, 130, .16), transparent 34%),
    radial-gradient(ellipse at 5% 18%, rgba(47, 91, 145, .13), transparent 30%),
    var(--page);
  color: var(--ink);
}
[data-testid="stHeader"] {
  background: rgba(7, 17, 31, .78);
}
[data-testid="stToolbar"] { right: 1.5rem; }
.block-container {
  max-width: 1500px;
  padding-top: 2rem;
  padding-bottom: 3.5rem;
}
h1, h2, h3, h4 {
  color: #f2f7fc !important;
  font-family: 'Manrope', sans-serif !important;
  letter-spacing: -.035em;
  font-weight: 750 !important;
}
h1 { line-height: 1.12 !important; }
h2, h3 { letter-spacing: -.025em; }
p, li, .stMarkdown, .stText, .stCaption, small {
  color: #c2d1df;
}
[data-testid="stCaptionContainer"] p, .stCaption {
  color: var(--muted) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b192a 0%, #091522 100%);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div {
  background: transparent;
}
section[data-testid="stSidebar"] .block-container {
  padding: 1.5rem 1.15rem 2rem;
}
section[data-testid="stSidebar"] h2 {
  color: #f3f8fc !important;
  font-family: 'Manrope', sans-serif !important;
  letter-spacing: -.04em;
}
section[data-testid="stSidebar"] h3 {
  color: #dceaf4 !important;
  font-size: .92rem !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
  color: #b7c8d8 !important;
}
section[data-testid="stSidebar"] hr {
  border-color: var(--border) !important;
}

/* Hero */
.hero {
  position: relative;
  overflow: hidden;
  padding: 2.15rem 2.25rem;
  border: 1px solid rgba(110, 194, 197, .24);
  border-radius: 24px;
  background:
    radial-gradient(circle at 88% 12%, rgba(69, 214, 197, .17), transparent 28%),
    linear-gradient(118deg, #10243a 0%, #102d40 54%, #123b48 100%);
  box-shadow: 0 22px 55px rgba(0, 0, 0, .24);
  margin-bottom: 1.5rem;
}
.hero:after {
  content: "";
  position: absolute;
  width: 260px;
  height: 260px;
  right: -105px;
  bottom: -165px;
  border: 1px solid rgba(116, 229, 214, .2);
  border-radius: 50%;
  box-shadow: 0 0 0 26px rgba(116, 229, 214, .035), 0 0 0 54px rgba(116, 229, 214, .025);
  pointer-events: none;
}
.eyebrow {
  color: var(--teal) !important;
  text-transform: uppercase;
  letter-spacing: .17em;
  font-size: .69rem;
  font-weight: 800;
}
.hero h1 {
  color: #f4fbff !important;
  font-size: clamp(2rem, 3.4vw, 2.9rem);
  margin: .55rem 0 .55rem 0;
}
.hero p {
  color: #c1d5e3 !important;
  margin: 0;
  max-width: 850px;
  font-size: 1.02rem;
  line-height: 1.75;
}
.pill {
  display: inline-block;
  border: 1px solid rgba(90, 218, 202, .3);
  background: rgba(58, 190, 177, .11);
  color: #aaf4e9 !important;
  border-radius: 99px;
  padding: .38rem .78rem;
  font-size: .72rem;
  font-weight: 700;
  margin: .7rem .35rem 0 0;
  letter-spacing: .015em;
}
.section-kicker {
  color: var(--teal) !important;
  font-size: .7rem;
  letter-spacing: .17em;
  text-transform: uppercase;
  font-weight: 800;
  margin: 1.45rem 0 .7rem;
}

/* Panels and custom metric cards */
.panel {
  background: linear-gradient(145deg, rgba(19, 39, 60, .96), rgba(13, 29, 46, .96));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1.15rem 1.25rem;
  box-shadow: var(--shadow);
}
.metric-card {
  background: linear-gradient(145deg, #142a40, #0e1e30);
  border: 1px solid var(--border);
  border-top: 2px solid var(--teal-dark);
  border-radius: 16px;
  padding: 1rem 1.05rem;
  min-height: 112px;
  box-shadow: 0 10px 26px rgba(0, 0, 0, .16);
}
.metric-label {
  color: #a9bdcf !important;
  font-size: .75rem;
  font-weight: 700;
  letter-spacing: .025em;
}
.metric-value {
  color: #f1f8fd !important;
  font-family: 'Manrope', sans-serif;
  font-size: 1.65rem;
  font-weight: 800;
  margin-top: .32rem;
  letter-spacing: -.035em;
}
.metric-help {
  color: #829bb0 !important;
  font-size: .7rem;
  margin-top: .18rem;
}

/* Native Streamlit metrics */
div[data-testid="stMetric"] {
  background: linear-gradient(145deg, #142a40, #0e1e30);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: .9rem 1rem;
  box-shadow: 0 9px 24px rgba(0, 0, 0, .14);
}
div[data-testid="stMetricLabel"] p {
  color: #a9bdcf !important;
  font-weight: 650;
  font-size: .78rem;
}
div[data-testid="stMetricValue"] {
  color: #f3f8fc !important;
  font-family: 'Manrope', sans-serif;
  font-weight: 800;
}
div[data-testid="stMetricDelta"] {
  color: var(--teal) !important;
}

/* Widget labels and help */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stRadio > label,
.stCheckbox > label,
.stSlider > label,
.stSelectSlider > label,
.stNumberInput > label,
.stFileUploader > label {
  color: #d5e2ed !important;
  font-weight: 650 !important;
}
[data-testid="stHelp"] { color: #91a8bc !important; }

/* Inputs and select controls */
input, textarea {
  color: #edf5fb !important;
  background: #102238 !important;
  -webkit-text-fill-color: #edf5fb !important;
  border-color: #2b435a !important;
}
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div,
[data-baseweb="select"] > div {
  background: #102238 !important;
  border-color: #2b435a !important;
  border-radius: 10px !important;
}
[data-baseweb="select"] * { color: #edf5fb !important; }
[role="listbox"], [role="option"], [data-baseweb="popover"] {
  background: #102238 !important;
  color: #edf5fb !important;
}
[role="option"]:hover, [aria-selected="true"] {
  background: #1a3a50 !important;
  color: #b9fff4 !important;
}

/* Radios, checkboxes and sliders */
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label,
[data-testid="stRadio"] p {
  color: #cbd9e5 !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"] { color: #91a8bc !important; }
[data-testid="stSlider"] [role="slider"] { border-color: var(--teal) !important; }

/* File uploader button: use the requested dark text colour */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button *,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button * {
  color: #0B192C !important;
  -webkit-text-fill-color: #0B192C !important;
}
[data-testid="stFileUploader"] button svg {
  color: #0B192C !important;
  fill: #0B192C !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
  background: rgba(16, 34, 54, .7);
  border: 1px dashed #45647c;
  border-radius: 14px;
  padding: .65rem;
}
[data-testid="stFileUploader"] section {
  background: #12283e !important;
  border-radius: 10px;
  border: 1px solid rgba(157, 184, 207, .1);
}
[data-testid="stFileUploader"] section *,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
  color: #c6d6e3 !important;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
  color: #0B192C !important;
  -webkit-text-fill-color: #0B192C !important;
  background: linear-gradient(105deg, #55dfcc 0%, #62b9ed 100%) !important;
  border: 1px solid rgba(91, 222, 207, .65) !important;
  border-radius: 11px !important;
  font-weight: 800 !important;
  min-height: 2.75rem;
  box-shadow: 0 8px 20px rgba(38, 184, 171, .16);
  transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
}
.stButton > button *, .stDownloadButton > button * {
  color: #0B192C !important;
  -webkit-text-fill-color: #0B192C !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  filter: brightness(1.07);
  transform: translateY(-1px);
  box-shadow: 0 11px 25px rgba(38, 184, 171, .24);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: .45rem;
  border-bottom: 1px solid var(--border);
  background: transparent;
}
.stTabs [data-baseweb="tab"] {
  background: #102238;
  border: 1px solid transparent;
  border-radius: 10px 10px 0 0;
  padding: .72rem 1rem;
}
.stTabs [data-baseweb="tab"] * {
  color: #a9bdcf !important;
  font-weight: 700;
}
.stTabs [aria-selected="true"] {
  background: #173149 !important;
  border: 1px solid var(--border) !important;
  border-bottom: 2px solid var(--teal) !important;
}
.stTabs [aria-selected="true"] * { color: #aaf4e9 !important; }

/* Alerts: dark text on the light warning surface */
div[data-testid="stAlert"] {
  background: #FFF8E1 !important;
  background-color: #FFF8E1 !important;
  border: 1px solid rgba(246, 190, 80, .42) !important;
  border-radius: 13px !important;
  color: #1C1C1E !important;
}
div[data-testid="stAlert"] *,
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] div {
  color: #1C1C1E !important;
  -webkit-text-fill-color: #1C1C1E !important;
  opacity: 1 !important;
}
div[data-testid="stAlert"] svg {
  fill: #1C1C1E !important;
  color: #1C1C1E !important;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  color: #e5eff7 !important;
}
[data-testid="stDataFrame"] {
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}
hr { border-color: var(--border) !important; }
footer { visibility: hidden; }

/* Streamlit sanitizes HTML wrappers between separate markdown calls.
   Hide the resulting empty wrapper shells so they do not look like blank cards. */
div[data-testid="stMarkdownContainer"]:has(> div.panel) {
  display: none !important;
}

/* Image presentation */
[data-testid="stImage"] img {
  border-radius: 12px;
}
[data-testid="stImage"] {
  border-radius: 12px;
}

/* Responsive spacing */
@media (max-width: 800px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .hero { padding: 1.45rem; border-radius: 18px; }
  .hero p { font-size: .93rem; }
  .metric-value { font-size: 1.4rem; }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------- HELPERS ---------------------------------
@st.cache_resource
def load_inpainting_unet():
    """
    Load the trained U-Net once and cache it for the Streamlit session.

    The U-Net checkpoint is separate from the DQN policy checkpoint:
      models/best_inpainting_model.pth
    """
    checkpoint_path = Path("models/best_inpainting_model.pth")
    device = torch.device("cpu")

    if not checkpoint_path.exists():
        return None, device, "U-Net checkpoint not found; classical hybrid inpainting will be used."

    try:
        model = load_inpainting_model(str(checkpoint_path), device=device)
        return model, device, "Trained U-Net loaded successfully."
    except Exception as exc:
        return None, device, f"U-Net loading failed; classical hybrid fallback will be used. Details: {exc}"


def load_agent():
    """Load the compatible 13-state, 3-action DQN checkpoint if available."""
    path = Path("models/recon_dqn.pt")

    if not path.exists():
        return None, (
            "No trained 3-action DQN checkpoint found. "
            "The app will use a clearly labelled rule-based fallback."
        )

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        if (checkpoint.get("state_dim") != 13 or checkpoint.get("action_dim") != 3
                or checkpoint.get("state_version") != 2):
            return None, (
                "Incompatible or legacy DQN checkpoint: expected state version 2, "
                "13 states and 3 actions. Retrain using experiments/train_dqn_brats.py. "
                "The app will use the rule-based fallback."
            )

        agent = DQNAgent(state_dim=13, action_dim=3, device="cpu")
        agent.load(str(path))
        return agent, "Loaded trained 3-action DQN checkpoint."

    except Exception as exc:
        return None, f"DQN checkpoint loading failed; using rule-based fallback. Details: {exc}"


def png_bytes(arr):
    out = io.BytesIO()
    Image.fromarray(arr).save(out, format="PNG")
    return out.getvalue()


def run_pipeline(original, roi_mask, tile_size, initial_loss, bandwidth, latency,
                 jitter, packet_loss, seed):
    """
    Execute packet-loss simulation and adaptive image recovery.

    The trained U-Net is loaded before constructing the environment because
    the environment receives the model in its constructor.
    """
    # Load the two independent learned components.
    agent, dqn_status = load_agent()
    inpainting_model, inpainting_device, unet_status = load_inpainting_unet()

    tiles = divide_into_tiles(original, tile_size=tile_size)
    _, lost = simulate_packet_loss(
        tiles, loss_probability=float(initial_loss), seed=int(seed)
    )

    # Ensure the demonstration contains at least one affected tile.
    if not lost:
        tid = max(tiles)
        lost = {tid: tiles[tid]}

    # NOTE: ReconstructionTransmissionEnv must accept these two keyword
    # arguments, and its _inpaint_current_tile() must call reconstruct_with_unet
    # when inpainting_model is not None.
    env = ReconstructionTransmissionEnv(
        tiles=tiles,
        lost_tiles=lost,
        roi_mask=roi_mask,
        roi_priority_map=roi_mask,
        evaluation_roi_mask=roi_mask,
        seed=int(seed),
        payload_bytes=1200,
        max_retries=4,
        inpainting_model=inpainting_model,
        inpainting_device=inpainting_device,
    )

    env.reset()
    env.network.update_conditions(
        bandwidth_mbps=float(bandwidth),
        latency_ms=float(latency),
        jitter_ms=float(jitter),
        loss_probability=float(packet_loss),
    )

    # Recompute the observation after applying the UI-selected network values.
    state = env._state()
    receiver_before = env.canvas.copy()
    decisions = []
    total_reward = 0.0
    policy = "Not selected"

    for _ in range(len(lost)):
        if agent is not None:
            action = agent.act(state, training=False)
            policy = "Trained 3-action DQN"
        else:
            # Explicitly labelled heuristic fallback. This is NOT a trained
            # reinforcement-learning policy.
            tid = env.ids[env.i]
            tile = env.lost_tiles[tid]
            roi_fraction = 0.0

            if roi_mask is not None:
                y, x = tile["position"]
                h, w = tile["data"].shape[:2]
                # Receiver-visible metadata is a single priority flag per tile.
                roi_fraction = float(
                    np.any(roi_mask[y:y + h, x:x + w] > 0)
                )

            config = env.network.config
            action = (
                2
                if (
                    roi_fraction >= 0.03
                    and config.loss_probability < 0.30
                    and config.bandwidth_mbps >= 0.5
                )
                else (
                    1
                    if (
                        config.loss_probability < 0.12
                        and config.bandwidth_mbps >= 1.0
                        and config.latency_ms < 100
                    )
                    else 0
                )
            )
            policy = "Rule-based fallback — not trained RL"

        state, reward, done, info = env.step(action)
        total_reward += float(reward)
        decisions.append(info)

        if done:
            break

    reconstructed = np.clip(env.canvas, 0.0, 1.0)

    metrics = evaluate_image(
        original,
        reconstructed,
        missing_mask=env.initial_missing_mask,
        data_range=1.0,
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

    status = f"{dqn_status} | {unet_status}"

    return (
        reconstructed,
        receiver_before,
        metrics,
        decisions,
        total_reward,
        policy,
        status,
        env,
    )


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
        unsafe_allow_html=True,
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
