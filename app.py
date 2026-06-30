import streamlit as st
import numpy as np
from PIL import Image

# ── Page config ───────────────────────────────────
st.set_page_config(
    page_title="LWDCNN — Diabetic Retinopathy Classifier",
    page_icon="👁️",
    layout="wide"
)

# ── Load weights ──────────────────────────────────
@st.cache_resource
def load_weights():
    try:
        w = np.load("lwdcnn_dr_pynq_weights.npz")
        return dict(w)
    except FileNotFoundError:
        st.error("lwdcnn_dr_pynq_weights.npz not found!")
        st.stop()

# ── Pure numpy inference ──────────────────────────
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def batchnorm(x, gamma, beta, mean, var, eps=1e-3):
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def conv2d_same(x, w):
    H, W, C = x.shape
    kH, kW, _, F = w.shape
    pad_h, pad_w = kH // 2, kW // 2
    x_pad = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    w_flat = w.reshape(-1, F)
    cols = np.zeros((H * W, kH * kW * C), dtype=np.float32)
    idx = 0
    for i in range(H):
        for j in range(W):
            patch = x_pad[i:i+kH, j:j+kW, :]
            cols[idx] = patch.flatten()
            idx += 1
    out = cols @ w_flat
    return out.reshape(H, W, F)

def maxpool2x2(x):
    H, W, C = x.shape
    oH, oW = H // 2, W // 2
    out = np.zeros((oH, oW, C), dtype=np.float32)
    for h in range(oH):
        for ww in range(oW):
            out[h, ww, :] = np.max(x[h*2:h*2+2, ww*2:ww*2+2, :], axis=(0, 1))
    return out

def gap(x):
    return np.mean(x, axis=(0, 1))

def predict_dr_lite(img_array, w):
    x = img_array.astype(np.float32)

    x = conv2d_same(x, w['conv2d_0'])
    x = batchnorm(x, w['batch_normalization_0'], w['batch_normalization_1'],
                  w['batch_normalization_2'], w['batch_normalization_3'])
    x = relu(x); x = maxpool2x2(x)

    x = conv2d_same(x, w['conv2d_1_0'])
    x = batchnorm(x, w['batch_normalization_1_0'], w['batch_normalization_1_1'],
                  w['batch_normalization_1_2'], w['batch_normalization_1_3'])
    x = relu(x); x = maxpool2x2(x)

    x = conv2d_same(x, w['conv2d_2_0'])
    x = batchnorm(x, w['batch_normalization_2_0'], w['batch_normalization_2_1'],
                  w['batch_normalization_2_2'], w['batch_normalization_2_3'])
    x = relu(x); x = maxpool2x2(x)

    x = conv2d_same(x, w['conv2d_3_0'])
    x = batchnorm(x, w['batch_normalization_3_0'], w['batch_normalization_3_1'],
                  w['batch_normalization_3_2'], w['batch_normalization_3_3'])
    x = relu(x); x = maxpool2x2(x)

    x = gap(x)

    x = x @ w['dense_0']
    x = batchnorm(x, w['batch_normalization_4_0'], w['batch_normalization_4_1'],
                  w['batch_normalization_4_2'], w['batch_normalization_4_3'])
    x = relu(x)

    x = x @ w['dense_1_0']
    x = batchnorm(x, w['batch_normalization_5_0'], w['batch_normalization_5_1'],
                  w['batch_normalization_5_2'], w['batch_normalization_5_3'])
    x = relu(x)

    x = x @ w['dense_2_0'] + w['dense_2_1']
    prob = float(sigmoid(x).flatten()[0])
    return prob

# ── Preprocess (CLAHE on L channel, matches training pipeline) ──
def preprocess(img):
    import cv2
    img = img.convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (128, 128))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img.astype(np.float32) / 255.0

# ── UI ────────────────────────────────────────────
st.title("👁️ LWDCNN — Diabetic Retinopathy Classifier")
st.markdown("Lightweight Deep CNN for diabetic retinopathy screening — **Pure NumPy inference**")
st.divider()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Model Info")
    st.metric("Parameters", "18,953")
    st.metric("Accuracy", "97.81%")
    st.metric("AUC-ROC", "98.94%")
    st.metric("Platform", "Streamlit Cloud")
    st.markdown("**Architecture:**")
    st.markdown("- Conv(8)→BN→Pool\n- Conv(16)→BN→Pool\n- Conv(32)→BN→Pool\n- Conv(32)→BN→Pool\n- GAP\n- Dense(64)→BN\n- Dense(16)→BN\n- Dense(1)→Sigmoid")

with col_right:
    weights = load_weights()

    uploaded = st.file_uploader(
        "Upload a retinal fundus image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:
        img = Image.open(uploaded)

        c1, c2 = st.columns(2)
        with c1:
            st.image(img, caption="Uploaded Fundus Image", use_column_width=True)

        with c2:
            with st.spinner("Running inference..."):
                img_array = preprocess(img)
                prob = predict_dr_lite(img_array, weights)

            label = "DR Detected" if prob >= 0.5 else "No DR"
            color = "🔴" if prob >= 0.5 else "🟢"
            conf  = prob if prob >= 0.5 else 1 - prob

            st.subheader("Prediction")
            st.metric("Classification", f"{color} {label}")
            st.metric("Confidence", f"{conf*100:.1f}%")
            st.progress(float(prob), text=f"DR probability: {prob:.4f}")

            if prob >= 0.5:
                st.error("⚠️ Likely **DR present** — please consult an ophthalmologist.")
            else:
                st.success("✅ Likely **No DR** — please consult a medical professional for confirmation.")

            st.caption("⚕️ For research purposes only. Not a medical diagnosis.")

st.divider()
st.caption("Built with LWDCNN | Streamlit Cloud | Pure NumPy — no TensorFlow required")
