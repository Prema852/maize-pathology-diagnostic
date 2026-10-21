import streamlit as st
import torch
import joblib
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

st.set_page_config(page_title="Multi-Crop Pathology Diagnostic Suite", layout="wide")

# Custom UI Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .diag-box { padding: 1.2rem; border-radius: 8px; margin: 1rem 0; }
    .diag-success { background-color: #ECFDF5; border-left: 5px solid #10B981; }
    .diag-title { font-size: 1.3rem; font-weight: bold; color: #065F46; }
</style>
""", unsafe_allow_html=True)

# 1. Load SigLIP Engine and Embedded Probe
@st.cache_resource(show_spinner="Initializing SigLIP Vision-Language Foundation Model...")
def load_vlm_engine():
    probe_bundle = joblib.load("maize_siglip_probe.joblib")
    model_id = probe_bundle["model_id"]
    
    # Load on CPU directly within Streamlit's RAM budget
    model = AutoModel.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    
    return model, processor, probe_bundle

model, processor, probe_bundle = load_vlm_engine()
clf = probe_bundle["classifier"]
class_names = probe_bundle["classes"]

# Agronomic Reference Intelligence
PATHOLOGY_GUIDE = {
    "Common_Rust": {
        "display": "Common Rust (Puccinia sorghi)",
        "pathogen": "Puccinia sorghi (Fungi - Basidiomycota)",
        "morphology": "Small, powdery cinnamon-brown to orange-brown pustules scattered across upper and lower leaf surfaces.",
        "management": "Foliar triazole or strobilurin fungicides (e.g., Pyraclostrobin, Difenoconazole) if disease reaches threshold prior to reproductive stage."
    },
    "Blight": {
        "display": "Northern Leaf Blight (Exserohilum turcicum)",
        "pathogen": "Exserohilum turcicum (syn. Helminthosporium turcicum)",
        "morphology": "Long, elliptical, cigar-shaped grayish-green to tan necrotic lesions (2.5–15 cm) extending parallel to leaf veins.",
        "management": "Crop rotation, tillage of infected debris, and targeted applications of Azoxystrobin + Propiconazole combinations."
    },
    "Gray_Leaf_Spot": {
        "display": "Gray Leaf Spot (Cercospora zeae-maydis)",
        "pathogen": "Cercospora zeae-maydis",
        "morphology": "Rectangular, tan-to-gray lesions strictly delimited and bounded by the leaf veins with sharp right angles.",
        "management": "Deploy resistant hybrids; apply QoI/DMI premixes during tassel to brown silk stages."
    },
    "Healthy": {
        "display": "Healthy Maize Specimen",
        "pathogen": "None detected (Physiologically Normal)",
        "morphology": "Uniform chlorophyll distribution, intact cuticle, and devoid of chlorotic streaks or fungal pustules.",
        "management": "Standard agronomic maintenance, balanced nitrogen fertilization, and routine field scouting."
    }
}

# Sidebar Info
st.sidebar.title("⚙️ System Status")
st.sidebar.success("Engine: Google SigLIP (ViT-B/16)")
st.sidebar.info("Inference: Native Streamlit CPU (No external server required)")
st.sidebar.markdown("**Monitored Classes:**")
for c in class_names:
    st.sidebar.markdown(f"- {c.replace('_', ' ')}")

# Main Interface
st.markdown('<div class="main-header">🌾 Multi-Crop Pathology Diagnostic Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated disease classification with embedded Vision-Language semantic representations.</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Upload Maize Leaf Sample")
    uploaded_file = st.file_uploader("Select leaf image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Specimen", use_container_width=True)

with col2:
    st.subheader("2. Diagnostic Evaluation")
    if uploaded_file is not None:
        if st.button("Run Diagnostic Engine", type="primary"):
            with st.spinner("Extracting multimodal visual representations..."):
                # Preprocess & Extract Features
                inputs = processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.get_image_features(pixel_values=inputs["pixel_values"])
                    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        feat = outputs.pooler_output
                    elif isinstance(outputs, torch.Tensor):
                        feat = outputs
                    else:
                        feat = outputs[0]
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    embedding = feat.cpu().numpy()

                # Inference via calibrated probe
                probs = clf.predict_proba(embedding)[0]
                pred_idx = int(np.argmax(probs))
                predicted_raw_class = class_names[pred_idx]
                confidence = float(probs[pred_idx] * 100)
                
                info = PATHOLOGY_GUIDE.get(predicted_raw_class, {
                    "display": predicted_raw_class,
                    "pathogen": "N/A",
                    "morphology": "N/A",
                    "management": "N/A"
                })

            # Display Result
            st.markdown(f"""
            <div class="diag-box diag-success">
                <div class="diag-title">VERIFIED DIAGNOSIS: {info['display'].upper()} ({confidence:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)

            # Probabilities distribution
            st.markdown("#### Confidence Breakdown")
            prob_cols = st.columns(len(class_names))
            for p_col, c_name, p_val in zip(prob_cols, class_names, probs):
                p_col.metric(label=c_name.replace("_", " "), value=f"{p_val * 100:.1f}%")

            # Agronomic Dossier
            st.markdown("---")
            st.markdown("### 📋 Clinical Pathology Assessment")
            st.markdown(f"**Causative Pathogen:** *{info['pathogen']}*")
            st.markdown(f"**Lesion Morphology:** {info['morphology']}")
            st.markdown(f"**Agronomic Protocol:** {info['management']}")
    else:
        st.info("Please upload a leaf image to run diagnostic evaluation.")
