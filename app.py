import streamlit as st
import torch
import joblib
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

st.set_page_config(
    page_title="Multi-Crop Pathology Diagnostic Suite",
    page_icon="🌽",
    layout="wide"
)

# 1. Domain Knowledge Base for Maize Foliar Diseases
BOTANICAL_GUIDE = {
    "Northern Leaf Blight": {
        "pathogen": "Exserohilum turcicum (syn. Helminthosporium turcicum)",
        "shape": "Long, cigar-shaped, elliptical necrotic lesions crossing leaf veins.",
        "fungicides": "Strobilurins (QoI: Azoxystrobin, Pyraclostrobin) or Triazoles (DMI: Difenoconazole).",
        "cultural": "Crop rotation with non-hosts (soybean) and deep tillage to bury crop debris."
    },
    "Gray Leaf Spot": {
        "pathogen": "Cercospora zeae-maydis",
        "shape": "Narrow, rectangular lesions strictly bounded by parallel leaf veins with blunt right-angle margins.",
        "fungicides": "Triazole (DMI) and Strobilurin (QoI) premixes applied around VT/R1.",
        "cultural": "Planting resistant hybrids and plowing under infected stover."
    },
    "Common Rust": {
        "pathogen": "Puccinia sorghi",
        "shape": "Small, raised, powdery cinnamon-brown to golden pustules scattered across both leaf surfaces.",
        "fungicides": "Foliar triazole or strobilurin fungicides if pustules appear prior to silking.",
        "cultural": "Deploying resistant hybrids bearing specific Rp genes."
    },
    "Healthy": {
        "pathogen": "None (Asymptomatic foliage)",
        "shape": "Uniform green parenchyma with intact cuticle and no fungal lesions or chlorosis.",
        "fungicides": "No chemical intervention needed.",
        "cultural": "Maintain balanced fertility (N-P-K) and standard irrigation regimes."
    }
}

TARGET_CLASSES = ["Gray Leaf Spot", "Common Rust", "Northern Leaf Blight", "Healthy"]
MAPPING = {0: "Gray Leaf Spot", 1: "Common Rust", 2: "Northern Leaf Blight", 3: "Healthy"}

# 2. Resource Loader (Configured strictly for CPU inference)
@st.cache_resource(show_spinner="Loading Qwen2-VL backbone and diagnostic probe into memory...")
def load_maize_models():
    probe = joblib.load("vlm_linear_probe_94acc.joblib")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    model.eval()
    return probe, model, processor

# 3. Streamlit Interface
st.title("🌿 Multimodal Crop Pathology Diagnostic Suite")
st.markdown("**Automated disease classification with zero-hallucination agronomic explanations.**")

crop_choice = st.sidebar.selectbox(
    "Select Crop Domain:",
    ["Maize (Cereals) — Active Module", "Beans (Legumes) — Pending Teammate Integration"]
)

if "Beans" in crop_choice:
    st.info("ℹ️ **Bean Pathology Module (MiniCPM-V 2.6):** Currently undergoing final calibration by teammate. Model weights and CoT logit pipeline will be linked here shortly.")
else:
    probe, model, processor = load_maize_models()

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("1. Upload Maize Leaf Sample")
        uploaded_file = st.file_uploader("Upload leaf image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Leaf Specimen", use_container_width=True)

    with col2:
        st.subheader("2. Diagnostic Evaluation")
        
        if uploaded_file is not None:
            if st.button("Run Diagnostic Engine", type="primary"):
                with st.spinner("Extracting 1,536-dim multimodal hidden state on CPU..."):
                    prompt = "Diagnose the condition of this maize leaf strictly as one of: ['Gray Leaf Spot', 'Common Rust', 'Northern Leaf Blight', 'Healthy']."
                    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
                    
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, _ = process_vision_info(messages)
                    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cpu")
                    
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)
                        embedding = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy().reshape(1, -1)
                    
                    probs = probe.predict_proba(embedding)[0]
                    pred_idx = np.argmax(probs)
                    diagnosis = MAPPING[pred_idx]
                    confidence = probs[pred_idx] * 100
                    info = BOTANICAL_GUIDE[diagnosis]
                
                st.success(f"### Verified Diagnosis: {diagnosis.upper()} ({confidence:.2f}%)")
                
                # Class Confidence Metrics
                st.markdown("#### Confidence Breakdown")
                prob_cols = st.columns(4)
                for i, cls in enumerate(TARGET_CLASSES):
                    prob_cols[i].metric(label=cls, value=f"{probs[i]*100:.1f}%")
                
                # Clinical Grounded Report
                st.markdown("---")
                st.markdown("### 📋 Clinical Pathology Assessment")
                st.markdown(f"**Causative Pathogen:** *{info['pathogen']}*")
                st.markdown(f"**Lesion Morphology:** {info['shape']}")
                st.markdown(f"**Chemical Control:** {info['fungicides']}")
                st.markdown(f"**Cultural Management:** {info['cultural']}")
        else:
            st.info("Upload a maize leaf photo on the left to begin diagnosis.")