import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(
    page_title="Multi-Crop Pathology Diagnostic Suite",
    page_icon="🌽",
    layout="wide"
)

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

st.title("🌿 Multimodal Crop Pathology Diagnostic Suite")
st.markdown("**Automated disease classification with zero-hallucination agronomic explanations.**")

st.sidebar.title("Configuration")
backend_url = st.sidebar.text_input(
    "Inference Engine API URL:",
    placeholder="https://xxxx.loca.lt"
)

crop_choice = st.sidebar.selectbox(
    "Select Crop Domain:",
    ["Maize (Cereals) — Active Module", "Beans (Legumes) — Pending Teammate Integration"]
)

if "Beans" in crop_choice:
    st.info("ℹ️ **Bean Pathology Module (MiniCPM-V 2.6):** Undergoing calibration by teammate. Model weights and CoT pipeline will be linked here shortly.")
else:
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
                if not backend_url:
                    st.error("Please enter the inference backend URL in the sidebar.")
                else:
                    with st.spinner("Processing through 1,536-dim multimodal hidden state representation..."):
                        img_bytes = io.BytesIO()
                        image.save(img_bytes, format='JPEG')
                        img_bytes.seek(0)
                        
                        try:
                            response = requests.post(
                                f"{backend_url.rstrip('/')}/predict",
                                files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
                                timeout=60
                            )
                            
                            if response.status_code == 200:
                                res = response.json()
                                diagnosis = res["diagnosis"]
                                confidence = res["confidence"]
                                probs = res["probabilities"]
                                info = BOTANICAL_GUIDE[diagnosis]
                                
                                st.success(f"### Verified Diagnosis: {diagnosis.upper()} ({confidence:.2f}%)")
                                
                                st.markdown("#### Confidence Breakdown")
                                prob_cols = st.columns(4)
                                for i, cls in enumerate(TARGET_CLASSES):
                                    prob_cols[i].metric(label=cls, value=f"{probs[i]*100:.1f}%")
                                
                                st.markdown("---")
                                st.markdown("### 📋 Clinical Pathology Assessment")
                                st.markdown(f"**Causative Pathogen:** *{info['pathogen']}*")
                                st.markdown(f"**Lesion Morphology:** {info['shape']}")
                                st.markdown(f"**Chemical Control:** {info['fungicides']}")
                                st.markdown(f"**Cultural Management:** {info['cultural']}")
                            else:
                                st.error(f"Inference error: {response.text}")
                        except Exception as e:
                            st.error(f"Connection error: {e}")
        else:
            st.info("Upload a maize leaf photo on the left to begin diagnosis.")
