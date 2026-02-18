import streamlit as st
import cv2
import numpy as np
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import inference.isolated_pipeline
st.write(dir(inference.isolated_pipeline))  # Lists available names

# Your existing imports...
from inference.isolated_pipeline import IsolatedPlantAnalyzer
from utils.visualization import draw_leaf_box, draw_pests
from inference.genAI_report import GenAIReportGenerator

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Leaf Health Analyzer", layout="wide")

# -------------------------------------------------
# Styling
# -------------------------------------------------
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 48px;
            font-weight: 700;
        }
        .description {
            text-align: center;
            font-size: 20px;
            margin-bottom: 30px;
        }

        /* Center and evenly space tabs */
        div[data-baseweb="tab-list"] {
            justify-content: center;
            gap: 60px;
        }

        div[data-baseweb="tab"] {
            font-size: 18px;
            font-weight: 500;
        }

        h3 {
            font-size: 26px !important;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.markdown("<div class='main-title'>🌿 Leaf Health Analysis System</div>", unsafe_allow_html=True)
st.markdown("<div class='description'>AI-powered modular plant health evaluation</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Tabs (Normal Tabs — CSS Handles Centering)
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Analytics",
    "📘 Documentation",
    "🧠 Report"
])

# -------------------------------------------------
# Load Pipeline
# -------------------------------------------------
@st.cache_resource
def load_pipeline():
    return IsolatedPlantAnalyzer()

analyzer = load_pipeline()

def format_value(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return value

@st.cache_resource
def load_genai():
    return GenAIReportGenerator()

genai_generator = load_genai()

# =================================================
# TAB 1 — ANALYTICS
# =================================================
with tab1:

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Upload Leaf Image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

    if uploaded_file is not None:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Failed to read image.")
            st.stop()

        with st.spinner("Running analysis..."):
            results = analyzer.analyze(image)
            st.session_state["analysis_results"] = results


        # Annotated Images
        leaf_image = draw_leaf_box(image.copy(), results.get("leaf_detection"))
        pest_image = draw_pests(image.copy(), results.get("pests"))

        st.markdown("## 🖼 Annotated Outputs")

        colA, colB = st.columns(2)
        with colA:
            st.subheader("Leaf Detection")
            st.image(cv2.cvtColor(leaf_image, cv2.COLOR_BGR2RGB))

        with colB:
            st.subheader("Pest Detection")
            st.image(cv2.cvtColor(pest_image, cv2.COLOR_BGR2RGB))

        st.markdown("---")
        st.markdown("## 📊 Detailed Analysis")

        r1c1, r1c2 = st.columns(2)

        with r1c1:
            with st.expander("🌿 Leaf Detection", expanded=True):
                leaf_result = results.get("leaf_detection")
                if leaf_result:
                    st.write(f"Bounding Box: {leaf_result}")
                else:
                    st.warning("No leaf detected.")

        with r1c2:
            with st.expander("🦠 Disease Analysis", expanded=True):
                disease_result = results.get("disease")
                if disease_result and "error" not in disease_result:
                    st.write(f"Prediction: {disease_result['label']}")
                    st.write(f"Confidence: {format_value(disease_result['confidence'])}")
                else:
                    st.warning("Disease module unavailable or error.")

        r2c1, r2c2 = st.columns(2)

        with r2c1:
            with st.expander("💧 Dryness Analysis"):
                dryness_result = results.get("dryness")
                if dryness_result:
                    for key, value in dryness_result.items():
                        if isinstance(value, dict):
                            st.markdown(f"**{key.replace('_',' ').title()}**")
                            for sub_key, sub_val in value.items():
                                st.write(f"{sub_key.replace('_',' ').title()}: {format_value(sub_val)}")
                        else:
                            st.write(f"{key.replace('_',' ').title()}: {format_value(value)}")

        with r2c2:
            with st.expander("🌱 Green Index"):
                green_result = results.get("green_index")
                if green_result:
                    for key, value in green_result.items():
                        st.write(f"{key.replace('_',' ').title()}: {format_value(value)}")

        r3c1, r3c2 = st.columns(2)

        with r3c1:
            with st.expander("🐛 Pest Detection"):
                pest_result = results.get("pests")
                if pest_result:
                    st.write(f"Pest Count: {pest_result.get('pest_count')}")
                    for idx, pest in enumerate(pest_result.get("pests", []), 1):
                        st.markdown(f"• Detection {idx}")
                        for k, v in pest.items():
                            st.write(f"  {k.replace('_',' ').title()}: {format_value(v)}")

        with r3c2:
            with st.expander("📌 System Status"):
                st.write("All modules executed independently.")
                st.write("Failures in one module do not affect others.")


# =================================================
# TAB 2 — METRIC INFORMATION
# =================================================

with tab2:

    st.header("📘 Understanding the Metrics")

    # -------------------------------------------------
    # Leaf Detection
    # -------------------------------------------------
    with st.expander("🌿 Leaf Detection"):
        st.markdown("""
        Detects the primary leaf region using YOLO object detection.

        **Fundamental Use:**  
        The bounding box defines the region of interest (ROI) so that all further analysis
        (disease, dryness, greenness, pests) focuses only on the plant tissue and not background noise.
        """)

    # -------------------------------------------------
    # Disease Classification
    # -------------------------------------------------
    with st.expander("🦠 Disease Classification"):
        st.markdown("""
        Classifies the leaf using a deep learning model.

        **Fundamental Use:**  
        Identifies the most probable disease class along with a confidence score.
        This enables early intervention and targeted treatment.

        The confidence score represents the model's certainty in its prediction.
        """)

    # -------------------------------------------------
    # Dryness Metrics
    # -------------------------------------------------
    with st.expander("💧 Dryness Metrics"):

        st.markdown("""
        ### 1️⃣ Dryness Score  
        **Composite Index**

        It is a fused value combining color and texture information.

        **Fundamental Use:**  
        Represents probability or intensity of water stress.  
        Used to trigger irrigation or intervention systems.
        """)

        st.markdown("""
        ### 2️⃣ Dryness Level  
        **Categorical Classifier**

        Maps numeric dryness scores to human-readable categories.

        **Fundamental Use:**  
        Enables rapid dashboard interpretation (Low, Moderate, High).
        """)

        st.markdown("""
        ### 3️⃣ Green Ratio  
        **Biomass Coverage Metric**

        Measures percentage of healthy green canopy.

        **Fundamental Use:**  
        Tracks crop growth stages and canopy coverage over time.
        """)

        st.markdown("""
        ### 4️⃣ Brown Ratio  
        **Senescence / Background Metric**

        Quantifies non-photosynthetic material.

        **Fundamental Use:**  
        Increasing Brown Ratio indicates crop stress, soil exposure,
        harvest readiness, or crop failure.
        """)

        st.markdown("""
        ### 5️⃣ Saturation Mean  
        **Color Purity Indicator**

        Measures intensity of green coloration.

        **Fundamental Use:**  
        Acts as proxy for chlorophyll density.  
        Low saturation may indicate nutrient deficiency or early stress.
        """)

        st.markdown("""
        ### 6️⃣ Texture Variance  
        **Structural Roughness Metric**

        Measures spatial roughness patterns.

        - Low Variance → Smooth, hydrated leaves  
        - High Variance → Crinkled, wilted, or dehydrated tissue  

        **Fundamental Use:**  
        Detects physical changes like wilting and curling.
        """)

    # -------------------------------------------------
    # Green Index & Greenness Metrics
    # -------------------------------------------------
    with st.expander("🌱 Green Index & Vegetation Metrics"):

        st.markdown("""
        ### 1️⃣ Excess Green (ExG)  
        **Contrast Enhancement Metric**

        Formula: 2G − R − B

        **Fundamental Use:**  
        Cancels out soil and non-green background elements,
        making plant regions highly distinguishable.

        Turns complex images into high-contrast plant maps.
        """)

        st.markdown("""
        ### 2️⃣ Normalized Green  
        **Lighting Standardization Metric**

        Removes lighting condition effects.

        **Fundamental Use:**  
        Enables consistent comparison across time,
        weather, and camera conditions.
        """)

        st.markdown("""
        ### 3️⃣ Green Score  
        **Health Quality Rating**

        A calibrated 0–1 scale derived from greenness metrics.

        **Fundamental Use:**  
        Detects early chlorosis (yellowing)  
        Indicates nutrient deficiencies  
        Evaluates plant vitality relative to growth stage
        """)

    # -------------------------------------------------
    # Pest Detection
    # -------------------------------------------------
    with st.expander("🐛 Pest Detection"):
        st.markdown("""
        Detects visible insect presence using object detection.

        **Fundamental Use:**  
        Provides pest count, species label, bounding boxes,
        and confidence scores.

        Enables early pest control and targeted treatment decisions.
        """)


# =================================================
# TAB 3 — GenAI Report
# =================================================
with tab3:

    st.header("🧠 AI-Generated Plant Health Report")

    if "analysis_results" not in st.session_state:
        st.info("Please analyze an image in the Analytics tab first.")
    else:
        results = st.session_state["analysis_results"]

        st.write("Generate an intelligent, farmer-friendly summary based on the detected metrics.")

        if st.button("Generate AI Report"):

            with st.spinner("Generating intelligent report..."):

                try:
                    report = genai_generator.generate(results)

                    if isinstance(report, dict):

                        structured = report.get("structured_analysis")
                        narrative = report.get("narrative_report")

                        if structured:
                            st.subheader("📊 Structured Analysis")
                            for key, value in structured.items():
                                st.write(f"**{key.replace('_',' ').title()}**: {value}")

                        st.markdown("---")

                        if narrative:
                            st.subheader("📝 Narrative Report")

                            # ✅ FIXED BLOCK — JSON parsing added here
                            if isinstance(narrative, str):
                                try:
                                    parsed = json.loads(narrative)

                                    if isinstance(parsed, dict):
                                        for key, value in parsed.items():
                                            st.markdown(f"### {key.replace('_',' ').title()}")
                                            st.write(value)
                                    else:
                                        st.write(parsed)

                                except json.JSONDecodeError:
                                    st.write(narrative)
                            else:
                                st.write(narrative)

                    else:
                        st.error("Unexpected response format from GenAI module.")

                except Exception as e:
                    st.error(f"GenAI report generation failed: {str(e)}")





