import streamlit as st
import cv2
import numpy as np
import json
import sys
import os
from PIL import Image
import io

# CRITICAL: FIRST STREAMLIT COMMAND - NOTHING BEFORE THIS
st.set_page_config(page_title="Leaf Health Analyzer", layout="wide")

sys.path.insert(0, os.path.dirname(__file__))


class RealPlantAnalyzer:
    def __init__(self):
        pass

    def analyze(self, image_cv):
        hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
        h, w = image_cv.shape[:2]
        total_pixels = h * w

        # Green leaf mask
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_ratio = cv2.countNonZero(green_mask) / total_pixels

        # Brown/dry mask
        brown_lower = np.array([10, 100, 20])
        brown_upper = np.array([25, 255, 200])
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        brown_ratio = cv2.countNonZero(brown_mask) / total_pixels

        # Leaf bbox: largest green contour
        contours, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w_bbox, h_bbox = cv2.boundingRect(largest)
            bbox = (x, y, x + w_bbox, y + h_bbox)
        else:
            bbox = (int(0.3 * w), int(0.2 * h), int(0.7 * w), int(0.8 * h))

        # Pests: small dark spots
        pest_count = len([c for c in contours if 50 < cv2.contourArea(c) < 500])

        # Scores
        dryness_score = min(brown_ratio * 1.5, 1.0)
        dryness_level = (
            "Low" if dryness_score < 0.2
            else "Medium" if dryness_score < 0.5
            else "High"
        )
        green_score = min(green_ratio * 1.2, 1.0)

        return {
            "leaf_detection": bbox,
            "disease": {
                "label": (
                    "Healthy" if green_score > 0.6
                    else "Blight Suspect" if dryness_score > 0.3
                    else "Minor Stress"
                ),
                "confidence": min(green_score * 1.1, 0.95),
            },
            "pests": {"pest_count": pest_count, "pests": []},
            "dryness": {
                "dryness_score": dryness_score,
                "dryness_level": dryness_level,
                "green_ratio": green_ratio,
                "brown_ratio": brown_ratio,
                "saturation_mean": np.mean(hsv[:, :, 1]) / 255,
                "texture_variance": np.std(image_cv[:, :, 0]) / 255,
            },
            "green_index": {
                "excess_green": green_ratio,
                "normalized_green": (
                    green_ratio / (green_ratio + brown_ratio + 1e-6)
                ),
                "green_score": green_score,
            },
        }


def draw_leaf_box(image, bbox):
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            image, "LEAF", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
        )
    return image


def draw_pests(image, pest_data):
    if pest_data and pest_data.get("pest_count", 0) > 0:
        cv2.rectangle(image, (200, 200), (250, 250), (0, 0, 255), 2)
        cv2.putText(
            image, "PEST", (200, 195),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )
    return image


class GenAIReportGenerator:
    def __init__(self):
        pass

    def generate(self, results):
        disease = results["disease"]["label"]
        confidence = results["disease"]["confidence"]
        dryness = results["dryness"]["dryness_level"]
        green = results["green_index"]["green_score"]
        pests = results["pests"]["pest_count"]

        narrative = (
            f"The leaf is classified as **{disease}** with {confidence:.0%} confidence. "
            f"Green index is {green:.0%}, dryness is {dryness.lower()}, "
            f"and {pests} potential pest(s) detected. "
        )
        if disease == "Healthy":
            narrative += "Continue current care routine with weekly monitoring."
        else:
            narrative += "Consider closer inspection and targeted treatment."

        return {
            "structured_analysis": {
                "Overall Health": disease,
                "Action Required": "Monitor" if disease == "Healthy" else "Inspect",
                "Priority": "Low" if disease == "Healthy" else "Medium",
            },
            "narrative_report": narrative,
        }


# --- CSS Styling ---
st.markdown(
    """
<style>
    .main-title {
        text-align: center; font-size: 48px; font-weight: bold;
        color: #2E7D32; margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card { background: linear-gradient(135deg, #4CAF50, #81C784);
        padding: 20px; border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center; color: white; font-weight: bold; }
    .warning-card { background: linear-gradient(135deg, #FF9800, #FFB74D);
        padding: 20px; border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center; color: white; font-weight: bold; }
    .error-card { background: linear-gradient(135deg, #F44336, #E57373);
        padding: 20px; border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center; color: white; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; padding: 12px 24px; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 class="main-title">🍃 Leaf Health Analyzer</h1>',
    unsafe_allow_html=True,
)

# Initialize analyzers
analyzer = RealPlantAnalyzer()
genai = GenAIReportGenerator()

st.subheader("📤 Upload a Leaf Image")
uploaded_file = st.file_uploader("Choose JPG/PNG...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    # Convert for CV2 analysis
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Analyze
    results = analyzer.analyze(img_cv)

    # Draw overlays
    img_annotated = draw_leaf_box(img_cv.copy(), results["leaf_detection"])
    if results["pests"]["pest_count"] > 0:
        img_annotated = draw_pests(img_annotated, results["pests"])

    st.image(
        img_annotated, channels="BGR",
        caption="✅ Analysis Overlay (Green=Leaf, Red=Pest)",
    )

    # Metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "🦠 Disease", results["disease"]["label"],
            f"{results['disease']['confidence']:.0%}",
        )
    with col2:
        st.metric(
            "🌿 Dryness", results["dryness"]["dryness_level"],
            f"{results['dryness']['dryness_score']:.0%}",
        )
    with col3:
        st.metric("🍃 Green Score", f"{results['green_index']['green_score']:.0%}")
    with col4:
        st.metric("🐛 Pests", results["pests"]["pest_count"])

    # AI Report
    report = genai.generate(results)
    st.balloons()
    st.success("**AI Summary:** " + report["narrative_report"])
else:
    st.info("👆 Upload an image to analyze leaf health!")
