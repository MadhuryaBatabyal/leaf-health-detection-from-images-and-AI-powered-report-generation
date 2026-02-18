import streamlit as st
import cv2
import numpy as np
import json
import sys
import os
from PIL import Image  # REQUIRED for Image.open()
import io  # Optional: safer file handling

# CRITICAL: FIRST STREAMLIT COMMAND - NOTHING BEFORE THIS
st.set_page_config(page_title="Leaf Health Analyzer", layout="wide")

sys.path.insert(0, os.path.dirname(__file__))

# Your classes and functions here (unchanged)
class SimplePlantAnalyzer:
    def __init__(self):
        pass

    def analyze(self, image):
        return {
            "leaf_detection": (100, 100, 400, 400),
            "disease": {"label": "Healthy", "confidence": 0.92},
            "pests": {"pest_count": 0, "pests": []},
            "dryness": {
                "dryness_score": 0.15, "dryness_level": "Low",
                "green_ratio": 0.78, "brown_ratio": 0.12,
                "saturation_mean": 0.65, "texture_variance": 45.2
            },
            "green_index": {
                "excess_green": 0.72, "normalized_green": 0.68, "green_score": 0.85
            }
        }

def draw_leaf_box(image, bbox):
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(image, "LEAF", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

def draw_pests(image, pest_data):
    if pest_data and pest_data.get('pest_count', 0) > 0:
        cv2.rectangle(image, (200, 200), (250, 250), (0, 0, 255), 2)
        cv2.putText(image, "PEST", (200, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return image

class GenAIReportGenerator:
    def __init__(self):
        pass

    def generate(self, results):
        return {
            "structured_analysis": {
                "Overall Health": "Good",
                "Action Required": "Monitor",
                "Priority": "Low"
            },
            "narrative_report": "The leaf shows good health indicators. Green index is strong (85%), dryness is low (15%), no pests detected, and disease confidence for 'Healthy' is 92%. Continue current care routine with weekly monitoring recommended."
        }

# NOW SAFE: CSS styling after config
st.markdown("""
<style>
    .main-title {
        text-align: center; font-size: 48px; font-weight: bold;
        color: #2E7D32; margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card { background: linear-gradient(135deg, #4CAF50, #81C784);
        padding: 20px; border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center; color: white; font-weight: bold; }
    .warning-card { background: linear-gradient(135deg, #FF9800, #FFB74D);
        padding: 20px; border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center; color: white; font-weight: bold; }
    .error-card { background: linear-gradient(135deg, #F44336, #E57373);
        padding: 20px; border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center; color: white; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 12px 24px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Title now works with CSS
st.markdown('<h1 class="main-title">🍃 Leaf Health Analyzer</h1>', unsafe_allow_html=True)
# Main app logic - SAFE VERSION
analyzer = SimplePlantAnalyzer()
genai = GenAIReportGenerator()

st.subheader("📤 Upload a Leaf Image")
uploaded_file = st.file_uploader("Choose JPG/PNG...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read & display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", use_column_width=True)
    
    # Convert for CV2 analysis
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Analyze
    results = analyzer.analyze(img_cv)
    
    # Draw overlays
    img_annotated = draw_leaf_box(img_cv.copy(), results['leaf_detection'])
    if results['pests']['pest_count'] > 0:
        img_annotated = draw_pests(img_annotated, results['pests'])
    
    st.image(img_annotated, channels="BGR", caption="✅ Analysis Overlay (Green=Leaf, Red=Pest)")
    
    # Metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🦠 Disease", results['disease']['label'], f"{results['disease']['confidence']:.0%}")
    with col2:
        st.metric("🌿 Dryness", results['dryness']['dryness_level'], f"{results['dryness']['dryness_score']:.0%}")
    with col3:
        st.metric("🍃 Green Score", f"{results['green_index']['green_score']:.0%}")
    with col4:
        st.metric("🐛 Pests", results['pests']['pest_count'])
    
    # AI Report
    report = genai.generate(results)
    st.balloons()
    st.success("**AI Summary:** " + report['narrative_report'])
else:
    st.info("👆 Upload an image to analyze leaf health!")



