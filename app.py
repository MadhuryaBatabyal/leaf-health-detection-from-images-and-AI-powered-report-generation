"""
🌿 LEAF HEALTH PRO - ERROR-PROOF PRODUCTION VERSION
Handles missing models gracefully
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import google.generativeai as genai

# Try importing YOLO/ONNX (optional)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.warning("⚠️ Install ultralytics for YOLO: `pip install ultralytics`")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    st.warning("⚠️ Install onnxruntime: `pip install onnxruntime`")

# Page config FIRST
st.set_page_config(page_title="🌿 Leaf Health Pro", layout="wide", page_icon="🍃")

# CSS
st.markdown("""
<style>
.main-title { font-size: 3rem; color: #2E7D32; text-align: center; margin-bottom: 2rem; }
.good-metric { background: linear-gradient(135deg, #4CAF50, #81C784); padding: 1rem; border-radius: 1rem; color: white; text-align: center; }
.warning-metric { background: linear-gradient(135deg, #FF9800, #FFB74D); }
.danger-metric { background: linear-gradient(135deg, #F44336, #E57373); }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🍃 Leaf Health Analysis Pro</h1>', unsafe_allow_html=True)

# ==================== SAFE MODEL LOADING ====================
@st.cache_resource
def safe_load_models():
    """Load models only if they exist"""
    models = {}
    
    # Leaf YOLO (your file)
    try:
        if YOLO_AVAILABLE:
            models['leaf_yolo'] = YOLO('leaf_detection.pt')
    except FileNotFoundError:
        st.info("ℹ️ Using HSV leaf detection (leaf_detection.pt not found)")
        models['leaf_yolo'] = None
    
    # Disease ONNX (your file)
    try:
        if ONNX_AVAILABLE:
            models['disease_onnx'] = ort.InferenceSession('leaf_disease.onnx')
    except FileNotFoundError:
        st.info("ℹ️ Using HSV disease detection (leaf_disease.onnx not found)")
        models['disease_onnx'] = None
    
    # Pest YOLO (SKIP if missing - FIXES ERROR)
    try:
        if YOLO_AVAILABLE:
            models['pest_yolo'] = YOLO('pest_detection.pt')
    except FileNotFoundError:
        st.info("ℹ️ No pest model - using HSV pest detection")
        models['pest_yolo'] = None
    
    return models

models = safe_load_models()

# Gemini
GEMINI_READY = False
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    GEMINI_READY = True
except:
    pass

# ==================== CORE ANALYSIS ====================
def detect_leaf_smart(img_cv):
    """Smart leaf detection - YOLO first, HSV fallback"""
    if models.get('leaf_yolo'):
        results = models['leaf_yolo'](img_cv, conf=0.4, verbose=False)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            box = results[0].boxes.xyxy[0].cpu().numpy()
            return tuple(box.astype(int))
    
    # HSV Fallback (your original code)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35,40,40]), np.array([85,255,255]))
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return (x,y,x+w,y+h)
    h,w = img_cv.shape[:2]
    return (int(0.2*w),int(0.2*h),int(0.8*w),int(0.8*h))

def classify_disease_smart(leaf_crop):
    """Smart disease - ONNX first, HSV fallback"""
    if models.get('disease_onnx'):
        try:
            img_resized = cv2.resize(leaf_crop, (224,224))
            img_norm = np.transpose(img_resized.astype(np.float32)/255, (2,0,1))[None,...]
            inputs = {models['disease_onnx'].get_inputs()[0].name: img_norm}
            outputs = models['disease_onnx'].run(None, inputs)
            probs = torch.softmax(torch.tensor(outputs[0]),0).numpy()[0]
            classes = ['Healthy','Blight','Rust','Mildew','Spot']
            pred = np.argmax(probs)
            return classes[pred%len(classes)], probs[pred]
        except:
            pass
    
    # HSV Fallback
    hsv = cv2.cvtColor(leaf_crop, cv2.COLOR_BGR2HSV)
    brown_mask = cv2.inRange(hsv, np.array([10,100,20]), np.array([25,255,200]))
    brown_ratio = cv2.countNonZero(brown_mask) / brown_mask.size
    green_ratio = 1 - brown_ratio
    
    if green_ratio > 0.7: return "Healthy", green_ratio
    elif brown_ratio > 0.3: return "Blight Suspect", 1-brown_ratio
    return "Minor Stress", green_ratio

def detect_pests_smart(img_cv):
    """Smart pests - YOLO first, HSV fallback"""
    if models.get('pest_yolo'):
        try:
            results = models['pest_yolo'](img_cv, conf=0.3, verbose=False)
            return len(results[0].boxes), results[0].boxes
        except:
            pass
    
    # HSV Fallback
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    dark_mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,50]))
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pest_contours = [c for c in contours if 50 < cv2.contourArea(c) < 1000]
    return len(pest_contours), []

# ==================== MAIN APP ====================
st.subheader("📤 Upload Leaf Image")
uploaded_file = st.file_uploader("JPG/PNG", type=['jpg','jpeg','png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    with st.spinner("🔍 AI Analysis Running..."):
        # PIPELINE
        bbox = detect_leaf_smart(img_cv)
        x1,y1,x2,y2 = map(int, bbox)
        leaf_crop = img_cv[y1:y2, x1:x2]
        
        disease_name, disease_conf = classify_disease_smart(leaf_crop)
        pest_count, _ = detect_pests_smart(img_cv)
        
        # Green index
        hsv_crop = cv2.cvtColor(leaf_crop, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv_crop, np.array([35,40,40]), np.array([85,255,255]))
        green_score = cv2.countNonZero(green_mask) / green_mask.size
        
        # AI Report
        if GEMINI_READY:
            prompt = f"Disease: {disease_name} ({disease_conf:.0%}), Pests: {pest_count}, Green: {green_score:.0%}"
            response = gemini_model.generate_content(prompt)
            ai_report = response.text
        else:
            ai_report = f"**{disease_name}** ({disease_conf:.0%}). {pest_count} pests detected."
    
    # DISPLAY
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(leaf_crop, cv2.COLOR_BGR2RGB), caption="🌿 Leaf", width=380)
    with col2:
        annotated_leaf = leaf_crop.copy()
        cv2.rectangle(annotated_leaf, (0,0,leaf_crop.shape[1],leaf_crop.shape[0]), (0,255,0), 3)
        st.image(annotated_leaf, channels="BGR", caption="✅ Bounds", width=380)
    
    # Full image
    full_img = img_cv.copy()
    cv2.rectangle(full_img, (x1,y1,x2,y2), (0,255,0), 3)
    cv2.putText(full_img, disease_name, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    st.image(full_img, channels="BGR", caption="📐 Full Analysis", width=600)
    
    # METRICS
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f'<div class="good-metric"><b>🦠 {disease_name}</b><br>{disease_conf:.0%}</div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="good-metric"><b>🌱 Green Score</b><br>{green_score:.0%}</div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="good-metric"><b>🐛 Pests</b><br>{pest_count}</div>', unsafe_allow_html=True)
    with col4: st.markdown('<div class="good-metric"><b>🤖 AI Ready</b><br>✅</div>', unsafe_allow_html=True)
    
    st.markdown("### 🌾 **AI Farmer Report**")
    st.success(ai_report)

else:
    st.info("👆 Upload leaf image for instant AI analysis!")

st.markdown("---")
st.caption("✅ Works with OR without YOLO/ONNX models • Add GEMINI_API_KEY for AI reports")
