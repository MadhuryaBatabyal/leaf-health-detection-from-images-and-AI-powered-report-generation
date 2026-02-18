"""
🌿 LEAF HEALTH ANALYSIS PRO - FULL PRODUCTION SYSTEM
Integrates YOLO + ONNX + Gemini with your uploaded models
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import onnxruntime as ort
import google.generativeai as genai

# Page config - FIRST LINE
st.set_page_config(page_title="🌿 Leaf Health Pro", layout="wide", page_icon="🍃")

# Custom CSS
st.markdown("""
<style>
.main-title { 
    font-size: 3rem; font-weight: bold; color: #2E7D32; 
    text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}
.leaf-metric { 
    background: linear-gradient(135deg, #4CAF50, #81C784); 
    padding: 1rem; border-radius: 1rem; text-align: center; color: white;
}
.warning-metric { 
    background: linear-gradient(135deg, #FF9800, #FFB74D);
}
.danger-metric { 
    background: linear-gradient(135deg, #F44336, #E57373);
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">🍃 Leaf Health Analysis Pro</h1>', unsafe_allow_html=True)

# ==================== MODEL INITIALIZATION ====================
@st.cache_resource
def load_production_models():
    """Load all your uploaded models"""
    
    # YOLO Leaf Detection (your leaf_detection.pt)
    leaf_model = YOLO('leaf_detection.pt')
    
    # ONNX Disease Model (your leaf_disease.onnx)
    disease_session = ort.InferenceSession('leaf_disease.onnx')
    
    # YOLO Pest Detection
    pest_model = YOLO('pest_detection.pt')  # Assuming you have this
    
    return leaf_model, disease_session, pest_model

leaf_model, disease_session, pest_model = load_production_models()

# Gemini AI
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    GEMINI_READY = True
except:
    GEMINI_READY = False
    st.warning("⚠️ Add GEMINI_API_KEY to secrets.toml for AI reports")

# ==================== ANALYSIS FUNCTIONS ====================
def detect_leaf_bbox(img_cv):
    """YOLO Leaf Detection"""
    results = leaf_model(img_cv, conf=0.4, verbose=False)
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        box = results[0].boxes.xyxy[0].cpu().numpy()
        return tuple(box.astype(int))
    # Fallback to HSV if YOLO fails
    return hsv_leaf_detection(img_cv)

def classify_disease_onnx(leaf_crop):
    """ONNX Disease Classification"""
    # Preprocess for MobileNetV2 (224x224)
    img_resized = cv2.resize(leaf_crop, (224, 224))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = np.transpose(img_norm, (2, 0, 1))[None, ...]  # CHW + batch
    
    # Predict
    inputs = {disease_session.get_inputs()[0].name: img_norm}
    outputs = disease_session.run(None, inputs)
    probs = torch.softmax(torch.tensor(outputs[0]), dim=1).numpy()[0]
    
    DISEASE_CLASSES = ['Healthy', 'Early Blight', 'Late Blight', 'Rust', 
                      'Mildew', 'Anthracnose', 'Leaf Spot']
    pred_idx = np.argmax(probs)
    return DISEASE_CLASSES[pred_idx % len(DISEASE_CLASSES)], probs[pred_idx]

def detect_pests_yolo(img_cv):
    """YOLO Pest Detection"""
    results = pest_model(img_cv, conf=0.3, verbose=False)
    pests = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            pests.append({
                'bbox': tuple(box.xyxy[0].cpu().numpy().astype(int)),
                'conf': float(box.conf[0])
            })
    return len(pests), pests

def hsv_leaf_detection(img_cv):  # Your original fallback
    """HSV-based leaf detection (backup)"""
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, green_lower, green_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return (x, y, x+w, y+h)
    h, w = img_cv.shape[:2]
    return (int(0.2*w), int(0.2*h), int(0.8*w), int(0.8*h))

def advanced_green_indices(leaf_crop):
    """Vegetation indices"""
    hsv = cv2.cvtColor(leaf_crop, cv2.COLOR_BGR2HSV)
    h, w = leaf_crop.shape[:2]
    
    # Excess Green
    exg = 2 * hsv[:,:,1] - hsv[:,:,0] - hsv[:,:,2]
    exg_ratio = np.sum(exg > np.mean(exg)) / (h * w)
    
    # Green ratio
    green_mask = cv2.inRange(hsv, np.array([35,40,40]), np.array([85,255,255]))
    green_ratio = cv2.countNonZero(green_mask) / (h * w)
    
    return {
        'exg': exg_ratio,
        'green_score': min(green_ratio * 1.2, 1.0),
        'health_status': 'Healthy' if green_ratio > 0.6 else 'Stressed'
    }

# ==================== MAIN APP ====================
st.subheader("📤 Upload Leaf Image")
uploaded_file = st.file_uploader("Choose JPG/PNG", type=['jpg','jpeg','png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 🔥 FULL PRODUCTION PIPELINE
    with st.spinner("🚀 Running YOLO + ONNX + Gemini Analysis..."):
        # 1. Leaf Detection
        bbox = detect_leaf_bbox(img_cv)
        x1, y1, x2, y2 = map(int, bbox)
        leaf_crop = img_cv[y1:y2, x1:x2]
        
        # 2. Disease Classification
        disease_name, disease_conf = classify_disease_onnx(leaf_crop)
        
        # 3. Pest Detection
        pest_count, pest_bboxes = detect_pests_yolo(img_cv)
        
        # 4. Green Indices
        green_data = advanced_green_indices(leaf_crop)
        
        # 5. Gemini Report
        if GEMINI_READY:
            prompt = f"""
            Indian farmer leaf report (simple English + Hindi):
            Disease: {disease_name} ({disease_conf:.0%})
            Pests: {pest_count}
            Green Health: {green_data['health_status']}
            
            Give: 1-line summary, Risk (Low/Medium/High), 3 actions.
            """
            response = gemini_model.generate_content(prompt)
            ai_report = response.text
        else:
            ai_report = f"**{disease_name}** detected ({disease_conf:.0%} confidence). {pest_count} pests found."
    
    # 🎨 VISUALIZATION (Fixed sizes!)
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(leaf_crop, cv2.COLOR_BGR2RGB), 
                caption="🌿 AI Detected Leaf", width=380)
    with col2:
        # Draw annotations on crop
        annotated_crop = leaf_crop.copy()
        cv2.rectangle(annotated_crop, (0,0,leaf_crop.shape[1],leaf_crop.shape[0]), (0,255,0), 3)
        st.image(annotated_crop, channels="BGR", caption="✅ Analysis Crop", width=380)
    
    # Full image
    full_annotated = img_cv.copy()
    cv2.rectangle(full_annotated, (x1,y1,x2,y2), (0,255,0), 3)
    cv2.putText(full_annotated, f"{disease_name}", (x1, y1-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    st.image(full_annotated, channels="BGR", caption="📐 Full Image Context", width=600)
    
    # 📊 PRODUCTION METRICS
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        conf_color = "metric-good" if disease_conf > 0.8 else "metric-warning" if disease_conf > 0.6 else "metric-danger"
        st.metric("🦠 Disease", disease_name, f"{disease_conf:.0%}", 
                 delta=None, label_visibility="collapsed")
    with col2:
        st.markdown('<div class="leaf-metric">🌱 Green Score<br><h2>{:.0%}</h2></div>'.format(green_data['green_score']), 
                   unsafe_allow_html=True)
    with col3:
        st.metric("💧 ExG Index", f"{green_data['exg']:.1%}")
    with col4:
        st.metric("🐛 Pests", pest_count)
    
    # 🤖 AI FARMER REPORT
    st.markdown("### 🌾 **AI Farmer Report**")
    st.success(ai_report)

else:
    st.info("👆 **Upload a leaf image** to get AI-powered analysis with YOLO + ONNX + Gemini!")

# Footer
st.markdown("---")
st.markdown("*Powered by YOLOv8 + ONNX MobileNetV2 + Gemini 1.5 Flash*")
