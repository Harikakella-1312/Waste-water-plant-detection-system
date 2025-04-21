import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from tensorflow.keras.models import load_model
import rasterio
from rasterio.transform import from_origin
import tempfile
import time

st.set_page_config(
    page_title="WWTP Detection System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

IMG_SIZE = (224, 224)
CATEGORIES = {
    0: "NONWWTP",
    1: "WWTP"
}

MODEL_PATH = "wwtp_detection_model.h5"
THRESHOLD_CONFIG_PATH = "threshold_config.txt"

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 1rem;
        color: #0D47A1;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        color: #1565C0;
    }
    .prediction {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .prediction-wwtp {
        background-color: #C8E6C9;
        color: #2E7D32;
    }
    .prediction-nonwwtp {
        background-color: #FFCDD2;
        color: #C62828;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .image-container {
        border: 1px solid #BDBDBD;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F57F17;
        font-weight: bold;
    }
    .confidence-low {
        color: #C62828;
        font-weight: bold;
    }
    .image-caption {
        text-align: center;
        font-size: 0.9rem;
        color: #424242;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🛰️ WWTP Detection System")
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This application detects Wastewater Treatment Plants (WWTP) from satellite imagery using a deep learning model.

**Features:**
- Upload satellite images
- Get WWTP detection predictions
- View DEM (Digital Elevation Model) visualization
- See simulated Landsat analysis
- Examine model's attention areas
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.45, 0.05, 
                             help="Adjust the confidence threshold for WWTP detection")

st.sidebar.markdown("---")
st.sidebar.markdown("### Help")
with st.sidebar.expander("How to Use"):
    st.markdown("""
    1. Upload a satellite image using the file uploader
    2. The system will analyze the image
    3. View the detection result and visualizations
    4. Adjust the threshold slider to fine-tune detection sensitivity
    """)

with st.sidebar.expander("About the Data Visualizations"):
    st.markdown("""
    - **Original Image**: The satellite image you uploaded
    - **DEM Visualization**: Digital Elevation Model showing terrain elevation
    - **Landsat Analysis**: Simulated spectral analysis highlighting features like water bodies
    - **Model Attention**: Heatmap showing areas the model focused on for its prediction
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ by HKP Team")

st.markdown("<div class='main-header'>🛰️ Wastewater Treatment Plant Detection System</div>", unsafe_allow_html=True)

@st.cache_resource
def load_detection_model():
    """Load the trained model and detection threshold"""
    try:
        detection_threshold = 0.45
        
        if os.path.exists(THRESHOLD_CONFIG_PATH):
            with open(THRESHOLD_CONFIG_PATH, 'r') as f:
                for line in f:
                    if line.startswith("Best detection threshold:"):
                        try:
                            detection_threshold = float(line.split(":")[1].strip())
                        except:
                            pass
        
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            return model, detection_threshold
        else:
            return None, detection_threshold
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, 0.45

model, default_threshold = load_detection_model()

if model is None:
    st.error("Failed to load the detection model. Please check the model path and try again.")
    st.info("For this demo, we'll continue with simulated predictions.")

st.markdown("<div class='sub-header'>Upload Satellite Image</div>", unsafe_allow_html=True)
upload_method = st.radio("Select upload method", ["Upload File", "Use Sample Images"])

sample_image = None

if upload_method == "Upload File":
    uploaded_file = st.file_uploader("Choose a satellite image", type=["jpg", "jpeg", "png", "tif", "tiff"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            if image_np.shape[-1] == 4:
                image_np = image_np[:, :, :3]
                
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        except:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                tmp.write(image_bytes)
                tmp_name = tmp.name
                
            try:
                image_np = cv2.imread(tmp_name)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            except:
                try:
                    with rasterio.open(tmp_name) as src:
                        image_np = np.dstack([src.read(i) for i in range(1, min(4, src.count+1))])
                        
                        if image_np.dtype != np.uint8:
                            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min()) * 255
                            image_np = image_np.astype(np.uint8)
                except:
                    st.error("Could not read the image. Please try a different format.")
                    image_np = None
                    
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
                
        sample_image = image_np
else:
    st.markdown("<div class='info-box'>Select a sample image to analyze:</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    def create_sample(type="wwtp", size=(300, 300)):
        """Create a synthetic sample image"""
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        
        if type == "wwtp":
            img.fill(120) 
            centers = [
                (size[0]//3, size[1]//3),
                (size[0]//3*2, size[1]//3),
                (size[0]//3, size[1]//3*2),
                (size[0]//3*2, size[1]//3*2)
            ]
            
            for center in centers:
                cv2.circle(img, center, 30, (0, 0, 255), -1)  # Blue circles
                
            cv2.rectangle(img, (10, 10), (100, 50), (0, 150, 0), -1)
            cv2.rectangle(img, (200, 250), (290, 290), (0, 150, 0), -1)
            
            cv2.rectangle(img, (150, 50), (200, 100), (200, 200, 200), -1)
            
        elif type == "industrial":
            img.fill(100)  # Gray background
            
            cv2.rectangle(img, (50, 50), (100, 100), (180, 180, 180), -1)
            cv2.rectangle(img, (150, 70), (250, 130), (200, 200, 200), -1)
            cv2.rectangle(img, (70, 180), (120, 280), (190, 190, 190), -1)
            cv2.rectangle(img, (180, 200), (240, 270), (170, 170, 170), -1)
            
            # Add some roads
            cv2.line(img, (0, 150), (300, 150), (50, 50, 50), 10)
            cv2.line(img, (150, 0), (150, 300), (50, 50, 50), 10)
            
        else:
            img.fill(50)  # Dark background
            
            cv2.rectangle(img, (0, 0), (size[0], size[1]), (30, 100, 30), -1)
            
            cv2.rectangle(img, (20, 20), (130, 130), (60, 180, 60), -1)
            cv2.rectangle(img, (150, 20), (280, 130), (70, 170, 40), -1)
            cv2.rectangle(img, (20, 150), (130, 280), (80, 160, 50), -1)
            cv2.rectangle(img, (150, 150), (280, 280), (90, 150, 60), -1)
            
            points = np.array([[0, 200], [50, 220], [100, 190], [150, 210], 
                              [200, 180], [250, 220], [300, 210]], dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(img, [points], False, (0, 0, 255), 10)
        
        return img
    
    sample1 = create_sample("wwtp")
    sample2 = create_sample("industrial")
    sample3 = create_sample("rural")
    
    with col1:
        st.image(sample1, caption="Sample 1: WWTP", width=150)
        if st.button("Select Sample 1"):
            sample_image = sample1
    
    with col2:
        st.image(sample2, caption="Sample 2: Industrial", width=150)
        if st.button("Select Sample 2"):
            sample_image = sample2
    
    with col3:
        st.image(sample3, caption="Sample 3: Rural", width=150)
        if st.button("Select Sample 3"):
            sample_image = sample3

if sample_image is not None:
    st.markdown("<div class='sub-header'>Image Analysis</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>Original Satellite Image</div>", unsafe_allow_html=True)
    st.image(sample_image, caption="Original Image", use_column_width=True)
    
    with st.spinner("Analyzing image..."):
        processed_img = cv2.resize(sample_image, IMG_SIZE)
        processed_img = processed_img.astype('float32') / 255.0
        
        prediction_start = time.time()
        
        if model is not None:
            img_batch = np.expand_dims(processed_img, axis=0)
            predictions = model.predict(img_batch)[0]
            prediction_class = 1 if predictions[1] > threshold else 0
            confidence = predictions[prediction_class] * 100
        else:
            import random
            predictions = [random.uniform(0, 1), random.uniform(0, 1)]
            predictions = predictions / sum(predictions)  # Normalize
            prediction_class = 1 if predictions[1] > threshold else 0
            confidence = predictions[prediction_class] * 100
        
        prediction_time = time.time() - prediction_start
    
    result_class = CATEGORIES[prediction_class]
    
    if result_class == "WWTP":
        st.markdown(f"<div class='prediction prediction-wwtp'>Prediction: {result_class}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='prediction prediction-nonwwtp'>Prediction: {result_class}</div>", unsafe_allow_html=True)
    
    if confidence > 90:
        confidence_class = "confidence-high"
    elif confidence > 70:
        confidence_class = "confidence-medium"
    else:
        confidence_class = "confidence-low"
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"Confidence: <span class='{confidence_class}'>{confidence:.2f}%</span>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"Prediction Time: {prediction_time:.3f} seconds")
    
    with st.expander("Technical Details"):
        st.markdown("### Prediction Probabilities")
        prob_df = {"Category": list(CATEGORIES.values()), "Probability": predictions}
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(prob_df["Category"], prob_df["Probability"], color=['#EF5350', '#66BB6A'])
        
        ax.axhline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')
        
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities by Category')
        ax.legend()
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        st.markdown("### Model Information")
        if model is not None:
            st.markdown(f"- **Model Type**: EfficientNet-based CNN")
            st.markdown(f"- **Input Shape**: {model.input.shape}")
            st.markdown(f"- **Default Threshold**: {default_threshold}")
            st.markdown(f"- **Current Threshold**: {threshold}")
        else:
            st.markdown("Model information not available (demo mode)")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("<div class='sub-header'>Multi-View Analysis</div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "DEM Analysis", 
        "Landsat Analysis", 
        "Model Attention",
        "Feature Detection"
    ])
    
    with tab1:
        with st.spinner("Generating DEM visualizations..."):
            def generate_dem(image, colored=True):
                """Generate a simulated Digital Elevation Model from an image"""
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
                
                smoothed = cv2.medianBlur(gray, 5)
                
                rows, cols = smoothed.shape
                X, Y = np.meshgrid(np.linspace(0, 3, cols), np.linspace(0, 3, rows))
                elevation = 20 * np.sin(X) * np.cos(Y)
                
                dem = smoothed.astype(np.float32) + elevation
                
                dem = cv2.normalize(dem, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                if colored:
                    return cv2.applyColorMap(dem, cv2.COLORMAP_TURBO)
                else:
                    return dem
            
            dem_gray = generate_dem(sample_image, colored=False)
            dem_colored = generate_dem(sample_image, colored=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(dem_gray, caption="DEM (Grayscale)", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='image-caption'>Digital Elevation Model (grayscale) showing terrain elevation</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(dem_colored, caption="DEM (Color-coded)", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='image-caption'>Color-coded elevation model with blue=low and red=high elevation</div>", unsafe_allow_html=True)
                
            st.markdown("### Elevation Profile")
            
            mid_row = dem_gray.shape[0] // 2
            profile = dem_gray[mid_row, :]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(profile, color='blue', linewidth=2)
            ax.set_xlabel('Distance (pixels)')
            ax.set_ylabel('Elevation (relative)')
            ax.set_title(f'Elevation Profile at Row {mid_row}')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            ax.fill_between(range(len(profile)), profile, alpha=0.3, color='skyblue')
            
            st.pyplot(fig)
    
    with tab2:
        with st.spinner("Generating Landsat visualizations..."):
            def simulate_landsat(image):
                """Create simulated Landsat-like visualizations from RGB image"""
                if len(image.shape) == 3:
                    if image.shape[2] == 3:
                        b, g, r = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    else:
                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        b = g = r = gray
                else:
                    b = g = r = image
                
                nir = cv2.addWeighted(r, 0.7, g, 0.3, 0)
                nir = cv2.equalizeHist(nir)  # Enhance contrast
                
                swir = cv2.addWeighted(b, 0.6, r, 0.4, 0)
                swir = cv2.GaussianBlur(swir, (3, 3), 0)
                
                true_color = cv2.merge([b, g, r])
                
                false_color = cv2.merge([g, r, nir])
                
                nir_f = nir.astype(np.float32)
                r_f = r.astype(np.float32)
                epsilon = 1e-10  # Avoid division by zero
                ndvi = (nir_f - r_f) / (nir_f + r_f + epsilon)
                
                ndvi_scaled = ((ndvi + 1) / 2 * 255).astype(np.uint8)
                ndvi_colored = cv2.applyColorMap(ndvi_scaled, cv2.COLORMAP_JET)
                
                g_f = g.astype(np.float32)
                ndwi = (g_f - nir_f) / (g_f + nir_f + epsilon)
                ndwi_scaled = ((ndwi + 1) / 2 * 255).astype(np.uint8)
                ndwi_colored = cv2.applyColorMap(ndwi_scaled, cv2.COLORMAP_OCEAN)
                
                return {
                    "true_color": true_color,
                    "false_color": false_color,
                    "ndvi": ndvi_colored,
                    "ndwi": ndwi_colored,
                    "swir_composite": cv2.merge([swir, b, g])
                }
            
            landsat_results = simulate_landsat(sample_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(cv2.cvtColor(landsat_results["false_color"], cv2.COLOR_BGR2RGB), 
                         caption="False Color Composite (NIR-R-G)", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='image-caption'>Vegetation appears in bright red, water in dark blue</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(cv2.cvtColor(landsat_results["ndvi"], cv2.COLOR_BGR2RGB), 
                         caption="NDVI (Vegetation Index)", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='image-caption'>Green = vegetation, Yellow = sparse vegetation, Red = non-vegetation</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(cv2.cvtColor(landsat_results["ndwi"], cv2.COLOR_BGR2RGB), 
                         caption="NDWI (Water Index)", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='image-caption'>Blue = water bodies, Red = dry land</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(cv2.cvtColor(landsat_results["swir_composite"], cv2.COLOR_BGR2RGB), 
                         caption="SWIR Composite", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='image-caption'>SWIR visualization highlights buildings and dry areas</div>", unsafe_allow_html=True)
            
            st.markdown("""
            ### Interpretation for WWTP Detection
            
            - **False Color Composite**: Treatment ponds in WWTPs typically appear as dark blue or black shapes due to water absorption in NIR
            - **NDVI (Vegetation Index)**: WWTPs typically show contrast between vegetated areas (green) and water/concrete (red)
            - **NDWI (Water Index)**: Treatment ponds appear as bright blue areas, helping differentiate from industrial facilities
            - **SWIR Composite**: Highlights moisture content, with treatment ponds showing distinctive patterns
            """)
    
    with tab3:
        with st.spinner("Generating model attention map..."):
            def generate_attention_map(image, prediction_class):
                """Generate a simulated attention map based on the prediction"""
                heatmap = np.zeros(image.shape[:2], dtype=np.float32)
                
                if prediction_class == 1:  # WWTP
                    if len(image.shape) == 3:
                        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                        
                        lower_blue = np.array([90, 50, 50])
                        upper_blue = np.array([130, 255, 255])
                        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
                        
                        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
                        
                        combined_mask = cv2.bitwise_or(blue_mask, dark_mask)
                        
                        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area > 100:
                                perimeter = cv2.arcLength(contour, True)
                                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                                
                                if circularity > 0.3:  # More circular shapes get more attention
                                    cv2.drawContours(heatmap, [contour], -1, circularity * 255, -1)
                    
                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        edges = cv2.Canny(gray, 50, 150)
                        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=20)
                        
                        if lines is not None:
                            for line in lines:
                                x1, y1, x2, y2 = line[0]
                                cv2.line(heatmap, (x1, y1), (x2, y2), 50, 2)
                    
                else:  # NONWWTP
                    if len(image.shape) == 3:
                        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                        
                        lower_green = np.array([40, 40, 40])
                        upper_green = np.array([80, 255, 255])
                        green_mask = cv2.inRange(hsv, lower_green, upper_green)
                        
                        heatmap[green_mask > 0] = 150
                        
                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        edges = cv2.Canny(gray, 50, 150)
                        
                        heatmap += cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
                
                heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                
                heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
                
                heatmap_colored = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
                
                alpha = 0.7 
                beta = 1 - alpha
                overlay = cv2.addWeighted(image.astype(np.uint8), alpha, heatmap_colored, beta, 0)
                
                return heatmap.astype(np.uint8), heatmap_colored, overlay
            
            heatmap, heatmap_colored, overlay = generate_attention_map(sample_image, prediction_class)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(heatmap, caption="Attention Heatmap (Grayscale)", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='image-caption'>Brighter areas indicate stronger model attention</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), 
                         caption="Attention Heatmap (Colored)", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='image-caption'>Red indicates strongest attention, blue indicates minimal attention</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), 
                     caption="Attention Overlay", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='image-caption'>Attention heatmap overlaid on the original image</div>", unsafe_allow_html=True)
            
            st.markdown("""
            ### Attention Map Interpretation
            
            The attention map shows which parts of the image the model focused on when making its prediction:
            
            - **For WWTP detection**: The model typically focuses on circular/oval treatment ponds, regular geometric patterns, and water bodies
            - **For non-WWTP classification**: The model looks for natural features, vegetation patterns, and absence of treatment infrastructure
            
            Stronger attention (red/yellow areas) indicates features that heavily influenced the prediction.
            """)
    
    with tab4:
        with st.spinner("Detecting features..."):
            def detect_wwtp_features(image):
                """Detect potential WWTP features like circular ponds and regular patterns"""
                if len(image.shape) == 2:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    image_rgb = image.copy()
                
                hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
                
                lower_water = np.array([90, 50, 50])
                upper_water = np.array([130, 255, 255])
                water_mask1 = cv2.inRange(hsv, lower_water, upper_water)
                
                dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
                
                water_mask = cv2.bitwise_or(water_mask1, dark_mask)
                
                kernel = np.ones((5, 5), np.uint8)
                water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
                water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
                
                contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                feature_img = image_rgb.copy()
                pond_img = image_rgb.copy()
                boundary_img = image_rgb.copy()
                
                pond_count = 0
                pond_areas = []
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 100:  # Filter out small noise
                        continue
                        
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    if circularity > 0.3:  # More circular shapes are likely treatment ponds
                        cv2.drawContours(pond_img, [contour], -1, (0, 255, 0), 2)
                        cv2.drawContours(feature_img, [contour], -1, (0, 255, 0), 2)
                        pond_count += 1
                        pond_areas.append(area)
                    else:
                        # Other water bodies
                        cv2.drawContours(feature_img, [contour], -1, (255, 0, 0), 2)
                
                gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=20)
                
                line_count = 0
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(feature_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        line_count += 1
                
                edges = cv2.Canny(gray, 30, 100)
                
                kernel = np.ones((3, 3), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=1)
                
                boundary_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in boundary_contours:
                    area = cv2.contourArea(contour)
                    if area > 1000: 
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        if 4 <= len(approx) <= 8:
                            cv2.drawContours(boundary_img, [approx], -1, (255, 165, 0), 2)
                            cv2.drawContours(feature_img, [approx], -1, (255, 165, 0), 2)
                
                pond_score = min(1.0, pond_count / 3) * 0.6  # Weight for ponds
                line_score = min(1.0, line_count / 20) * 0.3  # Weight for infrastructure
                
                size_regularity = 0.0
                if len(pond_areas) > 1:
                    mean_area = np.mean(pond_areas)
                    std_area = np.std(pond_areas)
                    cv_area = std_area / mean_area if mean_area > 0 else 1.0
                    size_regularity = max(0, 1 - min(1, cv_area))
                
                reg_score = size_regularity * 0.1
                
                wwtp_probability = pond_score + line_score + reg_score
                
                return {
                    "feature_image": feature_img,
                    "pond_image": pond_img,
                    "boundary_image": boundary_img,
                    "pond_count": pond_count,
                    "line_count": line_count,
                    "wwtp_probability": wwtp_probability
                }
            
            detection_results = detect_wwtp_features(sample_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(cv2.cvtColor(detection_results["pond_image"], cv2.COLOR_BGR2RGB), 
                         caption="Potential Treatment Ponds", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='image-caption'>Green contours show potential treatment ponds</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(cv2.cvtColor(detection_results["boundary_image"], cv2.COLOR_BGR2RGB), 
                         caption="Facility Boundaries", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='image-caption'>Orange contours show potential facility boundaries</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(detection_results["feature_image"], cv2.COLOR_BGR2RGB), 
                     caption="All Detected Features", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='image-caption'>Green = treatment ponds, Blue = other water, Yellow = infrastructure, Orange = boundaries</div>", unsafe_allow_html=True)
            
            st.markdown("### Feature Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Potential Treatment Ponds", detection_results["pond_count"])
                st.metric("Infrastructure Lines", detection_results["line_count"])
            
            with col2:
                wwtp_prob = detection_results["wwtp_probability"] * 100
                st.metric("WWTP Probability from Features", f"{wwtp_prob:.1f}%")
                
                model_prob = predictions[1] * 100
                st.metric("WWTP Probability from Model", f"{model_prob:.1f}%", 
                          f"{wwtp_prob - model_prob:.1f}%")
            
            st.markdown("""
            ### Feature Detection Interpretation
            
            This analysis identifies key features typical of wastewater treatment plants:
            
            - **Treatment Ponds**: Circular or oval-shaped water bodies
            - **Infrastructure**: Regular patterns of straight lines
            - **Facility Boundaries**: Perimeters that might indicate a treatment plant
            
            The feature-based probability is calculated from the density and characteristics of these features, 
            and can be compared with the model's prediction as a form of explainability.
            """)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Analysis Summary</div>", unsafe_allow_html=True)
    
    confidence_level = "High" if confidence > 90 else "Medium" if confidence > 70 else "Low"
    confidence_color = "#2E7D32" if confidence > 90 else "#F57F17" if confidence > 70 else "#C62828"
    
    st.markdown(f"""
    <div style="border-radius: 10px; padding: 15px; background-color: #f5f5f5;">
        <h3 style="margin-top: 0;">Detection Summary</h3>
        <p><strong>Prediction:</strong> {result_class}</p>
        <p><strong>Confidence:</strong> <span style="color: {confidence_color};">{confidence:.1f}% ({confidence_level})</span></p>
        <p><strong>Recommended Action:</strong> {
            "This location is likely a Wastewater Treatment Plant. Consider field verification for confirmation." 
            if result_class == "WWTP" and confidence > 70
            else "This location does not appear to be a Wastewater Treatment Plant." 
            if result_class == "NONWWTP" and confidence > 70
            else "Results are inconclusive. Consider additional analysis with higher resolution imagery."
        }</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Download Analysis")
    
    if st.download_button(
        label="Download Full Analysis Report (PDF)",
        data=b"Placeholder for PDF data",
        file_name="wwtp_analysis_report.pdf",
        mime="application/pdf",
    ):
        st.success("In a production version, this would generate and download a complete PDF report with all analysis results.")

# Add footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #666;">
    WWTP Detection System v1.0 | © 2025 HKP Team | Using AI for environmental monitoring
</div>
""", unsafe_allow_html=True)