import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os # Import os for file path checking

# IMPORTANT: Ensure the EfficientNet preprocessing module is correctly imported if 
# you relied on the separate 'efficientnet' package during training.
# Since the 'efficientnet' dependency is listed in requirements.txt, we assume it's available.
try:
    import efficientnet.tfkeras as efficientnet # Import for preprocessing if needed
except ImportError:
    st.warning("Could not import 'efficientnet.tfkeras'. Relying on tf.keras.applications.")
    efficientnet = None

# Set page configuration
st.set_page_config(
    page_title="Autism Detection AI",
    page_icon="🧩",
    layout="centered"
)

# --- GLOBAL VARIABLES ---
# Must match the input size used during training in Colab
IMG_SIZE = 224
CLASS_NAMES = ['Autistic', 'Non_Autistic']

@st.cache_resource
def load_model(model_name):
    """
    Loads the Keras model.
    Cached to prevent reloading on every user interaction.
    """
    model_path = ""
    if model_name == "ResNet50 (Fine-Tuned)":
        # The filename you saved in Step 9 of your Colab
        model_path = "autism_resnet50_finetuned.keras" 
    elif model_name == "EfficientNetB0":
        # The filename you saved in Step 4 of your Colab
        model_path = "autism_efficientnet.keras"
    
    # Check if the file exists before attempting to load
    if not os.path.exists(model_path):
        st.error(f"🔴 File not found: `{model_path}`. Please ensure this file is uploaded to your GitHub repository in the same directory as `app.py`. Check if the file is too large (>100MB) and requires Git LFS.")
        return None

    try:
        # Load the model using the correct path
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        # Specific error for model corruption or dependency issues
        st.error(f"Error loading model: {e}")
        st.error("This often indicates a model file corruption, incorrect saving format, or a missing layer/dependency issue (like `efficientnet` not being installed correctly).")
        return None

def import_and_predict(image_data, model, model_name):
    """
    Resizes the image, applies specific preprocessing, and predicts.
    """
    # 1. Resize the image to (224, 224)
    size = (IMG_SIZE, IMG_SIZE)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    
    # 2. Convert to numpy array
    img_array = np.asarray(image)
    
    # 3. Handle channels (ensure it's RGB)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3] # Remove alpha channel
    
    # 4. Expand dimensions to create a batch (1, 224, 224, 3)
    img_reshape = img_array[np.newaxis, ...]
    
    # 5. Apply Model-Specific Preprocessing
    # ResNet and EfficientNet expect different input scalings.
    if "ResNet50" in model_name:
        # ResNet preprocessing is built into Keras Applications
        preprocessed_image = tf.keras.applications.resnet.preprocess_input(img_reshape.astype(float))
    elif "EfficientNetB0" in model_name:
        # Try to use the imported efficientnet preprocessing first, otherwise use the TensorFlow version
        if efficientnet and hasattr(efficientnet, 'preprocess_input'):
            preprocessed_image = efficientnet.preprocess_input(img_reshape.astype(float))
        else:
            # Fallback to the one built into TensorFlow/Keras
            preprocessed_image = tf.keras.applications.efficientnet.preprocess_input(img_reshape.astype(float))
    else:
        # Default to simple scaling if model name is unknown
        preprocessed_image = img_reshape.astype(float) / 255.0

    prediction = model.predict(preprocessed_image)
    return prediction

# --- UI LAYOUT ---
st.title("🧩 Autism Spectrum Detection")
st.markdown("Upload a facial image (like the ones used in the training dataset) to detect potential signs of autism using a **ResNet50** or **EfficientNetB0** model.")

# Sidebar for model selection
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox(
    "Choose Model", 
    ["ResNet50 (Fine-Tuned)", "EfficientNetB0"]
)

# File Uploader
file = st.file_uploader("Please upload a facial image (jpg, png)", type=["jpg", "png", "jpeg"])

if file is None:
    st.text("Awaiting image upload...")
else:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Load Model
    model = None
    with st.spinner(f"⏳ Loading {model_choice} model..."):
        model = load_model(model_choice)

    if model is not None:
        # Predict Button
        if st.button("Analyze Image", key="analyze_button"):
            with st.spinner("🤖 Analyzing..."):
                predictions = import_and_predict(image, model, model_choice)
                
                # Interpret Results
                score = tf.nn.softmax(predictions[0]) # Get probabilities
                predicted_class = CLASS_NAMES[np.argmax(score)] # Use softmax score for argmax
                confidence = 100 * np.max(score)

                # Display Results
                st.write("---")
                st.subheader("✅ Prediction Results")
                
                # Custom result display
                if predicted_class == 'Autistic':
                    st.markdown(f"**Result:** <span style='color: red; font-size: 24px;'>**{predicted_class}**</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Result:** <span style='color: green; font-size: 24px;'>**{predicted_class}**</span>", unsafe_allow_html=True)
                
                st.write(f"Confidence: **{confidence:.2f}%**")
                
                # Detailed probability bar chart
                st.write("---")
                st.markdown("**Probability Distribution**")
                
                prob_data = {
                    "Prediction Probability": [score.numpy()[0], score.numpy()[1]]
                }
                prob_df = {
                    "Class": CLASS_NAMES,
                    "Probability": [score.numpy()[0], score.numpy()[1]]
                }
                
                st.bar_chart(prob_df, x="Class", y="Probability", color="#007bff") # Use Streamlit's built-in charting

    # Additional help if the model failed to load
    elif model is None and file is not None:
         st.markdown("---")
         st.error("🚨 **Deployment Error Alert**")
         st.markdown("Please re-read the error message above. The most common fix is **ensuring the `.keras` files are correctly uploaded to GitHub** (especially if they are large, which requires **Git LFS**).")
