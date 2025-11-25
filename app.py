import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Autism Detection AI",
    page_icon="PX",
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
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
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
    # Important: ResNet and EfficientNet expect different input scalings.
    if "ResNet50" in model_name:
        # ResNet expects inputs to be zero-centered (caffe style)
        preprocessed_image = tf.keras.applications.resnet.preprocess_input(img_reshape)
    else:
        # EfficientNet expects inputs in range [0, 255] or [0, 1] depending on version, 
        # but keras.applications handles it usually. 
        # In your code you used efficientnet.preprocess_input
        preprocessed_image = tf.keras.applications.efficientnet.preprocess_input(img_reshape)
        
    prediction = model.predict(preprocessed_image)
    return prediction

# --- UI LAYOUT ---
st.title("🧩 Autism Spectrum Detection")
st.write("Upload a facial image to detect potential signs of autism using Deep Learning.")

# Sidebar for model selection
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox(
    "Choose Model", 
    ["ResNet50 (Fine-Tuned)", "EfficientNetB0"]
)

# File Uploader
file = st.file_uploader("Please upload a facial image (jpg, png)", type=["jpg", "png", "jpeg"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    # Load Model
    with st.spinner(f"Loading {model_choice} model..."):
        model = load_model(model_choice)

    if model is not None:
        # Predict Button
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                predictions = import_and_predict(image, model, model_choice)
                
                # Interpret Results
                score = tf.nn.softmax(predictions[0]) # Get probabilities
                predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
                confidence = 100 * np.max(score)

                # Display Results
                st.write("---")
                st.subheader("Prediction Results")
                
                if predicted_class == 'Autistic':
                    st.error(f"Prediction: **{predicted_class}**")
                else:
                    st.success(f"Prediction: **{predicted_class}**")
                
                st.write(f"Confidence: **{confidence:.2f}%**")
                
                # Detailed probability bar chart
                st.write("Probability Distribution:")
                st.bar_chart({
                    "Autistic": predictions[0][0],
                    "Non_Autistic": predictions[0][1]
                })

    else:
        st.warning(f"⚠ Model file not found. Please ensure '{model_choice}' file is in the repository.")
