import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="GreenFix Community",
    page_icon="🌱",
    layout="wide"
)

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "waste_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "waste_classifier.keras")

# Load CSV dataset
@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()

    return pd.read_csv(CSV_PATH)

data = load_data()

# Load trained ML model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Categories must match the training class order.
# We will verify the exact order from your training script if needed.
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")

with open(CLASS_NAMES_PATH, "r") as file:
    CLASS_NAMES = json.load(file)

# Application heading
st.title("🌱 GreenFix Community")

st.subheader(
    "AI-Powered Repair, Reuse, Recycling and Green-Skill Platform"
)

st.write(
    "Don't throw it away! Discover ways to repair, reuse "
    "or responsibly recycle your electronic waste."
)

st.divider()

# AI image identification
st.header("📷 AI E-Waste Image Detection")

st.write(
    "Upload an electronic waste image to identify the item "
    "using your trained machine learning model."
)

if model is None:
    st.warning(
        "Model file not found. Please check that "
        "waste_classifier.keras is in your project folder."
    )

uploaded_file = st.file_uploader(
    "Choose an electronic waste image",
    type=["jpg", "jpeg", "png"]
)

predicted_item = None

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Uploaded E-Waste Image",
            width=300
        )

        if st.button("Identify Item", type="primary"):
            if model is None:
                st.error("The trained model could not be loaded.")
            else:
                with st.spinner("Analysing image..."):
                    resized_image = image.resize((224, 224))

                    image_array = np.array(
                        resized_image,
                        dtype=np.float32
                    )

                    image_array = np.expand_dims(
                        image_array,
                        axis=0
                    )

                    # MobileNetV2 preprocessing
                   # image_array = (
                       # tf.keras.applications.mobilenet_v2
                       # .preprocess_input(image_array)
                   # )

                    predictions = model.predict(
                        image_array,
                        verbose=0
                    )

                    predicted_index = int(
                        np.argmax(predictions[0])
                    )

                    confidence = float(
                        predictions[0][predicted_index]
                    ) * 100

                    if predicted_index >= len(CLASS_NAMES):
                        st.error(
                            "Model classes do not match CLASS_NAMES."
                        )
                    else:
                        predicted_item = CLASS_NAMES[predicted_index]

                        st.success(
                            f"Predicted item: {predicted_item}"
                        )

                        st.write(
                            f"Model confidence: {confidence:.2f}%"
                        )

                        if confidence < 60:
                            st.warning(
                                "Prediction confidence is low. "
                                "Try a clearer image with the item "
                                "fully visible."
                            )

                        st.session_state["predicted_item"] = (
                            predicted_item
                        )

    except Exception as error:
        st.error(f"Image prediction failed: {error}")

# Electronic Waste Advisor
st.divider()
st.header("🔍 Electronic Waste Advisor")

# Select item from CSV
if not data.empty and "item" in data.columns:
    item_list = data["item"].dropna().astype(str).tolist()

    default_item = st.session_state.get("predicted_item")

    matching_item = next(
        (
            item for item in item_list
            if default_item
            and item.strip().lower() == default_item.strip().lower()
        ),
        None
    )

    if matching_item:
        default_index = item_list.index(matching_item)
    else:
        default_index = 0

    selected_item = st.selectbox(
        "Select your electronic waste item:",
        item_list,
        index=default_index
    )

    item_rows = data[
        data["item"].astype(str).str.lower()
        == selected_item.lower()
    ]

    if not item_rows.empty:
        item_data = item_rows.iloc[0]

        st.info(f"Selected Item: {selected_item}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔧 Repair Recommendation")
            st.write(
                item_data.get(
                    "repair_suggestion",
                    "No repair recommendation available."
                )
            )

            st.subheader("♻️ Reuse Ideas")
            st.write(
                item_data.get(
                    "reuse_idea",
                    "No reuse idea available."
                )
            )

        with col2:
            st.subheader("💎 Potentially Recoverable Materials")
            st.write(
                item_data.get(
                    "materials",
                    "No material information available."
                )
            )

            st.subheader("🏭 Recommended Business Type")
            st.write(
                item_data.get(
                    "recycling_business",
                    "No recycling business information available."
                )
            )
    else:
        st.warning("No recommendations found for this item.")

else:
    selected_item = None
    st.warning(
        "waste_data.csv is missing or does not contain an 'item' column."
    )

# Problem description
st.divider()
st.header("📝 Describe Your Item")

problem = st.text_area(
    "What is the problem with your item?",
    placeholder="Example: My mobile phone is not turning on..."
)

if st.button("Get Recommendation", type="primary"):
    if not problem.strip():
        st.warning("Please describe the problem first.")

    elif selected_item is None:
        st.warning(
            "Please check your waste_data.csv file first."
        )

    else:
        st.success("Your item details have been recorded for review.")

        st.write("### Your Item")
        st.write(selected_item)

        st.write("### Your Problem")
        st.write(problem)

        if not data.empty:
            matching_rows = data[
                data["item"].astype(str).str.lower()
                == selected_item.lower()
            ]

            if not matching_rows.empty:
                repair = matching_rows.iloc[0].get(
                    "repair_suggestion",
                    "Consult a qualified repair technician."
                )

                st.write("### Suggested Next Step")
                st.write(repair)

        st.caption(
            "These are general recommendations, not a confirmed "
            "diagnosis. Get professional help for electrical repairs."
        )

# Green Skills section
st.divider()
st.header("🌿 Learn Green Skills")

st.write(
    "Learn basic repair, reuse, waste segregation and "
    "responsible e-waste management."
)

if st.button("Explore Green Skills"):
    st.write("1. Basic Electronics Repair")
    st.write("2. E-Waste Segregation")
    st.write("3. Component Reuse")
    st.write("4. Circular Economy")
    st.write("5. Safe Recycling Practices")

# Footer
st.divider()

st.caption(
    "GreenFix Community | Repair • Reuse • Recycle • Learn"
)