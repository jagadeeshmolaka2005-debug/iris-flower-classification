import streamlit as st
import pickle
import numpy as np
from PIL import Image
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)
model = pickle.load(open("random_forest_model.pkl","rb"))

scaler = pickle.load(open("scaler.pkl","rb"))

encoder = pickle.load(open("label_encoder.pkl","rb"))

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

/* Main Background */
.stApp{
    background: linear-gradient(to right,#EEF7FF,#F8FCFF);
}

.equal-card{
    background:white;
    padding:20px;
    border-radius:15px;
    border:1px solid #d9e3f0;
    box-shadow:0px 4px 10px rgba(0,0,0,0.10);
    min-height:700px;
}

/* Hide Streamlit Menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main Title */
.main-title{
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#0F4C81;
    margin-bottom:5px;
}

/* Subtitle */
.sub-title{
    text-align:center;
    font-size:20px;
    color:#5A5A5A;
    margin-bottom:30px;
}

/* Card */
.card{
    background:white;
    border-radius:18px;
    padding:25px;
    box-shadow:0px 8px 20px rgba(0,0,0,0.10);
    border:1px solid #E6EEF8;
}

/* Result Card */
.result-card{
    background:linear-gradient(135deg,#4CAF50,#81C784);
    color:white;
    border-radius:18px;
    padding:25px;
    text-align:center;
    box-shadow:0px 8px 20px rgba(0,0,0,.20);
}

/* Footer */
.footer{
    text-align:center;
    color:gray;
    font-size:16px;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
🌸 Iris Flower Classification
</div>

<div class="sub-title">
Predict Iris Flower Species using Random Forest Machine Learning
</div>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:

    st.title("🌸 Project")

    st.markdown("---")

    st.success("Random Forest Classifier")

    st.info("Dataset : Iris Dataset")

    st.write("### Features")

    st.write("• Sepal Length")

    st.write("• Sepal Width")

    st.write("• Petal Length")

    st.write("• Petal Width")

    st.markdown("---")

    st.write("👨‍💻 Developer")

    st.write("Jagadeesh")

    st.write("AI & DS")

# ======================================================
# MAIN LAYOUT
# ======================================================

left_column, right_column = st.columns([1, 1], gap="large")
# ======================================================
# FLOWER PREVIEW
# ======================================================

with left_column:

    with st.container(border=True):

        st.subheader("🌺 Flower Preview")

        # List of flower images
        flower_images = [
            ("Iris Setosa", "images/Iris Setosa.jpg"),
            ("Iris Versicolor", "images/Iris Versicolor.jpg"),
            ("Iris Virginica", "images/Iris Virginica.jpg")
        ]

        # Store current image index
        if "image_index" not in st.session_state:
            st.session_state.image_index = 0

        # Previous and Next buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("⬅ Previous"):
                st.session_state.image_index = (
                    st.session_state.image_index - 1
                ) % len(flower_images)

        with col2:
            if st.button("Next ➡"):
                st.session_state.image_index = (
                    st.session_state.image_index + 1
                ) % len(flower_images)

        # Display current image
        flower_name, image_path = flower_images[st.session_state.image_index]

        st.image(image_path, use_container_width=True)

        st.markdown(
            f"<h3 style='text-align:center;color:#0A58CA;'>{flower_name}</h3>",
            unsafe_allow_html=True
        )

        st.info(
            "Use the buttons above to preview all three Iris flower species."
        )

# ======================================================
# INPUT SECTION
# ======================================================

with right_column:

    with st.container(border=True):

        st.subheader("📝 Enter Flower Measurements")

        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=4.0,
            max_value=8.0,
            value=5.1,
            step=0.1
        )

        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=2.0,
            max_value=5.0,
            value=3.5,
            step=0.1
        )

        petal_length = st.slider(
            "Petal Length (cm)",
            min_value=1.0,
            max_value=7.0,
            value=1.4,
            step=0.1
        )

        petal_width = st.slider(
            "Petal Width (cm)",
            min_value=0.1,
            max_value=3.0,
            value=0.2,
            step=0.1
        )

        st.markdown("")

        predict = st.button(
            "🔍 Predict Species",
            use_container_width=True
        )
st.divider()

# ======================================================
# PREDICTION
# ======================================================

if predict:

    # Create Input Array
    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    # Scale Input
    input_scaled = scaler.transform(input_data)

    # Predict Species
    prediction = model.predict(input_scaled)

    # Prediction Probability
    probability = model.predict_proba(input_scaled)

    # Decode Label
    flower = encoder.inverse_transform(prediction)[0]

    # Highest Confidence
    confidence = np.max(probability) * 100
    # ======================================================
    # SELECT FLOWER IMAGE
    # ======================================================

    if flower == "Iris-setosa":
        flower_image = Image.open("images/Iris Setosa.jpg")

    elif flower == "Iris-versicolor":
        flower_image = Image.open("images/Iris Versicolor.jpg")

    else:
        flower_image = Image.open("images/Iris Virginica.jpg")
    # ======================================================
    # RESULT CARD
    # ======================================================
    st.success("Prediction Completed Successfully ✅")
    st.markdown("## 🌸 Prediction Result")
    result_left, result_right = st.columns([1,2], gap="large")
    with result_left:
        with st.container(border=True):
            st.image(
                flower_image,
                use_container_width=True
            )
    with result_right:
        with st.container(border=True):
            st.markdown(f"## 🌼 {flower}")
            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )
            st.success("Model Prediction Successful")
            st.markdown("### 📋 Entered Values")
            st.write(f"📏 Sepal Length : **{sepal_length} cm**")
            st.write(f"📏 Sepal Width : **{sepal_width} cm**")
            st.write(f"🌸 Petal Length : **{petal_length} cm**")
            st.write(f"🌸 Petal Width : **{petal_width} cm**")
    # ======================================================
    # PREDICTION CONFIDENCE
    # ======================================================
    st.markdown("---")
    st.subheader("📊 Prediction Confidence")
    classes = encoder.classes_
    for name, prob in zip(classes, probability[0]):
        col1, col2 = st.columns([2,5])
        with col1:
            st.write(f"**{name}**")
        with col2:
            st.progress(float(prob))
        st.write(f"Confidence : **{prob*100:.2f}%**")
        st.markdown("")
    # ======================================================
    # PROBABILITY TABLE
    # ======================================================
    st.markdown("---")
    st.subheader("📋 Prediction Summary")
    import pandas as pd
    result_df = pd.DataFrame({
    "Flower Species": encoder.classes_,
    "Probability (%)": np.round(probability[0] * 100, 2)
})
    st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True
)
    st.balloons()
# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.markdown(
"""
<div style="text-align:center;
padding:20px;
background:#F8F9FA;
border-radius:12px;
border:1px solid #DDDDDD;">

<h3>🌸 Iris Flower Classification</h3>

<p>
Developed by <b>Jagadeesh</b><br>
Random Forest Classifier • Streamlit • Python
</p>

</div>
""",
unsafe_allow_html=True
)
