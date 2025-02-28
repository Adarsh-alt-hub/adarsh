import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and dataset
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    df = pickle.load(open('df.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or dataset file not found. Please ensure 'pipe.pkl' and 'df.pkl' exist.")
    st.stop()

# Apply Custom Cyberpunk Styling
st.markdown(
    """
    <style>
        /* Cyberpunk Background */
        .stApp {
            background: url('https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExaDYwNTM0MGZpamF3c2J4dmc0ajg3cGZicm90YXdvb3JpYmdxam41NyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o6fJ1JHgh4Xw4hoK4/giphy.gif') no-repeat center center fixed;
            background-size: cover;
            color: white;
        }
        /* Gradient Overlay */
        .gradient {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.6);
            z-index: -1;
        }
        /* Title with Neon Glow */
        .title {
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            color: #00FFFF;
            text-shadow: 0 0 15px #00FFFF, 0 0 30px cyan, 0 0 45px cyan;
            border: 2px solid cyan;
            padding: 15px;
            border-radius: 20px;
            box-shadow: 0 0 40px cyan;
            margin-bottom: 20px;
        }
        /* Card Style */
        .card {
            background: rgba(0, 0, 0, 0.8);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0, 255, 255, 0.4);
            backdrop-filter: blur(12px);
            margin: 20px auto;
        }
        /* Button with Neon Effect */
        .neon-button {
            display: block;
            width: 100%;
            padding: 15px;
            font-size: 1.2rem;
            font-weight: bold;
            text-align: center;
            color: black;
            background: linear-gradient(90deg, #00FFFF, #FF00FF);
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 0 20px #0FF;
            text-decoration: none;
        }
        .neon-button:hover {
            transform: scale(1.1);
            box-shadow: 0 0 50px #FF00FF;
        }
        /* Result Display */
        .result {
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            color: #0FF;
            background: rgba(0, 0, 0, 0.7);
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 0 20px #0FF;
            margin-top: 20px;
        }
        /* Centered Element */
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
    <div class="gradient"></div>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown("<div class='title'>💻 AI-Powered Laptop Price Predictor</div>", unsafe_allow_html=True)

# Input Section
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### Select Laptop Features")

col1, col2 = st.columns(2)

# Left Column (Brand, Type, RAM, Storage, CPU)
with col1:
    st.subheader("Hardware Specifications")
    company = st.selectbox('Brand', df['Company'].unique())
    laptop_type = st.selectbox('Laptop Type', df['TypeName'].unique())
    ram = st.slider('RAM (GB)', min_value=2, max_value=64, step=2, value=8)
    hdd = st.selectbox('HDD (GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('SSD (GB)', [0, 8, 128, 256, 512, 1024])
    cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# GPU Field - Centered in the layout
st.markdown("<div class='centered'>", unsafe_allow_html=True)
gpu = st.selectbox('GPU', df['Gpu Brand'].unique())
st.markdown("</div>", unsafe_allow_html=True)

# Right Column (Screen, Display Features, OS, Weight)
with col2:
    st.subheader("Display & Other Features")
    screen_size = st.number_input('Screen Size (inches)', min_value=10.0, max_value=20.0, step=0.1, value=15.6)
    resolution = st.selectbox('Screen Resolution',
                              ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600',
                               '2560x1440', '2304x1440'])
    touchscreen = st.radio('Touchscreen', ['No', 'Yes'], horizontal=True)
    ips = st.radio('IPS Display', ['No', 'Yes'], horizontal=True)
    weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, step=0.1, value=1.5)
    os = st.selectbox('Operating System', df['os'].unique())

st.markdown("</div>", unsafe_allow_html=True)

# Predict Price Button
if st.button('💰 Predict Price', key="predict", help="Click to predict laptop price", use_container_width=True):
    try:
        touchscreen_val = 1 if touchscreen == 'Yes' else 0
        ips_val = 1 if ips == 'Yes' else 0
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

        query_df = pd.DataFrame(
            [[company, laptop_type, ram, weight, touchscreen_val, ips_val, ppi, cpu, hdd, ssd, gpu, os]],
            columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'IPS', 'PPI', 'Cpu brand', 'HDD', 'SSD',
                     'Gpu Brand', 'os']
        )

        # Convert only the relevant columns
        feature_types = df.drop(columns=['Price'], errors='ignore').dtypes.to_dict()
        query_df = query_df.astype(feature_types)

        # Perform prediction
        predicted_price = np.exp(pipe.predict(query_df))[0]

        st.markdown(f"""
            <div class="result">💰 Predicted Price: ₹{predicted_price:,.2f}</div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
