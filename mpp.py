import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Mobile Price Prediction",
    page_icon="📱",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
h1, h2, h3, h4 {
    font-weight: 600;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    padding: 0.6rem;
    font-size: 16px;
}
a {
    text-decoration: none;
    color: #6a0dad;
    font-weight: bold;
}
a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
with open("mpp.pkl", "rb") as f:
    model = pickle.load(f)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(r"E:\mobile_price_predictioin\cell.csv")

df = load_data()

# --------------------------------------------------
# Price Category Function
# --------------------------------------------------
def price_category(price):
    if price < 15000:
        return "💸 Budget"
    elif price < 30000:
        return "⚡ Mid-Range"
    else:
        return "👑 Premium"

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div style="background-color:#6a0dad;padding:15px;border-radius:12px">
<h2 style="color:white;text-align:center;">
📱 Mobile Price Prediction Web App
</h2>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tabs = st.tabs([
    "📱 Predict Price",
    "📊 Project Overview",
    "📈 EDA",
    "🧠 Model",
    "💡 Feature Insights",
    "⚠️ Disclaimer",
    "👤 About"
])

# ==================================================
# TAB 1 — PREDICTION
# ==================================================
with tabs[0]:
    st.subheader("Predict Mobile Price")
    st.caption("Enter mobile specifications")

    col1, col2 = st.columns(2)

    with col1:
        ram_gb = st.number_input("RAM (GB)", 1, 16, 4)
        battery_mah = st.number_input("Battery (mAh)", 1000, 7000, 4500, step=100)
        screen_size_inch = st.number_input("Screen Size (inch)", 4.0, 7.5, 6.5, step=0.1)
        front_camera_mp = st.number_input("Front Camera (MP)", 0, 64, 16)

    with col2:
        storage_gb = st.number_input("Storage (GB)", 8, 512, 64, step=8)
        primary_camera_mp = st.number_input("Rear Camera (MP)", 5, 200, 48)
        weight_g = st.number_input("Weight (grams)", 100, 300, 180)
        fiveg = st.selectbox("5G Supported", ["No", "Yes"])
        fiveg = 1 if fiveg == "Yes" else 0

    if st.button("🚀 Predict Mobile Price"):
        input_df = pd.DataFrame(
            [[ram_gb, storage_gb, battery_mah,
              primary_camera_mp, front_camera_mp,
              screen_size_inch, weight_g, fiveg]],
            columns=[
                'ram_gb', 'storage_gb', 'battery_mah', 'primary_camera_mp',
                'front_camera_mp', 'screen_size_inch', 'weight_g', 'fiveg'
            ]
        )

        prediction = model.predict(input_df)[0]
        category = price_category(prediction)

        st.success(f"💰 Estimated Price: ₹ {prediction:,.0f}")
        st.info(f"📊 Price Segment: {category}")

        report = f"""
Mobile Price Prediction Report
------------------------------
Estimated Price : ₹ {prediction:,.0f}
Category        : {category}

Specifications
---------------
RAM             : {ram_gb} GB
Storage         : {storage_gb} GB
Battery         : {battery_mah} mAh
Rear Camera     : {primary_camera_mp} MP
Front Camera    : {front_camera_mp} MP
Screen Size     : {screen_size_inch} inch
Weight          : {weight_g} g
5G Support      : {'Yes' if fiveg == 1 else 'No'}
"""

        st.download_button(
            label="📥 Download Prediction Report",
            data=report,
            file_name="mobile_price_report.txt"
        )

# ==================================================
# TAB 2 — PROJECT OVERVIEW
# ==================================================
with tabs[1]:
    st.subheader("Project Overview")
    st.markdown("""
This web application estimates **mobile phone prices** based on hardware specifications using a **Linear Regression** model.

**Objectives:**
- Apply regression to real-world product pricing
- Maintain model interpretability
- Deploy as a professional SaaS-style ML web app
""")

# ==================================================
# TAB 3 — EDA
# ==================================================
with tabs[2]:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Price (₹)", f"{int(df['price_inr'].mean()):,}")
        st.metric("Maximum Price (₹)", f"{int(df['price_inr'].max()):,}")
    with col2:
        st.metric("Average RAM (GB)", round(df['ram_gb'].mean(), 1))
        st.metric("Average Battery (mAh)", int(df['battery_mah'].mean()))

    st.markdown("### Price Distribution")
    st.bar_chart(df["price_inr"])

    st.markdown("### RAM vs Price")
    st.line_chart(df.groupby("ram_gb")["price_inr"].mean())

    st.markdown("### Battery vs Price")
    st.line_chart(df.groupby("battery_mah")["price_inr"].mean())

# ==================================================
# TAB 4 — MODEL
# ==================================================
with tabs[3]:
    st.subheader("Model & Performance")
    st.markdown("""
**Model Used:** Linear Regression  

**Performance:**
- Train R² ≈ 0.79
- Test R² ≈ 0.80

Close scores indicate good generalization without overfitting.
""")

# ==================================================
# TAB 5 — FEATURE INSIGHTS
# ==================================================
with tabs[4]:
    st.subheader("Feature Importance (Coefficients)")

    coef_df = pd.DataFrame({
        "Feature": [
            'ram_gb', 'storage_gb', 'battery_mah',
            'primary_camera_mp', 'front_camera_mp',
            'screen_size_inch', 'weight_g', 'fiveg'
        ],
        "Impact": model.coef_
    }).sort_values(by="Impact", ascending=False)

    st.dataframe(coef_df, use_container_width=True)

    st.info(
        "Positive coefficients increase the price, "
        "while negative values reduce it. "
        "Linear Regression allows direct interpretation."
    )

# ==================================================
# TAB 6 — DISCLAIMER
# ==================================================
with tabs[5]:
    st.subheader("Disclaimer")
    st.markdown("""
⚠️ This application provides **estimated prices** for educational and professional demonstration purposes.

Actual market prices vary based on brand, offers, availability, and regional pricing.
""")

with tabs[6]:
    st.markdown("""
    </div>
    <div style= padding:20px; background-color:black; border-radius:12px;'>
        <h4 style='color:white;'>About</h4>
        <p style='font-size:16px; max-width:600px; margin:auto;'>
            Hi! I'm <b>Kamalesh</b>, a Data Scientist & ML professional.<br>
            Hosted as a professional SaaS-style Mobile Price Prediction Machine Learning web application.
        </p>
        </p>
        <p style='font-size:16px; max-width:600px; margin:auto;'>
                <div style="background-color:#111; padding:20px; border-radius:12px;">
        <h4 style='color:white;'>Technology Used</h4>        
        <ul style="font-size:16px; line-height:2;">
            <li>🐍 <b>Python</b> — Core programming language</li>
            <li>📦 <b>Pandas</b> — Data processing & analysis</li>
            <li>🧠 <b>Scikit-learn</b> — Machine Learning model</li>
            <li>🖥️ <b>Streamlit</b> — Web application framework</li>
            <li>🎨 <b>HTML / CSS</b> — UI customization</li>
        </ul>
        <p style="margin-top:15px; font-size:14px; color:#aaa;">
           
    </div>

    <div style= padding:20px; border-radius:12px;'>
        <h4 style='margin-top:20px; font-size:16px;'>
            Connect with me: 
            🔗<a href='https://www.linkedin.com/in/kamalesh-v-a1504a33a/' target='_blank'>LinkedIn</a> 
            🐙<a href='https://github.com/kamaleshcr7' target='_blank'>GitHub</a> 
            ✉️<a href='mailto:kamalesh7cr7@gmail.com'>Email</a> 
            🌐<a href='https://https://portfolio-zeta-livid-73.vercel.app/' target='_blank'>Portfolio</a>
        </h4>
    </div>""",unsafe_allow_html=True)
# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:13px;'>📱 Mobile Price Prediction ML Web Application by @kamalesh</p>",
    unsafe_allow_html=True
)
