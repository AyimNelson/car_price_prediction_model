import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load('car_price_model.pkl')

# Page config
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Theme-aware CSS with header styling
st.markdown("""
<style>
    /* Header card styling */
    .header-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 1.5rem;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .header-card h1 {
        color: white !important;
        margin: 0 0 0.5rem 0 !important;
        font-size: 2.5rem !important;
    }
    .header-card p {
        color: rgba(255,255,255,0.9) !important;
        margin: 0 !important;
        font-size: 1.1rem !important;
    }
    /* Card for form steps */
    .card {
        background-color: var(--secondary-background-color);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .step-active {
        background-color: #4CAF50;
        color: white;
        border-radius: 2rem;
        padding: 0.3rem 0.8rem;
        text-align: center;
        font-weight: bold;
        display: inline-block;
        width: 2rem;
        margin: 0 auto;
    }
    .step-inactive {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid var(--border-color);
        border-radius: 2rem;
        padding: 0.3rem 0.8rem;
        text-align: center;
        font-weight: bold;
        display: inline-block;
        width: 2rem;
        margin: 0 auto;
    }
    /* Full-width price result */
    .price-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        background-color: rgba(76, 175, 80, 0.1);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #4CAF50;
        width: 100%;
    }
    /* Disclaimer styling */
    .disclaimer {
        font-size: 0.8rem;
        text-align: center;
        color: var(--text-color);
        opacity: 0.7;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid var(--border-color);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'show_result' not in st.session_state:
    st.session_state.show_result = False
if 'predicted_price' not in st.session_state:
    st.session_state.predicted_price = None

# Sidebar
with st.sidebar:
    st.markdown("## 🚗 Car Price Predictor")
    st.markdown("---")
    st.markdown("### 📋 Steps")
    cols = st.columns(3)
    for i in range(3):
        with cols[i]:
            if st.session_state.page == i+1:
                st.markdown(f'<div class="step-active">{i+1}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="step-inactive">{i+1}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        "This model predicts car prices based on specifications like year, mileage, "
        "brand, and features. Trained on real-world data with **Random Forest**.\n\n"
        "📊 **Model Performance:**\n"
        "- R² Train Score: 0.9330\n"
        "- R² Test Score: 0.7664\n"
        "- Average error: ~$4,109"
    )
    st.markdown("---")
    st.markdown("### 📌 Current Selections")
    selected = {
        'Year': st.session_state.get('pyear', 2015),
        'Mileage': f"{st.session_state.get('mileage', 50000):,} km",
        'Engine': f"{st.session_state.get('engine_volume', 2.0)} L",
        'Manufacturer': st.session_state.get('manufacturer', 'TOYOTA'),
        'Fuel': st.session_state.get('fuel_type', 'Petrol')
    }
    for k, v in selected.items():
        st.markdown(f"**{k}:** {v}")
    st.caption("Change values using the main panel.")

# Main area - Styled header
st.markdown("""
<div class="header-card">
    <h1>🚗 Car Price Prediction</h1>
    <p>Fill in the details below. Navigate using the buttons.</p>
</div>
""", unsafe_allow_html=True)

def get_form_data():
    return {
        'production_year': st.session_state.get('pyear', 2015),
        'levy': st.session_state.get('levy', 100),
        'mileage': st.session_state.get('mileage', 50000),
        'cylinders': st.session_state.get('cylinders', 4),
        'airbags': st.session_state.get('airbags', 4),
        'doors': st.session_state.get('doors', 4),
        'engine_volume': st.session_state.get('engine_volume', 2.0),
        'manufacturer': st.session_state.get('manufacturer', 'TOYOTA'),
        'model': st.session_state.get('model', 'Camry'),
        'fuel_type': st.session_state.get('fuel_type', 'Petrol'),
        'category': st.session_state.get('category', 'Sedan'),
        'leather_interior': st.session_state.get('leather_interior', 'Yes'),
        'gear_box_type': st.session_state.get('gear_box_type', 'Automatic'),
        'drive_wheels': st.session_state.get('drive_wheels', 'Front'),
        'wheel': st.session_state.get('wheel', 'Left wheel'),
        'color': st.session_state.get('color', 'White')
    }

def compute_price():
    form_data = get_form_data()
    input_df = pd.DataFrame([form_data])
    
    current_year = 2026
    input_df['car_age'] = current_year - input_df['production_year']
    input_df['age_group'] = pd.cut(input_df['car_age'],
                                   bins=[0,5,10,15,100],
                                   labels=['New','Recent','Mid-age','Old'])
    input_df['mileage_group'] = pd.cut(input_df['mileage'],
                                       bins=[0,50000,100000,150000,1_000_000],
                                       labels=['Low','Medium','High','Very High'])
    input_df['engine_per_cylinder'] = input_df['engine_volume'] / input_df['cylinders']
    input_df['production_year_squared'] = input_df['production_year'] ** 2
    
    feature_cols = [
        'production_year','levy','mileage','cylinders','airbags','doors',
        'manufacturer','model','fuel_type','category','leather_interior',
        'gear_box_type','drive_wheels','wheel','color','engine_volume',
        'car_age','age_group','mileage_group','engine_per_cylinder','production_year_squared'
    ]
    input_df = input_df[feature_cols]
    
    price = model.predict(input_df)[0]
    return price

def page1():
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📅 Step 1: Basic Specifications")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Production Year", min_value=1990, max_value=2025, value=2015, key='pyear')
            st.number_input("Levy (tax/insurance)", min_value=0, value=100, key='levy', step=50)
            st.number_input("Mileage (km)", min_value=0, value=50000, key='mileage', step=5000)
        with col2:
            st.number_input("Cylinders", min_value=2, max_value=12, value=4, key='cylinders')
            st.number_input("Airbags", min_value=0, max_value=16, value=4, key='airbags')
            st.selectbox("Doors", [2,3,4,5], index=2, key='doors')
        st.number_input("Engine volume (L)", min_value=0.5, max_value=8.0, step=0.1, value=2.0, key='engine_volume')
        st.markdown('</div>', unsafe_allow_html=True)

def page2():
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🏷️ Step 2: Model & Interior")
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Manufacturer", ["TOYOTA","HONDA","FORD","CHEVROLET","HYUNDAI","BMW","MERCEDES-BENZ","LEXUS","NISSAN","KIA"], index=0, key='manufacturer')
            st.text_input("Model", value="Camry", key='model')
            st.selectbox("Fuel type", ["Petrol","Diesel","Hybrid","LPG","CNG"], index=0, key='fuel_type')
        with col2:
            st.selectbox("Category", ["Sedan","Jeep","Hatchback","Minivan","Coupe","Universal"], index=0, key='category')
            st.selectbox("Leather interior", ["Yes","No"], index=0, key='leather_interior')
        st.markdown('</div>', unsafe_allow_html=True)

def page3():
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("⚙️ Step 3: Transmission, Drive & Color")
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Gear box type", ["Automatic","Manual","Tiptronic","Variator"], index=0, key='gear_box_type')
            st.selectbox("Drive wheels", ["Front","Rear","4x4"], index=0, key='drive_wheels')
        with col2:
            st.selectbox("Wheel", ["Left wheel","Right-hand drive"], index=0, key='wheel')
            st.text_input("Color", value="White", key='color')
        st.markdown('</div>', unsafe_allow_html=True)

def render_page():
    if st.session_state.page == 1:
        page1()
    elif st.session_state.page == 2:
        page2()
    elif st.session_state.page == 3:
        page3()
    
    if st.session_state.page == 1:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("Next ▶", type="primary", use_container_width=True):
                st.session_state.page += 1
                st.session_state.show_result = False
                st.rerun()
    elif st.session_state.page == 2:
        col_left, col_right = st.columns(2)
        with col_left:
            if st.button("◀ Previous", use_container_width=True):
                st.session_state.page -= 1
                st.session_state.show_result = False
                st.rerun()
        with col_right:
            if st.button("Next ▶", type="primary", use_container_width=True):
                st.session_state.page += 1
                st.session_state.show_result = False
                st.rerun()
    else:
        col_left, col_right = st.columns(2)
        with col_left:
            if st.button("◀ Previous", use_container_width=True):
                st.session_state.page -= 1
                st.session_state.show_result = False
                st.rerun()
        with col_right:
            if st.button("💰 Predict Price", type="primary", use_container_width=True):
                st.session_state.predicted_price = compute_price()
                st.session_state.show_result = True
                st.rerun()

# Render the form and buttons
render_page()

# Display result in its own full-width row (like disclaimer)
if st.session_state.show_result and st.session_state.predicted_price is not None:
    st.markdown(f'<div class="price-result">✨ Estimated Price: ${st.session_state.predicted_price:,.2f}</div>', unsafe_allow_html=True)
    st.balloons()

# AI Disclaimer (full-width row)
st.markdown("""
<div class="disclaimer">
🤖 <strong>AI Disclaimer:</strong> This prediction is generated by a machine learning model based on historical data. 
Actual car prices may vary due to market conditions, location, vehicle condition, and other factors not captured by the model. 
Use this estimate as a reference only, not as a definitive valuation.
</div>
""", unsafe_allow_html=True)