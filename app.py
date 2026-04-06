import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('car_price_model.pkl')

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Prediction App")
st.markdown("Fill in the car details across multiple steps. Use **Next** and **Previous** to navigate.")

# --- Initialize session state for page only ---
if 'page' not in st.session_state:
    st.session_state.page = 1

# --- Helper to get form data from all widgets (keys) ---
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

# --- Page 1: Basic Specs (single column) ---
def page1():
    st.subheader("📅 Page 1: Basic Specifications")
    st.number_input("Production Year", min_value=1990, max_value=2025, value=2015, key='pyear')
    st.number_input("Levy (tax/insurance)", min_value=0, value=100, key='levy')
    st.number_input("Mileage (km)", min_value=0, value=50000, key='mileage')
    st.number_input("Cylinders", min_value=2, max_value=12, value=4, key='cylinders')
    st.number_input("Airbags", min_value=0, max_value=16, value=4, key='airbags')
    st.selectbox("Doors", [2,3,4,5], index=2, key='doors')
    st.number_input("Engine volume (L)", min_value=0.5, step=0.1, value=2.0, key='engine_volume')

# --- Page 2: Categorical Details (single column) ---
def page2():
    st.subheader("🏷️ Page 2: Model & Interior")
    st.selectbox("Manufacturer", ["TOYOTA","HONDA","FORD","CHEVROLET","HYUNDAI","BMW","MERCEDES-BENZ","LEXUS","NISSAN","KIA"], index=0, key='manufacturer')
    st.text_input("Model", value="Camry", key='model')
    st.selectbox("Fuel type", ["Petrol","Diesel","Hybrid","LPG","CNG"], index=0, key='fuel_type')
    st.selectbox("Category", ["Sedan","Jeep","Hatchback","Minivan","Coupe","Universal"], index=0, key='category')
    st.selectbox("Leather interior", ["Yes","No"], index=0, key='leather_interior')

# --- Page 3: Drive & Color (single column) ---
def page3():
    st.subheader("⚙️ Page 3: Transmission, Drive & Color")
    st.selectbox("Gear box type", ["Automatic","Manual","Tiptronic","Variator"], index=0, key='gear_box_type')
    st.selectbox("Drive wheels", ["Front","Rear","4x4"], index=0, key='drive_wheels')
    st.selectbox("Wheel", ["Left wheel","Right-hand drive"], index=0, key='wheel')
    st.text_input("Color", value="White", key='color')

# --- Navigation and prediction ---
def render_page():
    if st.session_state.page == 1:
        page1()
    elif st.session_state.page == 2:
        page2()
    elif st.session_state.page == 3:
        page3()

    col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
    with col_btn1:
        if st.session_state.page > 1:
            if st.button("◀ Previous"):
                st.session_state.page -= 1
                st.rerun()
    with col_btn3:
        if st.session_state.page < 3:
            if st.button("Next ▶"):
                st.session_state.page += 1
                st.rerun()
        else:
            if st.button("💰 Predict Price", type="primary"):
                predict_price()

def predict_price():
    form_data = get_form_data()
    input_data = pd.DataFrame([form_data])

    # Feature engineering (exactly as in training)
    current_year = 2026
    input_data['car_age'] = current_year - input_data['production_year']
    input_data['age_group'] = pd.cut(input_data['car_age'],
                                     bins=[0,5,10,15,100],
                                     labels=['New','Recent','Mid-age','Old'])
    input_data['mileage_group'] = pd.cut(input_data['mileage'],
                                         bins=[0,50000,100000,150000,1_000_000],
                                         labels=['Low','Medium','High','Very High'])
    input_data['engine_per_cylinder'] = input_data['engine_volume'] / input_data['cylinders']
    input_data['production_year_squared'] = input_data['production_year'] ** 2

    feature_cols = [
        'production_year','levy','mileage','cylinders','airbags','doors',
        'manufacturer','model','fuel_type','category','leather_interior',
        'gear_box_type','drive_wheels','wheel','color','engine_volume',
        'car_age','age_group','mileage_group','engine_per_cylinder','production_year_squared'
    ]
    input_data = input_data[feature_cols]

    price = model.predict(input_data)[0]
    st.success(f"✨ Estimated Price: **${price:,.2f}**")
    st.balloons()

# --- Run the app ---
render_page()