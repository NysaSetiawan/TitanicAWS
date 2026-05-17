import streamlit as st
import boto3
import json

REGION = "us-east-1"
ENDPOINT_NAME = "titanic-endpoint-v18"

st.title("Space Titanic Passenger Survival Predictor")
st.write("Form input ini murni menggunakan fitur asli dari dataset Anda.")


with st.form("passenger_form"):
    st.subheader("Informasi Profil Penumpang")
    passenger_id = st.text_input("Passenger ID", value="0001_01")
    home_planet = st.selectbox("Home Planet", ["Earth", "Europa", "Mars"])
    cryo_sleep = st.selectbox("Apakah CryoSleep?", [1.0, 0.0], format_func=lambda x: "Yes" if x == 1.0 else "No")
    
    cabin = st.text_input("Cabin (Format: Deck/Num/Side)", value="G/150/S")
    
    destination = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
    age = st.number_input("Age (Usia)", min_value=0, max_value=100, value=10)
    vip = st.selectbox("Apakah Penumpang VIP?", [1.0, 0.0], format_func=lambda x: "Yes" if x == 1.0 else "No")
    name = st.text_input("Full Name", value="Nysa Setiawan")
    
    st.subheader("Pengeluaran Fasilitas Kapal ($)")
    room_service = st.number_input("Room Service", value=0.0)
    food_court = st.number_input("Food Court", value=0.0)
    shopping_mall = st.number_input("Shopping Mall", value=0.0)
    spa = st.number_input("Spa", value=0.0)
    vr_deck = st.number_input("VR Deck", value=0.0)
    
    submit_button = st.form_submit_button("Mulai Prediksi Keselamatan")

if submit_button:

    try:
        deck, cabin_num, side = cabin.split("/")
        cabin_num = float(cabin_num)
    except:
        deck, cabin_num, side = "G", 150.0, "S"
        
    if age < 13:
        age_group = "Child"
    elif age < 20:
        age_group = "Teenager"
    elif age < 60:
        age_group = "Adult"
    else:
        age_group = "Elderly"
        
    try:
        group_id = passenger_id.split("_")[0]
        group_size = 1.0  
        solo = 1.0
        family_size = 1.0
    except:
        group_size, solo, family_size = 1.0, 1.0, 1.0

    total_spending = room_service + food_court + shopping_mall + spa + vr_deck
    has_spending = 1.0 if total_spending > 0 else 0.0
    no_spending = 1.0 if total_spending == 0 else 0.0
    
    age_missing = 0.0
    cryo_sleep_missing = 0.0


    instance_data = [
        home_planet,       # HomePlanet
        cryo_sleep,        # CryoSleep
        destination,       # Destination
        vip,               # VIP
        deck,              # Deck (Hasil Feature Engineering)
        side,              # Side (Hasil Feature Engineering)
        age_group,         # Age_group (Hasil Feature Engineering)
        
        float(age),        # Age
        float(room_service),# RoomService
        float(food_court),  # FoodCourt
        float(shopping_mall),# ShoppingMall
        float(spa),        # Spa
        float(vr_deck),    # VRDeck
        float(cabin_num),  # Cabin_num (Hasil Feature Engineering)
        float(group_size), # Group_size (Hasil Feature Engineering)
        float(solo),       # Solo (Hasil Feature Engineering)
        float(family_size),# Family_size (Hasil Feature Engineering)
        float(total_spending), # TotalSpending (Hasil Feature Engineering)
        float(has_spending),# HasSpending (Hasil Feature Engineering)
        float(no_spending),# NoSpending (Hasil Feature Engineering)
        float(age_missing),# Age_missing
        float(cryo_sleep_missing) # CryoSleep_missing
    ]
    
    payload = {"instances": [instance_data]}
    
    try:
        runtime = boto3.client("sagemaker-runtime", region_name=REGION)
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        
        result = json.loads(response["Body"].read().decode("utf-8"))
        
        if result["status"] == "SUCCESS":
            prediction = result["predictions"][0]
            st.subheader("Hasil Analisis Model ML:")
            if prediction == 1:
                st.success("Transported")
            else:
                st.error("Not Transported")
        else:
            st.error(f"Gagal memproses prediksi: {result}")
            
    except Exception as e:
        st.error(f"Eror komunikasi dengan SageMaker: {str(e)}")