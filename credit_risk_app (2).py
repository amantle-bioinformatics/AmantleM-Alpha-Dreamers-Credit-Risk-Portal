
import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load('credit_risk_model.joblib')
scaler = joblib.load("scaler.joblib")

PROFESSION_MEANS = PROFESSION_MAPPING = {"Air_traffic_controller": 0.195338512763596, "Analyst": 0.18505338078291814, "Architect": 0.2098159509202454, "Army_officer": 0.21473158551810237, "Artist": 0.2134433962264151, "Aviator": 0.19312796208530805, "Biomedical_Engineer": 0.2041522491349481,
                                         "Chartered_Accountant": 0.213768115942029, "Chef": 0.20476190476190476, "Chemical_engineer": 0.18567961165048544, "Civil_engineer": 0.20116279069767443, "Civil_servant": 0.18931475029036005, "Comedian": 0.19672131147540983, "Computer_hardware_engineer": 0.18870056497175142,
                                         "Computer_operator": 0.18870056497175142, "Consultant": 0.18787878787878787, "Dentist": 0.18466898954703834, "Design_Engineer": 0.19952774498229045, "Designer": 0.19434628975265017, "Drafter": 0.19026047565118911, "Economist": 0.19626168224299065, "Engineer": 0.20358514724711907,
                                         "Fashion_Designer": 0.19410977242302543, "Financial_Analyst": 0.19572953736654805, "Firefighter": 0.20812807881773399, "Flight_attendant": 0.1951219512195122, "Geologist": 0.19544364508393286, "Graphic_Designer": 0.21173762945914845, "Hotel_Manager": 0.20317820658342792,
                                         "Industrial_Engineer": 0.176, "Lawyer": 0.18170878459687123, "Librarian": 0.2018140589569161, "Magistrate": 0.1872791519434629, "Mechanical_engineer": 0.1772885283893395, "Microbiologist": 0.19489559164733178, "Official": 0.19036427732079905, "Petroleum_Engineer": 0.17446270543615677,
                                         "Physician": 0.18314606741573033, "Police_officer": 0.21452894438138478, "Politician": 0.1891891891891892, "Psychologist": 0.22488038277511962,
                                         "Scientist": 0.18579881656804734, "Secretary": 0.18557919621749408, "Software_Developer": 0.2018348623853211, "Statistician": 0.17042889390519186, "Surgeon": 0.20076238881829733,
                                         "Surveyor": 0.21027479091995221, "Technical_writer": 0.19577960140679954, "Technician": 0.21885913853317812, "Technology_specialist": 0.19240196078431374,
                                         "Web_designer": 0.17149478563151796}

CITY_MAPPING = {"Adoni": 0.21019108280254778, "Agartala": 0.22666666666666666, "Agra": 0.1949685534591195, "Ahmedabad": 0.17886178861788618, "Ahmednagar": 0.16793893129770993, "Aizawl": 0.15037593984962405, "Ajmer": 0.19285714285714287, "Akola": 0.24087591240875914, "Alappuzha": 0.168, "Aligarh": 0.2230769230769231, "Allahabad": 0.24324324324324326,
                "Alwar": 0.16981132075471697, "Amaravati": 0.2465753424657534, "Ambala": 0.1736111111111111, "Ambarnath": 0.21323529411764705, "Ambattur": 0.17293233082706766, "Amravati": 0.21212121212121213, "Amritsar": 0.16417910447761194, "Amroha": 0.16083916083916083, "Anand": 0.17699115044247787, "Anantapur": 0.19402985074626866, "Anantapuram": 0.2191780821917808,
                "Arrah": 0.1417910447761194, "Asansol": 0.128, "Aurangabad": 0.15810276679841898, "Avadi": 0.2463768115942029, "Bahraich": 0.174496644295302, "Ballia": 0.19083969465648856, "Bally": 0.18181818181818182, "Bangalore": 0.2109375, "Baranagar": 0.18382352941176472, "Barasat": 0.2689655172413793,
                "Bardhaman": 0.2097902097902098, "Bareilly": 0.16778523489932887, "Bathinda": 0.19827586206896552, "Begusarai": 0.19834710743801653, "Belgaum": 0.12213740458015267, "Bellary": 0.19047619047619047, "Berhampore": 0.15555555555555556, "Berhampur": 0.20967741935483872, "Bettiah": 0.24347826086956523, "Bhagalpur": 0.21014492753623187, "Bhalswa_Jahangir_Pur": 0.21428571428571427,
                "Bharatpur": 0.1724137931034483, "Bhatpara": 0.15833333333333333, "Bhavnagar": 0.21333333333333335, "Bhilai": 0.19548872180451127, "Bhilwara": 0.1484375, "Bhimavaram": 0.16, "Bhind": 0.1773049645390071, "Bhiwandi": 0.21705426356589147, "Bhiwani": 0.16793893129770993, "Bhopal": 0.2222222222222222, "Bhubaneswar": 0.29457364341085274, "Bhusawal": 0.1956521739130435,
                "Bidar": 0.25384615384615383, "Bidhannagar": 0.17647058823529413, "Bihar_Sharif": 0.15702479338842976, "Bijapur": 0.15873015873015872, "Bikaner": 0.17164179104477612, "Bilaspur": 0.18461538461538463, "Bokaro": 0.25, "Bongaigaon": 0.22764227642276422, "Bulandshahr": 0.22818791946308725, "Burhanpur": 0.18705035971223022, "Buxar": 0.2052980132450331, "Chandigarh_city": 0.1984732824427481,
                "Chandrapur": 0.2, "Chapra": 0.18115942028985507, "Chennai": 0.2558139534883721, "Chinsurah": 0.13768115942028986, "Chittoor": 0.20175438596491227, "Coimbatore": 0.2550335570469799, "Cuttack": 0.16666666666666666, "Danapur": 0.1506849315068493, "Darbhanga": 0.19834710743801653, "Davanagere": 0.2781954887218045, "Dehradun": 0.17692307692307693, "Dehri": 0.1678832116788321, "Delhi_city": 0.1559633027522936,
                "Deoghar": 0.2185430463576159, "Dewas": 0.19205298013245034, "Dhanbad": 0.21875, "Dharmavaram": 0.18699186991869918, "Dhule": 0.1875, "Dibrugarh": 0.20125786163522014, "Dindigul": 0.2781954887218045, "Durg": 0.21678321678321677, "Durgapur": 0.1793103448275862, "Eluru": 0.1678832116788321, "Erode": 0.1987179487179487, "Etawah": 0.2556390977443609, "Faridabad": 0.16783216783216784, "Farrukhabad": 0.1888111888111888,
                "Fatehpur": 0.1951219512195122, "Firozabad": 0.21379310344827587, "Gandhidham": 0.18045112781954886, "Gandhinagar": 0.14166666666666666, "Gangtok": 0.18064516129032257, "Gaya": 0.16666666666666666, "Ghaziabad": 0.25874125874125875, "Giridih": 0.24647887323943662, "Gopalpur": 0.17647058823529413, "Gorakhpur": 0.23484848484848486, "Gudivada": 0.21428571428571427, "Gulbarga": 0.12605042016806722, "Guna": 0.14788732394366197,
                "Guntakal": 0.2, "Guntur": 0.19285714285714287, "Gurgaon": 0.18461538461538463, "Guwahati": 0.1875, "Gwalior": 0.25, "Hajipur": 0.20422535211267606, "Haldia": 0.16546762589928057, "Hapur": 0.16312056737588654, "Haridwar": 0.175, "Hazaribagh": 0.22900763358778625, "Hindupur": 0.19047619047619047, "Hospet": 0.2013888888888889, "Hosur": 0.14166666666666666, "Howrah": 0.15584415584415584, "Hubliâ€“Dharwad": 0.21487603305785125,
                "Hyderabad": 0.16535433070866143, "Ichalkaranji": 0.248, "Imphal": 0.2440944881889764, "Indore": 0.24489795918367346, "Jabalpur": 0.2222222222222222, "Jaipur": 0.20915032679738563, "Jalandhar": 0.1953125, "Jalgaon": 0.16666666666666666, "Jalna": 0.19393939393939394, "Jamalpur": 0.19594594594594594, "Jammu": 0.22137404580152673, "Jamnagar": 0.1640625, "Jamshedpur": 0.20491803278688525, "Jaunpur": 0.2108843537414966, "Jehanabad": 0.19230769230769232,
                "Jhansi": 0.18120805369127516, "Jodhpur": 0.16993464052287582, "Jorhat": 0.15602836879432624, "Junagadh": 0.15827338129496402, "Kadapa": 0.2302158273381295, "Kakinada": 0.16535433070866143, "Kalyan-Dombivli": 0.22758620689655173, "Kamarhati": 0.23357664233576642, "Kanpur": 0.16901408450704225, "Karaikudi": 0.2231404958677686, "Karawal_Nagar": 0.25, "Karimnagar": 0.19285714285714287, "Karnal": 0.23134328358208955, "Katihar": 0.1865671641791045,
                "Katni": 0.20388349514563106, "Kavali": 0.24183006535947713, "Khammam": 0.23741007194244604, "Khandwa": 0.19230769230769232, "Kharagpur": 0.2129032258064516, "Khora,_Ghaziabad": 0.14492753623188406, "Kirari_Suleman_Nagar": 0.208955223880597, "Kishanganj": 0.17482517482517482, "Kochi": 0.24675324675324675, "Kolhapur": 0.18791946308724833, "Kolkata": 0.1773049645390071, "Kollam": 0.1619718309859155, "Korba": 0.2206896551724138, "Kota": 0.17647058823529413,
                "Kottayam": 0.273972602739726, "Kozhikode": 0.21875, "Kulti": 0.19186046511627908, "Kumbakonam": 0.13636363636363635, "Kurnool": 0.17647058823529413, "Latur": 0.18110236220472442, "Loni": 0.20300751879699247, "Lucknow": 0.1415929203539823, "Ludhiana": 0.15584415584415584, "Machilipatnam": 0.22666666666666666, "Madanapalle": 0.21014492753623187, "Madhyamgram": 0.20496894409937888, "Madurai": 0.21323529411764705, "Mahbubnagar": 0.16783216783216784,
                "Maheshtala": 0.24812030075187969, "Malda": 0.176056338028169, "Malegaon": 0.16993464052287582, "Mangalore": 0.2214765100671141, "Mango": 0.2, "Mathura": 0.18253968253968253, "Mau": 0.1721311475409836, "Medininagar": 0.1732283464566929, "Meerut": 0.26356589147286824, "Mehsana": 0.12781954887218044, "Mira-Bhayandar": 0.12213740458015267, "Miryalaguda": 0.2112676056338028, "Mirzapur": 0.13740458015267176, "Moradabad": 0.19718309859154928, "Morbi": 0.20261437908496732,
                "Morena": 0.17647058823529413, "Motihari": 0.12598425196850394, "Mumbai": 0.21739130434782608, "Munger": 0.18181818181818182, "Muzaffarnagar": 0.2076923076923077, "Muzaffarpur": 0.16296296296296298, "Mysore": 0.2152777777777778, "Nadiad": 0.15625, "Nagaon": 0.16153846153846155, "Nagercoil": 0.2833333333333333, "Nagpur": 0.1777777777777778, "Naihati": 0.19444444444444445, "Nanded": 0.1640625, "Nandyal": 0.18461538461538463, "Nangloi_Jat": 0.17880794701986755, "Narasaraopet": 0.2540983606557377,
                "Nashik": 0.2595419847328244, "Navi_Mumbai": 0.1702127659574468, "Nellore": 0.19672131147540983, "New_Delhi": 0.15384615384615385, "Nizamabad": 0.17985611510791366, "Noida": 0.15384615384615385, "North_Dumdum": 0.19548872180451127, "Ongole": 0.19285714285714287, "Orai": 0.15384615384615385, "Ozhukarai": 0.2204724409448819, "Pali": 0.2396694214876033, "Pallavaram": 0.2037037037037037, "Panchkula": 0.2, "Panihati": 0.125, "Panipat": 0.1951219512195122, "Panvel": 0.2446043165467626, "Parbhani": 0.21428571428571427,
                "Patiala": 0.1652892561983471, "Patna": 0.23076923076923078, "Phagwara": 0.15833333333333333, "Phusro": 0.18518518518518517, "Pimpri-Chinchwad": 0.1917808219178082, "Pondicherry": 0.24444444444444444, "Proddatur": 0.19424460431654678, "Pudukkottai": 0.17857142857142858, "Pune": 0.22388059701492538, "Purnia": 0.17355371900826447, "Raebareli": 0.2054794520547945, "Raichur": 0.1865671641791045, "Raiganj": 0.22900763358778625, "Raipur": 0.17054263565891473, "Rajahmundry": 0.2553191489361702, "Rajkot": 0.23809523809523808,
                "Rajpur_Sonarpur": 0.11666666666666667, "Ramagundam": 0.2695035460992908, "Ramgarh": 0.18045112781954886, "Rampur": 0.16393442622950818, "Ranchi": 0.21710526315789475, "Ratlam": 0.18045112781954886, "Raurkela_Industrial_Township": 0.20125786163522014, "Rewa": 0.21739130434782608, "Rohtak": 0.21568627450980393, "Rourkela": 0.16891891891891891, "Sagar": 0.1953125, "Saharanpur": 0.20588235294117646, "Saharsa": 0.2, "Salem": 0.2605633802816901, "Sambalpur": 0.16428571428571428, "Sambhal": 0.1223021582733813,
                "Sangli-Miraj_&_Kupwad": 0.15748031496062992, "Sasaram": 0.14084507042253522, "Satara": 0.18796992481203006, "Satna": 0.20394736842105263, "Secunderabad": 0.2, "Serampore": 0.14285714285714285, "Shahjahanpur": 0.18705035971223022, "Shimla": 0.208955223880597, "Shimoga": 0.16911764705882354, "Shivpuri": 0.15555555555555556, "Sikar": 0.22535211267605634, "Silchar": 0.1910828025477707, "Siliguri": 0.2112676056338028, "Singrauli": 0.23636363636363636, "Sirsa": 0.22666666666666666, "Siwan": 0.18439716312056736, "Solapur": 0.19402985074626866,
                "Sonipat": 0.1564625850340136, "South_Dumdum": 0.2116788321167883, "Sri_Ganganagar": 0.21212121212121213, "Srikakulam": 0.25892857142857145, "Srinagar": 0.2125, "Sultan_Pur_Majra": 0.16216216216216217, "Surat": 0.22580645161290322, "Surendranagar_Dudhrej": 0.22302158273381295, "Suryapet": 0.17123287671232876, "Tadepalligudem": 0.23484848484848486, "Tadipatri": 0.24528301886792453, "Tenali": 0.17105263157894737, "Tezpur": 0.2, "Thane": 0.14285714285714285, "Thanjavur": 0.208955223880597, "Thiruvananthapuram": 0.2229299363057325,
                "Thoothukudi": 0.23308270676691728, "Thrissur": 0.22142857142857142, "Tinsukia": 0.16993464052287582, "Tiruchirappalli": 0.16891891891891891, "Tirunelveli": 0.14285714285714285, "Tirupati": 0.2013888888888889, "Tiruppur": 0.18518518518518517, "Tiruvottiyur": 0.17543859649122806, "Tumkur": 0.2426470588235294, "Udaipur": 0.2440944881889764, "Udupi": 0.2518518518518518, "Ujjain": 0.17692307692307693, "Ulhasnagar": 0.18, "Uluberia": 0.19117647058823528, "Unnao": 0.15625, "Vadodara": 0.19696969696969696, "Varanasi": 0.2,
                "Vasai-Virar": 0.15503875968992248, "Vellore": 0.125, "Vijayanagaram": 0.20915032679738563, "Vijayawada": 0.14906832298136646, "Visakhapatnam": 0.22627737226277372, "Warangal": 0.1984126984126984, "Yamunanagar": 0.29838709677419356}

STATE_MAPPING = {"Andhra_Pradesh": 0.20622119815668202, "Assam": 0.1863013698630137, "Bihar": 0.18038632986627043, "Chandigarh": 0.1984732824427481, "Chhattisgarh": 0.19852941176470587, "Delhi": 0.19148936170212766, "Gujarat": 0.1858974358974359, "Haryana": 0.20027434842249658, "Himachal_Pradesh": 0.208955223880597, "Jammu_and_Kashmir": 0.21649484536082475, "Jharkhand": 0.21185876082611593, "Karnataka": 0.20367610531544958, "Kerala": 0.21774193548387097, "Madhya_Pradesh": 0.19934906427990237, "Maharashtra": 0.1933179723502304,
                 "Manipur": 0.2440944881889764, "Mizoram": 0.15037593984962405, "Odisha": 0.19928400954653938, "Puducherry": 0.23282442748091603, "Punjab": 0.17205692108667528, "Rajasthan": 0.19341317365269461, "Sikkim": 0.18064516129032257, "Tamil_Nadu": 0.2052744119743407, "Telangana": 0.19956616052060738, "Tripura": 0.22666666666666666, "Uttar_Pradesh": 0.188964945032151, "Uttar_Pradesh[5]": 0.24324324324324326, "Uttarakhand": 0.17586206896551723, "West_Bengal": 0.18513916500994035}


st.set_page_config(page_title="Alpha Dreamers Credit Risk Portal", layout="wide")

st.title("🛡️ Alpha Dreamers Banking Consortium")
st.subheader("Automated Loan Risk Assessment Portal")
st.write("Enter the applicant's details below to calculate the risk profile.")

# Use columns to organize the input form
col1, col2 = st.columns(2)

with col1:
    st.header("👤 Personal & Professional")
    age = st.number_input("Age", min_value=18, max_value=100)
    marital = st.selectbox("Marital Status", ["single", "married"])
    profession = st.selectbox("Select Profession", options=list(PROFESSION_MAPPING.keys()))
    experience = st.number_input("Total Work Experience (Years)", min_value=0, max_value=50, value=5)
    job_years = st.number_input("Years in Current Job", min_value=0, max_value=50, value=3)


with col2:
    st.header("💰 Financial & Residential")
    income = st.number_input("Annual Income", min_value=0)
    house_ownership = st.selectbox("House Ownership", ["rented", "owned", "norent_noown"])
    car_ownership = st.selectbox("Do you own a car?", ["yes", "no"])
    house_years = st.number_input("Years in Current House", min_value=0, max_value=50, value=5)
    city = st.selectbox("Select City", options=list(CITY_MAPPING.keys()))
    state = st.selectbox("Select State", options=list(STATE_MAPPING.keys()))

# --- STAGE 3: PREDICTION LOGIC ---
# --- STAGE 3: PREDICTION LOGIC ---
if st.button("RUN RISK ASSESSMENT"):
    # Everything inside this block is indented exactly 4 spaces
    encoded_prof = PROFESSION_MAPPING.get(profession, 0.12)
    encoded_city = CITY_MAPPING.get(city, 0.123)
    encoded_state = STATE_MAPPING.get(state, 0.123)

    # 1. Create a base dictionary with numeric inputs
    input_dict = {
        'Income': income,
        'Age': age,
        'Experience': experience,
        'Married/Single': 1 if marital == "single" else 0,
        'Car_Ownership': 1 if car_ownership == "yes" else 0,
        'Profession': encoded_prof,
        'CITY': encoded_city,
        'STATE': encoded_state,
        'CURRENT_JOB_YRS': job_years,
        'CURRENT_HOUSE_YRS': house_years
    }

    # 2. Handle the One-Hot Encoding for House_Ownership
    # Ensure these lines line up perfectly with 'input_dict' above
    input_dict['House_Ownership_norent_noown'] = 0
    input_dict['House_Ownership_owned'] = 0
    input_dict['House_Ownership_rented'] = 0

    if house_ownership == "norent_noown":
        input_dict['House_Ownership_norent_noown'] = 1
    elif house_ownership == "owned":
        input_dict['House_Ownership_owned'] = 1
    elif house_ownership == "rented":
        input_dict['House_Ownership_rented'] = 1

    # 3. Create DataFrame and enforce the CORRECT column order
    input_df = pd.DataFrame([input_dict])

    # IMPORTANT: This list must exactly match the columns in your X_train
    COLUMN_ORDER = [
        'Income', 'Age', 'Experience', 'Married/Single', 'Car_Ownership',
        'Profession', 'CITY', 'STATE', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS',
        'House_Ownership_norent_noown', 'House_Ownership_owned', 'House_Ownership_rented'
    ]

    input_df = input_df[COLUMN_ORDER]
    input_df_scaled = scaler.transform(input_df)

    # Execute Prediction
    prediction = model.predict(input_df_scaled)
    probability = model.predict_proba(input_df_scaled)[0][1]

    # --- STAGE 4: DISPLAY RESULTS ---
    st.divider()
    if prediction[0] == 1:
        st.error(f"⚠️ **HIGH RISK DETECTED** (Probability: {probability:.2%})")
        st.write("Recommendation: Reject automated approval. Refer to Senior Underwriter.")
    else:
        st.success(f"✅ **LOW RISK PROFILE** (Risk Probability: {probability:.2%})")
        st.write("Recommendation: Proceed with automated loan processing.")
