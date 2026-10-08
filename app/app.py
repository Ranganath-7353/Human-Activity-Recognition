
import os
import joblib
import pandas as pd
import streamlit as st




st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🚶",
    layout="wide"
)




st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    font-size: 36px;
    font-weight: 700;
}

.subtitle {
    font-size: 16px;
    color: #666666;
}

.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #f7f7f7;
    border: 1px solid #dddddd;
    text-align: center;
}

.card-title {
    font-size: 15px;
    color: #666666;
}

.card-value {
    font-size: 28px;
    font-weight: 700;
}

.result {
    padding: 30px;
    border-radius: 15px;
    background-color: #f7f7f7;
    border: 2px solid #dddddd;
    text-align: center;
}

.result-title {
    font-size: 18px;
    color: #666666;
}

.result-value {
    font-size: 42px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)




BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "human_activity_model.pkl"
)

model = joblib.load(MODEL_PATH)




st.markdown(
    '<div class="title">🚶 Human Activity Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Sensor-based Standing & Walking Detection Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.divider()




col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">Model</div>
        <div class="card-value">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">ROC-AUC</div>
        <div class="card-value">94.81%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-title">Activities</div>
        <div class="card-value">2</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
        <div class="card-title">Sensors</div>
        <div class="card-value">6</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")




st.subheader("📊 Sensor Input")

st.caption(
    "Enter accelerometer and gyroscope readings."
)

col1, col2, col3 = st.columns(3)

with col1:

    accX = st.number_input(
        "Accelerometer X",
        value=0.0,
        format="%.4f"
    )

    accY = st.number_input(
        "Accelerometer Y",
        value=0.0,
        format="%.4f"
    )


with col2:

    accZ = st.number_input(
        "Accelerometer Z",
        value=0.0,
        format="%.4f"
    )

    gyroX = st.number_input(
        "Gyroscope X",
        value=0.0,
        format="%.4f"
    )


with col3:

    gyroY = st.number_input(
        "Gyroscope Y",
        value=0.0,
        format="%.4f"
    )

    gyroZ = st.number_input(
        "Gyroscope Z",
        value=0.0,
        format="%.4f"
    )


st.write("")




predict_button = st.button(
    "🔍 Predict Activity",
    use_container_width=True
)


if predict_button:

 

    new_data = pd.DataFrame({
        "accX": [accX],
        "accY": [accY],
        "accZ": [accZ],
        "gyroX": [gyroX],
        "gyroY": [gyroY],
        "gyroZ": [gyroZ]
    })


   

    prediction = model.predict(new_data)[0]

    probability = model.predict_proba(new_data)[0]

    confidence = probability[prediction] * 100


   
    if prediction == 0:
        activity_name = "Standing"
    else:
        activity_name = "Walking"


    

    st.write("")

    st.subheader("🎯 Prediction Result")

    st.markdown(
        f"""
        <div class="result">
            <div class="result-title">Detected Activity</div>
            <div class="result-value">{activity_name}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.write("")



    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )


    with col2:

        st.metric(
            "Model Prediction",
            f"Activity {prediction}"
        )


   

    st.subheader("📈 Activity Probability")

    st.write(
        f"Standing — {probability[0] * 100:.2f}%"
    )

    st.progress(
        float(probability[0])
    )


    st.write(
        f"Walking — {probability[1] * 100:.2f}%"
    )

    st.progress(
        float(probability[1])
    )


   

    st.subheader("📋 Sensor Summary")

    sensor_summary = pd.DataFrame({
        "Sensor": [
            "Accelerometer X",
            "Accelerometer Y",
            "Accelerometer Z",
            "Gyroscope X",
            "Gyroscope Y",
            "Gyroscope Z"
        ],
        "Value": [
            accX,
            accY,
            accZ,
            gyroX,
            gyroY,
            gyroZ
        ]
    })

    st.dataframe(
        sensor_summary,
        use_container_width=True,
        hide_index=True
    )

