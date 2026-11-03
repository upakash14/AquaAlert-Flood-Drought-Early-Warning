import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AquaAlert", page_icon="🌧️", layout="wide")
st.title("🌧️ AquaAlert")
st.subheader("AI-Powered Flood & Drought Early Warning System")
st.write("Neural Storm 2026 | Green Planet | GP-03 | Akash U P")
st.warning("Educational prototype only. Built-in data is synthetic and this app is not an official warning system.")

@st.cache_data
def make_data():
    rng = np.random.default_rng(42)
    n = 1500
    rainfall = rng.uniform(0, 300, n)
    soil = rng.uniform(5, 100, n)
    river = rng.uniform(0.5, 10, n)
    temp = rng.uniform(15, 45, n)
    humidity = rng.uniform(15, 100, n)
    labels = []
    for r, s, rv, t in zip(rainfall, soil, river, temp):
        if r > 180 or (r > 120 and rv > 6):
            labels.append("Flood")
        elif r < 20 and s < 25 and t > 30:
            labels.append("Drought")
        else:
            labels.append("Normal")
    return pd.DataFrame({
        "Rainfall": rainfall, "Soil Moisture": soil,
        "River Level": river, "Temperature": temp,
        "Humidity": humidity, "Risk": labels
    })

data = make_data()
features = ["Rainfall", "Soil Moisture", "River Level", "Temperature", "Humidity"]
X, y = data[features], data["Risk"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(
    n_estimators=150, max_depth=10, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)
score = accuracy_score(y_test, model.predict(X_test))

m1, m2, m3 = st.columns(3)
m1.metric("Demo records", f"{len(data):,}")
m2.metric("Algorithm", "Random Forest")
m3.metric("Demo test score", f"{score:.1%}")
st.caption("The score only measures agreement with synthetic demonstration labels, not real-world forecasting performance.")

st.header("🔎 Environmental Risk Assessment")
c1, c2 = st.columns(2)
with c1:
    rainfall = st.number_input("Rainfall in last 24 hours (mm)", 0.0, 500.0, 75.0)
    soil = st.slider("Soil moisture (%)", 0, 100, 45)
    river = st.number_input("River level (m)", 0.0, 15.0, 3.0)
with c2:
    temperature = st.number_input("Temperature (°C)", 0.0, 55.0, 29.0)
    humidity = st.slider("Humidity (%)", 0, 100, 60)
    location = st.selectbox("Location", ["Hassan", "Mysuru", "Kodagu", "Shivamogga",
        "Uttara Kannada", "Belagavi", "Kalaburagi", "Raichur", "Ballari", "Vijayapura"])

if st.button("Analyze environmental risk", type="primary"):
    row = pd.DataFrame([[rainfall, soil, river, temperature, humidity]], columns=features)
    result = model.predict(row)[0]
    confidence = float(model.predict_proba(row).max())
    st.subheader(f"{location}: {result.upper()}")
    st.metric("Model confidence", f"{confidence:.1%}")
    if result == "Flood":
        st.error("Flood-like conditions in the demo model")
        tips = ["Monitor official weather and river-level alerts.",
                "Avoid riverbanks and low-lying areas.",
                "Follow instructions from local authorities."]
    elif result == "Drought":
        st.warning("Drought-like conditions in the demo model")
        tips = ["Conserve available water.",
                "Monitor local water resources.",
                "Follow local agricultural advisories."]
    else:
        st.success("Normal conditions in the demo model")
        tips = ["Continue monitoring local conditions.",
                "Keep emergency contacts accessible.",
                "Follow official weather updates."]
    st.markdown("**Suggested preparedness actions**")
    for tip in tips:
        st.write("• " + tip)
    st.caption("Experimental output only; not an official alert or emergency advice.")

st.header("🗺️ Illustrative Karnataka Location Map")
places = pd.DataFrame({
    "latitude": [13.0033, 12.2958, 12.4244, 13.9299, 14.8182, 15.8497, 17.3297, 16.2076, 15.1394, 16.8302],
    "longitude": [76.1004, 76.6394, 75.7382, 75.5681, 74.1260, 74.4977, 76.8343, 77.3550, 76.9214, 75.7100]
}, index=["Hassan", "Mysuru", "Kodagu", "Shivamogga", "Uttara Kannada",
          "Belagavi", "Kalaburagi", "Raichur", "Ballari", "Vijayapura"])
st.map(places)
st.caption("These are illustrative district locations, not live risk markers.")

st.header("📊 Climate Data Analytics")
st.bar_chart(data["Risk"].value_counts())
st.line_chart(data["Rainfall"].head(50))
st.subheader("Sample dataset")
st.dataframe(data.head(20), use_container_width=True)

st.header("🤖 Model Information")
st.write("Inputs: rainfall, soil moisture, river level, temperature and humidity.")
st.write("Algorithm: Random Forest Classifier. Output classes: Normal, Flood and Drought.")
importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
st.bar_chart(importance.set_index("Feature"))

st.download_button("Download synthetic demo dataset",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="aquaalert_demo_data.csv", mime="text/csv")

st.divider()
st.write("Developed by Akash U P | Malnad College of Engineering | Neural Storm 2026 GP-03")
