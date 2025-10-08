# ==============================================================
# 🌾 Crop Recommendation & Scheduling Dashboard for Sri Lanka
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ==============================================================
# 1️⃣ LOAD DATA
# ==============================================================

st.set_page_config(page_title="Crop Recommendation Dashboard", layout="wide")
st.title("🌾 Crop Recommendation & Scheduling Dashboard for Sri Lanka")

data_path = r"C:\Users\Sanjali\Downloads\SriLanka_Crop.csv"
df = pd.read_csv(data_path)

# --- Data Cleaning ---
df = df.drop_duplicates()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

# --- Label Encode Target ---
le_crop = LabelEncoder()
df["Suitable_Crop_Label"] = le_crop.fit_transform(df["Suitable_Crop"])

# ==============================================================
# 2️⃣ MODEL TRAINING
# ==============================================================

feature_cols = [
    "Soil_pH", "Organic_Carbon", "Clay_pct", "Silt_pct", "Sand_pct",
    "Avg_Temp_C", "Annual_Rainfall_mm", "Humidity_pct", "Sunlight_hours", "Elevation_m"
]
X = df[feature_cols]
y = df["Suitable_Crop_Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# ==============================================================
# 3️⃣ HEADER / OVERVIEW METRICS
# ==============================================================

st.markdown("### 📊 Overview Metrics")
col1, col2 = st.columns(2)
col1.metric("🌍 Provinces Analyzed", df["Region"].nunique())
col2.metric("🌾 Unique Crops", df["Suitable_Crop"].nunique())

# ==============================================================
# 4️⃣ CROP RECOMMENDATION PANEL
# ==============================================================

st.markdown("### 🌾 Crop Recommendation Panel")
district = st.selectbox("Select a District / Region", sorted(df["Region"].unique()))

# Filter dataset for selected region
region_df = df[df["Region"] == district].copy()
selected_data = region_df.iloc[0]

# --- Predict (simulate recommendation) ---
input_features = np.array(selected_data[feature_cols]).reshape(1, -1)
pred_crop_label = rf_model.predict(input_features)[0]
pred_crop = le_crop.inverse_transform([pred_crop_label])[0]

st.success(f"🌾 **Recommended Crop for {district}: {pred_crop}**")
st.info(f"📅 **Optimal Planting Month:** {selected_data['Best_Planting_Month']}")
st.write(f"🌡️ Average Temperature: {selected_data['Avg_Temp_C']} °C")
st.write(f"💧 Annual Rainfall: {selected_data['Annual_Rainfall_mm']} mm")
st.write(f"🧪 Soil pH: {selected_data['Soil_pH']}")

# ==============================================================
# 5️⃣ CLIMATE AND SOIL INSIGHTS (Region-Specific)
# ==============================================================

st.markdown("### 🌤️ Climate and Soil Insights")
colA, colB = st.columns(2)

# --- Average Rainfall by Planting Month ---
rainfall_chart = px.bar(
    region_df.groupby("Best_Planting_Month")["Annual_Rainfall_mm"].mean().reset_index(),
    x="Best_Planting_Month",
    y="Annual_Rainfall_mm",
    title=f"🌧️ Average Rainfall by Planting Month for {district}",
    color="Annual_Rainfall_mm",
)
colA.plotly_chart(rainfall_chart, use_container_width=True)

# --- Soil pH Distribution by Crop ---
soil_chart = px.box(
    region_df,
    x="Suitable_Crop",
    y="Soil_pH",
    color="Suitable_Crop",
    title=f"🧪 Distribution of Soil pH by Crop Type in {district}"
)
soil_chart.update_layout(xaxis_title="Crop Type", yaxis_title="Soil pH", showlegend=False)
colB.plotly_chart(soil_chart, use_container_width=True)

# --- Temperature vs Yield (Scatter, clearer) ---
temp_yield_chart = px.scatter(
    region_df,
    x="Avg_Temp_C",
    y="Yield_kg_per_ha",
    color="Suitable_Crop",
    size="Annual_Rainfall_mm",
    hover_data=["Region", "Best_Planting_Month", "Soil_pH"],
    title=f"🌡️ Temperature vs Expected Yield for {district}",
    template="plotly_white",
)
colB.plotly_chart(temp_yield_chart, use_container_width=True)

# ==============================================================
# 6️⃣ PLANTING CALENDAR (Region-Specific)
# ==============================================================

st.markdown("### 🗓️ Planting Calendar View")
calendar_data = region_df.groupby(["Suitable_Crop", "Best_Planting_Month"])["Yield_kg_per_ha"].mean().reset_index()

calendar_fig = px.density_heatmap(
    calendar_data,
    x="Best_Planting_Month",
    y="Suitable_Crop",
    z="Yield_kg_per_ha",
    color_continuous_scale="Viridis",
    title=f"Crop Planting Schedule for {district}",
)
st.plotly_chart(calendar_fig, use_container_width=True)

# ==============================================================
# 7️⃣ DOWNLOAD & EXPORT SECTION (PDF)
# ==============================================================

st.markdown("### 📥 Download Recommendation Report (PDF)")

buffer = BytesIO()
pdf = canvas.Canvas(buffer, pagesize=A4)
pdf.setTitle("Crop Recommendation Report")

text = pdf.beginText(50, 800)
text.setFont("Helvetica-Bold", 14)
text.textLine("Crop Recommendation Report")
text.setFont("Helvetica", 12)
text.textLine("-------------------------------------")
text.textLine(f"Region: {district}")
text.textLine(f"Recommended Crop: {pred_crop}")
text.textLine(f"Optimal Planting Month: {selected_data['Best_Planting_Month']}")
text.textLine(f"Soil pH: {selected_data['Soil_pH']}")
text.textLine(f"Rainfall: {selected_data['Annual_Rainfall_mm']} mm")
text.textLine(f"Temperature: {selected_data['Avg_Temp_C']} °C")
text.textLine("")
text.textLine("This system uses machine learning to identify the most suitable")
text.textLine("crop and optimal planting schedule for each region in Sri Lanka")
text.textLine("based on soil and climatic conditions.")

pdf.drawText(text)
pdf.showPage()
pdf.save()

pdf_data = buffer.getvalue()
st.download_button(
    label="⬇️ Download PDF Report",
    data=pdf_data,
    file_name=f"{district}_Crop_Recommendation.pdf",
    mime="application/pdf"
)

st.success("✅ Dashboard Loaded Successfully — Explore insights above!")
