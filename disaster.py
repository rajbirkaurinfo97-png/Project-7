import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ==========================================
# 1. CONFIGURATION & OFFLINE HUB ENFORCEMENT
# ==========================================
# Force Hugging Face to read strictly from your hard drive
os.environ["TRANSFORMERS_OFFLINE"] = "1"

st.set_page_config(
    page_title="CrisisWatch: Real-Time Disaster Intelligence",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dashboard Styling
st.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: bold; color: #E74C3C; margin-bottom: 5px; }
    .subtitle { font-size: 18px; color: #7F8C8D; margin-bottom: 25px; }
    .metric-card { background-color: #F8F9F9; border-radius: 10px; padding: 15px; border-left: 5px solid #E74C3C; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOCAL MODEL LOADING (WINDOWS PATH)
# ==========================================
@st.cache_resource(show_spinner="Loading Fine-Tuned MiniLM Model Weights...")
def load_local_model():
    # Direct absolute reference to the exact folder we verified!
    model_path = "C:/Users/HUT2099/disaster_model"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    return tokenizer, model

try:
    tokenizer, model = load_local_model()
except Exception as e:
    st.error(f"⚠️ Error loading model from local directory. Details: {e}")
    st.stop()

# Helper inference function
def predict_tweet(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = F.softmax(outputs.logits, dim=-1).squeeze()
    prediction = torch.argmax(probabilities).item()
    confidence = probabilities[prediction].item()
    return prediction, confidence, probabilities[1].item() 

# ==========================================
# 3. MOCK DATA GENERATION FOR ANALYTICS
# ==========================================
@st.cache_data
def generate_historical_data():
    np.random.seed(42)
    regions = ["Northern Region", "Western Coast", "Southern Delta", "Eastern Hills"]
    keywords = ["flood", "earthquake", "monsoon", "wildfire", "evacuation", "shelter", "storm", "rescue"]
    
    start_time = datetime.now() - timedelta(days=2)
    data = []
    
    for i in range(150):
        timestamp = start_time + timedelta(minutes=int(np.random.randint(0, 2880)))
        region = np.random.choice(regions)
        keyword = np.random.choice(keywords)
        
        if region == "Northern Region" and keyword in ["monsoon", "flood", "evacuation"]:
            is_disaster = 1
            text = f"CRITICAL: Severe {keyword} hitting the Northern Region towns. Local teams ordering urgent evacuation."
            sentiment = "Negative / Panic"
        elif region == "Western Coast" and keyword in ["wildfire", "smoke"]:
            is_disaster = 1
            text = f"Alert: High winds spreading the {keyword} across the dry brush along the Western Coast."
            sentiment = "Negative / Warning"
        else:
            is_disaster = np.random.choice([0, 1], p=[0.75, 0.25])
            text = f"Discussing the upcoming weather patterns and possible {keyword} risks down the line." if is_disaster == 0 else f"Emergency dispatch reporting a sudden {keyword} outbreak!"
            sentiment = "Neutral" if is_disaster == 0 else "Negative"
            
        data.append({
            "Timestamp": timestamp,
            "Tweet": text,
            "Region": region,
            "Keyword": keyword,
            "Target": "Disaster" if is_disaster == 1 else "Not a Disaster",
            "Sentiment": sentiment
        })
    df = pd.DataFrame(data).sort_values("Timestamp")
    return df

df_historical = generate_historical_data()

# ==========================================
# 4. SIDEBAR CONTROLS & NAVIGATION
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=70)
st.sidebar.title("Dashboard Controls")
app_mode = st.sidebar.radio("Navigate App Modules:", ["Single Tweet Classifier", "Live Stream Monitor & Storytelling"])

# ==========================================
# MODULE 1: SINGLE TWEET CLASSIFIER
# ==========================================
if app_mode == "Single Tweet Classifier":
    st.markdown('<div class="main-title">🚨 CrisisWatch Sequence Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Fine-tuned MiniLM Transformer evaluation node for ad-hoc social media streams.</div>', unsafe_allow_html=True)
    
    st.subheader("Analyze Live Custom Input")
    user_tweet = st.text_area("Paste tweet content below:", placeholder="e.g., Heavy flash floods are destructive. Structural damage reported across highway 4...")
    
    if st.button("Evaluate Text Stream", type="primary"):
        if user_tweet.strip() == "":
            st.warning("Please enter a non-empty string sequence to classify.")
        else:
            with st.spinner("Processing token vectors via MiniLM backend..."):
                pred_class, confidence, disaster_prob = predict_tweet(user_tweet)
                
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Classification Output")
                if pred_class == 1:
                    st.error(f"🚨 **Prediction:** DISASTER EVENT DETECTED")
                else:
                    st.success(f"✅ **Prediction:** NOT A DISASTER")
                    
                st.metric(label="Model Confidence Score", value=f"{confidence * 100:.2f}%")
            
            with col2:
                st.markdown("### System Probability Metric Distribution")
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = disaster_prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Disaster Probability Index (%)", 'font': {'size': 16}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#E74C3C" if pred_class == 1 else "#2ECC71"},
                        'steps': [
                            {'range': [0, 50], 'color': '#E5E8E8'},
                            {'range': [50, 100], 'color': '#FADBD8' if pred_class == 1 else '#D4EFDF'}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

# ==========================================
# MODULE 2: LIVE STREAM MONITOR & STORYTELLING
# ==========================================
else:
    st.markdown('<div class="main-title">📊 Operational Risk Dashboard & Narrative Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Aggregated multi-channel event stream filtering and regional crisis analytics.</div>', unsafe_allow_html=True)
    
    # Automated Stakeholder Briefing Block
    st.markdown("""
    <div class="metric-card">
        <h4>📢 Regional Intelligence Briefing (Automated Operational Narrative)</h4>
        <p><strong>Current Active Pattern:</strong> Data indicates that the <strong>Northern Region</strong> is experiencing an active 
        <strong>monsoon system</strong>, generating a statistically significant spike in emergency vector signals containing keywords 
        like <em>flood</em> and <em>evacuation</em>. Response assets should monitor local network infrastructure continuously. 
        Conversely, the <strong>Western Coast</strong> presents secondary clustering anomalies regarding <em>wildfires</em>.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Stream Filters
    st.subheader("Interactive Stream Filters")
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        selected_regions = st.multiselect("Filter by Region Geography:", 
                                          options=list(df_historical["Region"].unique()), 
                                          default=list(df_historical["Region"].unique()))
    with f_col2:
        selected_keywords = st.multiselect("Filter by Core Lexicon Keywords:", 
                                           options=list(df_historical["Keyword"].unique()), 
                                           default=list(df_historical["Keyword"].unique()))
    with f_col3:
        selected_targets = st.multiselect("Filter by MiniLM Flag Status:", 
                                          options=list(df_historical["Target"].unique()), 
                                          default=list(df_historical["Target"].unique()))
        
    filtered_df = df_historical[
        (df_historical["Region"].isin(selected_regions)) & 
        (df_historical["Keyword"].isin(selected_keywords)) & 
        (df_historical["Target"].isin(selected_targets))
    ]
    
    # Key Performance Metric Rows
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Total Extracted Signals", len(filtered_df))
    m_col2.metric("Critical Disaster Validations", len(filtered_df[filtered_df["Target"] == "Disaster"]))
    m_col3.metric("Signal Density Ratio", f"{(len(filtered_df[filtered_df['Target'] == 'Disaster']) / max(len(filtered_df), 1)) * 100:.1f}%")

    st.markdown("---")
    
    # Data Visualizations
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.subheader("Temporal Disaster Signal Trends")
        trend_df = filtered_df.copy()
        trend_df['Time_Window'] = trend_df['Timestamp'].dt.floor('2h')
        chart_data = trend_df.groupby(['Time_Window', 'Target']).size().reset_index(name='Signal Count')
        
        fig_trend = px.line(chart_data, x='Time_Window', y='Signal Count', color='Target',
                            color_discrete_map={'Disaster': '#E74C3C', 'Not a Disaster': '#2ECC71'},
                            labels={'Time_Window': 'Timeline Scale', 'Signal Count': 'Volume of Incidents'})
        fig_trend.update_layout(margin=dict(l=20, r=20, t=10, b=20), height=300)
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with v_col2:
        st.subheader("Linguistic Keyword Vector Distribution")
        kw_data = filtered_df[filtered_df["Target"] == "Disaster"].groupby("Keyword").size().reset_index(name='Frequency')
        kw_data = kw_data.sort_values(by="Frequency", ascending=True)
        
        fig_bar = px.bar(kw_data, x='Frequency', y='Keyword', orientation='h',
                         title=None, color_discrete_sequence=['#34495E'])
        fig_bar.update_layout(margin=dict(l=20, r=20, t=10, b=20), height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

    v_col3, v_col4 = st.columns(2)
    with v_col3:
        st.subheader("Geographical Distribution Breakdown")
        geo_data = filtered_df.groupby(['Region', 'Target']).size().reset_index(name='Count')
        fig_geo = px.bar(geo_data, x='Region', y='Count', color='Target', barmode='group',
                         color_discrete_map={'Disaster': '#E74C3C', 'Not a Disaster': '#3498DB'})
        fig_geo.update_layout(margin=dict(l=20, r=20, t=10, b=20), height=300)
        st.plotly_chart(fig_geo, use_container_width=True)

    with v_col4:
        st.subheader("Categorical Sentiment Context Mapping")
        sent_data = filtered_df.groupby('Sentiment').size().reset_index(name='Weight')
        fig_pie = px.pie(sent_data, values='Weight', names='Sentiment', 
                         color_discrete_sequence=px.colors.sequential.RdBu)
        fig_pie.update_layout(margin=dict(l=20, r=20, t=10, b=20), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Real-Time Monitoring Simulator Block
    st.markdown("---")
    st.subheader("Simulated Real-Time Incident Feed")
    
    run_simulation = st.checkbox("Initialize Live Pipeline Simulation Backend")
    status_box = st.empty()
    
    if run_simulation:
        mock_live_tweets = [
            "BREAKING: Flash floods breaching dams in Northern Region! Requesting urgent sandbags.",
            "Normal sunny day out here down near the southern delta regions.",
            "Emergency fire engines dispatched to Western Coast brush fires. High risk warning.",
            "Traffic delays expected on the main inter-state link due to seasonal rain patterns."
        ]
        
        for item in mock_live_tweets:
            status_box.info(f"📥 **Intercepting Incoming Signal:** \"{item}\"")
            time.sleep(1.5)
            
            p_cls, conf, _ = predict_tweet(item)
            if p_cls == 1:
                status_box.error(f"🚨 **Classifier Flagged Alert:** Classified as CRITICAL DISASTER ({conf*100:.1f}% confidence)")
            else:
                status_box.success(f"✅ **Classifier Flagged Alert:** Cleared as SAFE INCIDENT ({conf*100:.1f}% confidence)")
            
            time.sleep(2.5)
        status_box.write("✨ Simulation loop cycle complete.")
    else:
        st.info("Check the checkbox above to simulate the live incoming social streaming connection pipeline.")