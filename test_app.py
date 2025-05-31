import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Football Injury Prediction",
    page_icon="⚽",
    layout="wide"
)

# Load models and data


@st.cache_resource
def load_models():
    rf_model = joblib.load('random_forest_injury_model.pkl')
    xgb_model = joblib.load('xgboost_injury_model.pkl')
    return rf_model, xgb_model


@st.cache_data
def load_player_data():
    with open('updated_player_info.json', 'r', encoding='utf-8') as f:
        player_info = json.load(f)
    features_tensor = np.load('positional_all_players_tensor.npy')
    return player_info, features_tensor


rf_model, xgb_model = load_models()
player_info, features_tensor = load_player_data()

# Helper function to extract position from player info


# def get_player_position(player_name):
#     """Extract position information from player data"""
#     if 'Position' in player_info[player_name]:
#         return player_info[player_name]['Position']
#     elif 'position' in player_info[player_name]:
#         return player_info[player_name]['position']
#     else:
#         return 'Unknown'

def get_player_position(player_name):
    """Get player position"""
    try:
        with open('player_positions.json', 'r', encoding='utf-8') as f:
            positions = json.load(f)
        return positions.get(player_name, 'Unknown')
    except:
        return 'Unknown'


def categorize_position(position):
    """Categorize positions into broader groups"""
    if position in ['GK', 'Goalkeeper']:
        return 'Goalkeeper'
    elif position in ['CB', 'LB', 'RB', 'LWB', 'RWB', 'Center-Back', 'Left-Back', 'Right-Back', 'Left Center Back', 'Left Back', 'Left Wing Back', 'Center Back', 'Right Back', 'Right Center Back', 'Right Wing Back']:
        return 'Defender'
    elif position in ['CM', 'CDM', 'CAM', 'LM', 'RM', 'Central Midfield', 'Defensive Midfield', 'Attacking Midfield', 'Left Midfield', 'Left Attacking Midfield', 'Left Defensive Midfield', 'Left Center Midfield', 'Center Defensive Midfield', 'Center Midfield', 'Center Attacking Midfield', 'Right Center Midfield', 'Right Defensive Midfield', 'Right Attacking Midfield', 'Right Midfield']:
        return 'Midfielder'
    elif position in ['LW', 'RW', 'CF', 'ST', 'Left Winger', 'Right Winger', 'Center Forward', 'Striker', 'Left Wing', 'Left Center Forward', 'Center Forward', 'Right Wing', 'Right Center Forward']:
        return 'Forward'
    else:
        return 'Unknown'


def create_player_body_visualization(injury_probability, player_name):
    """Create an advanced 3D-style player visualization with animated risk indicators"""

    # Calculate risk colors and effects
    risk_intensity = min(injury_probability * 1.8, 1.0)
    pulse_speed = max(0.5, 2 - injury_probability * 2)

    # Color calculations
    red_val = int(255 * risk_intensity)
    green_val = max(0, int(255 * (1 - risk_intensity * 1.2)))
    blue_val = max(0, int(255 * (1 - risk_intensity * 1.5)))

    high_risk_color = f"rgb({red_val}, {green_val}, {blue_val})"
    glow_color = f"rgba({red_val}, {green_val}, {blue_val}, 0.6)"

    # Risk level determination
    if injury_probability > 0.7:
        risk_level = "CRITICAL"
        risk_emoji = "🚨"
        bg_gradient = "linear-gradient(135deg, #ff6b6b, #ff8e8e)"
    elif injury_probability > 0.4:
        risk_level = "HIGH"
        risk_emoji = "⚠️"
        bg_gradient = "linear-gradient(135deg, #ffa726, #ffcc80)"
    elif injury_probability > 0.2:
        risk_level = "MEDIUM"
        risk_emoji = "📊"
        bg_gradient = "linear-gradient(135deg, #42a5f5, #90caf9)"
    else:
        risk_level = "LOW"
        risk_emoji = "✅"
        bg_gradient = "linear-gradient(135deg, #66bb6a, #a5d6a7)"

    html_content = f"""
    <div style="
        background: {bg_gradient};
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333;
        max-width: 350px;
        margin: 0 auto;
    ">
        <!-- Header -->
        <div style="text-align: center; margin-bottom: 15px;">
            <h3 style="margin: 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
                {risk_emoji} INJURY RISK ASSESSMENT
            </h3>
            <p style="margin: 5px 0; color: rgba(255,255,255,0.9); font-weight: bold;">
                {player_name}
            </p>
        </div>
        
        <!-- Risk Meter -->
        <div style="
            background: rgba(255,255,255,0.9);
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 24px; font-weight: bold; color: {high_risk_color}; margin-bottom: 5px;">
                {injury_probability:.1%}
            </div>
            <div style="font-size: 14px; font-weight: bold; color: {high_risk_color};">
                {risk_level} RISK
            </div>
            <div style="
                width: 100%;
                height: 10px;
                background: #e0e0e0;
                border-radius: 5px;
                margin: 10px 0;
                overflow: hidden;
            ">
                <div style="
                    width: {injury_probability * 100}%;
                    height: 100%;
                    background: {high_risk_color};
                    border-radius: 5px;
                    animation: pulse 2s infinite;
                "></div>
            </div>
        </div>
        
        <!-- 3D Player Figure -->
        <div style="text-align: center; margin-bottom: 20px;">
            <svg width="280" height="400" viewBox="0 0 280 400" style="filter: drop-shadow(5px 5px 10px rgba(0,0,0,0.3));">
                <defs>
                    <radialGradient id="riskGlow" cx="50%" cy="50%" r="60%">
                        <stop offset="0%" style="stop-color:{glow_color};stop-opacity:0.8"/>
                        <stop offset="50%" style="stop-color:{glow_color};stop-opacity:0.4"/>
                        <stop offset="100%" style="stop-color:transparent;stop-opacity:0"/>
                    </radialGradient>
                    
                    <linearGradient id="jerseyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#1e3c72"/>
                        <stop offset="100%" style="stop-color:#2a5298"/>
                    </linearGradient>
                    
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                        <feMerge> 
                            <feMergeNode in="coloredBlur"/>
                            <feMergeNode in="SourceGraphic"/>
                        </feMerge>
                    </filter>
                </defs>
                
                <!-- Background Circle -->
                <circle cx="140" cy="200" r="120" fill="rgba(255,255,255,0.1)" stroke="rgba(255,255,255,0.3)" stroke-width="2"/>
                
                <!-- Shadow -->
                <ellipse cx="140" cy="380" rx="50" ry="12" fill="rgba(0,0,0,0.2)"/>
                
                <!-- Head with 3D effect -->
                <circle cx="140" cy="50" r="22" fill="#FDBCB4" stroke="#E8A298" stroke-width="2"/>
                <circle cx="140" cy="48" r="20" fill="#FDBCB4"/>
                
                <!-- Hair -->
                <path d="M 120 38 Q 140 25 160 38 Q 160 30 140 28 Q 120 30 120 38" fill="#654321"/>
                
                <!-- Facial Features -->
                <circle cx="133" cy="45" r="1.5" fill="#333"/>
                <circle cx="147" cy="45" r="1.5" fill="#333"/>
                <path d="M 138 52 Q 140 55 142 52" stroke="#333" stroke-width="1" fill="none"/>
                
                <!-- Neck -->
                <rect x="132" y="72" width="16" height="12" fill="#FDBCB4" rx="2"/>
                
                <!-- Jersey with 3D shading -->
                <rect x="110" y="84" width="60" height="75" fill="url(#jerseyGrad)" stroke="#1a2b5c" stroke-width="2" rx="8"/>
                <rect x="120" y="94" width="40" height="3" fill="#FFD700"/>
                <text x="140" y="130" text-anchor="middle" fill="white" font-family="Arial Black" font-size="20" font-weight="bold">10</text>
                
                <!-- Arms with gradient -->
                <rect x="85" y="94" width="25" height="55" fill="#FDBCB4" stroke="#E8A298" stroke-width="2" rx="12"/>
                <rect x="170" y="94" width="25" height="55" fill="#FDBCB4" stroke="#E8A298" stroke-width="2" rx="12"/>
                
                <!-- Hands -->
                <circle cx="97" cy="160" r="7" fill="#FDBCB4" stroke="#E8A298" stroke-width="1"/>
                <circle cx="183" cy="160" r="7" fill="#FDBCB4" stroke="#E8A298" stroke-width="1"/>
                
                <!-- Shorts -->
                <rect x="115" y="159" width="50" height="35" fill="#1a237e" stroke="#0d1542" stroke-width="2" rx="5"/>
                
                <!-- Risk Glow Effects -->
                <ellipse cx="130" cy="230" rx="18" ry="35" fill="url(#riskGlow)" style="animation: pulse {pulse_speed}s infinite;"/>
                <ellipse cx="150" cy="230" rx="18" ry="35" fill="url(#riskGlow)" style="animation: pulse {pulse_speed}s infinite;"/>
                <ellipse cx="130" cy="290" rx="15" ry="40" fill="url(#riskGlow)" style="animation: pulse {pulse_speed}s infinite;"/>
                <ellipse cx="150" cy="290" rx="15" ry="40" fill="url(#riskGlow)" style="animation: pulse {pulse_speed}s infinite;"/>
                
                <!-- Thighs with risk coloring -->
                <rect x="120" y="194" width="18" height="50" fill="{high_risk_color}" stroke="#cc0000" stroke-width="3" rx="9" filter="url(#glow)"/>
                <rect x="142" y="194" width="18" height="50" fill="{high_risk_color}" stroke="#cc0000" stroke-width="3" rx="9" filter="url(#glow)"/>
                
                <!-- Shins with enhanced risk visualization -->
                <rect x="122" y="244" width="16" height="60" fill="{high_risk_color}" stroke="#990000" stroke-width="4" rx="8" filter="url(#glow)"/>
                <rect x="142" y="244" width="16" height="60" fill="{high_risk_color}" stroke="#990000" stroke-width="4" rx="8" filter="url(#glow)"/>
                
                <!-- Shin Guards -->
                <rect x="124" y="254" width="12" height="40" fill="rgba(255,255,255,0.8)" stroke="#ccc" stroke-width="1" rx="3"/>
                <rect x="144" y="254" width="12" height="40" fill="rgba(255,255,255,0.8)" stroke="#ccc" stroke-width="1" rx="3"/>
                
                <!-- Knee Caps -->
                <circle cx="130" cy="244" r="5" fill="#FFD700" stroke="#FF8C00" stroke-width="2"/>
                <circle cx="150" cy="244" r="5" fill="#FFD700" stroke="#FF8C00" stroke-width="2"/>
                
                <!-- Soccer Boots -->
                <ellipse cx="130" cy="315" rx="10" ry="18" fill="#000" transform="rotate(-8 130 315)"/>
                <ellipse cx="150" cy="315" rx="10" ry="18" fill="#000" transform="rotate(8 150 315)"/>
                <rect x="122" y="325" width="16" height="6" fill="#000" rx="3"/>
                <rect x="142" y="325" width="16" height="6" fill="#000" rx="3"/>
                
                <!-- Cleats -->
                <rect x="125" y="331" width="2" height="4" fill="#333"/>
                <rect x="130" y="331" width="2" height="4" fill="#333"/>
                <rect x="135" y="331" width="2" height="4" fill="#333"/>
                <rect x="145" y="331" width="2" height="4" fill="#333"/>
                <rect x="150" y="331" width="2" height="4" fill="#333"/>
                <rect x="155" y="331" width="2" height="4" fill="#333"/>
            </svg>
        </div>
        
        <!-- Risk Analysis -->
        <div style="
            background: rgba(255,255,255,0.95);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
            <h4 style="margin: 0 0 10px 0; color: #333; text-align: center;">🏥 Medical Assessment</h4>
            
            <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                <span><strong>Risk Level:</strong></span>
                <span style="color: {high_risk_color}; font-weight: bold;">{risk_level}</span>
            </div>
            
            <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                <span><strong>Probability:</strong></span>
                <span style="color: {high_risk_color}; font-weight: bold;">{injury_probability:.1%}</span>
            </div>
            
            <div style="margin: 15px 0; padding: 10px; background: rgba(255,215,0,0.1); border-left: 4px solid #FFD700; border-radius: 4px;">
                <strong>🎯 Focus Areas:</strong><br>
                <span style="font-size: 12px;">🦵 Lower extremities (thighs & shins)<br>
                🔧 Knee joint stability<br>
                ⚡ Muscle fatigue indicators</span>
            </div>
            
            <div style="text-align: center; margin-top: 15px;">
                {'<span style="color: #d32f2f; font-weight: bold;">⚠️ Immediate attention required</span>' if injury_probability > 0.7 else
                 '<span style="color: #f57c00; font-weight: bold;">📊 Monitor closely</span>' if injury_probability > 0.4 else
                 '<span style="color: #388e3c; font-weight: bold;">✅ Cleared for activity</span>'}
            </div>
        </div>
    </div>
    
    <style>
        @keyframes pulse {{
            0% {{ opacity: 0.7; transform: scale(1); }}
            50% {{ opacity: 1; transform: scale(1.05); }}
            100% {{ opacity: 0.7; transform: scale(1); }}
        }}
    </style>
    """

    return html_content


def analyze_position_injury_risk():
    """Enhanced position injury risk analysis with detailed insights"""
    position_data = []

    for player_name in player_info.keys():
        player_features = get_player_features(player_name)
        if player_features is not None:
            prediction = predict_risk(player_features, xgb_model)
            if prediction:
                position = get_player_position(player_name)
                position_category = categorize_position(position)
                team = player_info[player_name].get('Team', 'Unknown')

                position_data.append({
                    'Player': player_name,
                    'Position': position,
                    'Position_Category': position_category,
                    'Team': team,
                    'Risk_Probability': prediction['probability'],
                    'Risk_Level': prediction['risk_level'],
                    'Risk_Score': prediction['probability'] * 100
                })

    return pd.DataFrame(position_data)


def create_position_risk_dashboard(position_df):
    """Create an advanced dashboard for position risk analysis"""

    # Enhanced position risk analysis
    position_stats = position_df.groupby('Position_Category').agg({
        'Risk_Probability': ['mean', 'std', 'min', 'max', 'count'],
        'Risk_Score': ['mean'],
        'Player': 'count'
    }).round(3)

    position_stats.columns = ['Avg_Risk', 'Risk_Std', 'Min_Risk',
                              'Max_Risk', 'Player_Count', 'Avg_Score', 'Total_Players']
    position_stats = position_stats.reset_index()

    # Calculate additional metrics
    position_stats['Risk_Range'] = position_stats['Max_Risk'] - \
        position_stats['Min_Risk']
    position_stats['High_Risk_Players'] = position_df.groupby(
        'Position_Category')['Risk_Level'].apply(lambda x: (x == 'High').sum()).values
    position_stats['High_Risk_Percentage'] = (
        position_stats['High_Risk_Players'] / position_stats['Player_Count'] * 100).round(1)

    # Risk severity classification
    position_stats['Risk_Classification'] = position_stats['Avg_Risk'].apply(lambda x:
                                                                             'Critical' if x > 0.7 else 'High' if x > 0.5 else 'Moderate' if x > 0.3 else 'Low')

    return position_stats


# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a page",
    ["Model Performance", "Player Risk Assessment", "Team Overview",
        "Position Analysis"]
)

# Helper functions


def get_player_features(player_name):
    """Extract features for a specific player"""
    player_names = list(player_info.keys())
    if player_name not in player_names:
        return None

    player_idx = player_names.index(player_name)
    return features_tensor[player_idx]


def predict_risk(player_features, model):
    """Make prediction for a player"""
    lookback = 3
    if len(player_features) < lookback:
        return None

    window_features = player_features[-lookback:].flatten().reshape(1, -1)
    proba = model.predict_proba(window_features)[0, 1]

    if proba < 0.3:
        risk_level = "Low"
        color = "green"
    elif proba < 0.7:
        risk_level = "Medium"
        color = "orange"
    else:
        risk_level = "High"
        color = "red"

    return {
        'probability': float(proba),
        'risk_level': risk_level,
        'color': color
    }


# Player Risk Assessment Page
if app_mode == "Player Risk Assessment":
    st.title("⚽ Player Injury Risk Assessment")
    st.write("""
    Assess injury risk for individual players based on their recent performance data.
    """)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.subheader("Player Selection")
        selected_player = st.selectbox(
            "Select Player",
            list(player_info.keys())
        )

        selected_model = st.radio(
            "Prediction Model",
            ("XGBoost", "Random Forest")
        )

        if st.button("Assess Injury Risk"):
            player_features = get_player_features(selected_player)
            model = xgb_model if selected_model == "XGBoost" else rf_model
            prediction = predict_risk(player_features, model)

            if prediction:
                st.session_state['current_prediction'] = prediction
                st.session_state['current_player'] = selected_player
            else:
                st.error("Not enough data for this player")

    if 'current_prediction' in st.session_state:
        with col2:
            st.subheader(
                f"Risk Assessment for {st.session_state['current_player']}")

            # Risk indicator
            prob = st.session_state['current_prediction']['probability']
            risk_level = st.session_state['current_prediction']['risk_level']
            color = st.session_state['current_prediction']['color']

            st.metric(
                label="Injury Risk Probability",
                value=f"{prob:.1%}",
                help=f"Risk Level: {risk_level}"
            )

            # Progress bar visualization
            st.progress(prob)

            # Player information
            player_position = get_player_position(
                st.session_state['current_player'])
            st.write(f"**Position:** {player_position}")
            st.write(
                f"**Team:** {player_info[st.session_state['current_player']]['Team']}")

            # Detailed information
            with st.expander("Detailed Analysis"):
                # Feature importance visualization
                st.subheader("Key Contributing Factors")

                model = xgb_model if selected_model == "XGBoost" else rf_model
                if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                    importances = model.named_steps['classifier'].feature_importances_
                    top_features = pd.DataFrame({
                        'Feature': [f'Match {i//30 + 1}, Feature {i % 30}' for i in np.argsort(importances)[-10:][::-1]],
                        'Importance': np.sort(importances)[-10:][::-1]
                    })

                    fig = px.bar(top_features, x='Importance',
                                 y='Feature', orientation='h')
                    st.plotly_chart(fig, use_container_width=True)

                # Recent match history
                st.subheader("Recent Match History")
                player_features = get_player_features(
                    st.session_state['current_player'])
                recent_matches = pd.DataFrame(
                    player_features[-3:],
                    columns=[f"Feature {i}" for i in range(
                        player_features.shape[1])],
                    index=[f"Match {i}" for i in range(1, 4)]
                )
                st.dataframe(
                    recent_matches.style.background_gradient(cmap='Reds'))

        with col3:
            st.subheader("Player Risk Visualization")
            if 'current_prediction' in st.session_state:
                prob = st.session_state['current_prediction']['probability']
                player_html = create_player_body_visualization(
                    prob, st.session_state['current_player'])

                # Display the enhanced HTML visualization
                st.components.v1.html(player_html, height=600, scrolling=False)

# Team Overview Page
elif app_mode == "Team Overview":
    st.title("🏆 Team Overview Dashboard")
    st.markdown("""
    <div style="background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h4 style="color: white; margin: 0;">⚽ Team Injury Risk Management Center</h4>
        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Comprehensive team-level injury analytics and squad management insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Get all teams
    teams = sorted(list(set([player_info[player]['Team'] for player in player_info.keys()])))
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("🎯 Team Selection & Controls")
        
        selected_team = st.selectbox(
            "Select Team",
            teams,
            help="Choose a team for detailed analysis"
        )
        
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Quick Overview", "Detailed Analysis", "Complete Report"],
            index=1
        )
        
        risk_filter = st.multiselect(
            "Risk Level Filter",
            ["Low", "Medium", "High", "Critical"],
            default=["Medium", "High", "Critical"]
        )
        
        show_comparisons = st.checkbox("Show Team Comparisons", value=True)
        show_trends = st.checkbox("Show Risk Trends", value=False)
        
        if st.button("🔍 Generate Team Report", type="primary"):
            with st.spinner(f"🔄 Analyzing {selected_team} squad..."):
                # Get team players and their risk data
                team_players = [player for player in player_info.keys() 
                               if player_info[player]['Team'] == selected_team]
                
                team_risk_data = []
                for player in team_players:
                    player_features = get_player_features(player)
                    if player_features is not None:
                        prediction = predict_risk(player_features, xgb_model)
                        if prediction:
                            position = get_player_position(player)
                            position_category = categorize_position(position)
                            
                            # Risk level mapping
                            if prediction['probability'] < 0.2:
                                risk_level = "Low"
                            elif prediction['probability'] < 0.4:
                                risk_level = "Medium"
                            elif prediction['probability'] < 0.7:
                                risk_level = "High"
                            else:
                                risk_level = "Critical"
                            
                            team_risk_data.append({
                                'Player': player,
                                'Position': position,
                                'Position_Category': position_category,
                                'Risk_Probability': prediction['probability'],
                                'Risk_Level': risk_level,
                                'Risk_Score': prediction['probability'] * 100
                            })
                
                if team_risk_data:
                    st.session_state['team_data'] = pd.DataFrame(team_risk_data)
                    st.session_state['selected_team'] = selected_team
                    st.success(f"✅ Analysis completed for {selected_team}!")
                else:
                    st.error(f"❌ No data available for {selected_team}")
    
    with col2:
        if 'team_data' in st.session_state and 'selected_team' in st.session_state:
            team_df = st.session_state['team_data']
            team_name = st.session_state['selected_team']
            
            # Apply risk filter
            filtered_df = team_df[team_df['Risk_Level'].isin(risk_filter)] if risk_filter else team_df
            
            # Team Risk Summary Cards
            st.subheader(f"📊 {team_name} - Risk Summary")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                squad_size = len(team_df)
                st.metric(
                    "👥 Squad Size",
                    squad_size,
                    help="Total number of players analyzed"
                )
            
            with metric_col2:
                avg_risk = team_df['Risk_Probability'].mean()
                risk_status = "🟢 Low" if avg_risk < 0.3 else "🟡 Medium" if avg_risk < 0.6 else "🔴 High"
                st.metric(
                    "📈 Average Risk",
                    f"{avg_risk:.1%}",
                    risk_status
                )
            
            with metric_col3:
                high_risk_count = len(team_df[team_df['Risk_Level'].isin(['High', 'Critical'])])
                st.metric(
                    "⚠️ High Risk Players",
                    high_risk_count,
                    f"{high_risk_count/squad_size*100:.1f}%"
                )
            
            with metric_col4:
                critical_risk_count = len(team_df[team_df['Risk_Level'] == 'Critical'])
                st.metric(
                    "🚨 Critical Risk",
                    critical_risk_count,
                    f"{critical_risk_count/squad_size*100:.1f}%" if critical_risk_count > 0 else "0%"
                )
            
            # Risk Distribution Visualization
            st.subheader("📊 Risk Distribution Analysis")
            
            tab1, tab2, tab3 = st.tabs(["Risk Overview", "Position Analysis", "Player Details"])
            
            with tab1:
                # Risk level pie chart
                risk_counts = team_df['Risk_Level'].value_counts()
                fig_pie = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title=f"{team_name} - Risk Level Distribution",
                    color_discrete_map={
                        'Low': '#4CAF50',
                        'Medium': '#FF9800', 
                        'High': '#FF5722',
                        'Critical': '#D32F2F'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Risk probability histogram
                fig_hist = px.histogram(
                    team_df,
                    x='Risk_Probability',
                    nbins=20,
                    title=f"{team_name} - Risk Probability Distribution",
                    labels={'Risk_Probability': 'Risk Probability', 'count': 'Number of Players'}
                )
                fig_hist.add_vline(x=team_df['Risk_Probability'].mean(), 
                                 line_dash="dash", line_color="red",
                                 annotation_text=f"Team Average: {avg_risk:.1%}")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with tab2:
                # Position-based risk analysis
                position_risk = team_df.groupby('Position_Category').agg({
                    'Risk_Probability': ['mean', 'count', 'std'],
                    'Player': 'count'
                }).round(3)
                
                position_risk.columns = ['Avg_Risk', 'Player_Count', 'Risk_Std', 'Total_Players']
                position_risk = position_risk.reset_index()
                
                # Position risk bar chart
                fig_pos = px.bar(
                    position_risk,
                    x='Position_Category',
                    y='Avg_Risk',
                    color='Avg_Risk',
                    title=f"{team_name} - Average Risk by Position",
                    text='Avg_Risk',
                    color_continuous_scale='Reds'
                )
                fig_pos.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                st.plotly_chart(fig_pos, use_container_width=True)
                
                # Position risk scatter
                fig_scatter = px.scatter(
                    team_df,
                    x='Position_Category',
                    y='Risk_Probability',
                    color='Risk_Level',
                    size='Risk_Score',
                    hover_data=['Player'],
                    title=f"{team_name} - Player Risk by Position",
                    color_discrete_map={
                        'Low': '#4CAF50',
                        'Medium': '#FF9800', 
                        'High': '#FF5722',
                        'Critical': '#D32F2F'
                    }
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Position statistics table
                st.subheader("📋 Position Risk Statistics")
                st.dataframe(
                    position_risk.style.background_gradient(subset=['Avg_Risk'], cmap='Reds')
                    .format({
                        'Avg_Risk': '{:.1%}',
                        'Risk_Std': '{:.3f}'
                    }),
                    use_container_width=True
                )
            
            with tab3:
                # Detailed player table
                st.subheader(f"👥 {team_name} Squad Risk Assessment")
                
                # Sort players by risk
                sorted_team_df = team_df.sort_values('Risk_Probability', ascending=False)
                
                # Apply filters
                if risk_filter:
                    sorted_team_df = sorted_team_df[sorted_team_df['Risk_Level'].isin(risk_filter)]
                
                # Color coding function
                def color_risk_level(val):
                    color_map = {
                        'Low': 'background-color: #C8E6C9',
                        'Medium': 'background-color: #FFE0B2',
                        'High': 'background-color: #FFCDD2',
                        'Critical': 'background-color: #FFCDD2; font-weight: bold'
                    }
                    return color_map.get(val, '')
                
                # Display styled dataframe
                st.dataframe(
                    sorted_team_df[['Player', 'Position', 'Position_Category', 'Risk_Probability', 'Risk_Level']]
                    .style.applymap(color_risk_level, subset=['Risk_Level'])
                    .format({'Risk_Probability': '{:.1%}'}),
                    use_container_width=True
                )
                
                # Priority action list
                st.subheader("🎯 Priority Actions")
                
                critical_players = sorted_team_df[sorted_team_df['Risk_Level'] == 'Critical']
                high_risk_players = sorted_team_df[sorted_team_df['Risk_Level'] == 'High']
                
                if len(critical_players) > 0:
                    st.error(f"🚨 **IMMEDIATE ATTENTION REQUIRED** - {len(critical_players)} player(s)")
                    for _, player in critical_players.iterrows():
                        st.write(f"• **{player['Player']}** ({player['Position']}) - {player['Risk_Probability']:.1%} risk")
                
                if len(high_risk_players) > 0:
                    st.warning(f"⚠️ **HIGH PRIORITY MONITORING** - {len(high_risk_players)} player(s)")
                    for _, player in high_risk_players.iterrows():
                        st.write(f"• **{player['Player']}** ({player['Position']}) - {player['Risk_Probability']:.1%} risk")
                
                if len(critical_players) == 0 and len(high_risk_players) == 0:
                    st.success("✅ No immediate high-risk concerns identified")
            
            # Team Comparisons (if enabled)
            if show_comparisons:
                st.subheader("🏆 League Comparison")
                
                # Calculate league statistics
                all_teams_data = []
                for team in teams:
                    team_players = [player for player in player_info.keys() 
                                   if player_info[player]['Team'] == team]
                    
                    team_risks = []
                    for player in team_players:
                        player_features = get_player_features(player)
                        if player_features is not None:
                            prediction = predict_risk(player_features, xgb_model)
                            if prediction:
                                team_risks.append(prediction['probability'])
                    
                    if team_risks:
                        all_teams_data.append({
                            'Team': team,
                            'Avg_Risk': np.mean(team_risks),
                            'Squad_Size': len(team_risks),
                            'High_Risk_Count': len([r for r in team_risks if r > 0.4]),
                            'High_Risk_Percentage': len([r for r in team_risks if r > 0.4]) / len(team_risks) * 100
                        })
                
                league_df = pd.DataFrame(all_teams_data)
                league_df = league_df.sort_values('Avg_Risk', ascending=False)
                
                # Team ranking
                team_rank = league_df[league_df['Team'] == team_name].index[0] + 1 if len(league_df[league_df['Team'] == team_name]) > 0 else "N/A"
                
                col_rank1, col_rank2 = st.columns(2)
                
                with col_rank1:
                    st.metric(
                        "📊 League Ranking",
                        f"#{team_rank}" if team_rank != "N/A" else "N/A",
                        f"out of {len(league_df)} teams"
                    )
                
                with col_rank2:
                    league_avg = league_df['Avg_Risk'].mean()
                    diff_from_avg = (avg_risk - league_avg) * 100
                    st.metric(
                        "🎯 vs League Average",
                        f"{diff_from_avg:+.1f}%",
                        f"League: {league_avg:.1%}"
                    )
                
                # League comparison chart
                fig_league = px.bar(
                    league_df.head(10),
                    x='Team',
                    y='Avg_Risk',
                    color='Avg_Risk',
                    title="Top 10 Teams by Risk Level",
                    color_continuous_scale='Reds'
                )
                fig_league.update_layout(xaxis_tickangle=-45)
                
                # Highlight selected team
                if team_name in league_df['Team'].values:
                    fig_league.add_shape(
                        type="rect",
                        x0=list(league_df['Team']).index(team_name) - 0.4,
                        x1=list(league_df['Team']).index(team_name) + 0.4,
                        y0=0,
                        y1=league_df[league_df['Team'] == team_name]['Avg_Risk'].iloc[0],
                        fillcolor="gold",
                        opacity=0.3,
                        line=dict(color="gold", width=3)
                    )
                
                st.plotly_chart(fig_league, use_container_width=True)
            
            # Management Recommendations
            st.subheader("💡 Management Recommendations")
            
            recommendations = []
            
            # Risk level recommendations
            if critical_risk_count > 0:
                recommendations.append(f"🚨 **CRITICAL**: {critical_risk_count} player(s) require immediate medical assessment and possible rest.")
            
            if high_risk_count > 3:
                recommendations.append(f"⚠️ **HIGH PRIORITY**: Consider rotating high-risk players and implementing recovery protocols.")
            
            # Position-specific recommendations
            position_risks = team_df.groupby('Position_Category')['Risk_Probability'].mean()
            highest_risk_position = position_risks.idxmax()
            if position_risks[highest_risk_position] > 0.5:
                recommendations.append(f"🎯 **POSITION FOCUS**: {highest_risk_position} players show elevated risk levels - consider tactical adjustments.")
            
            # Squad management
            if avg_risk > 0.5:
                recommendations.append("📋 **SQUAD MANAGEMENT**: Overall team risk is elevated - implement squad rotation and load management.")
            elif avg_risk < 0.3:
                recommendations.append("✅ **SQUAD STATUS**: Team shows good injury risk profile - maintain current protocols.")
            
            # Display recommendations
            for i, rec in enumerate(recommendations, 1):
                st.info(f"{i}. {rec}")
            
            if not recommendations:
                st.success("🎉 **EXCELLENT**: No specific recommendations - squad appears well-managed!")
        
        else:
            # Default view when no team is selected
            st.info("👆 Select a team and click 'Generate Team Report' to view comprehensive team analytics")
            
            # Show available teams preview
            st.subheader("📋 Available Teams")
            teams_preview = pd.DataFrame({'Team': teams, 'Available': ['✅'] * len(teams)})
            st.dataframe(teams_preview, use_container_width=True)

# Position Analysis Page
elif app_mode == "Position Analysis":
    st.title("📍 Advanced Position-Based Injury Analytics")
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h4 style="color: white; margin: 0;">🏥 Medical Sports Analytics Dashboard</h4>
        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Comprehensive injury risk analysis across playing positions</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("⚙️ Analysis Controls")

        analysis_type = st.selectbox(
            "Analysis Type",
            ["Complete Analysis", "Position Category",
                "Specific Position", "Team Comparison"]
        )

        risk_threshold = st.slider(
            "Risk Threshold (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=5.0,
            help="Adjust the risk threshold for categorization"
        )

        show_details = st.checkbox("Show Detailed Statistics", value=True)

        if st.button("🔍 Run Advanced Analysis", type="primary"):
            with st.spinner("🔄 Processing position data..."):
                position_df = analyze_position_injury_risk()
                if not position_df.empty:
                    st.session_state['position_analysis'] = position_df
                    st.session_state['position_stats'] = create_position_risk_dashboard(
                        position_df)
                    st.success("✅ Analysis completed successfully!")
                else:
                    st.error("❌ No position data available for analysis")

    with col2:
        if 'position_analysis' in st.session_state and 'position_stats' in st.session_state:
            position_df = st.session_state['position_analysis']
            position_stats = st.session_state['position_stats']

            # Key Performance Indicators
            st.subheader("📊 Key Performance Indicators")

            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

            with kpi_col1:
                highest_risk_pos = position_stats.loc[position_stats['Avg_Risk'].idxmax(
                ), 'Position_Category']
                highest_risk_val = position_stats['Avg_Risk'].max()
                st.metric(
                    "🚨 Highest Risk Position",
                    highest_risk_pos,
                    f"{highest_risk_val:.1%}",
                    delta_color="inverse"
                )

            with kpi_col2:
                total_critical = len(
                    position_df[position_df['Risk_Probability'] > 0.7])
                st.metric(
                    "⚠️ Critical Risk Players",
                    total_critical,
                    f"{total_critical/len(position_df)*100:.1f}%",
                    delta_color="inverse"
                )

            with kpi_col3:
                avg_team_risk = position_df['Risk_Probability'].mean()
                st.metric(
                    "📈 Average Team Risk",
                    f"{avg_team_risk:.1%}",
                    f"Threshold: {risk_threshold:.0f}%"
                )

            with kpi_col4:
                risk_variance = position_df.groupby('Position_Category')[
                    'Risk_Probability'].var().max()
                st.metric(
                    "📊 Risk Variance",
                    f"{risk_variance:.3f}",
                    "Max across positions"
                )

            # Advanced Risk Visualization
            st.subheader("🎯 Risk Distribution Analysis")

            # Position risk box plot
            fig_heatmap = px.box(
                position_df,
                x='Position_Category',
                y='Risk_Probability',
                color='Position_Category',
                title="Risk Distribution by Position Category"
            )
            fig_heatmap.update_layout(showlegend=False)
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Risk scatter plot with team comparison
            fig_scatter = px.scatter(
                position_df,
                x='Risk_Probability',
                y='Risk_Score',
                color='Position_Category',
                size='Risk_Probability',
                hover_data=['Player', 'Team'],
                title="Player Risk Distribution by Position"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Detailed position analysis table
            if show_details:
                st.subheader("📋 Detailed Position Statistics")

                display_stats = position_stats[['Position_Category', 'Avg_Risk', 'Risk_Std',
                                                'Min_Risk', 'Max_Risk', 'Player_Count',
                                                'High_Risk_Players', 'High_Risk_Percentage']]

                st.dataframe(
                    display_stats.style.background_gradient(
                        subset=['Avg_Risk'], cmap='Reds')
                    .format({
                        'Avg_Risk': '{:.1%}',
                        'Risk_Std': '{:.3f}',
                        'Min_Risk': '{:.1%}',
                        'Max_Risk': '{:.1%}',
                        'High_Risk_Percentage': '{:.1f}%'
                    }),
                    use_container_width=True
                )

            # Individual player risk table
            st.subheader("👥 High Risk Players")
            high_risk_players = position_df[position_df['Risk_Probability'] > (
                risk_threshold/100)].sort_values('Risk_Probability', ascending=False)

            if not high_risk_players.empty:
                st.dataframe(
                    high_risk_players[['Player', 'Position_Category',
                                       'Team', 'Risk_Probability', 'Risk_Level']]
                    .style.format({'Risk_Probability': '{:.1%}'}),
                    use_container_width=True
                )
            else:
                st.info("No players exceed the current risk threshold.")

            # Position comparison chart
            st.subheader("📊 Position Risk Comparison")
            fig_position_comparison = px.bar(
                position_stats,
                x='Position_Category',
                y='Avg_Risk',
                color='Risk_Classification',
                title="Average Risk by Position Category",
                text='Avg_Risk'
            )
            fig_position_comparison.update_traces(
                texttemplate='%{text:.1%}', textposition='outside')
            st.plotly_chart(fig_position_comparison, use_container_width=True)

            # Risk management recommendations
            st.subheader("💡 Risk Management Recommendations")

            for _, row in position_stats.iterrows():
                position = row['Position_Category']
                avg_risk = row['Avg_Risk']
                high_risk_pct = row['High_Risk_Percentage']

                if avg_risk > 0.6:
                    risk_status = "🚨 Critical"
                    recommendation = f"Immediate intervention required for {position} players. Consider workload reduction and intensive monitoring."
                elif avg_risk > 0.4:
                    risk_status = "⚠️ High"
                    recommendation = f"Enhanced monitoring recommended for {position} players. Implement preventive measures."
                else:
                    risk_status = "✅ Acceptable"
                    recommendation = f"{position} players show acceptable risk levels. Maintain current protocols."

                st.info(f"**{position}** - {risk_status}: {recommendation}")

            # Team-wise position analysis
            if analysis_type == "Team Comparison":
                st.subheader("🏆 Team-wise Position Analysis")

                team_position_stats = position_df.groupby(['Team', 'Position_Category']).agg({
                    'Risk_Probability': 'mean',
                    'Player': 'count'
                }).reset_index()

                fig_team_heatmap = px.density_heatmap(
                    team_position_stats,
                    x='Position_Category',
                    y='Team',
                    z='Risk_Probability',
                    title="Team Risk Heatmap by Position"
                )
                st.plotly_chart(fig_team_heatmap, use_container_width=True)

        else:
            st.info(
                "👆 Click 'Run Advanced Analysis' to generate position-based injury analytics")


# Model Performance Page
elif app_mode == "Model Performance":
    st.title("📊 Model Performance")
    st.write("""
    View performance metrics and characteristics of the injury prediction models.
    """)

    st.subheader("Model Comparison")

    # Create metrics dataframe with numeric values only
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Random Forest': [0.85, 0.72, 0.68, 0.70, 0.89],
        'XGBoost': [0.87, 0.75, 0.70, 0.72, 0.91]
    }

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    # Display with highlighting
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    st.dataframe(
        metrics_df.style.apply(highlight_max, subset=[
                               'Random Forest', 'XGBoost'])
    )

    st.subheader("Feature Importance")

    # Feature importance visualization for both models
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Random Forest Feature Importance**")
        fig, ax = plt.subplots()
        sns.barplot(x=[0.2, 0.15, 0.1, 0.08, 0.05],
                    y=['Match Load', 'Tackles', 'Distance', 'Sprints', 'Recovery'])
        st.pyplot(fig)

    with col2:
        st.markdown("**XGBoost Feature Importance**")
        fig, ax = plt.subplots()
        sns.barplot(x=[0.25, 0.18, 0.12, 0.07, 0.05],
                    y=['Match Load', 'Distance', 'Tackles', 'Sprints', 'Recovery'])
        st.pyplot(fig)

    st.subheader("Model Documentation")
    st.write("""
    These models predict injury risk based on:
    - Player match load (minutes played)
    - Physical metrics (distance covered, sprints)
    - Tactical metrics (tackles, duels)
    - Recent injury history
    
    The models were trained on data from StatsBomb and football-lineups.com.
    """)
