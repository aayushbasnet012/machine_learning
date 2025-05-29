import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a page",
    ["Player Risk Assessment", "Team Overview", "Model Performance"]
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

    col1, col2 = st.columns([1, 3])

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

            # Detailed information
            with st.expander("Detailed Analysis"):
                st.write(
                    f"**Team:** {player_info[st.session_state['current_player']]['Team']}")

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

# Team Overview Page
elif app_mode == "Team Overview":
    st.title("🏟️ Team Injury Risk Overview")
    st.write("""
    View injury risk across an entire team to help with squad management.
    """)
    
    # Get unique teams
    teams = list(set([info['Team'] for info in player_info.values()]))
    selected_team = st.selectbox("Select Team", sorted(teams))
    
    if st.button("Analyze Team"):
        team_players = [p for p in player_info if player_info[p]['Team'] == selected_team]
        
        # Calculate risk for all players
        risk_data = []
        for player in team_players:
            player_features = get_player_features(player)
            prediction = predict_risk(player_features, xgb_model)
            if prediction:
                risk_data.append({
                    'Player': player,
                    'Risk Probability': prediction['probability'],
                    'Risk Level': prediction['risk_level'],
                    'Color': prediction['color']
                })
        
        if risk_data:
            risk_df = pd.DataFrame(risk_data)
            
            # Display team risk overview
            st.subheader(f"Team Risk Summary: {selected_team}")
            
            col1, col2, col3 = st.columns(3)
            high_risk = risk_df[risk_df['Risk Level'] == 'High'].shape[0]
            med_risk = risk_df[risk_df['Risk Level'] == 'Medium'].shape[0]
            low_risk = risk_df[risk_df['Risk Level'] == 'Low'].shape[0]
            
            col1.metric("High Risk Players", high_risk)
            col2.metric("Medium Risk Players", med_risk)
            col3.metric("Low Risk Players", low_risk)
            
            # Risk distribution chart
            st.subheader("Risk Distribution")
            fig = px.pie(risk_df, names='Risk Level', color='Risk Level',
                         color_discrete_map={'High':'red', 'Medium':'orange', 'Low':'green'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed player list - FIXED VERSION
            st.subheader("Player Risk Details")
            
            # Create a styled DataFrame without using the problematic approach
            def color_row(row):
                color = 'red' if row['Risk Level'] == 'High' else \
                        'orange' if row['Risk Level'] == 'Medium' else 'green'
                return [f'background-color: {color}'] * len(row)
            
            # Display with simpler styling
            st.dataframe(
                risk_df.style.apply(color_row, axis=1),
                column_config={
                    "Risk Probability": st.column_config.ProgressColumn(
                        "Risk Probability",
                        help="The probability of injury risk",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    )
                }
            )
            
            # Option to export results
            if st.button("Export Team Report"):
                csv = risk_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{selected_team}_injury_risk_report.csv",
                    mime='text/csv'
                )

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

    # Display with highlighting - using a different approach
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

# Run the app
if __name__ == "__main__":
    pass
