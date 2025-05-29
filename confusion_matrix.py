import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load tensors and player info
features_tensor = np.load("positional_all_players_tensor.npy")
heatmap_tensor = np.load("all_players_heatmap_tensor.npy")

with open("updated_player_info.json", "r", encoding="utf-8") as f:
    player_info = json.load(f)

# Define your existing supervised data preparation function


def prepare_supervised_data(heatmap_tensor, features_tensor, player_info):
    num_players = features_tensor.shape[0]
    num_matches = features_tensor.shape[1]
    num_features = features_tensor.shape[2]
    lookback = 3

    X, y, player_ids = [], [], []

    for player_idx in range(num_players):
        player_features = features_tensor[player_idx]
        for i in range(lookback, num_matches):
            window = player_features[i-lookback:i, :].flatten()
            target = player_features[i, -1]  # Assuming last column is injury
            X.append(window)
            y.append(target)
            player_ids.append(player_idx)

    return np.array(X), np.array(y), np.array(player_ids)


# Prepare the data
X, y, _ = prepare_supervised_data(heatmap_tensor, features_tensor, player_info)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=99)

# Load models
rf_model = joblib.load("random_forest_injury_model.pkl")
xgb_model = joblib.load("xgboost_injury_model.pkl")

# Predict
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Confusion matrices
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Random Forest Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title("XGBoost Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices_comparison.png")
plt.show()
