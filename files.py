import numpy as np

# Load the original tensors
features_tensor = np.load('positional_all_players_tensor.npy')

# Assume last column is injury_status (1 if injured, 0 otherwise)
# Use 3-match sliding window like your model

lookback = 3
X = []
y = []

num_players, num_matches, num_features = features_tensor.shape

for player_idx in range(num_players):
    player_data = features_tensor[player_idx]
    for i in range(lookback, num_matches):
        # Extract sliding window features
        window = player_data[i - lookback:i].flatten()
        # Injury status in the match right after the window
        injury_status = player_data[i, -1]
        X.append(window)
        y.append(injury_status)

X = np.array(X)
y = np.array(y)

# Save them
X_path = "X_with_heatmap.npy"
y_path = "y.npy"
np.save(X_path, X)
np.save(y_path, y)

X.shape, y.shape
