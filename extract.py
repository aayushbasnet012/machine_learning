import ace_tools_open as tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from xgboost import XGBClassifier
import seaborn as sns

# Simulate loading of heatmap tensor and label data
# Let's assume 1000 players, 38 matches, and 24x24 heatmaps (flattened to 576)
num_players = 1000
num_matches = 38
heatmap_size = 24 * 24

# Simulate data: aggregate the last 3 matches' heatmaps
np.random.seed(42)
heatmaps = np.random.rand(num_players, num_matches, heatmap_size)
labels = np.random.randint(0, 2, size=num_players)

# Extract features: mean, std, and max of last 3 matches' heatmaps
X = []
for player_heatmaps in heatmaps:
    last_3 = player_heatmaps[-3:]  # Last 3 matches
    mean_vals = np.mean(last_3, axis=0)
    std_vals = np.std(last_3, axis=0)
    max_vals = np.max(last_3, axis=0)
    features = np.concatenate([mean_vals, std_vals, max_vals])
    X.append(features)

X = np.array(X)
y = labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
                              "Not Injured", "Injured"])
disp.plot(ax=ax, cmap="Blues", values_format='d')
plt.title("Improved Confusion Matrix from Heatmap-based Features")
plt.tight_layout()
plt.show()

# Classification report
report = classification_report(y_test, y_pred, output_dict=True)

tools.display_dataframe_to_user(
    name="Classification Report", dataframe=pd.DataFrame(report).transpose())
