import matplotlib.pyplot as plt
import numpy as np

# Simulated top 20 features and importances (replace with actual data if available)
top_features = [f"Feature_{i}" for i in range(1, 21)]
xgb_importances = np.random.uniform(0.01, 0.1, size=20)
rf_importances = np.random.uniform(0.01, 0.1, size=20)

# Sorting for better visualization
xgb_sorted_idx = np.argsort(xgb_importances)
rf_sorted_idx = np.argsort(rf_importances)

# Create XGBoost feature importance plot
plt.figure(figsize=(10, 6))
plt.barh(np.array(top_features)[xgb_sorted_idx],
         xgb_importances[xgb_sorted_idx])
plt.title("Top 20 Feature Importances - XGBoost")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("xg_boost.png")

plt.close()

# Create Random Forest feature importance plot
plt.figure(figsize=(10, 6))
plt.barh(np.array(top_features)[rf_sorted_idx], rf_importances[rf_sorted_idx])
plt.title("Top 20 Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("randomforest.png")
plt.close()
