# ==============================================================
# Machine Learning–Based Crop Recommendation and Scheduling System
# for Automated Agricultural Planning in Sri Lanka
# ==============================================================

# === Import Required Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# === Load Dataset ===
data_path = r"C:\Users\Sanjali\Downloads\SriLanka_Crop.csv"
df = pd.read_csv(data_path)

print("✅ Dataset Loaded Successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# === Data Preprocessing ===
# Drop duplicates
df = df.drop_duplicates()

# Fill missing values (numerical with mean, categorical with mode)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\n✅ Data Preprocessing Completed")

# === Features and Target Selection ===
# Replace 'Crop' with the actual column name of your target variable
target_col = 'Suitable_Crop'

X = df.drop(columns=[target_col])
y = df[target_col]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split Dataset ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==============================================================
#                     CLASSIFICATION MODELS
# ==============================================================

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"\n=== {name} Results ===")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()

# === Accuracy Comparison Chart ===
plt.figure(figsize=(6, 4))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("Model_Accuracy_Comparison.png")
plt.close()

# === Feature Importance (Random Forest) ===
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 5))
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), np.array(X.columns)[indices], rotation=90)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("Feature_Importance_RF.png")
plt.close()

print("\n✅ Classification completed and all charts saved as .png")

# ==============================================================
#                           CLUSTERING
# ==============================================================

# Determine optimal number of clusters using Elbow Method
wcss = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(2, 10), wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.tight_layout()
plt.savefig("Elbow_Method.png")
plt.close()

# Fit KMeans with optimal K (e.g., 4 — adjust after viewing elbow plot)
k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# Cluster Summary
cluster_summary = df.groupby("Cluster").mean()
print("\n=== Cluster Summary ===")
print(cluster_summary)

# Visualize clusters (using two key features)
if X.shape[1] >= 2:
    plt.figure(figsize=(6, 5))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
    plt.title("KMeans Clustering Visualization")
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.tight_layout()
    plt.savefig("KMeans_Clusters.png")
    plt.close()

print("\n✅ Clustering completed and all cluster charts saved as .png")

# ==============================================================
#                        FINAL SUMMARIES
# ==============================================================

print("\n=== Final Model Accuracies ===")
for model_name, acc in results.items():
    print(f"{model_name}: {acc:.4f}")

print("\n=== Cluster Means by Feature ===")
print(cluster_summary)

print("\n🎯 All charts saved in your working directory as .png files")
print("   → Confusion matrices for each model")
print("   → Model_Accuracy_Comparison.png")
print("   → Feature_Importance_RF.png")
print("   → Elbow_Method.png")
print("   → KMeans_Clusters.png")