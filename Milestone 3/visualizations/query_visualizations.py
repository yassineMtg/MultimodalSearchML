import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw"))
REPORTING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "./reporting"))

query_features_path = os.path.join(BASE_DIR, "query_features_with_timestamp.parquet")
img1_path = os.path.join(REPORTING_DIR, "query_features_histograms.png")
img2_path = os.path.join(REPORTING_DIR, "query_features_corr_heatmap.png")
img3_path = os.path.join(REPORTING_DIR, "query_features_pca.png")
img4_path = os.path.join(REPORTING_DIR, "query_embedding_norm_distribution.png")

# Load dataset
df = pd.read_parquet(query_features_path)

# --- 1. Histogram of first 20 features ---
df.iloc[:, 1:21].hist(bins=30, figsize=(20, 15))
plt.suptitle("Distribution of Query Features (f0 to f19)")
plt.tight_layout()
plt.savefig(img1_path)
plt.show()

# --- 2. Correlation Heatmap ---
plt.figure(figsize=(15, 12))
corr = df.iloc[:, 1:21].corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap of Query Features (f0 to f19)")
plt.savefig(img2_path)
plt.show()

# --- 3. PCA Visualization ---

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df.iloc[:, 1:-1])

plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.title("PCA of Query Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.savefig(img3_path)
plt.grid(True)
plt.show()

# --- 4. Embedding Norm Distribution ---
embedding_columns = [col for col in df.columns if col.startswith('f')]
norms = np.linalg.norm(df[embedding_columns].values, axis=1)
plt.figure(figsize=(10, 6))
sns.histplot(norms, bins=50, kde=True)
plt.title("Distribution of Query Embedding Norms")
plt.xlabel("L2 Norm")
plt.savefig(img4_path)
plt.show()
