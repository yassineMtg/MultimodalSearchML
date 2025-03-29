import pandas as pd
import matplotlib.pyplot as plt

# Load the features
df = pd.read_csv("reporting/online_features_sample.csv")

# Basic statistics
print(df.describe())

# Plot distributions
for col in df.columns[1:]:  # skip product_id
    plt.figure()
    df[col].hist(bins=50)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.savefig(f"reporting/{col}_distribution.png")
    print(f"✅ Saved {col}_distribution.png")

print("✅ All plots saved inside reporting/")
