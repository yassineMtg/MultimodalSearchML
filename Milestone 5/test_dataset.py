# # import pandas as pd

# # # Load your processed metadata
# # df = pd.read_parquet("data/processed/product_metadata.parquet")

# # # Check how many products have non-empty images
# # df["has_image"] = df["image_urls"].apply(lambda x: bool(x.strip()))

# # total = len(df)
# # with_image = df["has_image"].sum()
# # without_image = total - with_image

# # print(f"📊 Total products: {total}")
# # print(f"🖼️ Products with images: {with_image}")
# # print(f"🚫 Products without images: {without_image}")
# # print(f"✅ % Products with images: {with_image/total*100:.2f}%")


import pandas as pd
import requests
from tqdm import tqdm

# Load metadata
df = pd.read_parquet("data/processed/product_metadata.parquet")

# Only keep rows with non-empty image_urls
df = df[df["image_urls"].str.strip() != ""]

# Result list
valid_rows = []

# Helper function to check if any image URL is reachable
def is_image_reachable(image_urls):
    urls = image_urls.split(",")
    for url in urls:
        url = url.strip()
        try:
            response = requests.head(url, timeout=5)  # Faster than full GET
            if response.status_code == 200:
                return True
        except:
            continue
    return False

# Process each row with progress bar
for idx, row in tqdm(df.iterrows(), total=len(df), desc="🌐 Checking images"):
    if is_image_reachable(row["image_urls"]):
        valid_rows.append(row)

# Create new DataFrame
clean_df = pd.DataFrame(valid_rows)

# Save clean dataset
output_path = "data/processed/product_metadata_reachable.parquet"
clean_df.to_parquet(output_path, index=False)

print(f"✅ Clean dataset saved: {len(clean_df)} valid products to {output_path}")






#checking the rows of the dataset file

# import pandas as pd

# # Load the product metadata
# df = pd.read_parquet("data/processed/product_metadata.parquet")

# # Show columns
# print("🧾 Columns:", df.columns.tolist())

# # Show few sample rows
# print("\n🔍 Sample rows:\n", df.head())
