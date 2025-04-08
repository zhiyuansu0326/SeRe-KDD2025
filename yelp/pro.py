import json
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn import preprocessing

# File paths and output directory
business_file = "yelp_academic_dataset_business.json"
review_file = "yelp_academic_dataset_review.json"
output_dir = "./processed_yelp"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

############################################################
# 1. Read JSON files and construct big_matrix
############################################################
# Process business data
business_rows = []
categories_to_keep = [
    "Restaurants",
    "Food",
    "Bars",
]  # Only focus on relevant categories
min_reviews = 50  # Minimum review count to include businesses

with open(business_file, "r", encoding="utf-8") as f:
    for line in f:
        business_data = json.loads(line.strip())
        if (
            business_data.get("categories")
            and any(cat in business_data["categories"] for cat in categories_to_keep)
            and business_data.get("review_count", 0) >= min_reviews
        ):
            business_rows.append(
                {
                    "business_id": business_data["business_id"],
                    "categories": business_data["categories"],
                    "stars": business_data["stars"],
                    "review_count": business_data["review_count"],
                }
            )

business_df = pd.DataFrame(business_rows)
print("Filtered businesses:", len(business_df))

# Process review data
# Process review data
review_rows = []

with open(review_file, "r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
        try:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            review_data = json.loads(line)
            review_rows.append(
                {
                    "user_id": review_data["user_id"],
                    "business_id": review_data["business_id"],
                    "stars": review_data["stars"],
                    "text": review_data["text"],
                    "date": review_data["date"],
                }
            )
        except json.JSONDecodeError:
            print(f"Skipping invalid line {line_number}: {line}")

review_df = pd.DataFrame(review_rows)
review_df["label"] = (review_df["stars"] >= 4).astype(int)  # 4+ stars = positive label
print("Total reviews after filtering invalid lines:", len(review_df))

# Merge filtered businesses with reviews
merged_df = review_df.merge(business_df, on="business_id", how="inner")

# Convert date to an integer format
merged_df["date"] = pd.to_datetime(merged_df["date"])
merged_df["date_int"] = merged_df["date"].dt.strftime("%Y%m%d").astype(int)

# Define periods based on the median date
median_date = merged_df["date_int"].median()
merged_df["period"] = merged_df["date_int"].apply(
    lambda x: 1 if x <= median_date else 2
)

############################################################
# 2. Filter users/items appearing in both periods
############################################################
user_period_count = (
    merged_df.groupby(["user_id", "period"]).size().reset_index(name="cnt")
)
active_users = user_period_count.groupby("user_id")["period"].nunique()
active_users = active_users[active_users == 2].index.tolist()

item_period_count = (
    merged_df.groupby(["business_id", "period"]).size().reset_index(name="cnt")
)
active_items = item_period_count.groupby("business_id")["period"].nunique()
active_items = active_items[active_items == 2].index.tolist()

filtered_df = merged_df[
    (merged_df["user_id"].isin(active_users))
    & (merged_df["business_id"].isin(active_items))
].copy()
filtered_df.sort_values(by="date_int", ascending=True, inplace=True)
print("Filtered interactions:", len(filtered_df))

############################################################
# 3. Construct User and Business Features
############################################################
# User Features
user_text_dict = defaultdict(list)
for _, row in filtered_df.iterrows():
    user_text_dict[row["user_id"]].append(str(row["text"]))

user_docs = {u: " ".join(texts) for u, texts in user_text_dict.items()}
user_ids = list(user_docs.keys())
user_texts = list(user_docs.values())

user_tfidf = TfidfVectorizer(min_df=2, max_features=1000).fit_transform(user_texts)
user_pca = PCA(n_components=25).fit_transform(user_tfidf.toarray())
user_features_df = pd.DataFrame(user_pca, columns=[f"u{i}" for i in range(25)])
user_features_df["user_id"] = user_ids

# Business Features
business_text_dict = defaultdict(list)
for _, row in filtered_df.iterrows():
    business_text_dict[row["business_id"]].append(str(row["categories"]))

business_docs = {b: " ".join(texts) for b, texts in business_text_dict.items()}
business_ids = list(business_docs.keys())
business_texts = list(business_docs.values())

business_tfidf = TfidfVectorizer(min_df=2, max_features=1000).fit_transform(
    business_texts
)
business_pca = PCA(n_components=25).fit_transform(business_tfidf.toarray())
business_features_df = pd.DataFrame(business_pca, columns=[f"v{i}" for i in range(25)])
business_features_df["business_id"] = business_ids

############################################################
# 4. Label Encode User and Business IDs
############################################################
user_le = preprocessing.LabelEncoder()
item_le = preprocessing.LabelEncoder()

filtered_df["user_id_le"] = user_le.fit_transform(filtered_df["user_id"])
filtered_df["business_id_le"] = item_le.fit_transform(filtered_df["business_id"])

user_features_df["user_id_le"] = user_le.transform(user_features_df["user_id"])
business_features_df["business_id_le"] = item_le.transform(
    business_features_df["business_id"]
)

############################################################
# 5. Save Processed Data
############################################################
# Full Data
full_data = filtered_df[
    ["user_id_le", "business_id_le", "date_int", "period", "label"]
].copy()
full_data.rename(
    columns={"user_id_le": "user_id", "business_id_le": "item_id"}, inplace=True
)
full_data.to_csv(os.path.join(output_dir, "full_data.csv"), sep="\t", index=False)

# User Features
user_features_df.rename(columns={"user_id_le": "user_id"}, inplace=True)
user_features_df.to_csv(
    os.path.join(output_dir, "user_feature_pca.csv"), sep="\t", index=False
)

# Business Features
business_features_df.rename(columns={"business_id_le": "item_id"}, inplace=True)
business_features_df.to_csv(
    os.path.join(output_dir, "item_feature_pca.csv"), sep="\t", index=False
)

print("Processing complete. Files saved in:", output_dir)
