import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

raw_data = pd.read_csv("dataset/customer_train.csv")

scaler = StandardScaler()
features_scaled = scaler.fit_transform(raw_data)

data = pd.DataFrame(features_scaled, columns=raw_data.columns)

k = 4
model = KMeans(n_clusters=k, random_state=42)
model.fit(raw_data)

data["cluster"] = model.fit_predict(data)

plt.figure(figsize=(15, 5))

# First subplot
plt.subplot(1, 3, 1)
plt.scatter(data["annual_income"], data["spending_score"], c=data["cluster"])
plt.xlabel("Annual income")
plt.ylabel("Spending Score")
plt.title("Income VS Spending Score")

plt.subplot(1, 3, 2)
plt.scatter(data["age"], data["spending_score"], c=data["cluster"])
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Age VS Spending Score")

plt.subplot(1, 3, 3)
plt.scatter(data["age"], data["annual_income"], c=data["cluster"])
plt.xlabel("Age")
plt.ylabel("Annual income")
plt.title("Age VS Annual Income")

plt.tight_layout()
plt.show()
