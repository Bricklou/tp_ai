import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree

train_data = pd.read_csv("dataset/fraud_train.csv")
test_data = pd.read_csv("dataset/fraud_test.csv")

X_train = train_data.drop("is_fraud", axis=1)
Y_train = train_data["is_fraud"]

X_test = test_data.drop("is_fraud", axis=1)
Y_test = test_data["is_fraud"]

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

plt.figure(figsize=(50, 50))
plot_tree(
    model,
    feature_names=X_train.columns,
    class_names=["Not fraud", "Fraud"],
    filled=True,
    rounded=True,
)
plt.title("Decision Tree for Fraud Detection")
plt.show()

# Plot confusion matrix
confusion_matrix = confusion_matrix(Y_test, Y_pred)
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()
