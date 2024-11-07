import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

train_data = pd.read_csv("dataset/salary_train.csv")
test_data = pd.read_csv("dataset/salary_test.csv")

X_train = train_data.drop("salary", axis=1)
Y_train = train_data["salary"]

X_test = test_data.drop("salary", axis=1)
Y_test = test_data["salary"]

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

plt.scatter(X_test["years_of_experience"], Y_test)
plt.plot(X_test["years_of_experience"], Y_pred, color="red")
plt.title("Linear Regression of salary over years of experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
