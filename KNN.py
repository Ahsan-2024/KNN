import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X = np.array([
    [25, 50000, 2],
    [30, 80000, 1],
    [35, 60000, 3],
    [20, 30000, 2],
    [40, 90000, 1],
    [45, 75000, 2]
])

labeling = np.array([1, 2, 1, 0, 2, 1])

x_train, x_test, y_train, y_test = train_test_split(X, labeling, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

age = float(input("Enter your age: "))
salary = float(input("Enter your salary: "))
items = float(input("Enter number of items you buy: "))

user_input = np.array([[age, salary, items]])
user_input_scaled = scaler.transform(user_input)

prediction = model.predict(user_input_scaled)[0]

categories = ['Low', 'Medium', 'High']
print(f"Predicted category: {categories[prediction]}")