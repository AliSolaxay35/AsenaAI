from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
data = load_iris()
x = data.data  
y = data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model accuracy:", accuracy_score(y_test, y_pred))

# Get input from user
print("\nEnter the following features of a flower:")
sepal_length = float(input("Sepal length (cm): "))
sepal_width = float(input("Sepal width (cm): "))
petal_length = float(input("Petal length (cm): "))
petal_width = float(input("Petal width (cm): "))

# Create input sample
new_sample = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(new_sample)

# Show result
class_names = data.target_names
print("\nPredicted class number:", prediction[0])
print("Predicted class name:", class_names[prediction[0]])