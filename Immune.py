import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset from absolute path
data = pd.read_csv("C:/Users/AliSolaxay/OneDrive/Desktop/AITest/immune_data.csv", header=None)

# Separate features and labels
X = data.iloc[:, :-1]  # All columns except last
y = data.iloc[:, -1]   # Last column is the label

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model accuracy:", accuracy_score(y_test, y_pred))

# Predict a new sample
sample = [[40, 9500, 2.0]]  # Example: [age, white blood cell count, antibody level]
prediction = model.predict(sample)
print("Disease prediction (1 = sick, 0 = healthy):", prediction[0])
