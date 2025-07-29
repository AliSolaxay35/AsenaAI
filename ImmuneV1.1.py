import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime

# Load dataset
data = pd.read_csv("C:/Users/AliSolaxay/OneDrive/Desktop/AITest/immune_data.csv", header=None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model accuracy:", accuracy_score(y_test, y_pred))

# Get input from user
print("\nEnter patient information:")
name = input("Patient name:")
age = float(input("age: "))
wbc = float(input("White blood cell count: "))
antibody = float(input("Antibody level: "))

# Predict
sample = [[age, wbc, antibody]]
prediction = model.predict(sample)[0]
print("Disease prediction (1 = sick, 0 = healthy):", prediction)

# Prepare log entry
log_data = {
    "timestamp": [datetime.now()],
    "name": [name],
    "age": [age],
    "wbc": [wbc],
    "antibody": [antibody],
    "prediction": [prediction]
}
df_log = pd.DataFrame(log_data)

# Save log
log_file =  "C:/Users/AliSolaxay/OneDrive/Desktop/AITest/prediction_log.csv"
try:
    df_log.to_csv(log_file, mode='a', header=False, index=False)
except:
    df_log.to_csv(log_file, index=False)
    
print("Prediction saved to prediction_log.csv")
