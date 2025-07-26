from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


data = load_iris()
x = data.data   
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Accuracy of the model", accuracy_score(y_test, y_pred))

new_sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_sample)

class_names = data.target_names  
print("Predicted class number:", prediction[0])
print("Predicted class name:", class_names[prediction[0]])