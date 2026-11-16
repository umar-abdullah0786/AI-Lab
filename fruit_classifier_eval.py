import numpy as np

probs = np.array([[0.7, 0.2, 0.1],
                  [0.1, 0.6, 0.3],
                  [0.2, 0.2, 0.6],
                  [0.5, 0.3, 0.2],
                  [0.1, 0.3, 0.6]])
y_test = np.array([0, 1, 2, 1, 2])

# Predicted class = most probable class in each row
y_pred = np.argmax(probs, axis=1)
print("Predicted:", y_pred)

# Accuracy by hand: True counts as 1, False as 0
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# MSE of the weight-prediction model
w_true = np.array([150.0, 120.0, 200.0, 180.0])
w_pred = np.array([140.0, 130.0, 210.0, 170.0])
mse = np.mean((w_true - w_pred) ** 2)
print("MSE:", mse)

# (added in Git step b) show which samples were misclassified
wrong = y_pred != y_test
print("Wrongly predicted sample indices:", np.where(wrong)[0])