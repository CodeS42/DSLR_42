import sys
import pandas as pd
import numpy as np

# z -> multi matri
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, theta):
    return sigmoid(X @ theta)

# m -> num samples
# y -> target class
def cost_function(X, y, theta):
    epsilon = 1e-15
    m = len(X)
    h = predict(X, theta)
    cost = (-1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

    return cost

def gradient_descent(learning_rate, theta, X, y):
    h = predict(X, theta)
    m = len(X)
    err = (h - y).reshape(-1)

    gradient = np.zeros((X.shape[1], 1))
    for i in range(X.shape[1]):
        gradient[i] = np.mean(X[:, i] * err)
    
    return theta - learning_rate * gradient


class logisticRegression:
    def __init__(self):
        self.theta = None
        self.feature_mean = {}
        self.feature_std = {}

    def standardize(self, data):
        standard = {}

        for name in data.columns:
            values = data[name]

            mean_value = np.mean(values)
            std_value = np.std(values)

            self.feature_mean[name] = mean_value
            self.feature_std[name] = std_value

            if std_value == 0:
                standard[name] = values - mean_value
            else:
                standard[name] = (values - mean_value) / std_value
        
        return pd.DataFrame(standard)

    def fit(self, features, target, learning_rate=0.1, iterations=10000):
        X = self.standardize(features)
        X.insert(0, "bias", 1)

        X = X.to_numpy()
        y = target.to_numpy().reshape(-1, 1)

        nb_features = X.shape[1]
        self.theta = np.zeros((nb_features, 1))
        cost_history = []

        for i in range(iterations):
            self.theta = gradient_descent(learning_rate, self.theta, X, y)
            if i % 200 == 0:
                cost_history.append(cost_function(X, y, self.theta).item())
        
        return cost_history

def main():
    if len(sys.argv) != 2:
        print("Error")
        return 1
    
    path = sys.argv[1]
    df = pd.read_csv(path).dropna()

    feature_names = (
        df.select_dtypes(include=np.number).drop(columns=["Index"]).columns
    )
    X_df = df[feature_names]
    target = df["Hogwarts House"]
    houses = [
        "Gryffindor",
        "Slytherin",
        "Hufflepuff",
        "Ravenclaw"
    ]
    final_theta = {}

    for house in houses:
        model = logisticRegression()
        binary_target = (target == house).astype(int)

        model.fit(
            X_df,
            binary_target,
            learning_rate=0.05,
            iterations=3000
        )
        final_theta[house] = model.theta.flatten()
    feature_labels = ["bias"] + list(feature_names)

    print("")
    for house in houses:
        print(f"\n=== {house} ===")
        weights = list(zip(feature_labels, final_theta[house]))
        weights.sort(key=lambda x:abs(x[1]), reverse=True)

        for feature, weight in weights:
            print(
                f"{feature:30} {weight:10.6f}"
                f"  abs ={abs(weight):12.6f}"
            )


if __name__ == "__main__":
    main()