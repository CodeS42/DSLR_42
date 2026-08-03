import sys
import pandas as pd
import numpy as np
import json as js
import matplotlib.pyplot as plt

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

        self.X_train = X
        self.Y_train = y

        nb_features = X.shape[1]
        self.theta = np.zeros((nb_features, 1))
        cost_history = []

        for i in range(iterations):
            self.theta = gradient_descent(learning_rate, self.theta, X, y)
            if i % 200 == 0:
                cost_history.append(cost_function(X, y, self.theta).item())
                # cost = cost_function(X, y, self.theta).item()
                # print(f"iteration {i:5d} | cost = {cost:.6f}")
                # cost_history.append(cost)
        
        return cost_history


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Wrong number of arguments.")
    
    path = sys.argv[1]
    df = pd.read_csv(path).dropna()

    # feature_names = (
    #     df.select_dtypes(include=np.number).drop(columns=["Index"]).columns
    # )
    all_features = df.select_dtypes(include=np.number).drop(columns=["Index"]).columns
    drop_features = ["Arithmancy", "Potions", "Transfiguration", "Care of Magical Creatures"]
    selected_features = all_features.drop(drop_features)
    X_df = df[selected_features]

    target = df["Hogwarts House"]
    houses = [
        "Gryffindor",
        "Slytherin",
        "Hufflepuff",
        "Ravenclaw"
    ]

    final_theta = {}
    model_stat = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, house in enumerate(houses):
        model = logisticRegression()
        binary_target = (target == house).astype(int) # one vs all

        history_cost = model.fit(
            X_df,
            binary_target,
            learning_rate=0.05,
            iterations=3000
        )
        final_theta[house] = model.theta.flatten().tolist()
        if not model_stat:
            model_stat['feature_mean'] = model.feature_mean
            model_stat['feature_std'] = model.feature_std
        
        axes[i].plot(
            range(0, len(history_cost) * 300, 300),
            history_cost
        )
        axes[i].set_title(f"cost function - {house}")
        axes[i].set_xlabel("iterations")
        axes[i].set_ylabel("cost")
        axes[i].grid(True)

    # feature_labels = ["bias"] + list(feature_names)
    # print("")
    # for house in houses:
    #     print(f"\n=== {house} ===")
    #     weights = list(zip(feature_labels, final_theta[house]))
    #     weights.sort(key=lambda x:abs(x[1]), reverse=True)

    #     for feature, weight in weights:
    #         print(
    #             f"{feature:30} {weight:10.6f}"
    #             f"  abs ={abs(weight):12.6f}"
    #         )
    
    # save
    model_data = {
        'thetas': final_theta,
        'feature_mean': model_stat['feature_mean'],
        'feature_std': model_stat['feature_std']
    }
    with open("thetas.json", "w") as file:
        js.dump(model_data, file, indent=2)
    plt.show()


if __name__ == "__main__":
    main()