import sys
import pandas as pd
import numpy as np
import json as js
import matplotlib.pyplot as plt


def ft_mean(values):
    """
    Calculate the arithmetic mean of an iterable of values.
    """
    res = 0.0
    count = 0

    for value in values:
        res += value
        count += 1

    if count == 0:
        return 0

    return res / count


def ft_std(values, mean):
    """
    Calculate the standard deviation of a precomputed mean.
    """
    res = 0.0
    count = 0

    for value in values:
        diff = value - mean
        res += diff * diff
        count += 1

    if count == 0:
        return 0
    variance = res / count

    return np.sqrt(variance)


def ft_clip(value, low, high):
    """
    Clip a value to stay within low and high.
    """
    if value < low:
        return low
    elif value > high:
        return high
    else:
        return value


def sigmoid(z):
    """
    Compute the sigmoid activation function for a given input.
    Returns values constrained between 0 and 1.
    """
    return 1 / (1 + np.exp(-z))


def predict(X, theta):
    """
    Compute predicted probabilities using logistic regression weights theta.
    """
    return sigmoid(X @ theta)


def cost_function(X, y, theta):
    """
    Compute the logistic regression cost,
    also called log loss for given weights theta.
    """
    epsilon = 1e-15
    m = X.shape[0]
    n_features = X.shape[1]
    total = 0.0

    for i in range(m):
        z = 0.0
        for j in range(n_features):
            z += theta[j][0] * X[i][j]

        h = 1.0 / (1.0 + np.exp(-z))
        safe_h = ft_clip(h, epsilon, 1 - epsilon)

        y_i = y[i][0]
        total += y_i * np.log(safe_h) + (1 - y_i) * np.log(1 - safe_h)

    return -total / m


def gradient_descent(learning_rate, theta, X, y):
    """
    Perform gradient descent step to reduce the cost,
    and return the updated theta.
    """
    h = predict(X, theta)
    err = (h - y).reshape(-1)

    gradient = np.zeros((X.shape[1], 1))
    for i in range(X.shape[1]):
        gradient[i] = ft_mean(X[:, i] * err)

    return theta - learning_rate * gradient


class logisticRegression:
    """
    Class for logistic regression classifier.
    """
    def __init__(self):
        """
        Initialize model weights and feature scaling parameters.
        """
        self.theta = None
        self.feature_mean = {}
        self.feature_std = {}

    def standardize(self, data):
        """
        Standardize features and store parameters.
        """
        standard = {}

        for name in data.columns:
            values = data[name]

            mean_value = ft_mean(values)
            std_value = ft_std(values, mean_value)

            self.feature_mean[name] = mean_value
            self.feature_std[name] = std_value

            if std_value == 0:
                standard[name] = values - mean_value
            else:
                standard[name] = (values - mean_value) / std_value

        return pd.DataFrame(standard)

    def fit(self, features, target, learning_rate=0.1, iterations=10000):
        """
        Train the model using gradient descent and return the cost history.
        """
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
    """
    Train one vs all logistic regression models per house,
    and save thetas to json file.
    """
    try:
        if len(sys.argv) != 2:
            raise SystemExit("Usage: python logreg_train.py dataset_train.csv")

        path = sys.argv[1]
        df = pd.read_csv(path).dropna()

        all_features = df.select_dtypes(
            include=np.number).drop(columns=["Index"]).columns
        drop_features = ["Arithmancy", "Potions",
                         "Transfiguration", "Care of Magical Creatures",
                         "Astronomy"]
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
            binary_target = (target == house).astype(int)

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
                range(0, len(history_cost) * 200, 200),
                history_cost
            )
            axes[i].set_title(f"cost function - {house}")
            axes[i].set_xlabel("iterations")
            axes[i].set_ylabel("cost")
            axes[i].grid(True)

        model_data = {
            'thetas': final_theta,
            'feature_mean': model_stat['feature_mean'],
            'feature_std': model_stat['feature_std']
        }
        with open("thetas.json", "w") as file:
            js.dump(model_data, file, indent=2)
        plt.show()

    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
