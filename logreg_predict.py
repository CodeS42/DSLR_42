import sys
import json as js
import pandas as pd
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict():
    df_test = pd.read_csv(sys.argv[1])
    with open(sys.argv[2]) as f:
        model = js.load(f)

    feature_means = model["feature_mean"]
    feature_stds = model["feature_std"]
    thetas = model["thetas"]
    df_notes = pd.DataFrame()

    for course_name in feature_means.keys():
        mean = feature_means[course_name]
        std = feature_stds[course_name]

        notes = df_test[course_name].fillna(mean)
        if std == 0:
            df_notes[course_name] = (notes - mean)
        else:
            df_notes[course_name] = (notes - mean) / std

    df_notes.insert(0, "bias", 1)
    probabilities = {}

    for house, theta_list in thetas.items():
        z = np.dot(df_notes, theta_list)
        probabilities[house] = sigmoid(z)

    df_probabilities = pd.DataFrame(probabilities)
    df_test["Hogwarts House"] = df_probabilities.idxmax(axis=1)
    df_test[["Index", "Hogwarts House"]].to_csv("houses.csv", index=False)


def main():
    try:
        if len(sys.argv) != 3:
            raise SystemExit(
                "Usage: python logreg_predict.py dataset_test.csv thetas.json")
        predict()
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
