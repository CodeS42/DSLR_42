import matplotlib.pyplot as plt
import pandas as pd
import sys
import math

house_colors = {
    "Gryffindor": "red",
    "Slytherin": "green",
    "Hufflepuff": "yellow",
    "Ravenclaw": "blue"
}


def ft_corr(x, y):
    valid_pair = [(x_val, y_val) for x_val, y_val in zip(x, y) if pd.notna(x_val) and pd.notna(y_val)]
    pairs = len(valid_pair)
    if pairs == 0:
        return 0
    
    x_valid, y_valid = zip(*valid_pair)
    mean_x = sum(x_valid) / pairs
    mean_y = sum(y_valid) / pairs

    num = sum((x_val - mean_x) * (y_val - mean_y) for x_val, y_val in zip(x_valid, y_valid)) # covariance
    variance_x = sum((x_val - mean_x) ** 2 for x_val in x_valid)
    variance_y = sum((y_val - mean_y) ** 2 for y_val in y_valid)
    den = math.sqrt(variance_x * variance_y)
    if den == 0:
        return 0
    
    return num / den


def display_scatter_plot(df):
    label_col = "Hogwarts House"
    num_col = df.select_dtypes(include=["float", "int"]).columns.tolist()
    if "Index" in num_col:
        num_col.remove("Index")

    # corr_max = 0
    corr_max = -1
    feature_corr = None
    for i in range(len(num_col)):
        for j in range(i + 1, len(num_col)):
            f1, f2 = num_col[i], num_col[j]
            corr_val = ft_corr(df[f1], df[f2])
            # if abs(corr_val) > corr_max:
            if corr_val > corr_max:
                corr_max = abs(corr_val)
                feature_corr = (f1, f2)
            # print(f"{f1} vs {f2} | corr: {corr_val:0.5f}")

    plt.figure(figsize=(10, 8))
    for house in df[label_col].unique():
        data_class = df[df[label_col] == house]
        plt.scatter(data_class[feature_corr[0]],
                    data_class[feature_corr[1]],
                    color=house_colors[house],
                    label=house, alpha=0.5)

    plt.xlabel(feature_corr[0])
    plt.ylabel(feature_corr[1])
    plt.title(f"Similar: {feature_corr[0]} vs {feature_corr[1]}")
    plt.legend()
    plt.show()


def main():
    try:
        if not len(sys.argv) == 1:
            raise SystemExit("Wrong number of arguments.")

        df = pd.read_csv("dataset_train.csv")
        display_scatter_plot(df)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
