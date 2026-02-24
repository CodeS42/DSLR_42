import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


house_colors = {
    "Gryffindor": "red",
    "Slytherin": "green",
    "Hufflepuff": "yellow",
    "Ravenclaw": "blue"
}


def display_pair_plot(df):
    label_col = "Hogwarts House"

    col_drop = ["First name", "Last name", "Birthday", "Best Hand"]
    df = df.drop(columns=col_drop, errors="ignore")
    df = df.dropna()

    houses = ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]
    features = df.select_dtypes(include=["float", "int"]).columns

    fig, axs = plt.subplots(len(features), len(features), figsize=(16, 20))
    fig.suptitle("Pair Plot", fontsize=20)

    for i in range(len(features)):
        for j in range(len(features)):
            if i == j:
                for house in houses:
                    data = df[df[label_col] == house]
                    axs[i, j].hist(data[features[i]],
                                   label=house,
                                   alpha=0.3,
                                   color=house_colors[house])
            else:
                for house in houses:
                    data = df[df[label_col] == house]
                    axs[i, j].scatter(
                        data[features[i]],
                        data[features[j]],
                        label=house,
                        alpha=0.3,
                        color=house_colors[house],
                        s=2
                    )
            if j == 0:
                axs[i, j].set_ylabel(features[i], fontsize=8)
            else:
                axs[i, j].set_yticklabels([])
            if i != len(features) - 1:
                axs[i, j].set_xticklabels([])
            else:
                axs[i, j].set_xlabel(features[j], fontsize=8)

    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right")
    plt.show()


def main():
    try:
        if not len(sys.argv) == 1:
            raise SystemExit("Wrong number of arguments.")

        df = pd.read_csv("dataset_train.csv", index_col="Index")
        display_pair_plot(df)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
