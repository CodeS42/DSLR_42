import pandas as pd
import sys


Q1 = 1
Q2 = 2
Q3 = 3


def ft_count(df, cols):
    count_lst = ["Count"]
    for col in range(cols):
        count = 0
        for nb in df.iloc[:, col]:
            if not pd.isna(nb):
                count += 1
        count_lst.append(count)
    return count_lst


def ft_mean(df, count_lst, cols):
    mean_lst = ["Mean"]
    for col, nb_values in zip(range(cols), count_lst):
        result = sum(nb for nb in df.iloc[:, col] if not pd.isna(nb)) / nb_values
        mean_lst.append(result)
    return mean_lst


def valid_numbers(df, col):
    nb_lst = []
    for nb in df.iloc[:, col]:
        if not pd.isna(nb):
            nb_lst.append(nb)
    return nb_lst


def variance(df, mean_lst, cols):
    var_lst = []
    for col, mean in zip(range(cols), mean_lst):
        nb_lst = valid_numbers(df, col)
        var = sum([(nb - mean) ** 2 for nb in nb_lst]) / len(nb_lst)
        var_lst.append(var)
    return var_lst


def ft_std(df, mean_lst, cols):
    var = variance(df, mean_lst, cols)
    std_lst = ["Std"]
    for v in var:
        std_lst.append(v ** 0.5)
    return std_lst


def ft_min(df, cols):
    min_lst = ["Min"]
    for col in range(cols):
        i = 0
        while pd.isna(df.iloc[i, col]):
            i += 1
        min_nb = df.iloc[i, col]
        for nb in df.iloc[i + 1:, col]:
            if not pd.isna(nb):
                if nb < min_nb:
                    min_nb = nb
        min_lst.append(min_nb)
    return min_lst


def ft_quartile(df, count_lst, cols, q):
    if q == Q1:
        quartile_lst = ["25%"]
    elif q == Q2:
        quartile_lst = ["50%"]
    elif q == Q3:
        quartile_lst = ["75%"]
    for col, nb_values in zip(range(cols), count_lst):
        nb_lst = valid_numbers(df, col)
        sorted_nb = sorted(nb_lst)
        if q == Q1:
            i_quartile = int(nb_values) // 4
        elif q == Q2:
            i_quartile = int(nb_values) * 2 // 4
        elif q == Q3:
            i_quartile = int(nb_values) * 3 // 4
        if i_quartile % 4 == 0:
            quartile = (sorted_nb[i_quartile - 1] + sorted_nb[i_quartile]) / 2
        else:
            quartile = sorted_nb[i_quartile]
        quartile_lst.append(quartile)
    return quartile_lst


def ft_max(df, cols):
    max_lst = ["Max"]
    for col in range(cols):
        i = 0
        while pd.isna(df.iloc[i, col]):
            i += 1
        max_nb = df.iloc[i, col]
        for nb in df.iloc[i + 1:, col]:
            if not pd.isna(nb):
                if nb > max_nb:
                    max_nb = nb
        max_lst.append(max_nb)
    return max_lst


def ft_nan(df, cols):
    nan_lst = ["Nan"]
    for col in range(cols):
        count = 0
        for nb in df.iloc[:, col]:
            if pd.isna(nb):
                count += 1
        nan_lst.append(count)
    return nan_lst


def analyze_csv(df):
    count_lst = ft_count(df, df.shape[1])
    mean_lst = ft_mean(df, count_lst[1:], df.shape[1])
    std_lst = ft_std(df, mean_lst[1:], df.shape[1])
    min_lst = ft_min(df, df.shape[1])
    q25_lst = ft_quartile(df, count_lst[1:], df.shape[1], Q1)
    q50_lst = ft_quartile(df, count_lst[1:], df.shape[1], Q2)
    q75_lst = ft_quartile(df, count_lst[1:], df.shape[1], Q3)
    max_lst = ft_max(df, df.shape[1])
    nan_lst = ft_nan(df, df.shape[1])

    return [count_lst, mean_lst, std_lst, min_lst, q25_lst, q50_lst, q75_lst, max_lst, nan_lst]


def print_statistics(titles, stats):
    len_titles_lst = [None]
    print(f"{'':>10}", end="")
    for title in titles:
        len_title = len(title) if len(title) > 14 else 14
        len_titles_lst.append(len_title)
        print(f" {title:>{len_title}} ", end="")
    print()
    for stat in stats:
        for data, len_title in zip(stat, len_titles_lst):
            if isinstance(data, str):
                print(f"{data:<10}", end="")
            else:
                data = f"{data:.6f}"
                if len_title == None or len_title <= 14:
                    print(f" {data:>14} ", end="")
                else:
                    print(f" {data:>{len_title}} ", end="")
        print()


def main():
    try:
        av = sys.argv
        if not len(av) == 2:
            raise SystemExit("Wrong number of arguments.")
        df = pd.read_csv(av[1])
        df = pd.concat([df.iloc[:, 0:1], df.iloc[:, 6:]], axis=1)
        stats = analyze_csv(df)
        print_statistics(df.columns, stats)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
