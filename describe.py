# Objectif
#   → Afficher les statistiques du dataset passe en argument du programme.

# Les statistiques a afficher sont:

# - Count
#   → Nombre de valeurs valides (nombres)
# - Mean
#   → Moyenne des notes
# - Std (Ecart-Type)
#   → Valeur qui represente a quel point les valeurs
#   sont dispersees autour de la moyenne.
#   Plus le chiffre est grand plus elles sont eloignees les une des autres,
#   plus il est petit, plus les chiffres sont homogenes.
# - Min
#   → Valeur minimale
# - 25%
#   → C’est le quartile 1.
#   Cela signifie que 25% des notes sont inferieures cette valeur.
# - 50%
#   → C’est le quartile 2.
#   Cela signifie que 50% des notes sont inferieures cette valeur.
# - 75%
#   → C’est le quartile 3.
#   Cela signifie que 75% des notes sont inferieures cette valeur.
# - Max
#   → Valeur maximale
# - NaN (Bonus)
#   → Nombre de valeurs non valides.


import pandas as pd
import sys


Q1 = 1
Q2 = 2
Q3 = 3


def ft_count(df, cols):
    """
    Count the number of non-null values in each column.
    Returns a list starting with the label 'Count'.
    """
    count_lst = ["Count"]
    for col in range(cols):
        count = 0
        for nb in df.iloc[:, col]:
            if not pd.isna(nb):
                count += 1
        count_lst.append(count)
    return count_lst


def ft_mean(df, count_lst, cols):
    """
    Calculate the arithmetic mean for each column.
    Returns a list starting with the label 'Mean'.
    """
    mean_lst = ["Mean"]
    for col, nb_values in zip(range(cols), count_lst):
        result = sum(nb for nb in df.iloc[:, col]
                     if not pd.isna(nb)) / nb_values
        mean_lst.append(result)
    return mean_lst


def valid_numbers(df, col):
    """
    Extract all non-null numerical values from a specific column.
    Returns a list of valid numbers.
    """
    nb_lst = []
    for nb in df.iloc[:, col]:
        if not pd.isna(nb):
            nb_lst.append(nb)
    return nb_lst


def variance(df, mean_lst, cols):
    """
    Calculate the variance for each column based on its mean.
    Returns a list of variance values.
    """
    var_lst = []
    for col, mean in zip(range(cols), mean_lst):
        nb_lst = valid_numbers(df, col)
        var = sum([(nb - mean) ** 2 for nb in nb_lst]) / (len(nb_lst) - 1)
        var_lst.append(var)
    return var_lst


def ft_std(df, mean_lst, cols):
    """
    Compute the standard deviation for each column.
    Returns a list starting with the label 'Std'.
    """
    # Je calcule la variance pour chaque colonne
    var = variance(df, mean_lst, cols)
    std_lst = ["Std"]
    for v in var:
        # Je calcule la racine carre de chaque variance
        # pour obtenir l'ecart type de chaque colonne
        std_lst.append(v ** 0.5)
    return std_lst


def ft_min(df, cols):
    """
    Find the minimum non-null value for each column.
    Returns a list starting with the label 'Min'.
    """
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
    """
    Compute the specified quartile (25%, 50%, or 75%) for each column.
    Returns a list labeled with the quartile percentage.
    """
    # Percentile (centile en francais)
    # -> Valeur qui permet de diviser
    #   un ensemble de donnees triees en 100 parties egales
    if q == Q1:
        quartile_lst = ["25%"]
        percentile = 0.25
    elif q == Q2:
        quartile_lst = ["50%"]
        percentile = 0.50
    elif q == Q3:
        quartile_lst = ["75%"]
        percentile = 0.75
    for col in range(cols):
        nb_lst = valid_numbers(df, col)
        # pour calculer les quartiles:
        #   il faut que les nombres soient tries en ordre croissant
        sorted_nb = sorted(nb_lst)
        n = len(sorted_nb)
        # n - 1 : index maximal de la liste
        # Calcule la position theorique du quartile dans la liste triee
        index = percentile * (n - 1)
        # lower et upper sont les index
        #   encadrant la position theorique du quartile
        # lower garde la partie entiere (ex: 2.75 devient 2)
        lower = int(index)
        upper = lower + 1
        # Extrait la partie decimale de l'index (ex: 2.75 - 2 = 0.75)
        weight = index - lower

        quartile = sorted_nb[lower] \
            + weight * (sorted_nb[upper] - sorted_nb[lower])

        quartile_lst.append(quartile)
    return quartile_lst


def ft_max(df, cols):
    """
    Find the maximum non-null value for each column.
    Returns a list starting with the label 'Max'.
    """
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
    """
    Count the number of missing (NaN) values in each column.
    Returns a list starting with the label 'Nan'.
    """
    nan_lst = ["Nan"]
    for col in range(cols):
        count = 0
        for nb in df.iloc[:, col]:
            if pd.isna(nb):
                count += 1
        nan_lst.append(count)
    return nan_lst


def analyze_csv(df):
    """
    Compute summary statistics for all columns in the DataFrame.
    Returns a list containing all statistical metrics.
    """
    # Chaque liste contient plusieurs valeurs.
    # Chaque valeur represente la statistique d'une colonne du dataset
    count_lst = ft_count(df, df.shape[1])
    mean_lst = ft_mean(df, count_lst[1:], df.shape[1])
    std_lst = ft_std(df, mean_lst[1:], df.shape[1])
    min_lst = ft_min(df, df.shape[1])
    q25_lst = ft_quartile(df, count_lst[1:], df.shape[1], Q1)
    q50_lst = ft_quartile(df, count_lst[1:], df.shape[1], Q2)
    q75_lst = ft_quartile(df, count_lst[1:], df.shape[1], Q3)
    max_lst = ft_max(df, df.shape[1])
    nan_lst = ft_nan(df, df.shape[1])

    return [count_lst, mean_lst, std_lst, min_lst,
            q25_lst, q50_lst, q75_lst, max_lst, nan_lst]


def print_statistics(titles, stats):
    """
    Format and display the statistical results in a clean table format.
    Prints the headers and rows to standard output.
    """
    # Affiche le tableau ligne par ligne:
    #   - d'abord les titres du haut
    #   - puis chaque ligne qui contient son propre titre
    #       et les valeurs calculees
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
                if len_title is None or len_title <= 14:
                    print(f" {data:>14} ", end="")
                else:
                    print(f" {data:>{len_title}} ", end="")
        print()


def main():
    """
    Entry point of the script. Reads a CSV file from command line arguments
    and triggers data analysis and display.
    """
    try:
        av = sys.argv
        if not len(av) == 2:
            raise SystemExit("Usage: python describe.py <dataset>")
        df = pd.read_csv(av[1])
        # J'isole les colonnes qui ne contiennent que des valeurs numeriques
        df = pd.concat([df.iloc[:, 0:1], df.iloc[:, 6:]], axis=1)
        stats = analyze_csv(df)
        print_statistics(df.columns, stats)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
