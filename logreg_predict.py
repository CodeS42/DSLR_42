# Objectif
#     -> Utiliser les poids des 4 modeles de regression logistique entraines
#        pour calculer la probabilite d'appartenance d'un eleve a chaque maison
#        et lui attribuer celle qui possede la probabilite la plus elevee.


import sys
import json as js
import pandas as pd
import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid activation function for a given input array or scalar.
    Returns values constrained between 0 and 1.
    """
    return 1 / (1 + np.exp(-z))


def predict():
    """
    Load test data and trained weights
    to compute class probabilities per house.
    Normalizes features and exports
    the predicted house assignments to 'houses.csv'.
    """
    df_test = pd.read_csv(sys.argv[1])
    with open(sys.argv[2]) as f:
        model = js.load(f)
    # Extrait les dictionnaires des moyennes et des ecarts types de chaque matiere
    #     -> Ces chiffres serviront a standardiser les nouvelles notes du fichier de test comme pendant l'entrainement
    feature_means = model["feature_mean"]
    feature_stds = model["feature_std"]
    # Extrait les poids appris par les modeles de regression logistique
    #     -> Ils mesurent l'importance de chaque matiere pour deviner la maison d'un eleve
    thetas = model["thetas"]
    # Creation d'un nouveau dataframe pour enregistrer les notes normalisees
    df_notes = pd.DataFrame()
    # Parcourt une par une toutes les matieres
    for course_name in feature_means.keys():
        # Stocke la moyenne et l'ecart-type de cette matiere
        mean = feature_means[course_name]
        std = feature_stds[course_name]
        # Remplace dans la colonne de la matiere du DF de test les valeurs manquantes par la moyenne enregistree
        notes = df_test[course_name].fillna(mean)
        # Normalise toutes les notes de cette matiere et les stocke dans le nouveau dataframe de notes 
        if std == 0:
            df_notes[course_name] = (notes - mean)
        else:
            df_notes[course_name] = (notes - mean) / std
    # Insere une premiere colonne nommee bias remplie de 1 pour permettre le produit matriciel avec thetas_list
    df_notes.insert(0, "bias", 1)
    probabilities = {}
    # Parcourt les 4 modeles entraines 
    # house: nom de la maison ; theta_list: biais et poids de chaque matiere pour une maison
    for house, theta_list in thetas.items():
        # Multiplie chaque note de l'élève par le poids de la matière correspondante, 
        # y ajoute le biais, puis additionne le tout pour obtenir son score global de compatibilité.
        z = np.dot(df_notes, theta_list)
        # Transforme le score de compatibilite en probabilite comprise entre 0 et 1 et l'enregistre dans un dictionnaire
        probabilities[house] = sigmoid(z)
    # Transforme le dictionnaire en DataFrame
    #     -> 4 colonnes: une pour chaque maison
    #     -> chaque ligne correspond a un eleve, et contient ses 4 probabilites
    df_probabilities = pd.DataFrame(probabilities)
    # Cherche pour chaque ligne du DF de probabilites la probabilite la plus elevee,
    # extrait le nom de la colonne correspondante (la maison) et l'enregistre dans 
    # la colonne de la maison du DF de test
    df_test["Hogwarts House"] = df_probabilities.idxmax(axis=1)
    # Exporte les colonnes de l'index de l'eleve et de sa maison dans un nouveau fichier .csv
    df_test[["Index", "Hogwarts House"]].to_csv("houses.csv", index=False)


def main():
    """
    Validate command-line arguments and trigger the prediction process.
    Handles potential execution errors and exits cleanly.
    """
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
