# Objectif
#   → Repondre a la question : Parmis tous les cours,
#   lequel possede la repartition des notes
#   la plus homogene entre les quatre maisons?

# Pour chaque matiere il faut :
#   - regrouper les notes en fonction des maisons et calculer leur moyenne
#   - calculer l'ecart type des moyennes de chaque maison
#   - voir pour quelle matiere l'ecart-type est le plus bas

import matplotlib.pyplot as plt
import pandas as pd
import sys


def house_means(course_marks):
    """
    Calculate the mean mark of each of the four houses for a single course.
    Returns a list of four float values (one mean per house).
    """
    return [sum(house) / len(house) for house in course_marks]


def var(means, global_mean):
    """
    Calculate the sample variance
        between the four house means for a single course.
    Returns the variance float value.
    """
    return sum([(m - global_mean) ** 2 for m in means]) / (len(means) - 1)


def std(marks):
    """
    Compute the standard deviation of house means for every course.
    Returns a list of standard deviations per course.
    """
    std_per_course = []
    # Recupere le nombre de matieres
    num_courses = len(marks[0])

    for course_i in range(num_courses):
        # Recupere la liste des notes de chaque maison dans une meme matiere
        course_marks = [house[course_i] for house in marks]
        # calcule la moyenne de chaque maison dans ce cours
        #   et renvoie la liste des moyennes
        means = house_means(course_marks)
        # Calcule la moyenne des 4 moyennes pour le calcul de la variance
        #   -> La formule de la variance exige de calculer la difference
        #   entre chaque donnee et la moyenne de l'ensemble de ces donnees
        global_mean = sum(means) / len(means)
        var_result = var(means, global_mean)
        std_result = var_result ** 0.5
        std_per_course.append(std_result)

    return std_per_course


def retrieve_marks(df, houses, courses):
    """
    Extract and group valid numerical marks by house for every course.
    Returns nested lists of marks grouped by house and course.
    """
    # Creer une liste de 4 listes correspondant a chaque maison
    # Chacune de ces quatre listes possede
    #   un nombre de liste egal au nombre de matieres
    # Chacune de ces listes contient
    #   l'ensemble des notes d'une maion dans une matiere
    gryffindor_marks = []
    slytherin_marks = []
    hufflepuff_marks = []
    ravenclaw_marks = []

    for course in courses:
        gryffindor_course_marks = []
        slytherin_course_marks = []
        hufflepuff_course_marks = []
        ravenclaw_course_marks = []
        i = 0

        for mark in df[course]:
            if pd.isna(mark):
                i += 1
                continue
            # Conversion en float pour garantir
            #   que la donnee est exploitable pour les futurs calculs
            mark = float(mark)
            house = houses[i]
            if house == "Gryffindor":
                gryffindor_course_marks.append(mark)
            elif house == "Slytherin":
                slytherin_course_marks.append(mark)
            elif house == "Hufflepuff":
                hufflepuff_course_marks.append(mark)
            elif house == "Ravenclaw":
                ravenclaw_course_marks.append(mark)
            i += 1
        gryffindor_marks.append(gryffindor_course_marks)
        slytherin_marks.append(slytherin_course_marks)
        hufflepuff_marks.append(hufflepuff_course_marks)
        ravenclaw_marks.append(ravenclaw_course_marks)

    return [gryffindor_marks, slytherin_marks,
            hufflepuff_marks, ravenclaw_marks]


def course_smallest_std(std_per_course, courses):
    """
    Identify the course with the lowest standard deviation among house means.
    Returns a tuple containing the course name and its index.
    """
    min_std = std_per_course[0]
    course_name = (courses[0], 0)
    i = 1

    for nb in std_per_course[1:]:
        if nb < min_std:
            min_std = nb
            course_name = (courses[i], i)
        i += 1
    # Renvoie un tupple avec le nom de la matiere
    #   et son index dans la structure marks pour chaque maison
    return course_name


def display_histogram(smallest_std, marks):
    """
    Plot and display a histogram of student marks for each house.
    Renders the plot using Matplotlib.
    """
    plt.hist(marks, label=["Gryffindor", "Slytherin",
                           "Hufflepuff", "Ravenclaw"],
             histtype="stepfilled", alpha=0.5)
    plt.title(smallest_std)
    plt.xlabel("Marks")
    plt.ylabel("Number of Students")
    plt.legend()
    plt.show()


def retrieve_course_marks(marks, course_i):
    """
    Extract the marks of all four houses for a single course index.
    Returns a list of mark lists corresponding to each house.
    """
    return [house[course_i] for house in marks]


def main():
    """
    Main execution flow to load data, compute statistics, and visualize
    the course with the most homogeneous score distribution across houses.
    """
    try:
        if not len(sys.argv) == 1:
            raise SystemExit("Wrong number of arguments.")
        df = pd.read_csv("dataset_train.csv")
        marks = retrieve_marks(df, df.iloc[:, 1], df.iloc[:, 6:].columns)
        std_per_course = std(marks)
        smallest_std = course_smallest_std(
            std_per_course, df.iloc[:, 6:].columns)
        course_marks = retrieve_course_marks(marks, smallest_std[1])
        display_histogram(smallest_std[0], course_marks)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
