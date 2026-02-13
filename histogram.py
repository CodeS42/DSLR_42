import matplotlib.pyplot as plt
import pandas as pd
import sys


def calculate_global_means(courses_means):
    global_means = []

    for means in courses_means:
        global_mean = sum(means) / 4
        global_means.append(global_mean)
    
    return global_means


def var(course_means, global_mean):
    return sum([(mean - global_mean) ** 2 for mean in course_means]) / 4


def std(courses_means):
    global_means = calculate_global_means(courses_means)
    std_per_course = []

    for course_means, global_mean in zip(courses_means, global_means):
        var_result = var(course_means, global_mean)
        std_result = var_result ** 0.5
        std_per_course.append(std_result)

    return std_per_course


def means_per_course(marks):
    courses_means = []

    for g_course, s_course, h_course, r_course in zip(marks[0], marks[1], marks[2], marks[3]):
        g_mean = sum(g_course) / len(g_course)
        s_mean = sum(s_course) / len(s_course)
        h_mean = sum(h_course) / len(h_course)
        r_mean = sum(r_course) / len(r_course)
        courses_means.append([g_mean, s_mean, h_mean, r_mean])
    
    return courses_means


def retrieve_marks(df, houses, courses):
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
    
    return [gryffindor_marks, slytherin_marks, hufflepuff_marks, ravenclaw_marks]


def course_smallest_std(std_per_course, courses):
    min_std = std_per_course[0]
    course_name = (courses[0], 0)
    i = 1

    for nb in std_per_course[1:]:
        if nb < min_std:
            min_std = nb
            course_name = (courses[i], i)
        i += 1
    
    return course_name


def display_histogram(smallest_std, marks):
    plt.hist(marks, label=["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"], histtype="stepfilled", alpha=0.5)
    plt.title(smallest_std)
    plt.xlabel("Marks")
    plt.ylabel("Number of Students")
    plt.legend()
    plt.show()


def retrieve_course_marks(marks, course_i):
    return [house[course_i] for house in marks]


def main():
    try:
        if not len(sys.argv) == 1:
            raise SystemExit("Wrong number of arguments.")
        df = pd.read_csv("dataset_train.csv")

        marks = retrieve_marks(df, df.iloc[:, 1], df.iloc[:, 6:].columns)
        std_per_course = std(means_per_course(marks))
        smallest_std = course_smallest_std(std_per_course, df.iloc[:, 6:].columns)
        course_marks = retrieve_course_marks(marks, smallest_std[1])
        display_histogram(smallest_std[0], course_marks)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
