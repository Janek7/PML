FILE = 'Punktetabelle.csv'
SEPARATOR_LENGTH = 100
LFD_KEY = 'Lfd'
GRADE_KEY = 'Note'
POINTS_KEY = 'Pkt.'
EXERCISES_KEY = 'Aufgaben'
MIN_KEY = 'Min'
MAX_KEY = 'Max'
AVG_KEY = 'Mittel'
EXERCISE_PREFIX = 'A'


def create_exams_dict(lines):
    exams = []
    for line in lines:
        exam = {LFD_KEY: line[0], EXERCISES_KEY: [int(points) for points in line[1:]]}
        exam[POINTS_KEY] = sum(exam[EXERCISES_KEY])
        exam[GRADE_KEY] = get_grade(exam[POINTS_KEY])
        exams.append(exam)
    return exams


def get_grade(points):
    nt = {
        (0, 49): 5.0,
        (50, 54): 4.0,
        (55, 59): 3.7,
        (60, 64): 3.3,
        (65, 69): 3.0,
        (70, 74): 2.7,
        (75, 79): 2.3,
        (80, 84): 2.0,
        (85, 89): 1.7,
        (90, 94): 1.3,
        (95, 100): 1.0
    }
    return [nt[i] for i in nt if (i[0] <= points <= i[1])][0]


def exam_stats(exams):
    print('Anzahl der Datensätze:', len(exams))
    print('Datenfelder pro Satz:', len(exams[0]['Aufgaben']))
    print()
    print(LFD_KEY, GRADE_KEY, POINTS_KEY, EXERCISES_KEY)
    for exam in exams:
        print(exam[LFD_KEY] + ':', exam[GRADE_KEY], exam[POINTS_KEY], exam[EXERCISES_KEY])


def exercise_stats(exams):
    exercise_stats = compute_exercise_stats(exams)
    print('Statistik')
    print('Auf: Min Max Mittel')
    for exercise in exercise_stats:
        exercise_statistic = exercise_stats[exercise]
        print(exercise + ': ', exercise_statistic['Min'], exercise_statistic['Max'], exercise_statistic['Mittel'])


def compute_exercise_stats(exams):
    exercise_stats = {}
    for i in range(len(exams[0][EXERCISES_KEY])):
        exercise_statistic = {}
        exercise_points_all_exams = [exam[EXERCISES_KEY][i] for exam in exams]
        exercise_statistic[MIN_KEY] = min(exercise_points_all_exams)
        exercise_statistic[MAX_KEY] = max(exercise_points_all_exams)
        exercise_statistic[AVG_KEY] = round(sum(exercise_points_all_exams) / len(exercise_points_all_exams), 2)
        exercise_stats[EXERCISE_PREFIX + str(i + 1)] = exercise_statistic
    return exercise_stats


def point_stats(exams):
    points = [exam[POINTS_KEY] for exam in exams]
    print(POINTS_KEY, min(points), max(points), round(sum(points) / len(points), 2))


def grade_stats(exams):
    grades = [exam[GRADE_KEY] for exam in exams]
    print(GRADE_KEY, min(grades), max(grades), round(sum(grades) / len(grades), 2))


if __name__ == '__main__':

    # print header
    print('Klausurauswertung')
    print('Datei:', FILE)
    print('*' * SEPARATOR_LENGTH)

    # prepare data
    lines = [line.split(';') for line in open(FILE, 'r').read().split('\n') if line != '']
    header = lines[0]
    data = lines[1:]
    exams = create_exams_dict(data)

    exam_stats(exams)
    print('*' * SEPARATOR_LENGTH)
    print()
    exercise_stats(exams)
    print()
    point_stats(exams)
    print()
    grade_stats(exams)
