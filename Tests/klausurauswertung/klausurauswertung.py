lines = [line.split(';') for line in open('Punktetabelle.csv', 'r').read().split('\n') if line != '']
print(lines)


def note(punkte):
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