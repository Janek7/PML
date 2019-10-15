def bs(l):
    length = len(l)
    for i in range(length):
        for j in range(0, length - i - 1):
            if l[j] > l[j + 1]:
                l[j], l[j + 1] = l[j + 1], l[j]
    return l


# def bs2(l):
    # return [[j for j in range(0, len(l) - i - 1)] for i in range(len(l))]


def qs(l):
    if len(l) <= 1: return l
    return qs([x for x in l[1:] if x < l[0]]) + l[0:1] + qs([x for x in l[1:] if x >= l[0]])

if __name__ == '__main__':
    l = [4, 6, 1, 43, 9, 8]
    print('Quicksort:', qs(l))
    print('Bubblesort:', bs(l))