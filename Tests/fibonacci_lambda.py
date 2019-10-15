def fib(i):
    return list(map(lambda x, f=lambda x, f:(f(x-1, f) + f(x-2, f)) if x>1 else 1: f(x,f), range(i)))

n = int(input('Fibonacci Liste bis: '))
if n > 0: print(fib(n))
else: print(n, 'ist zu klein')