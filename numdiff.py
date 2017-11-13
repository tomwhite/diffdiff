def derivative(f, x, eps):
  return (f(x + eps) - f(x)) / eps

def numdiff(f):
  return lambda x: derivative(f, x, 0.00001)

def f(x):
  return x * x

fdash = numdiff(f)

print(f(1))
print(fdash(1)) # not exactly 2.0