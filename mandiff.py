def f(x):
  return x * x

def fdash(x):
  return 2 * x # do the differentiation yourself

print(f(1))
print(fdash(1))

import math
def h(x):
  return math.tanh(x)

def hdash(x):
  return 1 / (math.cosh(x) * math.cosh(x))

print(h(1))
print(hdash(1))