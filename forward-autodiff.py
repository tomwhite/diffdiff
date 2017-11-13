# Autodiff

class Dual(object):
  def __init__(self, a, b = 1):
    self.a = a
    self.b = b
  def diff(self):
    return self.b
  def __mul__(self, other):
    return Dual(self.a * other.a, self.a * other.b + self.b * other.a)
  def __rmul__(self, other):
    return Dual(self.a * other.a, self.a * other.b + self.b * other.a)
  def __repr__(self):
    return "(%s, %s)" % (self.a, self.b)
 
def autodiff(f):
  return lambda x: f(Dual(x)).diff()

def f(x):
	return x * x

fdash = autodiff(f)

print(f(1))
print(fdash(1))