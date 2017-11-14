import ast
import inspect		
def walk(f):
	src = inspect.getsource(f)
	tree = ast.parse(src)
	for node in ast.walk(tree):
		print(node)
	print(ast.dump(tree))

def f(x):
	return x * x
	
#walk(f)

def fdash(x):
	ret = x * x
	
	# Backwards
	df_dn3 = 1.0
	
	df_dn1 = df_dn3 * x
	df_dn2 = df_dn3 * x
	
	df_dx = df_dn1 + df_dn2
	
	return df_dx
	

print(f(1))
print(fdash(1))

import math
def h(x):
	return math.tanh(x)

#walk(h)

class Variable(object):
	def __init__(self, parent, op, grad_op, val=None):
		self.parent = parent
		self.op = op
		self.grad_op = grad_op
		self.val = val
		
	def eval(self):
		if self.op is None:
			return self.val
		else:
			return self.op(self.parent.eval())
		
	def __repr__(self):
		return "Variable(parent: %s, op: %s, grad_op: %s, val: %s)" % (self.parent, self.op, self.grad_op, self.val)
	
class Graph(object):
	def __init__(self, vars):
		self.vars = vars
			
	def get_operation(self, v):
		return v.op
	
	def get_inputs(self, v):
		return (v.parent,) # just a single parent for the moment!
	
	def get_consumers(self, v):
		children = []
		for var in self.vars:
			if var.parent == v:
				children.append(var)
		return children
	
n1 = Variable(None, None, None, 1)
n2 = Variable(n1, lambda x: math.tanh(x), lambda x: 1 / (math.cosh(x) * math.cosh(x)))

g = Graph([n1, n2])

print("Op of n2:",  g.get_operation(n2))
print("Inputs to n2:",  g.get_inputs(n2))
print("Consumers of n1:", g.get_consumers(n1))
print("n1: ", n1.eval())
print("n2: ", n2.eval())
	
grad_table = {}
grad_table[n2] = 1 # final output has gradient of 1

def build_grad(v, g, grad_table):
	if v in grad_table:
		return grad_table[v]
	c = g.get_consumers(v)[0] # just one for the moment!
	op = c.op
	grad_op = c.grad_op
	d = build_grad(c, g, grad_table)
	grad_i = grad_op(g.get_inputs(c)[0].eval()) * d
	grad = grad_i # TODO: sum
	grad_table[v] = grad

for v in [n1]: # just start with one variable, x
  build_grad(v, g, grad_table)

print(grad_table)





