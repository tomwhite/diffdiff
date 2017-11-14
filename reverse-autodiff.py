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
	def __init__(self, name, parents, op, grad_ops):
		self.name = name
		self.parents = parents
		self.op = op
		self.grad_ops = grad_ops
		self.val = None
		
	def set(self, val):
		self.val = val
		
	def eval(self):
		if self.val is None:
			self.val = self.op(*[p.eval() for p in self.parents])
		return self.val
		
	def __repr__(self):
		return "Variable(name: %s, parents: %s, op: %s, grad_ops: %s, val: %s)" % (self.name, self.parents, self.op, self.grad_ops, self.val)
	
class Graph(object):
	def __init__(self, vars):
		self.vars = vars
			
	def get_operation(self, v):
		return v.op
	
	def get_inputs(self, v):
		return v.parents
	
	def get_consumers(self, v):
		children = []
		for var in self.vars:
			if var.parents != None and v in var.parents:
				children.append(var)
		return children
	
def reverse_autodiff(nodes, inputs, outputs, input_values):
	g = Graph(nodes)
	
	# Forward pass
	
	for (var, val) in zip(inputs, input_values):
		var.set(val)
	for n in outputs:
		n.eval()
	
	# Reverse pass
	
	grad_table = {}
	for n in outputs:
		grad_table[n] = 1 # final output has gradient of 1

	def build_grad(v, g, grad_table):
		if v in grad_table:
			return grad_table[v]
		grad = 0
		for c in g.get_consumers(v):
			d = build_grad(c, g, grad_table)
			grad += sum([grad_op(input.eval()) * d for (grad_op, input) in zip(c.grad_ops, g.get_inputs(c))])
		grad_table[v] = grad
		return grad_table[v]

	for v in inputs:
	  build_grad(v, g, grad_table)
	
	return [grad_table[n] for n in inputs]
	
def autodiff(nodes, inputs, outputs):
	return lambda input_values: reverse_autodiff(nodes, inputs, outputs, input_values)

# tanh(x)
n1 = Variable("n1", None, None, None)
n2 = Variable("n2", (n1, ), lambda x: math.tanh(x), (lambda x: 1 / (math.cosh(x) * math.cosh(x)),))

fdash = autodiff((n1, n2), (n1,), (n2,))
print(fdash((1,)))

# x * tanh(x)
n3 = Variable("n3", None, None, None)
n4 = Variable("n4", (n2, n3), lambda x, y: x * y, (lambda x: x, lambda x: x))

fdash = autodiff((n1, n2, n3, n4), (n1, n3), (n4,))
print(fdash((1, 1)))

# x * x
n1 = Variable("n1", None, None, None)
n2 = Variable("n2", None, None, None)
n3 = Variable("n3", (n1, n2), lambda x, y: x * y, (lambda x: x, lambda x: x))

fdash = autodiff((n1, n2, n3), (n1, n2), (n3,))
print(fdash((1, 1)))
